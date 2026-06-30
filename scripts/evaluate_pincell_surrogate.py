"""Rigorous, leakage-free evaluation and honest benchmark for the pin-cell response surrogate.

This script supersedes the in-trainer evaluation in train_pincell_surrogate.py, which
selected the "best" model by its score on the same test split it then reported (selection
on test -> optimistic bias) using a single 24-point split.

What this harness does instead:

1. Repeated holdout: the full dataset is split into a train/val pool and an untouched
   test set, repeated over many seeds. Every reported test number is aggregated as
   mean +/- std across repeats, so single-split luck is removed.
2. Honest model selection: for each repeat and each target, the best model is chosen by
   k-fold cross-validation *on the train/val pool only*. The test set is touched exactly
   once, after selection, to produce the reported metric.
3. Optimism-bias audit: for the same repeats we also compute the old "select on test"
   metric, so the bias of the previous methodology is quantified directly.
4. Baselines: linear, quadratic-ridge, and kNN are evaluated under the identical protocol
   so the gain from the ML models is anchored.
5. Noise-floor deconvolution: because OpenMC labels carry Monte-Carlo uncertainty, the
   intrinsic surrogate RMSE is reported as sqrt(max(0, RMSE_measured^2 - mean_label_var)).
6. Physics sanity: the final model is checked for the correct monotonic response of keff
   to soluble boron (down) and fuel temperature (Doppler, down).
7. Honest speedup: end-to-end ML inference cost (feature engineering + predict) is measured
   for single-case latency and batched throughput, then compared to OpenMC per-case cost,
   including the amortized cost of generating the training data and the resulting
   break-even query count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from train_pincell_surrogate import (
    CANDIDATE_FEATURES,
    ENGINEERED_FEATURES,
    TARGETS,
    add_engineered_features,
    pcm,
    relative_mae,
    resolve_features,
    target_std_values,
)

# Models that cross-validate cleanly (no per-sample alpha that CV cannot subset).
# The Gaussian processes learn their own noise level through a WhiteKernel, which is
# both CV-safe and a more honest treatment of label noise than a hand-fed alpha vector.
BASELINE_NAMES = {"linear", "quad_ridge", "knn"}

# Max support-set size for Gaussian processes; set from --gpr-fit-cap in main().
GPR_CAP = 400

# Fast mode (set from --fast): prune the slow 600-700 tree forests and skip the
# optimism-audit refit pass. The honest core (CV-on-train selection, test-once,
# baselines, repeated holdout, physics checks) is unchanged.
FAST = False
FAST_DROP = {"rf", "extra_trees"}


class SubsampledGPR(BaseEstimator, RegressorMixin):
    """Gaussian process that fits on at most `cap` rows.

    A Gaussian process is O(n^3) to train and O(n_train) per query, so on the larger
    high-stat dataset (~500+ train rows) an exact GP is both slow to fit and slow to
    serve. Capping the support set keeps training tractable and bounds inference cost,
    which is exactly the property a "fast surrogate" needs. The cap is applied
    identically inside every CV fold and the final fit, so the protocol stays honest.
    """

    def __init__(
        self,
        kernel=None,
        cap: int = 400,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 0,
        random_state: int = 0,
    ):
        self.kernel = kernel
        self.cap = cap
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) > self.cap:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.cap, replace=False)
            X, y = X[idx], y[idx]
        self.gpr_ = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.gpr_.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.gpr_.predict(np.asarray(X, dtype=float), return_std=return_std)


def build_candidate_models(random_state: int, gpr_cap: int | None = None) -> dict[str, Pipeline]:
    if gpr_cap is None:
        gpr_cap = GPR_CAP
    return {
        "linear": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "quad_ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
            ]
        ),
        "hgb": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=400,
                        learning_rate=0.05,
                        max_leaf_nodes=24,
                        l2_regularization=1.0e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=2,
                        max_features=0.8,
                        bootstrap=True,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gpr_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SubsampledGPR(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                        + WhiteKernel(noise_level=1.0e-3),
                        cap=gpr_cap,
                        normalize_y=True,
                        n_restarts_optimizer=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gpr_matern": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SubsampledGPR(
                        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                        + WhiteKernel(noise_level=1.0e-3),
                        cap=gpr_cap,
                        normalize_y=True,
                        n_restarts_optimizer=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def selection_metric(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lower is better. keff in pcm, everything else as relative MAE."""
    if target == "keff":
        return float(mean_absolute_error(y_true, y_pred) * 1.0e5)
    return relative_mae(y_true, y_pred)


def test_metrics(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "relative_mae": relative_mae(y_true, y_pred),
    }
    if target == "keff":
        err_pcm = np.abs(pcm(y_pred - y_true))
        out["mae_pcm"] = float(out["mae"] * 1.0e5)
        out["rmse_pcm"] = float(out["rmse"] * 1.0e5)
        out["max_abs_error_pcm"] = float(np.max(err_pcm))
        out["percent_within_300_pcm"] = float(np.mean(err_pcm <= 300.0) * 100.0)
        out["percent_within_500_pcm"] = float(np.mean(err_pcm <= 500.0) * 100.0)
        out["percent_within_1000_pcm"] = float(np.mean(err_pcm <= 1000.0) * 100.0)
    return out


def cv_select(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    n_folds: int,
    random_state: int,
    include_gpr: bool,
) -> tuple[str, dict[str, float]]:
    """Pick the model with the best k-fold CV score on the train pool only."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X = train_df[features]
    y = train_df[target].to_numpy()
    cv_scores: dict[str, float] = {}
    for name, model in build_candidate_models(random_state).items():
        if name in {"gpr_rbf", "gpr_matern"} and not include_gpr:
            continue
        if FAST and name in FAST_DROP:
            continue
        try:
            oof = cross_val_predict(model, X, y, cv=kf, n_jobs=1)
        except Exception:
            continue
        cv_scores[name] = selection_metric(target, y, oof)
    best = min(cv_scores, key=cv_scores.get)
    return best, cv_scores


def run_repeat(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    test_size: float,
    n_folds: int,
    seed: int,
    gpr_max_train_rows: int,
) -> dict:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    include_gpr = len(train_df) <= gpr_max_train_rows
    repeat_result: dict[str, dict] = {}
    for target in targets:
        best_name, cv_scores = cv_select(
            train_df, features, target, n_folds, seed, include_gpr
        )
        # Honest: refit selected model on full train pool, score once on untouched test.
        model = build_candidate_models(seed)[best_name]
        model.fit(train_df[features], train_df[target].to_numpy())
        pred = model.predict(test_df[features])
        honest = test_metrics(target, test_df[target].to_numpy(), pred)

        # Optimism audit: the OLD methodology picked the model that minimizes the test
        # metric, then reported that same metric. Reproduce it for direct comparison.
        # Skipped in --fast (already quantified on the 120-case dataset).
        old_best_metric = None
        old_best_name = None
        if not FAST:
            for name in build_candidate_models(seed):
                if name in {"gpr_rbf", "gpr_matern"} and not include_gpr:
                    continue
                if name in BASELINE_NAMES:
                    continue  # old method never had baselines; compare ML-pool to ML-pool
                m = build_candidate_models(seed)[name]
                m.fit(train_df[features], train_df[target].to_numpy())
                p = m.predict(test_df[features])
                score = selection_metric(target, test_df[target].to_numpy(), p)
                if old_best_metric is None or score < old_best_metric:
                    old_best_metric = score
                    old_best_name = name

        # baseline-only honest reference (best of linear/quad/knn under same protocol)
        baseline_best = min(
            (v for k, v in cv_scores.items() if k in BASELINE_NAMES),
            default=float("nan"),
        )

        repeat_result[target] = {
            "selected_model": best_name,
            "cv_scores": cv_scores,
            "honest_test": honest,
            "honest_selection_metric_on_test": selection_metric(
                target, test_df[target].to_numpy(), pred
            ),
            "old_select_on_test_model": old_best_name,
            "old_select_on_test_metric": old_best_metric,
            "baseline_best_cv_metric": baseline_best,
        }
        if target == "keff":
            std = target_std_values(test_df, "keff")
            if std is not None:
                repeat_result[target]["test_label_mean_var_pcm2"] = float(
                    np.mean((std * 1.0e5) ** 2)
                )
    return repeat_result


def aggregate(repeats: list[dict], targets: list[str]) -> dict:
    agg: dict[str, dict] = {}
    for target in targets:
        honest_mae = []
        honest_rmse = []
        honest_relmae = []
        honest_r2 = []
        within_500 = []
        within_1000 = []
        old_metric = []
        honest_sel_metric = []
        baseline_metric = []
        selected = []
        label_var = []
        for rep in repeats:
            r = rep[target]
            h = r["honest_test"]
            honest_relmae.append(h["relative_mae"])
            honest_r2.append(h["r2"])
            honest_sel_metric.append(r["honest_selection_metric_on_test"])
            if r["old_select_on_test_metric"] is not None:
                old_metric.append(r["old_select_on_test_metric"])
            baseline_metric.append(r["baseline_best_cv_metric"])
            selected.append(r["selected_model"])
            if target == "keff":
                honest_mae.append(h["mae_pcm"])
                honest_rmse.append(h["rmse_pcm"])
                within_500.append(h["percent_within_500_pcm"])
                within_1000.append(h["percent_within_1000_pcm"])
                if "test_label_mean_var_pcm2" in r:
                    label_var.append(r["test_label_mean_var_pcm2"])

        entry: dict = {
            "selected_model_counts": {m: selected.count(m) for m in set(selected)},
            "relative_mae_mean": float(np.mean(honest_relmae)),
            "relative_mae_std": float(np.std(honest_relmae)),
            "r2_mean": float(np.mean(honest_r2)),
            "honest_selection_metric_mean": float(np.mean(honest_sel_metric)),
            "old_select_on_test_metric_mean": float(np.mean(old_metric)) if old_metric else None,
            "baseline_best_metric_mean": float(np.mean(baseline_metric)),
        }
        # quantify optimism + baseline gain
        entry["optimism_bias_metric"] = (
            float(np.mean(honest_sel_metric) - np.mean(old_metric)) if old_metric else None
        )
        entry["honest_vs_baseline_ratio"] = (
            float(np.mean(honest_sel_metric) / np.mean(baseline_metric))
            if np.mean(baseline_metric) > 0
            else float("nan")
        )
        if target == "keff":
            entry["mae_pcm_mean"] = float(np.mean(honest_mae))
            entry["mae_pcm_std"] = float(np.std(honest_mae))
            entry["rmse_pcm_mean"] = float(np.mean(honest_rmse))
            entry["rmse_pcm_std"] = float(np.std(honest_rmse))
            entry["percent_within_500_pcm_mean"] = float(np.mean(within_500))
            entry["percent_within_1000_pcm_mean"] = float(np.mean(within_1000))
            if label_var:
                mean_label_var = float(np.mean(label_var))
                mean_rmse = float(np.mean(honest_rmse))
                intrinsic = float(np.sqrt(max(0.0, mean_rmse**2 - mean_label_var)))
                entry["label_noise_rmse_pcm"] = float(np.sqrt(mean_label_var))
                entry["intrinsic_surrogate_rmse_pcm"] = intrinsic
        agg[target] = entry
    return agg


def physics_monotonicity(
    df: pd.DataFrame, features: list[str], seed: int, gpr_max_train_rows: int
) -> dict:
    """Train on all data with the keff-selected model and check known monotonic trends."""
    include_gpr = len(df) <= gpr_max_train_rows
    best_name, _ = cv_select(df, features, "keff", 5, seed, include_gpr)
    model = build_candidate_models(seed)[best_name]
    model.fit(df[features], df["keff"].to_numpy())

    raw_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    nominal = {c: float(df[c].median()) for c in raw_cols}

    def sweep(var: str, lo: float, hi: float, n: int = 25) -> np.ndarray:
        rows = []
        for v in np.linspace(lo, hi, n):
            r = dict(nominal)
            r[var] = float(v)
            rows.append(r)
        grid = add_engineered_features(pd.DataFrame(rows))
        return model.predict(grid[features])

    checks = {}
    if "boron_ppm" in df.columns and df["boron_ppm"].nunique() > 1:
        k = sweep("boron_ppm", df["boron_ppm"].min(), df["boron_ppm"].max())
        slope = float(np.polyfit(range(len(k)), k, 1)[0])
        checks["keff_vs_boron"] = {
            "expected": "decreasing",
            "slope_sign": "negative" if slope < 0 else "positive",
            "delta_keff_pcm_full_range": float((k[-1] - k[0]) * 1.0e5),
            "monotonic_ok": bool(slope < 0),
        }
    if "fuel_temperature_K" in df.columns and df["fuel_temperature_K"].nunique() > 1:
        k = sweep(
            "fuel_temperature_K",
            df["fuel_temperature_K"].min(),
            df["fuel_temperature_K"].max(),
        )
        slope = float(np.polyfit(range(len(k)), k, 1)[0])
        checks["keff_vs_fuel_temperature_doppler"] = {
            "expected": "decreasing",
            "slope_sign": "negative" if slope < 0 else "positive",
            "delta_keff_pcm_full_range": float((k[-1] - k[0]) * 1.0e5),
            "monotonic_ok": bool(slope < 0),
        }
    if "enrichment_wt" in df.columns and df["enrichment_wt"].nunique() > 1:
        k = sweep("enrichment_wt", df["enrichment_wt"].min(), df["enrichment_wt"].max())
        slope = float(np.polyfit(range(len(k)), k, 1)[0])
        checks["keff_vs_enrichment"] = {
            "expected": "increasing",
            "slope_sign": "positive" if slope > 0 else "negative",
            "delta_keff_pcm_full_range": float((k[-1] - k[0]) * 1.0e5),
            "monotonic_ok": bool(slope > 0),
        }
    checks["model_used"] = best_name
    return checks


def honest_speedup(
    df: pd.DataFrame, features: list[str], seed: int, gpr_max_train_rows: int
) -> dict:
    """Measure real end-to-end ML cost and compare to OpenMC, including amortized training."""
    include_gpr = len(df) <= gpr_max_train_rows
    best_name, _ = cv_select(df, features, "keff", 5, seed, include_gpr)
    model = build_candidate_models(seed)[best_name]
    model.fit(df[features], df["keff"].to_numpy())

    raw_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    base = df[raw_cols].sample(1, random_state=seed).iloc[0].to_dict()

    # Single-case latency: includes feature engineering from raw inputs each call.
    n_single = 200
    t0 = time.perf_counter()
    for _ in range(n_single):
        one = add_engineered_features(pd.DataFrame([base]))
        model.predict(one[features])
    single_latency = (time.perf_counter() - t0) / n_single

    # Batched throughput: one feature-engineering + predict call over a large design set.
    n_batch = 20000
    rng = np.random.default_rng(seed)
    batch_raw = pd.DataFrame(
        {c: rng.uniform(df[c].min(), df[c].max(), n_batch) for c in raw_cols}
    )
    t1 = time.perf_counter()
    batch_feat = add_engineered_features(batch_raw)
    model.predict(batch_feat[features])
    batch_total = time.perf_counter() - t1
    batch_per_case = batch_total / n_batch

    openmc_per_case = float(df["openmc_elapsed_seconds"].replace(0.0, np.nan).mean())
    n_train = len(df)
    training_data_cost_s = n_train * openmc_per_case

    # Break-even: total cost to answer Q novel queries.
    #   OpenMC:    Q * openmc_per_case
    #   Surrogate: training_data_cost + Q * batch_per_case
    # Equal when Q* = training_data_cost / (openmc_per_case - batch_per_case)
    denom = openmc_per_case - batch_per_case
    break_even_queries = training_data_cost_s / denom if denom > 0 else float("inf")

    return {
        "keff_model": best_name,
        "openmc_seconds_per_case": openmc_per_case,
        "ml_single_case_latency_seconds": single_latency,
        "ml_batched_seconds_per_case": batch_per_case,
        "ml_batched_throughput_cases_per_second": 1.0 / batch_per_case,
        "naive_speedup_single_case": openmc_per_case / single_latency,
        "batched_speedup_per_case": openmc_per_case / batch_per_case,
        "training_data_n_cases": n_train,
        "training_data_cost_seconds": training_data_cost_s,
        "break_even_query_count": break_even_queries,
        "amortized_note": (
            "Surrogate is net-faster than OpenMC only after ~break_even_query_count "
            "distinct evaluations, because the training set itself costs "
            "training_data_cost_seconds of OpenMC. Beyond that, marginal cost is "
            "ml_batched_seconds_per_case per query."
        ),
    }


def save_final_models(df, features, targets, agg, base_seed, test_size, name):
    """Train the CV-preferred model per target on a train split and save model + honest figures.

    The selected model per target is the one most often chosen by CV across the repeats, so the
    saved artifact matches the honestly-reported metrics (no select-on-test).
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=base_seed)
    chosen = {}
    bundle_models = {}
    comparison = test_df.copy()
    for target in targets:
        counts = agg[target]["selected_model_counts"]
        sel = max(counts, key=counts.get)
        chosen[target] = sel
        model = build_candidate_models(base_seed)[sel]
        model.fit(train_df[features], train_df[target].to_numpy())
        bundle_models[target] = model
        comparison[f"ml_{target}"] = model.predict(test_df[features])
    joblib.dump(
        {"features": features, "targets": targets, "selected_models": chosen, "models": bundle_models},
        MODEL_DIR / f"{name}_response_surrogate_best.joblib",
    )
    comparison.to_csv(PROCESSED_DATA_DIR / f"{name}_surrogate_comparison.csv", index=False)

    # keff parity figure with OpenMC Monte-Carlo error bars
    plt.figure(figsize=(5.8, 5.4))
    plt.errorbar(comparison["keff"], comparison["ml_keff"],
                 xerr=comparison.get("keff_std"), fmt="o", ms=4, capsize=2, alpha=0.7)
    lo = min(comparison["keff"].min(), comparison["ml_keff"].min())
    hi = max(comparison["keff"].max(), comparison["ml_keff"].max())
    plt.plot([lo, hi], [lo, hi], "k-", lw=1.0)
    k = agg["keff"]
    plt.xlabel("OpenMC keff"); plt.ylabel("ML surrogate keff")
    plt.title(f"Pin-cell keff: {chosen['keff']}\n"
              f"MAE {k['mae_pcm_mean']:.0f}±{k['mae_pcm_std']:.0f} pcm "
              f"(honest, {len(test_df)}-pt test)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_keff_parity.png", dpi=220)
    plt.close()

    # multi-output parity grid
    plot_targets = [t for t in targets if t in comparison.columns][:8]
    ncols = 4
    nrows = int(np.ceil(len(plot_targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 3.2 * nrows), squeeze=False)
    for ax, t in zip(axes.ravel(), plot_targets):
        ax.scatter(comparison[t], comparison[f"ml_{t}"], s=18, alpha=0.7)
        lo = min(comparison[t].min(), comparison[f"ml_{t}"].min())
        hi = max(comparison[t].max(), comparison[f"ml_{t}"].max())
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.9)
        ax.set_title(f"{t}\n({chosen[t]}, {agg[t]['relative_mae_mean']*100:.2f}% relMAE)", fontsize=8)
    for ax in axes.ravel()[len(plot_targets):]:
        ax.axis("off")
    fig.suptitle("Multi-output pin-cell surrogate vs OpenMC (honest CV-selected models)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_multioutput_parity.png", dpi=220)
    plt.close(fig)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=str(PROCESSED_DATA_DIR / "pincell_lhs120_highstat_openmc.csv")
    )
    parser.add_argument("--name", default="pincell_rigorous")
    parser.add_argument("--n-repeats", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument(
        "--gpr-max-train-rows",
        type=int,
        default=100000,
        help="Include Gaussian-process candidates only when train pool <= this.",
    )
    parser.add_argument(
        "--gpr-fit-cap",
        type=int,
        default=400,
        help="Max support-set size for Gaussian processes (subsampled if exceeded).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Prune slow tree forests and skip the optimism-audit pass (talk-grade, ~minutes).",
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Train the CV-preferred model per target and save a deployable bundle + parity figures.",
    )
    args = parser.parse_args()

    global GPR_CAP, FAST
    GPR_CAP = args.gpr_fit_cap
    FAST = args.fast

    ensure_project_dirs()
    df = add_engineered_features(pd.read_csv(args.dataset))
    features = resolve_features(df)
    targets = [t for t in TARGETS if t in df.columns]

    print(
        f"Dataset: {args.dataset}\n"
        f"  rows={len(df)}  features={len(features)}  targets={len(targets)}\n"
        f"  protocol: {args.n_repeats} repeats x {args.n_folds}-fold CV selection, "
        f"test_size={args.test_size}"
    )

    repeats = []
    for r in range(args.n_repeats):
        seed = args.base_seed + r
        print(f"  repeat {r + 1}/{args.n_repeats} (seed={seed}) ...", flush=True)
        repeats.append(
            run_repeat(
                df, features, targets, args.test_size, args.n_folds, seed,
                args.gpr_max_train_rows,
            )
        )

    agg = aggregate(repeats, targets)
    physics = physics_monotonicity(df, features, args.base_seed, args.gpr_max_train_rows)
    speed = honest_speedup(df, features, args.base_seed, args.gpr_max_train_rows)

    report = {
        "dataset": args.dataset,
        "n_rows": int(len(df)),
        "features": features,
        "targets": targets,
        "protocol": {
            "n_repeats": args.n_repeats,
            "n_folds": args.n_folds,
            "test_size": args.test_size,
            "model_selection": "k-fold CV on train pool only; test touched once after selection",
        },
        "aggregate": agg,
        "physics_monotonicity": physics,
        "honest_speedup": speed,
    }
    if args.save_final:
        chosen = save_final_models(
            df, features, targets, agg, args.base_seed, args.test_size, args.name
        )
        report["final_saved_models"] = chosen
        print(f"\nSaved deployable surrogate bundle + parity figures ({args.name}). Models: {chosen}")

    out_path = MODEL_DIR / f"{args.name}_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2))

    k = agg["keff"]
    print("\n================ HONEST keff RESULT ================")
    print(f"  selected models across repeats: {k['selected_model_counts']}")
    print(f"  test MAE   = {k['mae_pcm_mean']:.1f} +/- {k['mae_pcm_std']:.1f} pcm")
    print(f"  test RMSE  = {k['rmse_pcm_mean']:.1f} +/- {k['rmse_pcm_std']:.1f} pcm")
    print(f"  within 500 pcm  = {k['percent_within_500_pcm_mean']:.1f}%")
    print(f"  within 1000 pcm = {k['percent_within_1000_pcm_mean']:.1f}%")
    if "intrinsic_surrogate_rmse_pcm" in k:
        print(
            f"  label-noise RMSE = {k['label_noise_rmse_pcm']:.1f} pcm  ->  "
            f"intrinsic surrogate RMSE = {k['intrinsic_surrogate_rmse_pcm']:.1f} pcm"
        )
    if k.get("old_select_on_test_metric_mean") is not None:
        print(
            f"  optimism bias of old 'select-on-test' method: "
            f"{k['old_select_on_test_metric_mean']:.1f} pcm reported vs "
            f"{k['honest_selection_metric_mean']:.1f} pcm honest "
            f"(gap {abs(k['optimism_bias_metric']):.1f} pcm)"
        )
    print(
        f"  best linear/quad/knn baseline = {k['baseline_best_metric_mean']:.1f} pcm; "
        f"ML is {1.0 / k['honest_vs_baseline_ratio']:.2f}x better"
    )
    print("\n================ PHYSICS CHECKS ====================")
    for name, c in physics.items():
        if isinstance(c, dict) and "monotonic_ok" in c:
            print(
                f"  {name}: {c['slope_sign']} ({c['delta_keff_pcm_full_range']:+.0f} pcm), "
                f"{'OK' if c['monotonic_ok'] else 'VIOLATION'}"
            )
    print("\n================ HONEST SPEEDUP ====================")
    print(f"  OpenMC per case          = {speed['openmc_seconds_per_case']:.3f} s")
    print(f"  ML single-case latency   = {speed['ml_single_case_latency_seconds']*1e3:.3f} ms")
    print(f"  ML batched per case      = {speed['ml_batched_seconds_per_case']*1e6:.2f} us")
    print(f"  batched speedup per case = {speed['batched_speedup_per_case']:.0f}x")
    print(
        f"  break-even after         = {speed['break_even_query_count']:.0f} queries "
        f"(training data cost {speed['training_data_cost_seconds']:.0f} s)"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
