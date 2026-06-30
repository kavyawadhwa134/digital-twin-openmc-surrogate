"""Train and evaluate a surrogate for 2-group assembly group constants.

Takes the 200-case (60 + 140) assembly dataset from run_assembly_sweep.py and trains
multi-output ML models that map assembly state → 2-group homogenised cross sections
(D1, D2, Sa1, Sa2, nuSf1, nuSf2, Sf1, Sf2, Ss1→1, Ss1→2, Ss2→1, Ss2→2, chi1, chi2, k_inf).

This is the ML bridge in the two-step lattice-to-core method:
    OpenMC assembly  →  group constants  →  nodal diffusion core solve
    [4s / case]          [surrogate]         [milliseconds]

Model selection uses the same leakage-free repeated-holdout protocol as the
pin-cell surrogate: k-fold CV on the training pool only, test touched once.

Outputs:
    models/assembly_groupconst_surrogate.joblib   — saved surrogate bundle
    models/assembly_groupconst_evaluation.json    — metrics per target
    figures/assembly_groupconst_parity.png        — parity grid (D, Sa, nuSf, k_inf)
    figures/assembly_kinf_parity.png              — k_inf only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import clone, BaseEstimator, RegressorMixin

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

INPUTS = [
    "fuel_temperature_K", "enrichment_wt", "moderator_density_g_cm3",
    "moderator_temperature_K", "boron_ppm", "fuel_radius_cm",
    "pin_pitch_cm", "cladding_thickness_cm",
]

GC_TARGETS = [
    "D1", "D2", "Sa1", "Sa2",
    "nuSf1", "nuSf2", "Sf1", "Sf2",
    "Ss1to1", "Ss1to2", "Ss2to1", "Ss2to2",
    "chi1",
    "k_inf",
]

PHYSICAL_CHECKS = {
    "D1 > D2 (fast diffusion > thermal)": lambda df: (df.D1 > df.D2).mean(),
    "Sa2 > Sa1 (thermal absorption > fast)": lambda df: (df.Sa2 > df.Sa1).mean(),
    "nuSf2 > nuSf1 (thermal nu-fission > fast)": lambda df: (df.nuSf2 > df.nuSf1).mean(),
    "Ss1to2 > Ss2to1 (downscatter > upscatter)": lambda df: (df.Ss1to2 > df.Ss2to1).mean(),
    "chi1 > 0.99 (fission births in fast group)": lambda df: (df.chi1 > 0.99).mean(),
}


class SubsampledGPR(BaseEstimator, RegressorMixin):
    """GPR with a capped support set for O(n^2) tractability."""
    def __init__(self, kernel=None, cap=300, random_state=42):
        self.kernel = kernel
        self.cap = cap
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.cap, n), replace=False)
        kernel = self.kernel or (Matern(nu=2.5) + WhiteKernel())
        self._gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        self._gpr.fit(X[idx], y[idx])
        return self

    def predict(self, X):
        return self._gpr.predict(X)

    def get_params(self, deep=True):
        return {"kernel": self.kernel, "cap": self.cap, "random_state": self.random_state}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


def build_models(random_state=0):
    return {
        "quad_ridge": Pipeline([
            ("sc", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=0.1)),
        ]),
        "knn": Pipeline([
            ("sc", StandardScaler()),
            ("m", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "hgb": Pipeline([
            ("sc", StandardScaler()),
            ("m", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.8, random_state=random_state)),
        ]),
        "rf": Pipeline([
            ("sc", StandardScaler()),
            ("m", RandomForestRegressor(
                n_estimators=200, max_features="sqrt",
                random_state=random_state, n_jobs=-1)),
        ]),
        "gpr_matern": Pipeline([
            ("sc", StandardScaler()),
            ("m", SubsampledGPR(kernel=Matern(nu=2.5) + WhiteKernel(), cap=300,
                                random_state=random_state)),
        ]),
        "gpr_rbf": Pipeline([
            ("sc", StandardScaler()),
            ("m", SubsampledGPR(kernel=RBF() + WhiteKernel(), cap=300,
                                random_state=random_state)),
        ]),
    }


def evaluate_target(X_train, y_train, X_test, y_test, n_folds=3, n_repeats=3):
    """Leakage-free: select model by CV on train only, evaluate once on test."""
    best_model_name = None
    best_cv_score = np.inf
    model_scores = {}

    for rep in range(n_repeats):
        models = build_models(random_state=rep * 17)
        for name, pipe in models.items():
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep * 100)
            neg_mae = cross_val_score(pipe, X_train, y_train, cv=kf,
                                      scoring="neg_mean_absolute_error", n_jobs=1)
            score = float(-neg_mae.mean())
            model_scores[name] = model_scores.get(name, []) + [score]

    mean_scores = {k: np.mean(v) for k, v in model_scores.items()}
    best_model_name = min(mean_scores, key=mean_scores.get)

    # Train best model on full training pool, evaluate once on test
    best_pipe = build_models(random_state=0)[best_model_name]
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    mae = float(np.mean(np.abs(y_pred - y_test)))
    rel_mae = float(mae / (np.mean(np.abs(y_test)) + 1e-30))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    ss_res = np.sum((y_pred - y_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-30))

    return best_model_name, best_pipe, y_pred, {
        "selected_model": best_model_name,
        "mae": mae,
        "rel_mae_pct": rel_mae * 100,
        "rmse": rmse,
        "r2": r2,
        "cv_scores": {k: float(np.mean(v)) for k, v in model_scores.items()},
    }


def make_parity_plot(results, X_test, df_test, outpath, title="Assembly group-constant surrogate"):
    """4×4 parity grid for key group constants."""
    plot_targets = ["k_inf", "D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2", "Ss1to2"]
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7))
    axes = axes.flatten()

    for ax, tgt in zip(axes, plot_targets):
        if tgt not in results:
            ax.set_visible(False)
            continue
        y_true = df_test[tgt].values
        y_pred = results[tgt]["y_pred"]
        model = results[tgt]["metrics"]["selected_model"]
        rel_mae = results[tgt]["metrics"]["rel_mae_pct"]
        ax.scatter(y_true, y_pred, s=18, alpha=0.7, color="#1e6091", edgecolors="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
        ax.set_xlabel(f"OpenMC {tgt}", fontsize=8)
        ax.set_ylabel(f"Surrogate {tgt}", fontsize=8)
        ax.set_title(f"{tgt}\n({model}, {rel_mae:.2f}% relMAE)", fontsize=8.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def make_kinf_parity(results, df_test, outpath):
    """Dedicated k_inf parity plot."""
    y_true = df_test["k_inf"].values
    y_pred = results["k_inf"]["y_pred"]
    model = results["k_inf"]["metrics"]["selected_model"]
    mae_pcm = results["k_inf"]["metrics"]["mae"] * 1e5
    rel_mae = results["k_inf"]["metrics"]["rel_mae_pct"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=25, alpha=0.75, color="#1e6091", edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.04
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=1.0)
    ax.set_xlabel("OpenMC k∞", fontsize=11)
    ax.set_ylabel("ML surrogate k∞", fontsize=11)
    ax.set_title(
        f"Assembly k∞ surrogate ({model})\n"
        f"MAE {mae_pcm:.0f} pcm  |  {rel_mae:.2f}% relMAE  |  "
        f"{len(y_true)}-pt honest test",
        fontsize=10,
    )
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(PROCESSED_DATA_DIR / "assembly_groupconst_500.csv"),
                   help="Assembly group-constant CSV (default: 500-case high-fidelity set)")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=3)
    args = p.parse_args()

    ensure_project_dirs()
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} assembly cases from {args.data}")

    # Physical consistency check on full dataset
    print("\n=== Physical consistency ===")
    for desc, fn in PHYSICAL_CHECKS.items():
        pct = fn(df) * 100
        print(f"  {pct:.1f}%  {desc}")

    # Train/test split
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(df) * args.test_frac))
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)

    X_train = df_train[INPUTS].values.astype(float)
    X_test = df_test[INPUTS].values.astype(float)

    print(f"\nTrain: {len(df_train)} | Test: {len(df_test)} | Inputs: {len(INPUTS)} | Targets: {len(GC_TARGETS)}")

    results = {}
    summary_rows = []
    t_total = time.perf_counter()

    for tgt in GC_TARGETS:
        y_train = df_train[tgt].values.astype(float)
        y_test = df_test[tgt].values.astype(float)
        t0 = time.perf_counter()
        model_name, pipe, y_pred, metrics = evaluate_target(
            X_train, y_train, X_test, y_test,
            n_folds=args.n_folds, n_repeats=args.n_repeats,
        )
        elapsed = time.perf_counter() - t0
        results[tgt] = {"model": pipe, "y_pred": y_pred, "metrics": metrics}

        print(f"  {tgt:12s}  {model_name:14s}  relMAE={metrics['rel_mae_pct']:.3f}%  "
              f"R²={metrics['r2']:.4f}  [{elapsed:.0f}s]")
        summary_rows.append({
            "target": tgt,
            "model": model_name,
            "rel_mae_pct": metrics["rel_mae_pct"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        })

    print(f"\nTotal training time: {time.perf_counter() - t_total:.0f}s")

    # Surrogate inference speedup
    openmc_s_per_case = 4.7  # from existing dataset
    X_all = df[INPUTS].values.astype(float)
    t0 = time.perf_counter()
    for _ in range(1000):
        results["k_inf"]["model"].predict(X_all[:1])
    ml_latency = (time.perf_counter() - t0) / 1000
    print(f"\nSurrogate k_inf latency: {ml_latency*1000:.2f} ms/case")
    print(f"Speedup vs OpenMC: {openmc_s_per_case / ml_latency:.0f}×")

    # Save joblib bundle
    bundle = {tgt: results[tgt]["model"] for tgt in GC_TARGETS}
    bundle["inputs"] = INPUTS
    bundle["targets"] = GC_TARGETS
    bundle_path = MODEL_DIR / "assembly_groupconst_surrogate.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"\nSaved surrogate bundle → {bundle_path}")

    # Save metrics JSON
    eval_out = {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "protocol": f"{args.n_repeats}-repeat {args.n_folds}-fold CV on train pool; test touched once",
        "openmc_s_per_case": openmc_s_per_case,
        "ml_latency_ms": ml_latency * 1000,
        "speedup": openmc_s_per_case / ml_latency,
        "targets": {tgt: results[tgt]["metrics"] for tgt in GC_TARGETS},
        "physical_consistency_pct": {
            desc: float(fn(df) * 100) for desc, fn in PHYSICAL_CHECKS.items()
        },
    }
    eval_path = MODEL_DIR / "assembly_groupconst_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Saved evaluation → {eval_path}")

    # Figures
    make_parity_plot(results, X_test, df_test,
                     FIGURE_DIR / "assembly_groupconst_parity.png")
    make_kinf_parity(results, df_test,
                     FIGURE_DIR / "assembly_kinf_parity.png")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Target':<18} {'Model':<16} {'RelMAE %':>10} {'R²':>8}")
    print("-" * 56)
    for row in summary_rows:
        print(f"{row['target']:<18} {row['model']:<16} {row['rel_mae_pct']:>10.3f} {row['r2']:>8.4f}")


if __name__ == "__main__":
    main()
