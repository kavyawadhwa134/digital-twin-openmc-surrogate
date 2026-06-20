from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import openmc.data
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_config import (
    DEFAULT_CROSS_SECTIONS,
    FIGURE_DIR,
    MODEL_DIR,
    NUCLIDE_PROPERTIES,
    PROCESSED_DATA_DIR,
    ensure_project_dirs,
)


FEATURES = [
    "log10_energy_eV",
    "temperature_K",
    "nuclide_Z",
    "nuclide_A",
    "nuclide_N",
    "nuclide_is_actinide",
    "nuclide_is_fissile",
    "nuclide_family",
    "reaction_type",
]
TARGET = "log10_xs_barns"
EPS = 1.0e-30


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                [
                    "log10_energy_eV",
                    "temperature_K",
                    "nuclide_Z",
                    "nuclide_A",
                    "nuclide_N",
                    "nuclide_is_actinide",
                    "nuclide_is_fissile",
                ],
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["nuclide_family", "reaction_type"],
            ),
        ],
        remainder="drop",
    )


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "hgb": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=220,
                        learning_rate=0.08,
                        max_leaf_nodes=45,
                        l2_regularization=1.0e-4,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 128, 64),
                        activation="relu",
                        alpha=1.0e-4,
                        batch_size=512,
                        learning_rate_init=1.0e-3,
                        max_iter=260,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=random_state,
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


def invert_log_xs(values: np.ndarray) -> np.ndarray:
    return np.power(10.0, values)


def relative_error(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    y_true = invert_log_xs(y_true_log)
    y_pred = invert_log_xs(y_pred_log)
    return np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), EPS)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rel = relative_error(y_true, y_pred)
    return {
        "rmse_log10": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log10": float(mean_absolute_error(y_true, y_pred)),
        "r2_log10": float(r2_score(y_true, y_pred)),
        "median_relative_error": float(np.median(rel)),
        "mean_relative_error": float(np.mean(rel)),
        "p95_relative_error": float(np.percentile(rel, 95)),
        "max_relative_error": float(np.max(rel)),
    }


def split_random(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["reaction_type"]
    )


def split_holdout_values(
    df: pd.DataFrame, column: str, holdout_values: list[Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_set = {v for v in holdout_values if v in set(df[column].unique())}
    if not holdout_set:
        raise ValueError(f"No holdout values from {holdout_values} were present in {column}")
    test_df = df[df[column].isin(holdout_set)].copy().reset_index(drop=True)
    train_df = df[~df[column].isin(holdout_set)].copy().reset_index(drop=True)
    return train_df, test_df


def train_and_score_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    save_dir: Path,
    include_mlp: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, Pipeline], str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    models = build_models(random_state)
    if not include_mlp:
        models.pop("mlp", None)
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f"Training {name} on {len(train_df):,} rows...")
        t0 = time.perf_counter()
        model.fit(train_df[FEATURES], train_df[TARGET])
        train_seconds = time.perf_counter() - t0
        pred = model.predict(test_df[FEATURES])
        model_metrics = metrics(test_df[TARGET].to_numpy(), pred)
        model_metrics["train_seconds"] = float(train_seconds)
        model_metrics["train_rows"] = int(len(train_df))
        model_metrics["test_rows"] = int(len(test_df))
        results[name] = model_metrics
        fitted[name] = model
        joblib.dump(model, save_dir / f"xs_surrogate_{name}.joblib")
        print(name, json.dumps(model_metrics, indent=2))

    best_name = min(results, key=lambda n: results[n]["p95_relative_error"])
    joblib.dump(fitted[best_name], save_dir / "xs_surrogate_best.joblib")
    return results, fitted, best_name


def load_openmc_reaction(
    cross_sections_xml: Path, nuclide: str, mt: int
) -> openmc.data.Reaction:
    root = ET.parse(cross_sections_xml).getroot()
    for lib in root.findall("library"):
        if lib.get("type") == "neutron" and nuclide in (lib.get("materials") or "").split():
            path = Path(lib.get("path"))
            if not path.is_absolute():
                path = cross_sections_xml.parent / path
            data = openmc.data.IncidentNeutron.from_hdf5(path)
            return data.reactions[mt]
    raise KeyError(f"{nuclide} MT={mt} not found")


def benchmark_lookup(
    model: Pipeline,
    cross_sections_xml: Path,
    nuclide: str = "U238",
    reaction_type: str = "capture",
    reaction_mt: int = 102,
    temperature_k: int = 900,
    n_queries: int = 30000,
) -> dict[str, float | str]:
    energies = np.logspace(-5, np.log10(2.0e7), n_queries)
    temp_label = f"{temperature_k}K" if temperature_k != 294 else "294K"
    rx = load_openmc_reaction(cross_sections_xml, nuclide, reaction_mt)
    props = NUCLIDE_PROPERTIES[nuclide]

    t0 = time.perf_counter()
    reference = rx.xs[temp_label](energies)
    lookup_seconds = time.perf_counter() - t0

    x = pd.DataFrame(
        {
            "log10_energy_eV": np.log10(energies),
            "temperature_K": temperature_k,
            "nuclide_Z": props["z"],
            "nuclide_A": props["a"],
            "nuclide_N": props["a"] - props["z"],
            "nuclide_is_actinide": props["is_actinide"],
            "nuclide_is_fissile": props["is_fissile"],
            "nuclide_family": props["family"],
            "reaction_type": reaction_type,
        }
    )
    t1 = time.perf_counter()
    pred_log = model.predict(x[FEATURES])
    model_seconds = time.perf_counter() - t1

    pred = invert_log_xs(pred_log)
    rel = np.abs(pred - reference) / np.maximum(np.abs(reference), EPS)

    return {
        "benchmark_nuclide": nuclide,
        "benchmark_reaction_type": reaction_type,
        "benchmark_temperature_K": temperature_k,
        "queries": int(n_queries),
        "openmc_lookup_seconds": float(lookup_seconds),
        "model_predict_seconds": float(model_seconds),
        "openmc_queries_per_second": float(n_queries / lookup_seconds),
        "model_queries_per_second": float(n_queries / model_seconds),
        "speedup_vs_openmc_lookup": float(lookup_seconds / model_seconds),
        "benchmark_mean_relative_error": float(np.mean(rel)),
        "benchmark_p95_relative_error": float(np.percentile(rel, 95)),
    }


def plot_predicted_vs_reference(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = np.random.default_rng(7).choice(len(y_true), size=min(12000, len(y_true)), replace=False)
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(y_true[sample], y_pred[sample], s=7, alpha=0.25, c=df.iloc[sample]["temperature_K"])
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], color="black", linewidth=1.2)
    plt.xlabel("Reference log10(cross section / barns)")
    plt.ylabel("Predicted log10(cross section / barns)")
    plt.title("Surrogate Accuracy Against OpenMC/ENDF-B-VIII.0")
    cbar = plt.colorbar()
    cbar.set_label("Temperature (K)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_error_vs_energy(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    rel = relative_error(y_true, y_pred)
    plot_df = df.copy()
    plot_df["relative_error"] = rel
    grouped = (
        plot_df.assign(energy_bin=pd.cut(plot_df["log10_energy_eV"], bins=80))
        .groupby(["energy_bin", "reaction_type"], observed=True)["relative_error"]
        .median()
        .reset_index()
    )
    grouped["bin_center"] = grouped["energy_bin"].apply(lambda b: (b.left + b.right) / 2)

    plt.figure(figsize=(8.2, 5.2))
    for reaction, sub in grouped.groupby("reaction_type", observed=True):
        plt.plot(10 ** sub["bin_center"].astype(float), sub["relative_error"], label=reaction)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Neutron energy (eV)")
    plt.ylabel("Median relative error")
    plt.title("Energy-Dependent Surrogate Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_resonance_zoom(model: Pipeline, df: pd.DataFrame, output_path: Path) -> None:
    mask = (
        (df["nuclide"] == "U238")
        & (df["reaction_type"] == "capture")
        & (df["temperature_K"] == 900)
        & (df["energy_eV"].between(1.0, 1.0e4))
    )
    zoom = df.loc[mask].sort_values("energy_eV")
    if zoom.empty:
        return
    pred = invert_log_xs(model.predict(zoom[FEATURES]))
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(zoom["energy_eV"], zoom["xs_barns"], label="OpenMC/ENDF reference", linewidth=1.4)
    plt.plot(zoom["energy_eV"], pred, label="ML surrogate", linewidth=1.2, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Neutron energy (eV)")
    plt.ylabel("U-238 capture cross section (barns)")
    plt.title("Resonance-Region Zoom: U-238 Capture at 900 K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_speed(bench: dict[str, float | str], output_path: Path) -> None:
    labels = ["OpenMC HDF5 lookup", "ML surrogate"]
    qps = [bench["openmc_queries_per_second"], bench["model_queries_per_second"]]
    plt.figure(figsize=(6.6, 4.6))
    bars = plt.bar(labels, qps, color=["#4C78A8", "#F58518"])
    plt.ylabel("Queries per second")
    plt.title("Batch Cross-Section Throughput")
    plt.yscale("log")
    for bar, val in zip(bars, qps):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train experimental microscopic cross-section surrogate models."
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help=(
            "Actually train the microscopic XS surrogate. This path is intentionally "
            "opt-in because the current sklearn implementation is slower than OpenMC "
            "vectorized lookup and should not be used as a headline speed result."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(PROCESSED_DATA_DIR / "xs_dataset.csv"),
        help="Path to extracted cross-section CSV.",
    )
    parser.add_argument("--max-rows", type=int, default=80000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument(
        "--min-speedup-for-claim",
        type=float,
        default=1.0,
        help="Minimum speedup over OpenMC lookup required before allowing a speed claim.",
    )
    parser.add_argument(
        "--heldout-nuclides",
        nargs="+",
        default=["Pu239"],
        help="Nuclides to hold out for grouped generalization testing.",
    )
    parser.add_argument(
        "--heldout-temperature",
        type=int,
        default=1200,
        help="Temperature to hold out for optional temperature-transfer testing.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    if not args.experimental:
        status = {
            "status": "disabled_by_default",
            "reason": (
                "The microscopic ENDF/OpenMC cross-section surrogate is not part of the "
                "default workflow because the current CPU sklearn implementation is slower "
                "than OpenMC vectorized lookup. Use --experimental only for future "
                "XSBench/GPU/batched-inference experiments."
            ),
            "how_to_run_experiment": (
                "python scripts/train_xs_surrogate.py --experimental --max-rows 100000"
            ),
        }
        disabled_path = MODEL_DIR / "xs_surrogate_disabled.json"
        disabled_path.write_text(json.dumps(status, indent=2))
        print(status["reason"])
        print(f"Wrote {disabled_path}")
        return

    df = pd.read_csv(args.dataset)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.random_state).reset_index(drop=True)

    random_train, random_test = split_random(df, args.test_size, args.random_state)
    nuclide_train, nuclide_test = split_holdout_values(df, "nuclide", args.heldout_nuclides)
    temp_train, temp_test = split_holdout_values(df, "temperature_K", [args.heldout_temperature])

    split_specs = {
        "random_split": (random_train, random_test),
        "heldout_nuclide": (nuclide_train, nuclide_test),
        "heldout_temperature": (temp_train, temp_test),
    }

    split_results: dict[str, dict[str, Any]] = {}
    best_random_model: Pipeline | None = None
    best_random_name: str | None = None
    best_random_pred: np.ndarray | None = None

    for split_name, (train_df, test_df) in split_specs.items():
        split_dir = MODEL_DIR / split_name
        results, fitted, best_name = train_and_score_models(
            train_df,
            test_df,
            args.random_state,
            split_dir,
            include_mlp=not args.skip_mlp,
        )
        split_results[split_name] = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_nuclides": sorted(train_df["nuclide"].unique().tolist()),
            "test_nuclides": sorted(test_df["nuclide"].unique().tolist()),
            "train_temperatures": sorted(map(int, train_df["temperature_K"].unique().tolist())),
            "test_temperatures": sorted(map(int, test_df["temperature_K"].unique().tolist())),
            "models": results,
            "best_model": best_name,
        }

        if split_name == "random_split":
            best_random_model = fitted[best_name]
            best_random_name = best_name
            best_random_pred = best_random_model.predict(test_df[FEATURES])

            for model_name, model in fitted.items():
                joblib.dump(model, MODEL_DIR / f"xs_surrogate_{model_name}.joblib")
            joblib.dump(best_random_model, MODEL_DIR / "xs_surrogate_best.joblib")

            cross_sections_xml = Path(os.environ.get("OPENMC_CROSS_SECTIONS", DEFAULT_CROSS_SECTIONS))
            if not cross_sections_xml.exists():
                cross_sections_xml = DEFAULT_CROSS_SECTIONS
            bench = benchmark_lookup(best_random_model, cross_sections_xml)
            split_results[split_name]["batch_lookup_benchmark"] = bench
            split_results[split_name]["speed_claim_allowed"] = bool(
                bench["speedup_vs_openmc_lookup"] >= args.min_speedup_for_claim
            )
            split_results[split_name]["speed_claim_note"] = (
                "Allowed only if speedup_vs_openmc_lookup meets or exceeds "
                f"{args.min_speedup_for_claim:.2f}x. Current value: "
                f"{bench['speedup_vs_openmc_lookup']:.3f}x."
            )
            split_results[split_name]["heldout_nuclides"] = args.heldout_nuclides
            split_results[split_name]["heldout_temperature"] = int(args.heldout_temperature)

            metrics_path = MODEL_DIR / "xs_surrogate_metrics.json"
            metrics_path.write_text(json.dumps(split_results, indent=2))

            plot_predicted_vs_reference(
                test_df,
                test_df[TARGET].to_numpy(),
                best_random_pred,
                FIGURE_DIR / "xs_predicted_vs_reference.png",
            )
            plot_error_vs_energy(
                test_df,
                test_df[TARGET].to_numpy(),
                best_random_pred,
                FIGURE_DIR / "xs_error_vs_energy.png",
            )
            plot_resonance_zoom(best_random_model, df, FIGURE_DIR / "xs_resonance_zoom.png")
            plot_speed(bench, FIGURE_DIR / "xs_lookup_speed.png")

    summary_rows = []
    for split_name, payload in split_results.items():
        for model_name, model_metrics in payload["models"].items():
            row = {"split": split_name, "model": model_name, **model_metrics}
            summary_rows.append(row)
    summary_path = MODEL_DIR / "validation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nBest interpolation model: {best_random_name}")
    print(f"Wrote validation summary to {summary_path}")
    print(f"Wrote models to {MODEL_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
