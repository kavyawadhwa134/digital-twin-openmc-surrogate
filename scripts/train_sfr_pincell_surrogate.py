"""Train and evaluate a surrogate for SFR (fast-spectrum) pin-cell group constants.

Generation IV extension: same leakage-free protocol as the LWR pin-cell and assembly
surrogates (k-fold CV on the training pool only, held-out test evaluated once,
error reported against the Monte Carlo noise floor), applied to the sodium-cooled
fast MOX pin cell of run_sfr_pincell_sweep.py (2-group condensation at 0.1 MeV).

Outputs:
    models/sfr_pincell_surrogate.joblib
    models/sfr_pincell_evaluation.json
    figures/sfr_pincell_groupconst_parity.png
    figures/sfr_pincell_kinf_parity.png
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from train_assembly_surrogate import build_models, evaluate_target, SubsampledGPR  # noqa: F401

INPUTS = [
    "pu_fraction", "fuel_temperature_K", "sodium_density_g_cm3",
    "coolant_temperature_K", "fuel_radius_cm", "cladding_thickness_cm",
    "pin_pitch_cm",
]

GC_TARGETS = [
    "D1", "D2", "Sa1", "Sa2",
    "nuSf1", "nuSf2", "Sf1", "Sf2",
    "Ss1to1", "Ss1to2", "Ss2to1", "Ss2to2",
    "chi1",
    "k_inf",
]

# fast-spectrum sanity checks (fractions reported, not asserted)
PHYSICAL_CHECKS = {
    "chi1 > 0.9 (fission births above 0.1 MeV)": lambda df: (df.chi1 > 0.9).mean(),
    "Ss1to2 > Ss2to1 (downscatter dominates)": lambda df: (df.Ss1to2 > df.Ss2to1).mean(),
    "Sa2 > Sa1 (resonance absorption below 0.1 MeV)": lambda df: (df.Sa2 > df.Sa1).mean(),
    "k_inf rises with pu_fraction (rank corr > 0)":
        lambda df: float(df.pu_fraction.corr(df.k_inf, method="spearman") > 0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(PROCESSED_DATA_DIR / "sfr_pincell_groupconst.csv"))
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=3)
    args = p.parse_args()

    ensure_project_dirs()
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} SFR pin-cell cases from {args.data}")
    noise_pcm = df.k_inf_std.mean() * 1e5
    print(f"Mean k_inf MC noise (label noise floor): {noise_pcm:.0f} pcm")

    print("\n=== Fast-spectrum physical consistency ===")
    for desc, fn in PHYSICAL_CHECKS.items():
        print(f"  {fn(df)*100:.1f}%  {desc}")

    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(df) * args.test_frac))
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)
    X_train = df_train[INPUTS].values.astype(float)
    X_test = df_test[INPUTS].values.astype(float)
    print(f"\nTrain: {len(df_train)} | Test: {len(df_test)} | "
          f"Inputs: {len(INPUTS)} | Targets: {len(GC_TARGETS)}")

    results, summary_rows = {}, []
    t_total = time.perf_counter()
    for tgt in GC_TARGETS:
        y_train = df_train[tgt].values.astype(float)
        y_test = df_test[tgt].values.astype(float)
        model_name, pipe, y_pred, metrics = evaluate_target(
            X_train, y_train, X_test, y_test,
            n_folds=args.n_folds, n_repeats=args.n_repeats)
        results[tgt] = {"model": pipe, "y_pred": y_pred, "metrics": metrics}
        print(f"  {tgt:12s}  {model_name:14s}  relMAE={metrics['rel_mae_pct']:.3f}%  "
              f"R²={metrics['r2']:.4f}")
        summary_rows.append({"target": tgt, "model": model_name,
                             "rel_mae_pct": metrics["rel_mae_pct"],
                             "mae": metrics["mae"], "r2": metrics["r2"]})
    print(f"Total training time: {time.perf_counter() - t_total:.0f}s")

    kin = results["k_inf"]["metrics"]
    mae_pcm = kin["mae"] * 1e5
    rmse_pcm = kin["rmse"] * 1e5
    intrinsic = np.sqrt(max(0.0, rmse_pcm**2 - noise_pcm**2))
    print(f"\nk_inf: MAE {mae_pcm:.0f} pcm | RMSE {rmse_pcm:.0f} pcm | "
          f"noise floor {noise_pcm:.0f} pcm | intrinsic ~{intrinsic:.0f} pcm")

    # linear baseline comparison for k_inf (honest metric)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipe
    lin = SkPipe([("sc", StandardScaler()), ("m", LinearRegression())])
    lin.fit(X_train, df_train["k_inf"].values)
    lin_mae_pcm = float(np.mean(np.abs(lin.predict(X_test) - df_test["k_inf"].values))) * 1e5
    print(f"linear-baseline k_inf MAE: {lin_mae_pcm:.0f} pcm "
          f"(surrogate is {lin_mae_pcm/max(mae_pcm,1e-9):.1f}x better)")

    bundle = {tgt: results[tgt]["model"] for tgt in GC_TARGETS}
    bundle["inputs"] = INPUTS
    bundle["targets"] = GC_TARGETS
    joblib.dump(bundle, MODEL_DIR / "sfr_pincell_surrogate.joblib")

    eval_out = {
        "n_train": len(df_train), "n_test": len(df_test),
        "protocol": f"{args.n_repeats}-repeat {args.n_folds}-fold CV on train pool; test touched once",
        "noise_floor_pcm": noise_pcm,
        "kinf_mae_pcm": mae_pcm, "kinf_rmse_pcm": rmse_pcm,
        "kinf_intrinsic_pcm": intrinsic,
        "kinf_linear_baseline_mae_pcm": lin_mae_pcm,
        "targets": {t: results[t]["metrics"] for t in GC_TARGETS},
        "physical_consistency_pct": {d: float(f(df) * 100) for d, f in PHYSICAL_CHECKS.items()},
    }
    with open(MODEL_DIR / "sfr_pincell_evaluation.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Saved evaluation → {MODEL_DIR / 'sfr_pincell_evaluation.json'}")

    # parity grid
    plot_targets = ["k_inf", "D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2", "Ss1to2"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, tgt in zip(axes.flatten(), plot_targets):
        y_true = df_test[tgt].values
        y_pred = results[tgt]["y_pred"]
        m = results[tgt]["metrics"]
        ax.scatter(y_true, y_pred, s=18, alpha=0.7, color="#b03a2e", edgecolors="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.04
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
        ax.set_xlabel(f"OpenMC {tgt}", fontsize=8)
        ax.set_ylabel(f"Surrogate {tgt}", fontsize=8)
        ax.set_title(f"{tgt}\n({m['selected_model']}, {m['rel_mae_pct']:.2f}% relMAE)", fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle("SFR pin-cell group-constant surrogate (fast spectrum, 0.1 MeV boundary)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sfr_pincell_groupconst_parity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # k_inf parity
    y_true = df_test["k_inf"].values
    y_pred = results["k_inf"]["y_pred"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=25, alpha=0.75, color="#b03a2e", edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.04
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=1.0)
    ax.set_xlabel("OpenMC k∞", fontsize=11)
    ax.set_ylabel("ML surrogate k∞", fontsize=11)
    ax.set_title(f"SFR pin-cell k∞ surrogate ({kin['selected_model']})\n"
                 f"MAE {mae_pcm:.0f} pcm | noise floor {noise_pcm:.0f} pcm | "
                 f"{len(y_true)}-pt honest test", fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sfr_pincell_kinf_parity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures → {FIGURE_DIR}/sfr_pincell_*.png")

    print("\n=== Summary ===")
    for row in summary_rows:
        print(f"{row['target']:<12} {row['model']:<16} relMAE {row['rel_mae_pct']:>8.3f}%  "
              f"R² {row['r2']:.4f}")


if __name__ == "__main__":
    main()
