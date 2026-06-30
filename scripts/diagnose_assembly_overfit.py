"""Overfitting diagnostic for the assembly group-constant surrogate.

Three honest tests for overfitting, each plotted:
  1. TRAIN vs TEST error per target. Overfitting <=> train error << test error.
     If the two bars are close, the model is NOT overfit.
  2. LEARNING CURVE (k_inf): train error and 5-fold CV error vs training-set size.
     Overfitting <=> large persistent gap; healthy <=> curves converge.
  3. k_inf parity against the Monte Carlo NOISE FLOOR. A surrogate cannot
     meaningfully beat the stochastic noise of its own training labels; if the
     test error sits at the noise band, there is no room to overfit into.

Outputs figures/assembly_overfit_diagnostic.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from train_assembly_surrogate import INPUTS, build_models

# Targets worth showing (drop chi1/chi2: literal constants, std ~ 1e-15, nothing to learn)
TARGETS = [
    "k_inf", "D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2",
    "Sf1", "Sf2", "Ss1to1", "Ss1to2", "Ss2to2",
]

# CV-selected model per target (from the main eval run)
SELECTED = {
    "k_inf": "quad_ridge", "D1": "quad_ridge", "D2": "quad_ridge",
    "Sa1": "gpr_rbf", "Sa2": "quad_ridge", "nuSf1": "quad_ridge",
    "nuSf2": "quad_ridge", "Sf1": "quad_ridge", "Sf2": "quad_ridge",
    "Ss1to1": "gpr_rbf", "Ss1to2": "quad_ridge", "Ss2to2": "quad_ridge",
}


def main():
    ensure_project_dirs()
    df = pd.read_csv(PROCESSED_DATA_DIR / "assembly_groupconst_500.csv")
    print(f"Loaded {len(df)} cases")

    rng = np.random.default_rng(42)
    n_test = int(len(df) * 0.25)
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    mask = np.ones(len(df), dtype=bool); mask[test_idx] = False
    tr, te = df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
    Xtr, Xte = tr[INPUTS].values.astype(float), te[INPUTS].values.astype(float)

    # --- Test 1: train vs test relMAE per target ---
    rows = []
    for tgt in TARGETS:
        ytr, yte = tr[tgt].values.astype(float), te[tgt].values.astype(float)
        pipe = build_models()[SELECTED[tgt]]
        pipe.fit(Xtr, ytr)
        tr_relmae = np.mean(np.abs(pipe.predict(Xtr) - ytr)) / (np.mean(np.abs(ytr)) + 1e-30) * 100
        te_relmae = np.mean(np.abs(pipe.predict(Xte) - yte)) / (np.mean(np.abs(yte)) + 1e-30) * 100
        rows.append((tgt, tr_relmae, te_relmae))
        gap = te_relmae - tr_relmae
        flag = "OVERFIT?" if (te_relmae > 2 * tr_relmae + 0.05) else "ok"
        print(f"  {tgt:8s} train={tr_relmae:.3f}%  test={te_relmae:.3f}%  gap={gap:+.3f}%  [{flag}]")

    # --- Test 2: learning curve for k_inf ---
    yk = df["k_inf"].values.astype(float)
    Xk = df[INPUTS].values.astype(float)
    pipe = build_models()["quad_ridge"]
    sizes, train_sc, cv_sc = learning_curve(
        pipe, Xk, yk, cv=KFold(5, shuffle=True, random_state=0),
        train_sizes=np.linspace(0.2, 1.0, 8),
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    train_mae = -train_sc.mean(axis=1) * 1e5  # pcm
    cv_mae = -cv_sc.mean(axis=1) * 1e5
    cv_std = cv_sc.std(axis=1) * 1e5

    # --- Test 3: noise floor for k_inf ---
    noise_floor_pcm = df["k_inf_std"].mean() * 1e5
    pipe_k = build_models()["quad_ridge"]; pipe_k.fit(Xtr, tr["k_inf"].values.astype(float))
    yk_pred = pipe_k.predict(Xte)
    yk_true = te["k_inf"].values.astype(float)
    test_mae_pcm = np.mean(np.abs(yk_pred - yk_true)) * 1e5

    # ===================== FIGURE =====================
    fig = plt.figure(figsize=(15, 5.2))

    # Panel 1: train vs test bars
    ax1 = fig.add_subplot(1, 3, 1)
    tgts = [r[0] for r in rows]
    trv = [r[1] for r in rows]
    tev = [r[2] for r in rows]
    x = np.arange(len(tgts))
    w = 0.38
    ax1.bar(x - w/2, trv, w, label="train", color="#90be6d")
    ax1.bar(x + w/2, tev, w, label="test (held-out)", color="#1e6091")
    ax1.set_xticks(x); ax1.set_xticklabels(tgts, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("Relative MAE (%)")
    ax1.set_title("1. Train vs Test error per target\n(close bars = NOT overfit)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: learning curve
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(sizes, train_mae, "o-", color="#90be6d", label="train error")
    ax2.plot(sizes, cv_mae, "s-", color="#1e6091", label="5-fold CV error")
    ax2.fill_between(sizes, cv_mae - cv_std, cv_mae + cv_std, color="#1e6091", alpha=0.15)
    ax2.axhline(noise_floor_pcm, color="#e63946", ls="--", lw=1.2,
                label=f"MC noise floor ({noise_floor_pcm:.0f} pcm)")
    ax2.set_xlabel("Training-set size")
    ax2.set_ylabel("k∞ MAE (pcm)")
    ax2.set_title("2. Learning curve (k∞)\n(curves converge = healthy fit)", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    # Panel 3: parity with noise band
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(yk_true, yk_pred, s=28, alpha=0.75, color="#1e6091", edgecolors="none")
    lo, hi = yk_true.min(), yk_true.max()
    pad = (hi - lo) * 0.04
    xline = np.array([lo - pad, hi + pad])
    ax3.plot(xline, xline, "k-", lw=1.0)
    band = noise_floor_pcm / 1e5
    ax3.fill_between(xline, xline - band, xline + band, color="#e63946", alpha=0.18,
                     label=f"±1σ MC noise ({noise_floor_pcm:.0f} pcm)")
    ax3.set_xlabel("OpenMC k∞"); ax3.set_ylabel("Surrogate k∞")
    ax3.set_title(f"3. k∞ vs MC noise floor\ntest MAE {test_mae_pcm:.0f} pcm ≈ noise {noise_floor_pcm:.0f} pcm",
                  fontsize=10)
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Assembly surrogate — overfitting diagnostic ({len(df)} cases, leakage-free)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "assembly_overfit_diagnostic.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # Save numeric verdict
    verdict = {
        "noise_floor_pcm": float(noise_floor_pcm),
        "kinf_test_mae_pcm": float(test_mae_pcm),
        "kinf_at_noise_floor": bool(test_mae_pcm <= 1.5 * noise_floor_pcm),
        "per_target": [
            {"target": t, "train_relmae_pct": float(tr_), "test_relmae_pct": float(te_),
             "gap_pct": float(te_ - tr_)}
            for (t, tr_, te_) in rows
        ],
        "learning_curve": {
            "sizes": [int(s) for s in sizes],
            "train_mae_pcm": [float(v) for v in train_mae],
            "cv_mae_pcm": [float(v) for v in cv_mae],
        },
    }
    vp = MODEL_DIR / "assembly_overfit_diagnostic.json"
    with open(vp, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"Saved {vp}")

    # Verdict line
    max_gap = max(te_ - tr_ for (_, tr_, te_) in rows)
    print("\n=== VERDICT ===")
    print(f"  Largest train->test gap: {max_gap:.3f}%  (overfit if >> train error)")
    print(f"  k∞ test MAE {test_mae_pcm:.0f} pcm vs MC noise floor {noise_floor_pcm:.0f} pcm")
    print(f"  {'NOT OVERFIT — error sits at the MC noise floor' if test_mae_pcm <= 1.5*noise_floor_pcm else 'check gap'}")


if __name__ == "__main__":
    main()
