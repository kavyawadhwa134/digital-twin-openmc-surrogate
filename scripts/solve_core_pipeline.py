"""Close the two-step pipeline: ML group constants -> 2-group diffusion core solve,
validated against the direct Monte Carlo colorset reference.

Pipeline under test (the thing the whole poster builds toward):

    OpenMC pin-cell/assembly sweeps  ->  ML group-constant surrogate (~0.1 ms/query)
        ->  2-group finite-difference nodal diffusion (this file, ~ms)
        ->  core k_eff + assembly power map

Validation reference: data/processed/core_colorset_reference.json (run_core_colorset.py)
    * direct continuous-energy MC of the 3x3 checkerboard colorset (k_eff + power map)
    * single-assembly OpenMC group constants for both assembly types

Error decomposition (the honest part):
    diffusion(OpenMC GCs) vs MC   -> method error (homogenization + diffusion + 2-group)
    diffusion(ML GCs)     vs MC   -> total pipeline error
    difference of the two         -> error added by the ML surrogate (should be small)

Outputs: models/core_pipeline_validation.json, figures/core_pipeline_validation.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import splu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
# needed so joblib can unpickle the saved surrogate bundle: it was pickled from a
# __main__ run of train_assembly_surrogate.py, so the class must exist on __main__.
import __main__
from train_assembly_surrogate import SubsampledGPR, INPUTS  # noqa: F401
__main__.SubsampledGPR = SubsampledGPR

CELLS_PER_ASM = 10   # FD cells per assembly side (fine enough that discretization << homogenization)


# ------------------------------------------------------------------ 2-group FD solver
def solve_two_group(core_types, gc_by_type, asm_pitch_cm, n_cell=CELLS_PER_ASM,
                    tol=1e-8, max_outer=2000):
    """core_types: 2D array of type labels ('A'/'B') per assembly.
    gc_by_type: {label: dict of D1,D2,Sa1,Sa2,nuSf1,nuSf2,Sf1,Sf2,Ss1to2,Ss2to1}.
    Reflective BCs on all sides. Returns k_eff, assembly fission-power map (mean=1)."""
    nA = core_types.shape[0]
    N = nA * n_cell
    h = asm_pitch_cm / n_cell

    # per-cell material maps
    def cellmap(key):
        M = np.empty((N, N))
        for J in range(nA):
            for I in range(nA):
                M[J*n_cell:(J+1)*n_cell, I*n_cell:(I+1)*n_cell] = gc_by_type[core_types[J, I]][key]
        return M

    D = [cellmap("D1"), cellmap("D2")]
    Sa = [cellmap("Sa1"), cellmap("Sa2")]
    nSf = [cellmap("nuSf1"), cellmap("nuSf2")]
    Sf = [cellmap("Sf1"), cellmap("Sf2")]
    S12 = cellmap("Ss1to2")   # fast -> thermal (downscatter)
    S21 = cellmap("Ss2to1")   # thermal -> fast (upscatter)

    idx = lambda j, i: j * N + i
    removal = [Sa[0] + S12, Sa[1] + S21]

    mats = []
    for g in range(2):
        A = lil_matrix((N * N, N * N))
        Dg = D[g]
        for j in range(N):
            for i in range(N):
                c = idx(j, i)
                diag = removal[g][j, i] * h * h
                for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    jn, in_ = j + dj, i + di
                    if 0 <= jn < N and 0 <= in_ < N:
                        Dface = 2.0 * Dg[j, i] * Dg[jn, in_] / (Dg[j, i] + Dg[jn, in_])
                        A[c, c] += Dface
                        A[c, idx(jn, in_)] = -Dface
                # reflective boundary: simply no coupling term (zero net current)
                A[c, c] = A[c, c] + diag if A[c, c] != 0 else diag
        mats.append(splu(csc_matrix(A)))

    phi1 = np.ones(N * N)
    phi2 = np.ones(N * N)
    nSf1, nSf2 = nSf[0].ravel(), nSf[1].ravel()
    S12v, S21v = S12.ravel(), S21.ravel()
    h2 = h * h
    k = 1.0
    F = (nSf1 * phi1 + nSf2 * phi2) * h2
    for it in range(max_outer):
        src1 = F / k + S21v * phi2 * h2          # chi = [1, 0]
        phi1 = mats[0].solve(src1)
        src2 = S12v * phi1 * h2
        phi2 = mats[1].solve(src2)
        Fnew = (nSf1 * phi1 + nSf2 * phi2) * h2
        knew = k * Fnew.sum() / F.sum()
        err_k = abs(knew - k)
        err_s = np.max(np.abs(Fnew / Fnew.sum() - F / F.sum()))
        k, F = knew, Fnew
        if err_k < tol and err_s < tol:
            break

    # assembly-wise fission power (Sf, not nuSf — matches the MC 'fission' tally)
    fis = (Sf[0].ravel() * phi1 + Sf[1].ravel() * phi2) * h2
    fis = fis.reshape(N, N)
    P = np.zeros((nA, nA))
    for J in range(nA):
        for I in range(nA):
            P[J, I] = fis[J*n_cell:(J+1)*n_cell, I*n_cell:(I+1)*n_cell].sum()
    P /= P.mean()
    return float(k), P, it + 1


def gc_subset(d):
    keys = ["D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2", "Sf1", "Sf2", "Ss1to2", "Ss2to1"]
    return {k: float(d[k]) for k in keys}


def main():
    ensure_project_dirs()
    ref = json.loads((PROCESSED_DATA_DIR / "core_colorset_reference.json").read_text())
    st = ref["state_common"]
    nA = ref["core_n"]
    asm_pitch = ref["lattice_n"] * st["pin_pitch_cm"]
    core_types = np.array([["A" if (i + j) % 2 == 0 else "B" for i in range(nA)]
                           for j in range(nA)])

    mc = ref["colorset"]
    P_mc = np.array(mc["power_map_mean1"])
    k_mc = mc["k_eff"]

    # ---------------- ML surrogate group constants ----------------
    bundle = joblib.load(MODEL_DIR / "assembly_groupconst_surrogate.joblib")
    inputs = bundle["inputs"]

    def state_vec(enr):
        v = dict(st); v["enrichment_wt"] = enr
        return np.array([[v[k] for k in inputs]], dtype=float)

    t0 = time.perf_counter()
    gc_ml = {}
    for lab, enr in (("A", ref["enrich_A"]), ("B", ref["enrich_B"])):
        x = state_vec(enr)
        gc_ml[lab] = {t: float(bundle[t].predict(x)[0]) for t in
                      ["D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2", "Sf1", "Sf2",
                       "Ss1to2", "Ss2to1", "k_inf"]}
    t_surrogate = time.perf_counter() - t0

    gc_ref = {"A": ref["gc_A"], "B": ref["gc_B"]}

    # ---------------- diffusion solves ----------------
    t0 = time.perf_counter()
    k_ml, P_ml, it_ml = solve_two_group(core_types, {l: gc_subset(gc_ml[l]) for l in "AB"}, asm_pitch)
    t_diffusion = time.perf_counter() - t0

    k_ref, P_ref, it_ref = solve_two_group(core_types, {l: gc_subset(gc_ref[l]) for l in "AB"}, asm_pitch)

    # ---------------- error decomposition ----------------
    def pcm(a, b): return (a - b) * 1e5
    dP_ml = (P_ml - P_mc) / P_mc * 100.0
    dP_ref = (P_ref - P_mc) / P_mc * 100.0

    res = {
        "reference_MC": {"k_eff": k_mc, "k_std_pcm": mc["k_eff_std"] * 1e5,
                          "elapsed_s": mc["elapsed_s"], "power_map": P_mc.tolist()},
        "diffusion_openmc_GC": {"k_eff": k_ref, "dk_vs_MC_pcm": pcm(k_ref, k_mc),
                                 "power_map": P_ref.tolist(),
                                 "power_err_pct_max": float(np.max(np.abs(dP_ref))),
                                 "power_err_pct_mean": float(np.mean(np.abs(dP_ref)))},
        "diffusion_ML_GC": {"k_eff": k_ml, "dk_vs_MC_pcm": pcm(k_ml, k_mc),
                             "power_map": P_ml.tolist(),
                             "power_err_pct_max": float(np.max(np.abs(dP_ml))),
                             "power_err_pct_mean": float(np.mean(np.abs(dP_ml)))},
        "surrogate_added_error": {
            "dk_pcm": pcm(k_ml, k_ref),
            "power_pct_max": float(np.max(np.abs(P_ml - P_ref) / P_ref * 100.0)),
        },
        "timing": {"MC_colorset_s": mc["elapsed_s"],
                    "surrogate_query_s": t_surrogate,
                    "diffusion_solve_s": t_diffusion,
                    "pipeline_total_s": t_surrogate + t_diffusion,
                    "speedup_vs_MC": mc["elapsed_s"] / (t_surrogate + t_diffusion)},
        "surrogate_GC_vs_openmc_GC_relerr_pct": {
            lab: {k: abs(gc_ml[lab][k] - gc_ref[lab][k]) / abs(gc_ref[lab][k]) * 100.0
                  for k in ["D1", "D2", "Sa1", "Sa2", "nuSf1", "nuSf2", "Ss1to2"]}
            for lab in "AB"},
        "k_inf_check": {lab: {"ML_pred": gc_ml[lab]["k_inf"], "openmc": gc_ref[lab]["k_inf"]}
                         for lab in "AB"},
        "solver": {"cells_per_assembly_side": CELLS_PER_ASM, "outer_iterations_ml": it_ml},
    }
    out_json = MODEL_DIR / "core_pipeline_validation.json"
    out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["reference_MC", "diffusion_openmc_GC", "diffusion_ML_GC",
                       "surrogate_added_error", "timing"]}, indent=2))
    print(f"\nWrote {out_json}")

    # ---------------- money figure ----------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    vmin = min(P_mc.min(), P_ml.min()) * 0.98
    vmax = max(P_mc.max(), P_ml.max()) * 1.02

    for ax, P, title in ((axes[0], P_mc, f"OpenMC Monte Carlo (reference)\nk={k_mc:.5f}  ·  {mc['elapsed_s']:.0f} s"),
                         (axes[1], P_ml, f"ML pipeline: surrogate + diffusion\nk={k_ml:.5f}  ·  "
                                          f"{(t_surrogate+t_diffusion)*1e3:.0f} ms")):
        im = ax.imshow(P, cmap="YlOrRd", vmin=vmin, vmax=vmax)
        for (j, i), v in np.ndenumerate(P):
            ax.text(i, j, f"{v:.3f}", ha="center", va="center", fontsize=11,
                    color="#1a1a2e", fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(nA)); ax.set_yticks(range(nA))
        ax.set_xticklabels([]); ax.set_yticklabels([])

    dmax = max(0.5, np.max(np.abs(dP_ml)) * 1.15)
    im2 = axes[2].imshow(dP_ml, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    for (j, i), v in np.ndenumerate(dP_ml):
        axes[2].text(i, j, f"{v:+.2f}%", ha="center", va="center", fontsize=10.5,
                     color="#1a1a2e", fontweight="bold")
    axes[2].set_title(f"power difference ML vs MC\nmax {np.max(np.abs(dP_ml)):.2f}%  ·  "
                      f"mean {np.mean(np.abs(dP_ml)):.2f}%", fontsize=10)
    axes[2].set_xticks(range(nA)); axes[2].set_yticks(range(nA))
    axes[2].set_xticklabels([]); axes[2].set_yticklabels([])

    ax3 = axes[3]; ax3.axis("off")
    txt = (
        "ERROR DECOMPOSITION\n"
        f"  diffusion + OpenMC GCs vs MC : {pcm(k_ref,k_mc):+.0f} pcm\n"
        f"     (method: homogenization + 2-group + diffusion)\n"
        f"  diffusion + ML GCs vs MC     : {pcm(k_ml,k_mc):+.0f} pcm\n"
        f"  error ADDED by ML surrogate  : {pcm(k_ml,k_ref):+.0f} pcm\n"
        f"     power shape added error   : "
        f"{res['surrogate_added_error']['power_pct_max']:.2f}% max\n\n"
        "TIMING\n"
        f"  direct Monte Carlo           : {mc['elapsed_s']:.0f} s\n"
        f"  surrogate (2 queries)        : {t_surrogate*1e3:.1f} ms\n"
        f"  2-group diffusion solve      : {t_diffusion*1e3:.0f} ms\n"
        f"  pipeline speedup             : ~{res['timing']['speedup_vs_MC']:.0f}x\n\n"
        f"MC noise: k ±{mc['k_eff_std']*1e5:.0f} pcm; power ±"
        f"{np.max(np.array(mc['power_rel_std']))*100:.2f}%"
    )
    ax3.text(0.0, 0.97, txt, va="top", ha="left", fontsize=9.5, family="monospace",
             color="#1a1a2e", linespacing=1.55)

    fig.suptitle("Two-step pipeline closed: OpenMC lattice data → ML group constants → "
                 "2-group diffusion → core power map  (3×3 checkerboard colorset, reflective BC)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[:2], shrink=0.85, label="assembly power (mean=1)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="%")
    out_fig = FIGURE_DIR / "core_pipeline_validation.png"
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
