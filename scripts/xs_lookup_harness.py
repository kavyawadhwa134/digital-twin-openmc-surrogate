"""Fair benchmark harness for ML/algorithmic acceleration of the macroscopic
cross-section lookup kernel (the XSBench kernel) on real ENDF/B-VIII.0 native grids.

The macroscopic lookup the kernel performs, per query energy E:

    Sigma_macro(E) = sum_i  N_i * sigma_i(E)

where sigma_i(E) is obtained by (1) finding the bracketing index in nuclide i's sorted
energy grid and (2) linear-interpolating.  Step (1) -- a binary search per nuclide into a
large, random-access table -- is the memory-latency-bound bottleneck (~80% of MC runtime).

This harness isolates that step so accelerators compete FAIRLY:
  * Every method only supplies an INDEX FINDER (build_index + find_lower_index).
  * Interpolation, accumulation, query batch, timing protocol, and the accuracy check
    against the binary-search ground truth are SHARED and identical for all methods.
  * Therefore every method is accuracy-exact by construction (it reads the true table
    values); a wrong index finder is caught by the accuracy gate.

Metric reported: macroscopic lookups / second (the XSBench figure of merit) and speedup
vs the numpy.searchsorted binary-search baseline.

Subcommands:
    build    pre-extract pin-cell and assembly material grids to .npz (fast reload)
    bench    benchmark the registered methods on a material
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import DEFAULT_CROSS_SECTIONS, PROCESSED_DATA_DIR, ensure_project_dirs

# ----------------------------------------------------------------------------- materials
# Representative number densities (atoms / barn-cm). Exact values are immaterial to lookup
# SPEED and identical across methods for the accuracy check; chosen to be physically sane.
PINCELL = {
    "U235": 5.0e-4, "U238": 2.1e-2, "O16": 4.6e-2, "H1": 4.7e-2,
    "B10": 5.0e-6, "Zr90": 2.2e-3,
}
ASSEMBLY = {
    "U235": 5.0e-4, "U238": 2.1e-2, "O16": 4.6e-2, "H1": 4.7e-2,
    "B10": 4.0e-6, "B11": 1.6e-5,
    "Zr90": 1.1e-3, "Zr91": 2.4e-4, "Zr92": 3.7e-4, "Zr94": 3.7e-4, "Zr96": 6.0e-5,
    "Fe56": 1.4e-3, "Cr52": 7.0e-4, "Ni58": 3.0e-4,
    "Gd155": 1.0e-6, "Gd157": 1.0e-6,
}
TARGET_TEMP_K = 900


def _library_paths(xml: Path) -> dict[str, Path]:
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    out: dict[str, Path] = {}
    for lib in root.findall("library"):
        if lib.get("type") != "neutron":
            continue
        rel = lib.get("path")
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = xml.parent / p
        for mat in (lib.get("materials") or "").split():
            out[mat] = p
    return out


def _total_xs(inc, tlabel):
    """Total microscopic cross section (MT=1) on its native grid; fall back to a sum."""
    if 1 in inc.reactions and tlabel in inc.reactions[1].xs:
        fun = inc.reactions[1].xs[tlabel]
        return np.asarray(fun.x, float), np.asarray(fun.y, float)
    # fallback: union of elastic+capture(+fission) grids, summed
    Es = []
    for mt in (2, 102, 18):
        if mt in inc.reactions and tlabel in inc.reactions[mt].xs:
            Es.append(np.asarray(inc.reactions[mt].xs[mt and tlabel].x, float))
    E = np.unique(np.concatenate(Es))
    S = np.zeros_like(E)
    for mt in (2, 102, 18):
        if mt in inc.reactions and tlabel in inc.reactions[mt].xs:
            fun = inc.reactions[mt].xs[tlabel]
            S = S + np.interp(E, np.asarray(fun.x, float), np.asarray(fun.y, float), left=0, right=0)
    return E, S


def build_material(name: str, densities: dict, e_min=1.0e-3, e_max=2.0e7) -> Path:
    import openmc.data
    xml = Path(str(DEFAULT_CROSS_SECTIONS))
    paths = _library_paths(xml)
    arrays = {}
    names = []
    dens = []
    for nuc, nd in densities.items():
        if nuc not in paths:
            print(f"  skip {nuc} (not in library)")
            continue
        inc = openmc.data.IncidentNeutron.from_hdf5(paths[nuc])
        labels = {int(round(float(t.rstrip('K')))): t for t in inc.temperatures}
        tk = min(labels, key=lambda k: abs(k - TARGET_TEMP_K))
        tlabel = labels[tk]
        E, S = _total_xs(inc, tlabel)
        m = np.isfinite(E) & np.isfinite(S) & (S > 0) & (E >= e_min) & (E <= e_max)
        E, S = E[m], S[m]
        order = np.argsort(E)
        E, S = E[order], S[order]
        # de-duplicate energies (strictly increasing grid for searchsorted/interp)
        keep = np.concatenate([[True], np.diff(E) > 0])
        E, S = E[keep], S[keep]
        i = len(names)
        arrays[f"E_{i}"] = E.astype(np.float64)
        arrays[f"S_{i}"] = S.astype(np.float64)
        names.append(nuc)
        dens.append(nd)
        print(f"  {nuc:>6} @ {tk}K : {E.size:>8,} pts  [{E[0]:.2e}-{E[-1]:.2e} eV]")
    arrays["densities"] = np.asarray(dens, float)
    out = PROCESSED_DATA_DIR / f"xslookup_{name}.npz"
    np.savez(out, **arrays)
    out.with_suffix(".names.json").write_text(json.dumps(names))
    tot = sum(arrays[f"E_{i}"].size for i in range(len(names)))
    print(f"Wrote {out}  ({len(names)} nuclides, {tot:,} total grid pts)")
    return out


def load_material(name: str):
    npz = PROCESSED_DATA_DIR / f"xslookup_{name}.npz"
    d = np.load(npz)
    names = json.loads((npz.with_suffix(".names.json")).read_text())
    grids = [(d[f"E_{i}"], d[f"S_{i}"]) for i in range(len(names))]
    return {"names": names, "grids": grids, "densities": d["densities"]}


def make_queries(n: int, seed: int, e_min=1.0e-3, e_max=2.0e7) -> np.ndarray:
    """Log-uniform energies over the spectrum (stresses search across the whole grid)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return (e_min * (e_max / e_min) ** u).astype(np.float64)

# ----------------------------------------------------------------------------- baseline
def baseline_lower_index(Eq, E_grid, _precompute=None):
    idx = np.searchsorted(E_grid, Eq, side="right") - 1
    return np.clip(idx, 0, E_grid.size - 2)


def macro_from_indices(Eq, mat, lower_idx_list):
    """Shared linear-interp + accumulate over nuclides given each nuclide's lower index."""
    out = np.zeros_like(Eq)
    for (E_grid, S_grid), nd, idx in zip(mat["grids"], mat["densities"], lower_idx_list):
        e0 = E_grid[idx]; e1 = E_grid[idx + 1]
        s0 = S_grid[idx]; s1 = S_grid[idx + 1]
        t = (Eq - e0) / (e1 - e0)
        out += nd * (s0 + t * (s1 - s0))
    return out

# ----------------------------------------------------------------------------- methods
# A "method" = {build(E_grid)->precompute, find(Eq,E_grid,precompute)->lower_idx}

def m_baseline():
    return {"name": "binary_search (baseline)",
            "build": lambda E: None,
            "find": baseline_lower_index}


def m_learned_rmi(n_buckets=16384, max_window=64):
    """Two-stage learned index (RMI): uniform-log first stage -> per-bucket linear model
    predicting the grid index -> bounded local correction. Adapts to grid density."""
    def build(E):
        logE = np.log10(E)
        lmin, lmax = logE[0], logE[-1]
        span = max(lmax - lmin, 1e-12)
        # first stage: which bucket each grid point falls in
        gb = np.clip(((logE - lmin) / span * n_buckets).astype(np.int64), 0, n_buckets - 1)
        # per-bucket linear model idx ~ a*logE + b, fit from first/last point in bucket
        a = np.zeros(n_buckets); b = np.zeros(n_buckets)
        # default: map bucket center to a representative index (nearest grid point)
        # gather first and last grid index per bucket
        first = np.full(n_buckets, -1, np.int64)
        last = np.full(n_buckets, -1, np.int64)
        # first occurrence
        order = np.arange(E.size)
        # first index per bucket
        f_idx = np.full(n_buckets, -1, np.int64)
        np.maximum.at(last, gb, order)         # last grid index in each bucket
        # for first: use minimum
        tmp = np.full(n_buckets, E.size, np.int64)
        np.minimum.at(tmp, gb, order)
        first = np.where(tmp == E.size, -1, tmp)
        for k in range(n_buckets):
            if first[k] < 0:
                # empty bucket: borrow neighbours later via fill
                continue
            i0, i1 = first[k], last[k]
            if i1 > i0:
                le0, le1 = logE[i0], logE[i1]
                a[k] = (i1 - i0) / max(le1 - le0, 1e-12)
                b[k] = i0 - a[k] * le0
            else:
                a[k] = 0.0; b[k] = i0
        # fill empty buckets with nearest non-empty (forward then backward)
        haveb = first >= 0
        last_good = 0
        for k in range(n_buckets):
            if haveb[k]:
                last_good = k
            else:
                a[k] = a[last_good]; b[k] = b[last_good]
        return {"lmin": lmin, "span": span, "nb": n_buckets, "a": a, "b": b,
                "W": max_window, "N": E.size}

    def find(Eq, E_grid, pc):
        logE = np.log10(Eq)
        k = np.clip(((logE - pc["lmin"]) / pc["span"] * pc["nb"]).astype(np.int64),
                    0, pc["nb"] - 1)
        idx = (pc["a"][k] * logE + pc["b"][k]).astype(np.int64)
        idx = np.clip(idx, 0, pc["N"] - 2)
        # bounded local correction: move so that E_grid[idx] <= Eq < E_grid[idx+1]
        W = pc["W"]
        # advance forward while next grid energy <= Eq (cap idx at N-2)
        for _ in range(W):
            nxt = np.minimum(idx + 1, pc["N"] - 2)
            adv = (E_grid[nxt] <= Eq) & (idx < pc["N"] - 2)
            if not adv.any():
                break
            idx = np.minimum(np.where(adv, idx + 1, idx), pc["N"] - 2)
        # retreat while current grid energy > Eq
        for _ in range(W):
            ret = (E_grid[idx] > Eq) & (idx > 0)
            if not ret.any():
                break
            idx = np.where(ret, idx - 1, idx)
        return np.clip(idx, 0, pc["N"] - 2)

    return {"name": f"learned_RMI(buckets={n_buckets},W={max_window})",
            "build": build, "find": find}

# ---------------------------------------------------------------- whole-macro methods
# These don't fit the per-nuclide find() API: they share ONE index across all nuclides.

def mm_unionized():
    """Unionized energy grid: ONE binary search into the union of all nuclide grids,
    then O(1) gather of each nuclide's bracket index. Exact. Classic XSBench optimization."""
    def build(mat):
        grids = mat["grids"]
        union = np.unique(np.concatenate([E for (E, S) in grids]))
        # per nuclide: lower index in that nuclide's grid for each union point
        maps = [np.clip(np.searchsorted(E, union, side="right") - 1, 0, E.size - 2).astype(np.int32)
                for (E, S) in grids]
        return {"union": union, "maps": maps}
    def lookup(Eq, mat, st):
        u = np.clip(np.searchsorted(st["union"], Eq, side="right") - 1, 0, st["union"].size - 1)
        out = np.zeros_like(Eq)
        for (E, S), nd, mp in zip(mat["grids"], mat["densities"], st["maps"]):
            idx = mp[u]
            e0 = E[idx]; e1 = E[idx + 1]; s0 = S[idx]; s1 = S[idx + 1]
            out += nd * (s0 + (Eq - e0) / (e1 - e0) * (s1 - s0))
        return out
    return {"name": "unionized_grid", "build_macro": build, "lookup_macro": lookup}


def mm_hash_union(bits=24, W=8):
    """Hash index on the UNION grid: O(1) uniform-log address -> start union index ->
    bounded local correction (exact), then O(1) per-nuclide gather. No binary search."""
    def build(mat):
        grids = mat["grids"]
        union = np.unique(np.concatenate([E for (E, S) in grids]))
        maps = [np.clip(np.searchsorted(E, union, side="right") - 1, 0, E.size - 2).astype(np.int32)
                for (E, S) in grids]
        logU = np.log10(union)
        lmin, lmax = logU[0], logU[-1]
        span = max(lmax - lmin, 1e-12)
        K = 1 << bits
        # bucket -> first union index whose logU >= bucket lower edge
        edges = lmin + (np.arange(K + 1) / K) * span
        start = np.searchsorted(logU, edges, side="left").astype(np.int32)
        start = np.clip(start, 0, union.size - 1)
        return {"union": union, "maps": maps, "logU": logU, "lmin": lmin,
                "span": span, "K": K, "start": start, "Nu": union.size}
    def lookup(Eq, mat, st):
        lg = np.log10(Eq)
        k = np.clip(((lg - st["lmin"]) / st["span"] * st["K"]).astype(np.int64), 0, st["K"] - 1)
        u = st["start"][k].astype(np.int64)
        u = np.clip(u, 0, st["Nu"] - 1)
        union = st["union"]; Nu = st["Nu"]
        # bounded forward/back correction to exact union lower bracket
        for _ in range(W):
            nxt = np.minimum(u + 1, Nu - 1)
            adv = (union[nxt] <= Eq) & (u < Nu - 1)
            if not adv.any():
                break
            u = np.minimum(np.where(adv, u + 1, u), Nu - 1)
        for _ in range(W):
            ret = (union[u] > Eq) & (u > 0)
            if not ret.any():
                break
            u = np.where(ret, u - 1, u)
        out = np.zeros_like(Eq)
        for (E, S), nd, mp in zip(mat["grids"], mat["densities"], st["maps"]):
            idx = mp[u]
            e0 = E[idx]; e1 = E[idx + 1]; s0 = S[idx]; s1 = S[idx + 1]
            out += nd * (s0 + (Eq - e0) / (e1 - e0) * (s1 - s0))
        return out
    return {"name": f"hash_union(bits={bits},W={W})", "build_macro": build, "lookup_macro": lookup}

# ----------------------------------------------------------------------------- benchmark
def benchmark(method, mat, queries, repeats=5):
    if "lookup_macro" in method:
        return _benchmark_macro(method, mat, queries, repeats)
    # build per-nuclide precompute (not timed)
    pc = [method["build"](E) for (E, S) in mat["grids"]]
    find = method["find"]

    def run_once():
        idxs = [find(queries, E, pc[i]) for i, (E, S) in enumerate(mat["grids"])]
        return macro_from_indices(queries, mat, idxs)

    val = run_once()  # warmup + result for accuracy
    # ground truth
    base_idx = [baseline_lower_index(queries, E) for (E, S) in mat["grids"]]
    truth = macro_from_indices(queries, mat, base_idx)
    rel = np.abs(val - truth) / np.maximum(np.abs(truth), 1e-30)
    max_rel = float(np.max(rel)); mean_rel = float(np.mean(rel))

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    lps = queries.size / t
    return {"name": method["name"], "median_s": t, "lookups_per_s": lps,
            "max_rel_err": max_rel, "mean_rel_err": mean_rel,
            "accuracy_ok": max_rel < 1e-6}


def _benchmark_macro(method, mat, queries, repeats=5):
    st = method["build_macro"](mat)
    fn = method["lookup_macro"]
    val = fn(queries, mat, st)
    base_idx = [baseline_lower_index(queries, E) for (E, S) in mat["grids"]]
    truth = macro_from_indices(queries, mat, base_idx)
    rel = np.abs(val - truth) / np.maximum(np.abs(truth), 1e-30)
    max_rel = float(np.max(rel)); mean_rel = float(np.mean(rel))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(queries, mat, st)
        times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    return {"name": method["name"], "median_s": t, "lookups_per_s": queries.size / t,
            "max_rel_err": max_rel, "mean_rel_err": mean_rel, "accuracy_ok": max_rel < 1e-6}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    b = sub.add_parser("bench")
    b.add_argument("--material", default="pincell", choices=["pincell", "assembly"])
    b.add_argument("--n-queries", type=int, default=2_000_000)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    ensure_project_dirs()
    if args.cmd == "build":
        print("=== pin-cell ==="); build_material("pincell", PINCELL)
        print("=== assembly ==="); build_material("assembly", ASSEMBLY)
        return

    mat = load_material(args.material)
    q = make_queries(args.n_queries, args.seed)
    tot = sum(E.size for (E, S) in mat["grids"])
    print(f"Material '{args.material}': {len(mat['names'])} nuclides, {tot:,} grid pts, "
          f"{args.n_queries:,} queries\n")
    methods = [m_baseline(), mm_unionized(), mm_hash_union()]
    base_lps = None
    print(f"{'method':<40}{'lookups/s':>16}{'speedup':>10}{'max_rel':>12}{'ok':>5}")
    for meth in methods:
        r = benchmark(meth, mat, q, args.repeats)
        if base_lps is None:
            base_lps = r["lookups_per_s"]
        sp = r["lookups_per_s"] / base_lps
        print(f"{r['name']:<40}{r['lookups_per_s']:>16,.0f}{sp:>9.2f}x"
              f"{r['max_rel_err']:>12.2e}{'Y' if r['accuracy_ok'] else 'N':>5}")


if __name__ == "__main__":
    main()
