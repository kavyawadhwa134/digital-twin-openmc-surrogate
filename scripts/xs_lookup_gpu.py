"""GPU (CuPy) benchmark of the macroscopic cross-section lookup kernel.

Self-contained: needs only `numpy` and (for GPU) `cupy`. NO OpenMC / nuclear-data
library required on the GPU box -- it reads the pre-extracted material grids
(xslookup_*.npz + .names.json) produced by xs_lookup_harness.py `build` on the
data machine.  Falls back to NumPy/CPU if CuPy is absent (for sanity checks).

It benchmarks three EXACT methods and reports lookups/s + speedups, with correct
GPU synchronisation so the timings are real (not async launch latency):
  * baseline      : per-nuclide binary search (xp.searchsorted), M searches
  * unionized     : one search into the unionized grid + O(1) per-nuclide gathers
  * hash_union    : O(1) uniform-log hash on the union grid + fixed-W local
                    correction (divergence-free, GPU-friendly), then gathers

Usage on the CUDA box:
    pip install cupy-cuda12x        # match your CUDA toolkit (or cupy-cuda11x)
    python xs_lookup_gpu.py --data-dir . --material assembly --n-queries 20000000
    python xs_lookup_gpu.py --data-dir . --material assembly --fp32   # consumer GPUs
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    xp = np
    GPU = False


def sync():
    if GPU:
        cp.cuda.runtime.deviceSynchronize()


def device_name():
    if not GPU:
        return "CPU (numpy fallback -- install cupy for GPU)"
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def load_material(data_dir: Path, material: str, dtype):
    npz = Path(data_dir) / f"xslookup_{material}.npz"
    names = json.loads((npz.with_suffix(".names.json")).read_text())
    d = np.load(npz)
    grids = []
    for i in range(len(names)):
        E = xp.asarray(d[f"E_{i}"], dtype=dtype)
        S = xp.asarray(d[f"S_{i}"], dtype=dtype)
        grids.append((E, S))
    dens = xp.asarray(d["densities"], dtype=dtype)
    # keep host copies of grids for ground-truth accuracy on CPU
    host = [(np.asarray(d[f"E_{i}"], np.float64), np.asarray(d[f"S_{i}"], np.float64))
            for i in range(len(names))]
    return {"names": names, "grids": grids, "densities": dens,
            "host_grids": host, "host_dens": np.asarray(d["densities"], np.float64)}


def make_queries(n, seed, dtype, e_min=1e-3, e_max=2e7):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    q = (e_min * (e_max / e_min) ** u)
    return xp.asarray(q, dtype=dtype), q  # device, host


def interp_accumulate(Eq, grids, dens, idx_list):
    out = xp.zeros_like(Eq)
    for (E, S), nd, idx in zip(grids, dens, idx_list):
        e0 = E[idx]; e1 = E[idx + 1]; s0 = S[idx]; s1 = S[idx + 1]
        out = out + nd * (s0 + (Eq - e0) / (e1 - e0) * (s1 - s0))
    return out


# ---- baseline: per-nuclide binary search ----
def build_baseline(mat):
    return None
def lookup_baseline(Eq, mat, st):
    idxs = [xp.clip(xp.searchsorted(E, Eq, side="right") - 1, 0, E.size - 2)
            for (E, S) in mat["grids"]]
    return interp_accumulate(Eq, mat["grids"], mat["densities"], idxs)


# ---- unionized grid ----
def build_unionized(mat):
    union = xp.unique(xp.concatenate([E for (E, S) in mat["grids"]]))
    maps = [xp.clip(xp.searchsorted(E, union, side="right") - 1, 0, E.size - 2).astype(xp.int32)
            for (E, S) in mat["grids"]]
    return {"union": union, "maps": maps}
def lookup_unionized(Eq, mat, st):
    u = xp.clip(xp.searchsorted(st["union"], Eq, side="right") - 1, 0, st["union"].size - 1)
    idxs = [mp[u] for mp in st["maps"]]
    return interp_accumulate(Eq, mat["grids"], mat["densities"], idxs)


# ---- hash-indexed unionized (fixed-W correction; divergence-free on GPU) ----
def build_hash(mat, bits=24, W=8):
    union = xp.unique(xp.concatenate([E for (E, S) in mat["grids"]]))
    maps = [xp.clip(xp.searchsorted(E, union, side="right") - 1, 0, E.size - 2).astype(xp.int32)
            for (E, S) in mat["grids"]]
    logU = xp.log10(union)
    lmin = float(logU[0]); span = max(float(logU[-1]) - lmin, 1e-12)
    K = 1 << bits
    edges = lmin + (xp.arange(K + 1, dtype=xp.float64) / K) * span
    start = xp.clip(xp.searchsorted(logU, edges, side="left"), 0, union.size - 1).astype(xp.int32)
    return {"union": union, "maps": maps, "lmin": lmin, "span": span, "K": K,
            "start": start, "Nu": int(union.size), "W": W}
def lookup_hash(Eq, mat, st):
    Nu = st["Nu"]; W = st["W"]
    lg = xp.log10(Eq)
    k = xp.clip(((lg - st["lmin"]) / st["span"] * st["K"]).astype(xp.int64), 0, st["K"] - 1)
    u = st["start"][k].astype(xp.int64)
    union = st["union"]
    # fixed-W bounded correction (no data-dependent early-break -> no warp divergence)
    for _ in range(W):
        nxt = xp.minimum(u + 1, Nu - 1)
        adv = (union[nxt] <= Eq) & (u < Nu - 1)
        u = xp.minimum(xp.where(adv, u + 1, u), Nu - 1)
    for _ in range(W):
        ret = (union[u] > Eq) & (u > 0)
        u = xp.where(ret, u - 1, u)
    idxs = [mp[u] for mp in st["maps"]]
    return interp_accumulate(Eq, mat["grids"], mat["densities"], idxs)


METHODS = [
    ("baseline (binary search)", build_baseline, lookup_baseline),
    ("unionized_grid", build_unionized, lookup_unionized),
    ("hash_union(bits=24,W=8)", build_hash, lookup_hash),
]


def host_truth(q_host, mat):
    out = np.zeros_like(q_host)
    for (E, S), nd in zip(mat["host_grids"], mat["host_dens"]):
        idx = np.clip(np.searchsorted(E, q_host, side="right") - 1, 0, E.size - 2)
        e0 = E[idx]; e1 = E[idx + 1]; s0 = S[idx]; s1 = S[idx + 1]
        out += nd * (s0 + (q_host - e0) / (e1 - e0) * (s1 - s0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--material", default="assembly")
    p.add_argument("--n-queries", type=int, default=10_000_000)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--fp32", action="store_true", help="use float32 (fast on consumer GPUs)")
    args = p.parse_args()

    dtype = xp.float32 if args.fp32 else xp.float64
    print(f"device : {device_name()}")
    print(f"backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'} | dtype={'fp32' if args.fp32 else 'fp64'}")

    mat = load_material(args.data_dir, args.material, dtype)
    Eq, q_host = make_queries(args.n_queries, args.seed, dtype)
    tot = sum(int(E.size) for (E, S) in mat["grids"])
    print(f"material '{args.material}': {len(mat['names'])} nuclides, {tot:,} grid pts, "
          f"{args.n_queries:,} queries\n")

    truth = host_truth(q_host, mat)
    base_lps = None
    print(f"{'method':<28}{'lookups/s':>16}{'speedup':>10}{'max_rel':>12}{'ok':>5}")
    for name, build, lookup in METHODS:
        st = build(mat) if build.__name__ != "build_baseline" else None
        # warmup + accuracy
        val = lookup(Eq, mat, st); sync()
        v = (cp.asnumpy(val) if GPU else val).astype(np.float64)
        rel = np.abs(v - truth) / np.maximum(np.abs(truth), 1e-30)
        max_rel = float(rel.max())
        # tolerance: fp32 interpolation has ~1e-6 rounding; fp64 should be ~0
        tol = 5e-6 if args.fp32 else 1e-9
        ts = []
        for _ in range(args.repeats):
            sync(); t0 = time.perf_counter()
            lookup(Eq, mat, st); sync()
            ts.append(time.perf_counter() - t0)
        lps = args.n_queries / float(np.median(ts))
        if base_lps is None:
            base_lps = lps
        ok = "Y" if max_rel < tol else "N"
        print(f"{name:<28}{lps:>16,.0f}{lps/base_lps:>9.2f}x{max_rel:>12.1e}{ok:>5}")

    print("\nNote: 'baseline' here is per-nuclide binary search; also compare hash vs unionized.")
    if args.fp32:
        print("fp32: tiny interpolation rounding is expected; use fp64 for bit-exact checks.")


if __name__ == "__main__":
    main()
