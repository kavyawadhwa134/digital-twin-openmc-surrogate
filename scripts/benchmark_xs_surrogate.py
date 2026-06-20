"""Fair speed benchmark for the neural cross-section surrogate vs OpenMC HDF5 lookup.

Three comparisons, each reported honestly:

  1. Raw batched inference throughput (queries/s) on accelerator vs CPU. This is the
     surrogate's intrinsic serving rate.

  2. Single-nuclide vectorised lookup: OpenMC's Tabulated1D interpolation over a sorted grid
     is a cache-friendly, vectorised numpy call and is genuinely very fast. This is the honest
     "lookup is hard to beat per nuclide" baseline.

  3. XSBench-style macroscopic accumulation: a material with M nuclides queried at N random
     neutron energies, accumulating the macroscopic cross section. This is the kernel the
     abstract targets (the ~80% MC bottleneck): real transport does irregular, memory-bound,
     per-nuclide energy-grid searches. The surrogate turns this into one dense batched forward
     pass. We measure both wall times and the resulting accuracy (which is limited by the
     surrogate's resonance-region error, reported alongside so the trade-off is explicit).
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import openmc.data
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import DEFAULT_CROSS_SECTIONS, MODEL_DIR, NUCLIDE_PROPERTIES
from train_xs_surrogate_torch import NUMERIC, XSNet, pick_device


def library_paths(xml: Path) -> dict[str, Path]:
    root = ET.parse(xml).getroot()
    out = {}
    for lib in root.findall("library"):
        if lib.get("type") != "neutron":
            continue
        rel = lib.get("path")
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = xml.parent / p
        for m in (lib.get("materials") or "").split():
            out[m] = p
    return out


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    meta = ckpt["meta"]
    net = XSNet(
        n_nuclides=len(meta["nuclide_ids"]),
        n_reactions=len(meta["reaction_ids"]),
        n_fourier=cfg["fourier_features"],
        fourier_scale=cfg["fourier_scale"],
        hidden=cfg["hidden"],
        depth=cfg["depth"],
        use_embedding=ckpt["use_embedding"],
        emb_dim=cfg["emb_dim"],
        seed=cfg["seed"],
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def featurize(energies, temp_k, nuclide, reaction, ckpt, device):
    sc = ckpt["scaler"]
    meta = ckpt["meta"]
    props = NUCLIDE_PROPERTIES[nuclide]
    n = len(energies)
    logE = np.log10(energies).astype(np.float32)
    numeric = np.stack([
        np.full(n, temp_k, dtype=np.float32),
        np.full(n, props["z"], dtype=np.float32),
        np.full(n, props["a"], dtype=np.float32),
        np.full(n, props["a"] - props["z"], dtype=np.float32),
        np.full(n, props["is_actinide"], dtype=np.float32),
        np.full(n, props["is_fissile"], dtype=np.float32),
    ], axis=1)
    logE_s = (logE - sc["logE_mu"]) / sc["logE_sd"]
    num_s = (numeric - np.array(sc["num_mu"], dtype=np.float32)) / np.array(sc["num_sd"], dtype=np.float32)
    n_rxn = len(meta["reaction_ids"])
    rxn_oh = np.eye(n_rxn, dtype=np.float32)[np.full(n, meta["reaction_ids"][reaction])]
    nuc = np.full(n, meta["nuclide_ids"][nuclide], dtype=np.int64)
    return (
        torch.from_numpy(logE_s).view(-1, 1).to(device),
        torch.from_numpy(num_s).to(device),
        torch.from_numpy(rxn_oh).to(device),
        torch.from_numpy(nuc).to(device),
    )


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def describe_device(device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return {
            "device": str(device),
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / 1024**3,
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        }
    if device.type == "mps":
        return {"device": str(device), "device_name": "Apple Metal Performance Shaders"}
    return {"device": str(device), "device_name": "CPU"}


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def time_forward(net, packs, device, reps=3, use_amp=False):
    # warmup
    with torch.no_grad():
        with autocast_context(device, use_amp):
            net(*packs)
    sync(device)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            with autocast_context(device, use_amp):
                net(*packs)
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def predict_log_batched(
    net,
    energies: np.ndarray,
    temp_k: int,
    nuclide: str,
    reaction: str,
    ckpt: dict,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(energies), chunk_size):
            chunk = energies[start : start + chunk_size]
            packs = featurize(chunk, temp_k, nuclide, reaction, ckpt, device)
            with autocast_context(device, use_amp):
                pred = net(*packs)
            out.append(pred.float().detach().cpu().numpy())
    sync(device)
    return np.concatenate(out), time.perf_counter() - t0


def predict_macro_from_micro_batched(
    net,
    energies: np.ndarray,
    temp_k: int,
    material_nuclides: list[str],
    reactions: list[str],
    number_densities: dict[str, float],
    ckpt: dict,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    macro_chunks = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(energies), chunk_size):
            chunk = energies[start : start + chunk_size]
            macro = torch.zeros(len(chunk), device=device, dtype=torch.float32)
            for n in material_nuclides:
                for rname in reactions:
                    packs = featurize(chunk, temp_k, n, rname, ckpt, device)
                    with autocast_context(device, use_amp):
                        pred_log = net(*packs)
                    macro += float(number_densities[n]) * torch.pow(10.0, pred_log.float())
            macro_chunks.append(macro.detach().cpu().numpy())
    sync(device)
    return np.concatenate(macro_chunks), time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(MODEL_DIR / "xs_torch_surrogate.pt"))
    p.add_argument("--cross-sections", default=str(DEFAULT_CROSS_SECTIONS))
    p.add_argument("--n-queries", type=int, default=1_000_000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Inference chunk size. Lower this if the GPU runs out of memory.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA bfloat16 autocast for inference. Faster on A100, slightly different numerics.",
    )
    p.add_argument("--material-nuclides", nargs="+",
                   default=["U235", "U238", "O16", "Fe56", "Na23"])
    p.add_argument("--temperature", type=int, default=900)
    p.add_argument("--out", default=str(MODEL_DIR / "xs_torch_benchmark.json"))
    args = p.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    xml = Path(args.cross_sections).resolve()
    paths = library_paths(xml)
    device = pick_device(args.device)
    configure_torch(device)
    cpu = torch.device("cpu")
    net_accel, ckpt = load_model(Path(args.model), device)
    net_cpu, _ = load_model(Path(args.model), cpu)
    report = {
        **describe_device(device),
        "n_queries": args.n_queries,
        "temperature_K": args.temperature,
        "chunk_size": args.chunk_size,
        "amp": bool(args.amp and device.type == "cuda"),
    }

    # ---- Benchmark 1: raw inference throughput (U238 capture energies) ----
    N = args.n_queries
    energies = np.logspace(-5, np.log10(2.0e7), N)
    raw_chunk = energies[: min(N, args.chunk_size)]
    packs_accel = featurize(raw_chunk, args.temperature, "U238", "capture", ckpt, device)
    packs_cpu = featurize(raw_chunk, args.temperature, "U238", "capture", ckpt, cpu)
    t_accel_chunk = time_forward(net_accel, packs_accel, device, use_amp=args.amp)
    t_cpu = time_forward(net_cpu, packs_cpu, cpu)
    report["raw_inference"] = {
        "accelerator_seconds_for_chunk": t_accel_chunk,
        "accelerator_chunk_size": len(raw_chunk),
        "accelerator_queries_per_second": len(raw_chunk) / t_accel_chunk,
        "cpu_seconds_for_chunk": t_cpu,
        "cpu_chunk_size": len(raw_chunk),
        "cpu_queries_per_second": len(raw_chunk) / t_cpu,
    }

    # ---- Benchmark 2: single-nuclide vs OpenMC vectorised interpolation ----
    nuc = "U238"
    data = openmc.data.IncidentNeutron.from_hdf5(paths[nuc])
    tlab = {int(round(float(t.rstrip("K")))): t for t in data.temperatures}[args.temperature]
    rx = data.reactions[102]
    t0 = time.perf_counter()
    ref = rx.xs[tlab](energies)
    openmc_seconds = time.perf_counter() - t0
    pred_log, surrogate_seconds = predict_log_batched(
        net_accel,
        energies,
        args.temperature,
        nuc,
        "capture",
        ckpt,
        device,
        args.chunk_size,
        args.amp,
    )
    pred = np.power(10.0, pred_log)
    rel = np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-30)
    report["single_nuclide_vs_openmc"] = {
        "nuclide": nuc, "reaction": "capture",
        "openmc_lookup_seconds": openmc_seconds,
        "openmc_queries_per_second": N / openmc_seconds,
        "surrogate_seconds": surrogate_seconds,
        "surrogate_queries_per_second": N / surrogate_seconds,
        "surrogate_speedup_vs_openmc": openmc_seconds / surrogate_seconds,
        "surrogate_median_rel_error": float(np.median(rel)),
        "surrogate_p95_rel_error": float(np.percentile(rel, 95)),
        "note": "OpenMC per-nuclide vectorised interpolation is a strong, cache-friendly baseline.",
    }

    # ---- Benchmark 3: XSBench-style macroscopic accumulation ----
    M = [n for n in args.material_nuclides if n in paths]
    rng = np.random.default_rng(0)
    qE = np.power(10.0, rng.uniform(-5, np.log10(2.0e7), N)).astype(np.float64)
    # number densities (arbitrary but fixed) for a mock material
    ndens = {n: 1.0 for n in M}

    # OpenMC path: per-nuclide vectorised interpolation (capture+elastic) + accumulate.
    # Load HDF5 data before timing so the number is a kernel/lookup benchmark rather than
    # a file-system benchmark.
    reactions_mt = {"capture": 102, "elastic": 2}
    macro_lookup = {}
    for n in M:
        d = openmc.data.IncidentNeutron.from_hdf5(paths[n])
        tl = {int(round(float(t.rstrip("K")))): t for t in d.temperatures}[args.temperature]
        macro_lookup[n] = []
        for mt in reactions_mt.values():
            if mt in d.reactions and tl in d.reactions[mt].xs:
                macro_lookup[n].append(d.reactions[mt].xs[tl])
    t0 = time.perf_counter()
    macro_ref = np.zeros(N)
    for n, funcs in macro_lookup.items():
        for xsfun in funcs:
            macro_ref += ndens[n] * xsfun(qE)
    openmc_macro_seconds = time.perf_counter() - t0

    # Surrogate path: chunked batched forward per (nuclide, reaction), accumulate on accelerator.
    macro_sur_np, surrogate_macro_seconds = predict_macro_from_micro_batched(
        net_accel,
        qE,
        args.temperature,
        M,
        list(reactions_mt),
        ndens,
        ckpt,
        device,
        args.chunk_size,
        args.amp,
    )
    macro_rel = np.abs(macro_sur_np - macro_ref) / np.maximum(np.abs(macro_ref), 1e-30)
    report["xsbench_style_macroscopic"] = {
        "material_nuclides": M,
        "reactions": list(reactions_mt),
        "n_energy_samples": N,
        "openmc_seconds": openmc_macro_seconds,
        "surrogate_seconds": surrogate_macro_seconds,
        "surrogate_speedup": openmc_macro_seconds / surrogate_macro_seconds,
        "macroscopic_median_rel_error": float(np.median(macro_rel)),
        "macroscopic_p95_rel_error": float(np.percentile(macro_rel, 95)),
        "note": (
            "Surrogate replaces per-nuclide grid lookups with dense batched inference. "
            "Speed and accuracy are both reported; resonance-region error propagates into "
            "the macroscopic accuracy, which is why a UQ-gated hybrid is the honest design."
        ),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
