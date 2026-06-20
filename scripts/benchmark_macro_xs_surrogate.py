"""Benchmark the direct macroscopic XS surrogate against OpenMC accumulation.

This benchmark is the honest GPU test for the A100 path:

  OpenMC reference:
    sum_i N_i sigma_i(E, T) using HDF5 Tabulated1D interpolation.

  ML surrogate:
    one batched neural call for Sigma_r(E, T, composition).

The model is only a candidate accelerator when both speed and error are acceptable
inside the trained material/temperature/reaction envelope.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_macro_xs_dataset import MATERIAL_DEFINITIONS, NUCLIDES, library_paths, load_incident_neutron, macro_xs, normalize_fractions
from project_config import DEFAULT_CROSS_SECTIONS, MODEL_DIR
from train_macro_xs_surrogate_torch import MacroXSNet, configure_torch
from train_xs_surrogate_torch import pick_device, region_of


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    meta = ckpt["meta"]
    net = MacroXSNet(
        n_reactions=len(meta["reaction_ids"]),
        n_numeric=len(ckpt["numeric_columns"]),
        n_fourier=cfg["fourier_features"],
        fourier_scale=cfg["fourier_scale"],
        hidden=cfg["hidden"],
        depth=cfg["depth"],
        seed=cfg["seed"],
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def material_numeric(material_name: str, temp_k: int, meta: dict, n_rows: int) -> np.ndarray:
    material = MATERIAL_DEFINITIONS[material_name]
    fractions = normalize_fractions(material["fractions"])
    cols = [
        np.full(n_rows, temp_k, dtype=np.float32),
        np.full(n_rows, float(material["atom_density_atom_per_bcm"]), dtype=np.float32),
    ]
    cols.extend(np.full(n_rows, fractions.get(nuclide, 0.0), dtype=np.float32) for nuclide in meta["nuclides"])
    return np.stack(cols, axis=1)


def featurize(
    energies: np.ndarray,
    temp_k: int,
    material_name: str,
    reaction: str,
    ckpt: dict,
    device: torch.device,
):
    sc = ckpt["scaler"]
    meta = ckpt["meta"]
    n = len(energies)
    loge = np.log10(energies).astype(np.float32)
    numeric = material_numeric(material_name, temp_k, meta, n)
    loge_s = ((loge - sc["logE_mu"]) / sc["logE_sd"]).astype(np.float32)
    num_s = ((numeric - np.asarray(sc["num_mu"], dtype=np.float32)) / np.asarray(sc["num_sd"], dtype=np.float32)).astype(np.float32)
    rxn_id = meta["reaction_ids"][reaction]
    rxn_oh = np.eye(len(meta["reaction_ids"]), dtype=np.float32)[np.full(n, rxn_id, dtype=np.int64)]
    return (
        torch.from_numpy(loge_s).view(-1, 1).to(device),
        torch.from_numpy(num_s).to(device),
        torch.from_numpy(rxn_oh).to(device),
    )


def predict(
    net,
    energies: np.ndarray,
    temp_k: int,
    material_name: str,
    reaction: str,
    ckpt: dict,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    out = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(energies), chunk_size):
            chunk = energies[start : start + chunk_size]
            packs = featurize(chunk, temp_k, material_name, reaction, ckpt, device)
            with autocast_context(device, use_amp):
                pred = net(*packs)
            out.append(pred.float().detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.power(10.0, np.concatenate(out)), time.perf_counter() - t0


def error_metrics(energies: np.ndarray, ref: np.ndarray, pred: np.ndarray) -> dict:
    rel = np.abs(pred - ref) / np.maximum(np.abs(ref), 1.0e-30)
    regions = region_of(energies)
    out = {}
    for region in ["thermal", "epithermal_resonance", "fast", "all"]:
        mask = np.ones_like(rel, dtype=bool) if region == "all" else (regions == region)
        if not np.any(mask):
            continue
        r = rel[mask]
        out[region] = {
            "n": int(mask.sum()),
            "median_rel_error": float(np.median(r)),
            "mean_rel_error": float(np.mean(r)),
            "p95_rel_error": float(np.percentile(r, 95)),
            "p99_rel_error": float(np.percentile(r, 99)),
            "within_1pct": float(np.mean(r <= 0.01)),
            "within_5pct": float(np.mean(r <= 0.05)),
            "within_10pct": float(np.mean(r <= 0.10)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_DIR / "macro_xs_surrogate.pt"))
    parser.add_argument("--cross-sections", default=str(DEFAULT_CROSS_SECTIONS))
    parser.add_argument("--material", default="sfr_mox_fast", choices=sorted(MATERIAL_DEFINITIONS))
    parser.add_argument("--reaction", default="major")
    parser.add_argument("--temperature", type=int, default=900)
    parser.add_argument("--n-queries", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA bfloat16 autocast.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(MODEL_DIR / "macro_xs_benchmark.json"))
    args = parser.parse_args()

    if args.n_queries <= 0:
        raise ValueError("--n-queries must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    device = pick_device(args.device)
    configure_torch(device)
    net, ckpt = load_model(Path(args.model), device)

    xml = Path(args.cross_sections).expanduser().resolve()
    paths = library_paths(xml)
    needed = sorted(set(NUCLIDES) & set(paths))
    cache = load_incident_neutron(paths, needed)

    rng = np.random.default_rng(args.seed)
    energies = np.power(10.0, rng.uniform(-5.0, np.log10(2.0e7), args.n_queries)).astype(np.float64)

    t0 = time.perf_counter()
    ref = macro_xs(cache, MATERIAL_DEFINITIONS[args.material], args.reaction, energies, args.temperature)
    openmc_seconds = time.perf_counter() - t0

    pred, surrogate_seconds = predict(
        net,
        energies,
        args.temperature,
        args.material,
        args.reaction,
        ckpt,
        device,
        args.chunk_size,
        args.amp,
    )

    report = {
        **describe_device(device),
        "model": str(Path(args.model).resolve()),
        "cross_sections": str(xml),
        "material": args.material,
        "reaction": args.reaction,
        "temperature_K": args.temperature,
        "n_queries": args.n_queries,
        "chunk_size": args.chunk_size,
        "amp": bool(args.amp and device.type == "cuda"),
        "openmc_seconds": openmc_seconds,
        "openmc_queries_per_second": args.n_queries / openmc_seconds,
        "surrogate_seconds": surrogate_seconds,
        "surrogate_queries_per_second": args.n_queries / surrogate_seconds,
        "surrogate_speedup_vs_openmc": openmc_seconds / surrogate_seconds,
        "error_metrics": error_metrics(energies, ref, pred),
        "speed_claim_allowed": bool(openmc_seconds > surrogate_seconds),
        "note": (
            "This is a restricted-domain macroscopic benchmark. A speedup here supports "
            "a GPU-batched material-response accelerator claim, not replacement of full "
            "OpenMC transport or a universal XSBench kernel."
        ),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
