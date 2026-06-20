"""Train a direct neural surrogate for macroscopic cross sections.

The target is log10(Sigma_r) for homogenized Gen-IV-inspired material proxies.
This is the recommended A100 experiment because one network call predicts the
macroscopic value directly, instead of calling a microscopic network once per
nuclide and reaction channel.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from train_xs_surrogate_torch import FourierFeatures, pick_device, region_of


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class MacroXSNet(nn.Module):
    def __init__(
        self,
        n_reactions: int,
        n_numeric: int,
        n_fourier: int,
        fourier_scale: float,
        hidden: int,
        depth: int,
        seed: int,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.ff = FourierFeatures(n_fourier, fourier_scale, seed=seed)
        in_dim = 2 * n_fourier + 1 + n_numeric + n_reactions
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.LayerNorm(hidden)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, loge_scaled, numeric_scaled, reaction_onehot):
        x = torch.cat([self.ff(loge_scaled), loge_scaled, numeric_scaled, reaction_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def load_dataset(path: Path, max_rows: int | None, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    npz = np.load(path)
    data = {k: npz[k] for k in npz.files}
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    n = int(data["log10_macro_xs_cm_inv"].size)
    if max_rows is not None and n > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
        data = {k: v[idx] for k, v in data.items()}
    return data, meta


def numeric_columns(meta: dict) -> list[str]:
    return ["temperature_K", "atom_density_atom_per_bcm"] + [f"frac_{n}" for n in meta["nuclides"]]


def build_arrays(data: dict[str, np.ndarray], meta: dict):
    cols = numeric_columns(meta)
    loge = data["log10_energy_eV"].astype(np.float32)
    numeric = np.stack([data[c].astype(np.float32) for c in cols], axis=1)
    reaction = data["reaction_id"].astype(np.int64)
    y = data["log10_macro_xs_cm_inv"].astype(np.float32)
    return loge, numeric, reaction, y


def standardize(train_idx, loge, numeric):
    loge_mu = float(loge[train_idx].mean())
    loge_sd = float(loge[train_idx].std() + 1.0e-8)
    num_mu = numeric[train_idx].mean(axis=0).astype(np.float32)
    num_sd = (numeric[train_idx].std(axis=0) + 1.0e-8).astype(np.float32)
    return loge_mu, loge_sd, num_mu, num_sd


def make_split(split: str, data: dict[str, np.ndarray], meta: dict, rng, args):
    n = data["log10_macro_xs_cm_inv"].size
    if split == "random":
        idx = rng.permutation(n)
        cut = int(0.85 * n)
        return idx[:cut], idx[cut:]
    if split == "heldout_material":
        material = args.heldout_material or list(meta["material_ids"])[-1]
        if material not in meta["material_ids"]:
            raise KeyError(f"Unknown held-out material '{material}'.")
        hold = meta["material_ids"][material]
        test = np.where(data["material_id"] == hold)[0]
        train = np.where(data["material_id"] != hold)[0]
        return train, test
    if split == "heldout_temperature":
        temp = args.heldout_temperature
        if temp is None:
            temp = int(np.max(data["temperature_K"]))
        test = np.where(np.isclose(data["temperature_K"], temp))[0]
        train = np.where(~np.isclose(data["temperature_K"], temp))[0]
        return train, test
    raise ValueError(split)


def batch_to_device(loge_s, num_s, rxn_oh, y, idx, device):
    return (
        torch.from_numpy(loge_s[idx]).view(-1, 1).to(device),
        torch.from_numpy(num_s[idx]).to(device),
        torch.from_numpy(rxn_oh[idx]).to(device),
        torch.from_numpy(y[idx]).to(device),
    )


def validation_loss(net, loge_s, num_s, rxn_oh, y, idx, device, batch_size, use_amp):
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    net.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            xb = batch_to_device(loge_s, num_s, rxn_oh, y, b, device)
            with autocast_context(device, use_amp):
                pred = net(xb[0], xb[1], xb[2])
                loss = loss_fn(pred.float(), xb[3])
            total += float(loss.detach().cpu())
            count += len(b)
    return total / max(count, 1)


def predict_log(net, loge_s, num_s, rxn_oh, idx, device, batch_size, use_amp) -> np.ndarray:
    out = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            xb = (
                torch.from_numpy(loge_s[b]).view(-1, 1).to(device),
                torch.from_numpy(num_s[b]).to(device),
                torch.from_numpy(rxn_oh[b]).to(device),
            )
            with autocast_context(device, use_amp):
                pred = net(*xb)
            out.append(pred.float().detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(out)


def train_one(net, loge_s, num_s, rxn_oh, y, tr_idx, va_idx, device, args, seed: int):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    bad = 0

    for epoch in range(args.epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for start in range(0, len(perm), args.batch_size):
            b = perm[start : start + args.batch_size]
            xb = batch_to_device(loge_s, num_s, rxn_oh, y, b, device)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                pred = net(xb[0], xb[1], xb[2])
                loss = loss_fn(pred.float(), xb[3])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
        sched.step()

        val = validation_loss(net, loge_s, num_s, rxn_oh, y, va_idx, device, args.eval_batch_size, args.amp)
        if val < best - 1.0e-6:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
        if args.verbose:
            print(f"    epoch {epoch + 1:03d}: val_mse_log10={val:.6f}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)
    return best


def metrics(energy_ev: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    yt = np.power(10.0, y_true_log)
    yp = np.power(10.0, y_pred_log)
    rel = np.abs(yp - yt) / np.maximum(np.abs(yt), 1.0e-30)
    regions = region_of(energy_ev)
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
            "rmse_log10": float(np.sqrt(np.mean((y_pred_log[mask] - y_true_log[mask]) ** 2))),
        }
    return out


def run_split(split: str, data: dict[str, np.ndarray], meta: dict, device: torch.device, args) -> dict:
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = make_split(split, data, meta, rng, args)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"Split '{split}' produced empty train/test arrays.")

    rng.shuffle(train_idx)
    val_cut = int(0.9 * len(train_idx))
    tr_idx = train_idx[:val_cut]
    va_idx = train_idx[val_cut:]

    loge, numeric, reaction, y = build_arrays(data, meta)
    loge_mu, loge_sd, num_mu, num_sd = standardize(tr_idx, loge, numeric)
    loge_s = ((loge - loge_mu) / loge_sd).astype(np.float32)
    num_s = ((numeric - num_mu) / num_sd).astype(np.float32)
    n_rxn = len(meta["reaction_ids"])
    rxn_oh = np.eye(n_rxn, dtype=np.float32)[reaction]

    preds = []
    val_losses = []
    t0 = time.perf_counter()
    for k in range(args.ensemble):
        net = MacroXSNet(
            n_reactions=n_rxn,
            n_numeric=numeric.shape[1],
            n_fourier=args.fourier_features,
            fourier_scale=args.fourier_scale,
            hidden=args.hidden,
            depth=args.depth,
            seed=args.seed + 131 * k,
        ).to(device)
        val = train_one(net, loge_s, num_s, rxn_oh, y, tr_idx, va_idx, device, args, args.seed + k)
        val_losses.append(val)
        pred = predict_log(net, loge_s, num_s, rxn_oh, test_idx, device, args.eval_batch_size, args.amp)
        preds.append(pred)
        if split == "random" and k == 0:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "config": vars(args),
                    "scaler": {
                        "logE_mu": loge_mu,
                        "logE_sd": loge_sd,
                        "num_mu": num_mu.tolist(),
                        "num_sd": num_sd.tolist(),
                    },
                    "meta": meta,
                    "numeric_columns": numeric_columns(meta),
                },
                MODEL_DIR / f"{args.name}_surrogate.pt",
            )

    pred_mean = np.stack(preds, axis=0).mean(axis=0)
    result = {
        "split": split,
        "n_train": int(len(tr_idx)),
        "n_validation": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "ensemble": int(args.ensemble),
        "mean_val_loss_log10_mse": float(np.mean(val_losses)),
        "train_seconds": float(time.perf_counter() - t0),
        "metrics": metrics(data["energy_eV"][test_idx], y[test_idx], pred_mean),
    }
    if split == "heldout_material":
        result["heldout_material"] = args.heldout_material or list(meta["material_ids"])[-1]
    if split == "heldout_temperature":
        result["heldout_temperature_K"] = int(args.heldout_temperature or np.max(data["temperature_K"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROCESSED_DATA_DIR / "macro_xs_dataset.npz"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--eval-batch-size", type=int, default=262_144)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fourier-features", type=int, default=160)
    parser.add_argument("--fourier-scale", type=float, default=10.0)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true", help="Use CUDA bfloat16 autocast.")
    parser.add_argument("--max-rows", type=int, default=None, help="Diagnostic subsample for smoke tests.")
    parser.add_argument("--heldout-material", default="msr_flibe_fuel_salt")
    parser.add_argument("--heldout-temperature", type=int, default=1200)
    parser.add_argument("--splits", nargs="+", default=["random", "heldout_material", "heldout_temperature"])
    parser.add_argument("--name", default="macro_xs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    device = pick_device(args.device)
    configure_torch(device)
    print(f"device = {device}; amp = {bool(args.amp and device.type == 'cuda')}")

    data, meta = load_dataset(Path(args.dataset), args.max_rows, args.seed)
    print(f"dataset rows = {data['log10_macro_xs_cm_inv'].size:,}")

    report = {"device": str(device), "dataset": args.dataset, "config": vars(args), "splits": {}}
    out = MODEL_DIR / f"{args.name}_surrogate_metrics.json"
    for split in args.splits:
        print(f"\n=== split: {split} ===", flush=True)
        result = run_split(split, data, meta, device, args)
        report["splits"][split] = result
        for region, values in result["metrics"].items():
            print(
                f"  {region:<22} median={100*values['median_rel_error']:7.3f}% "
                f"p95={100*values['p95_rel_error']:7.3f}% "
                f"within5={100*values['within_5pct']:6.2f}% n={values['n']:,}",
                flush=True,
            )
        out.write_text(json.dumps(report, indent=2))
        print(f"  [saved {out.name} after split '{split}']", flush=True)

    print(f"\nWrote {out}")
    print(f"Saved serving model to {MODEL_DIR / f'{args.name}_surrogate.pt'}")


if __name__ == "__main__":
    main()
