"""GPU (Apple MPS / CUDA) neural surrogate for microscopic cross sections, with UQ.

This is the abstract's headline component: replace memory-bound microscopic cross-section
lookups with dense neural inference, with honest accuracy reporting across the epithermal
resonance region and bounded predictive uncertainty.

Design choices that matter:
  * Trained on the NATIVE, resonance-resolved ENDF grid (see extract_xs_resonance_dataset.py),
    not a coarse resampled grid, so resonance fidelity is actually testable.
  * Target is log10(sigma): cross sections span ~10 orders of magnitude.
  * Random Fourier features on log10(E) (Tancik et al. 2020) counter the spectral bias of a
    plain MLP, which otherwise blurs the rapidly-oscillating resonance structure.
  * A deep ensemble (K independently-initialised nets) gives a predictive mean and standard
    deviation, i.e. the "rigorously bounded prediction uncertainty" the abstract promises.
  * Accuracy is reported per energy region (thermal / epithermal-resonance / fast) so the hard
    region is never hidden inside a global average.
  * Three splits: random interpolation, held-out nuclide (generalisation), held-out temperature.

Honesty note: a single smooth network cannot perfectly reproduce thousands of narrow
resonances. The point is to characterise *where* the surrogate is trustworthy (and bound it
with UQ), not to claim resonance-perfect fidelity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

NUMERIC = ["temperature_K", "nuclide_Z", "nuclide_A", "nuclide_N", "is_actinide", "is_fissile"]


def pick_device(prefer: str) -> torch.device:
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def region_of(energy_ev: np.ndarray) -> np.ndarray:
    r = np.full(energy_ev.shape, "fast", dtype=object)
    r[energy_ev < 1.0e5] = "epithermal_resonance"
    r[energy_ev < 0.625] = "thermal"
    return r


class FourierFeatures(nn.Module):
    """Fixed random Fourier features on a scalar input to represent high-frequency structure."""

    def __init__(self, n_features: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(n_features, generator=g) * scale
        self.register_buffer("b", b)

    def forward(self, x):  # x: (N,1) scaled log-energy
        proj = 2.0 * np.pi * x * self.b  # (N, n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class XSNet(nn.Module):
    def __init__(
        self,
        n_nuclides: int,
        n_reactions: int,
        n_fourier: int,
        fourier_scale: float,
        hidden: int,
        depth: int,
        use_embedding: bool,
        emb_dim: int,
        seed: int,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.use_embedding = use_embedding
        self.ff = FourierFeatures(n_fourier, fourier_scale, seed=seed)
        self.n_reactions = n_reactions
        in_dim = 2 * n_fourier + 1 + len(NUMERIC) + n_reactions  # +1 raw logE
        if use_embedding:
            self.nuc_emb = nn.Embedding(n_nuclides, emb_dim)
            in_dim += emb_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, logE_scaled, numeric, reaction_onehot, nuclide_id):
        feats = [self.ff(logE_scaled), logE_scaled, numeric, reaction_onehot]
        if self.use_embedding:
            feats.append(self.nuc_emb(nuclide_id))
        return self.mlp(torch.cat(feats, dim=-1)).squeeze(-1)


def build_tensors(data: dict, meta: dict, device: torch.device):
    logE = data["log10_energy_eV"].astype(np.float32)
    numeric = np.stack([data[c].astype(np.float32) for c in NUMERIC], axis=1)
    y = data["log10_xs_barns"].astype(np.float32)
    nuc = data["nuclide_id"].astype(np.int64)
    rxn = data["reaction_id"].astype(np.int64)
    return logE, numeric, y, nuc, rxn


def standardize(train_idx, logE, numeric):
    logE_mu, logE_sd = logE[train_idx].mean(), logE[train_idx].std() + 1e-8
    num_mu = numeric[train_idx].mean(axis=0)
    num_sd = numeric[train_idx].std(axis=0) + 1e-8
    return (float(logE_mu), float(logE_sd), num_mu, num_sd)


def make_split(split, data, meta, rng):
    n = data["log10_xs_barns"].size
    if split == "random":
        idx = rng.permutation(n)
        cut = int(0.85 * n)
        return idx[:cut], idx[cut:]
    if split == "heldout_nuclide":
        hold = meta["nuclide_ids"].get("Pu239")
        test = np.where(data["nuclide_id"] == hold)[0]
        train = np.where(data["nuclide_id"] != hold)[0]
        return train, test
    if split == "heldout_temperature":
        test = np.where(np.isclose(data["temperature_K"], 1200))[0]
        train = np.where(~np.isclose(data["temperature_K"], 1200))[0]
        return train, test
    raise ValueError(split)


def train_one(
    net, batches, device, epochs, lr, val_pack, patience=4
):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    vlogE, vnum, vrxn, vnuc, vy = val_pack
    for ep in range(epochs):
        net.train()
        for logE_b, num_b, rxn_b, nuc_b, y_b in batches():
            opt.zero_grad()
            pred = net(logE_b, num_b, rxn_b, nuc_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vpred = net(vlogE, vnum, vrxn, vnuc)
            vloss = float(loss_fn(vpred, vy))
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val


def region_metrics(energy_ev, y_true_log, y_pred_log):
    yt = np.power(10.0, y_true_log)
    yp = np.power(10.0, y_pred_log)
    rel = np.abs(yp - yt) / np.maximum(np.abs(yt), 1e-30)
    reg = region_of(energy_ev)
    out = {}
    for name in ["thermal", "epithermal_resonance", "fast", "all"]:
        m = np.ones_like(rel, dtype=bool) if name == "all" else (reg == name)
        if m.sum() == 0:
            continue
        out[name] = {
            "n": int(m.sum()),
            "median_rel_error": float(np.median(rel[m])),
            "mean_rel_error": float(np.mean(rel[m])),
            "p95_rel_error": float(np.percentile(rel[m], 95)),
            "rmse_log10": float(np.sqrt(np.mean((y_pred_log[m] - y_true_log[m]) ** 2))),
        }
    return out


def run_split(split, data, meta, device, args):
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = make_split(split, data, meta, rng)
    # carve a validation slice out of train
    rng.shuffle(train_idx)
    vcut = int(0.9 * len(train_idx))
    tr_idx, va_idx = train_idx[:vcut], train_idx[vcut:]

    logE, numeric, y, nuc, rxn = build_tensors(data, meta, device)
    logE_mu, logE_sd, num_mu, num_sd = standardize(tr_idx, logE, numeric)
    logE_s = ((logE - logE_mu) / logE_sd).astype(np.float32)
    num_s = ((numeric - num_mu) / num_sd).astype(np.float32)
    n_rxn = len(meta["reaction_ids"])
    rxn_oh = np.eye(n_rxn, dtype=np.float32)[rxn]

    def to_dev(idx):
        return (
            torch.from_numpy(logE_s[idx]).view(-1, 1).to(device),
            torch.from_numpy(num_s[idx]).to(device),
            torch.from_numpy(rxn_oh[idx]).to(device),
            torch.from_numpy(nuc[idx]).to(device),
            torch.from_numpy(y[idx]).to(device),
        )

    va_pack = to_dev(va_idx)[:4] + (torch.from_numpy(y[va_idx]).to(device),)
    va_pack = (va_pack[0], va_pack[1], va_pack[2], va_pack[3], va_pack[4])
    tr_logE, tr_num, tr_rxn, tr_nuc, tr_y = to_dev(tr_idx)

    n_tr = len(tr_idx)
    bs = args.batch_size

    def batches():
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            yield tr_logE[b], tr_num[b], tr_rxn[b], tr_nuc[b], tr_y[b]

    use_emb = (split != "heldout_nuclide")  # embedding is useless for an unseen nuclide
    preds = []
    val_losses = []
    t0 = time.perf_counter()
    for k in range(args.ensemble):
        net = XSNet(
            n_nuclides=len(meta["nuclide_ids"]),
            n_reactions=n_rxn,
            n_fourier=args.fourier_features,
            fourier_scale=args.fourier_scale,
            hidden=args.hidden,
            depth=args.depth,
            use_embedding=use_emb,
            emb_dim=args.emb_dim,
            seed=args.seed + 101 * k,
        ).to(device)
        net, vloss = train_one(net, batches, device, args.epochs, args.lr, va_pack)
        val_losses.append(vloss)
        net.eval()
        with torch.no_grad():
            te_logE = torch.from_numpy(logE_s[test_idx]).view(-1, 1).to(device)
            te_num = torch.from_numpy(num_s[test_idx]).to(device)
            te_rxn = torch.from_numpy(rxn_oh[test_idx]).to(device)
            te_nuc = torch.from_numpy(nuc[test_idx]).to(device)
            p = net(te_logE, te_num, te_rxn, te_nuc).cpu().numpy()
        preds.append(p)
        if split == "random" and k == 0:
            # save first net of the interpolation model for the benchmark / serving
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "config": vars(args),
                    "scaler": {
                        "logE_mu": logE_mu, "logE_sd": logE_sd,
                        "num_mu": num_mu.tolist(), "num_sd": num_sd.tolist(),
                    },
                    "meta": meta,
                    "use_embedding": use_emb,
                },
                MODEL_DIR / "xs_torch_surrogate.pt",
            )
    train_seconds = time.perf_counter() - t0

    preds = np.stack(preds, axis=0)  # (K, n_test)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0) if args.ensemble > 1 else np.zeros_like(mean_pred)

    y_test = y[test_idx]
    e_test = data["energy_eV"][test_idx]
    rm = region_metrics(e_test, y_test, mean_pred)

    # UQ calibration: in log space, fraction of points within +/-2 ensemble-std.
    calib = None
    if args.ensemble > 1:
        within2 = float(np.mean(np.abs(mean_pred - y_test) <= 2.0 * np.maximum(std_pred, 1e-9)))
        calib = {
            "mean_ensemble_std_log10": float(np.mean(std_pred)),
            "fraction_within_2std_log10": within2,
        }
    return {
        "split": split,
        "use_embedding": use_emb,
        "n_train": int(n_tr),
        "n_test": int(len(test_idx)),
        "ensemble": args.ensemble,
        "mean_val_loss_log10_mse": float(np.mean(val_losses)),
        "train_seconds": train_seconds,
        "region_metrics": rm,
        "uq_calibration": calib,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(PROCESSED_DATA_DIR / "xs_resonance_dataset.npz"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fourier-features", type=int, default=128)
    p.add_argument("--fourier-scale", type=float, default=8.0)
    p.add_argument("--emb-dim", type=int, default=8)
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--splits", nargs="+",
        default=["random", "heldout_nuclide", "heldout_temperature"],
    )
    p.add_argument("--name", default="xs_torch")
    p.add_argument("--only-nuclides", nargs="+", default=None,
                   help="Diagnostic: restrict dataset to these nuclides.")
    p.add_argument("--only-reactions", nargs="+", default=None,
                   help="Diagnostic: restrict dataset to these reactions.")
    args = p.parse_args()

    ensure_project_dirs()
    device = pick_device(args.device)
    print(f"device = {device}")
    npz = np.load(args.dataset)
    data = {k: npz[k] for k in npz.files}
    meta = json.loads(Path(args.dataset).with_suffix(".meta.json").read_text())
    if args.only_nuclides:
        keep_ids = {meta["nuclide_ids"][n] for n in args.only_nuclides}
        m = np.isin(data["nuclide_id"], list(keep_ids))
        data = {k: v[m] for k, v in data.items()}
        print(f"filtered to nuclides {args.only_nuclides}")
    if args.only_reactions:
        keep_rx = {meta["reaction_ids"][r] for r in args.only_reactions}
        m = np.isin(data["reaction_id"], list(keep_rx))
        data = {k: v[m] for k, v in data.items()}
        print(f"filtered to reactions {args.only_reactions}")
    print(f"dataset rows = {data['log10_xs_barns'].size:,}")

    report = {"device": str(device), "dataset": args.dataset, "config": vars(args), "splits": {}}
    out = MODEL_DIR / f"{args.name}_surrogate_metrics.json"
    for split in args.splits:
        print(f"\n=== split: {split} ===", flush=True)
        res = run_split(split, data, meta, device, args)
        report["splits"][split] = res
        rm = res["region_metrics"]
        for reg in ["thermal", "epithermal_resonance", "fast", "all"]:
            if reg in rm:
                print(
                    f"  {reg:<22} median_rel={rm[reg]['median_rel_error']*100:7.3f}%  "
                    f"p95={rm[reg]['p95_rel_error']*100:8.3f}%  n={rm[reg]['n']:,}",
                    flush=True,
                )
        if res["uq_calibration"]:
            print(
                f"  UQ within 2std: {res['uq_calibration']['fraction_within_2std_log10']*100:.1f}%",
                flush=True,
            )
        # Write incrementally so a later split failure never discards earlier results.
        out.write_text(json.dumps(report, indent=2))
        print(f"  [saved {out.name} after split '{split}']", flush=True)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
