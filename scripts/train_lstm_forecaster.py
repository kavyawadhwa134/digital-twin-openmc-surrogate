from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project_config import PROCESSED_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an LSTM forecaster on reactor-state sequences if PyTorch is installed."
    )
    parser.add_argument(
        "--data",
        default=str(PROCESSED_DATA_DIR / "state_sequences_lstm_ready.npz"),
        help="NPZ file produced by generate_state_sequences.py",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in the current environment. "
            "The LSTM-ready dataset is still available. Install PyTorch in the openmc "
            "environment when you want to train the LSTM, then rerun this script."
        ) from exc

    data = np.load(Path(args.data), allow_pickle=True)
    x = data["X"].astype("float32")
    y = data["y"].astype("float32")
    n_features = x.shape[-1]
    horizon = y.shape[1]

    mean = x.reshape(-1, n_features).mean(axis=0)
    std = x.reshape(-1, n_features).std(axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    x_norm = (x - mean) / std
    y_norm = (y - mean) / std

    split = int(0.8 * len(x_norm))
    train_ds = TensorDataset(torch.from_numpy(x_norm[:split]), torch.from_numpy(y_norm[:split]))
    test_x = torch.from_numpy(x_norm[split:])
    test_y = torch.from_numpy(y_norm[split:])
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    class LSTMForecaster(nn.Module):
        def __init__(self, features: int, hidden: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(features, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon * features),
            )

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.lstm(batch)
            out = self.head(hidden[-1])
            return out.reshape(batch.shape[0], horizon, n_features)

    model = LSTMForecaster(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(80):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0 or epoch == 79:
            model.eval()
            with torch.no_grad():
                test_loss = float(loss_fn(model(test_x), test_y))
            print(f"epoch={epoch:03d} train_mse={np.mean(losses):.6f} test_mse={test_loss:.6f}")

    out = Path("models")
    out.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": data["feature_names"],
            "horizon": horizon,
        },
        out / "lstm_state_forecaster.pt",
    )
    print(f"Wrote {out / 'lstm_state_forecaster.pt'}")


if __name__ == "__main__":
    main()
