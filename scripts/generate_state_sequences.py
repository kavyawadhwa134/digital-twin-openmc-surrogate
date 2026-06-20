from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import FIGURE_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


STATE_FEATURES = [
    "fuel_temperature_K",
    "coolant_temperature_K",
    "coolant_density_g_cm3",
    "control_rod_position_pct",
    "power_norm",
    "keff",
    "flux_fast_norm",
    "flux_thermal_norm",
    "fission_rate_norm",
    "capture_rate_norm",
]

ANOMALY_KINDS = [
    "coolant_loss",
    "control_rod_withdrawal",
    "flux_detector_bias",
    "coolant_heating",
    "sensor_drift",
]


def simulate_state_series(
    n_steps: int,
    anomaly_start: int,
    seed: int,
    anomaly_kind: str = "coolant_loss",
    severity: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)

    demand = 1.0 + 0.025 * np.sin(2 * np.pi * t / 180.0) + 0.010 * np.sin(2 * np.pi * t / 47.0)
    control_rod = 55.0 - 4.0 * np.sin(2 * np.pi * t / 240.0)
    coolant_density = 0.72 - 0.006 * (demand - 1.0) + rng.normal(0.0, 0.0006, n_steps)
    coolant_temp = 580.0 + 9.0 * (demand - 1.0) + rng.normal(0.0, 0.35, n_steps)
    fuel_temp = 900.0 + 115.0 * (demand - 1.0) + rng.normal(0.0, 1.2, n_steps)

    # Physics-informed response approximations: rod insertion, density, and temperature feedbacks.
    reactivity = (
        0.0018 * (55.0 - control_rod)
        + 0.095 * (coolant_density - 0.72)
        - 2.2e-5 * (fuel_temp - 900.0)
    )
    keff = 1.0 + reactivity + rng.normal(0.0, 1.2e-4, n_steps)
    power = demand * (1.0 + 18.0 * (keff - 1.0)) + rng.normal(0.0, 0.003, n_steps)
    power = np.clip(power, 0.65, 1.35)
    flux_fast = 1.0 + 0.70 * (power - 1.0) + 0.12 * (1.0 - coolant_density / 0.72)
    flux_thermal = 1.0 + 0.55 * (power - 1.0) + 0.26 * (coolant_density / 0.72 - 1.0)
    fission_rate = power * (1.0 + 0.8 * (keff - 1.0))
    capture_rate = flux_thermal * (1.0 + 0.00035 * (fuel_temp - 900.0))

    is_anomaly = np.zeros(n_steps, dtype=int)
    anomaly_kind = anomaly_kind.lower()
    severity = float(severity)
    if 0 <= anomaly_start < n_steps and severity > 0.0:
        width = min(55, n_steps - anomaly_start)
        idx = slice(anomaly_start, anomaly_start + width)
        is_anomaly[idx] = 1
        if anomaly_kind == "coolant_loss":
            # Coolant-density-loss anomaly with delayed temperature and flux response.
            coolant_density[idx] -= severity * np.linspace(0.015, 0.045, width)
            coolant_temp[idx] += severity * np.linspace(2.0, 18.0, width)
            fuel_temp[idx] += severity * np.linspace(4.0, 65.0, width)
            keff[idx] -= severity * np.linspace(0.0015, 0.0075, width)
            power[idx] += severity * np.linspace(0.01, 0.08, width)
            flux_fast[idx] += severity * np.linspace(0.015, 0.11, width)
            flux_thermal[idx] -= severity * np.linspace(0.015, 0.09, width)
            fission_rate[idx] += severity * np.linspace(0.01, 0.075, width)
            capture_rate[idx] -= severity * np.linspace(0.01, 0.08, width)
        elif anomaly_kind == "control_rod_withdrawal":
            # Positive-reactivity transient: rod withdrawal drives power and flux upward.
            control_rod[idx] -= severity * np.linspace(0.8, 6.5, width)
            keff[idx] += severity * np.linspace(0.0008, 0.0055, width)
            power[idx] += severity * np.linspace(0.006, 0.075, width)
            fuel_temp[idx] += severity * np.linspace(1.5, 45.0, width)
            coolant_temp[idx] += severity * np.linspace(0.8, 10.0, width)
            coolant_density[idx] -= severity * np.linspace(0.001, 0.012, width)
            flux_fast[idx] += severity * np.linspace(0.010, 0.090, width)
            flux_thermal[idx] += severity * np.linspace(0.007, 0.060, width)
            fission_rate[idx] += severity * np.linspace(0.008, 0.080, width)
            capture_rate[idx] += severity * np.linspace(0.004, 0.035, width)
        elif anomaly_kind == "flux_detector_bias":
            # Instrumentation fault: flux channels drift away from otherwise consistent states.
            flux_fast[idx] += severity * np.linspace(0.006, 0.075, width)
            flux_thermal[idx] -= severity * np.linspace(0.004, 0.055, width)
            fission_rate[idx] += severity * np.linspace(0.003, 0.030, width)
            capture_rate[idx] -= severity * np.linspace(0.002, 0.026, width)
        elif anomaly_kind == "coolant_heating":
            # Held-out thermal transient: coolant heats first, then density and fuel respond.
            coolant_temp[idx] += severity * np.linspace(2.5, 24.0, width)
            coolant_density[idx] -= severity * np.linspace(0.002, 0.024, width)
            fuel_temp[idx] += severity * np.linspace(1.0, 38.0, width)
            keff[idx] -= severity * np.linspace(0.0004, 0.0038, width)
            power[idx] += severity * np.linspace(0.002, 0.030, width)
            flux_fast[idx] += severity * np.linspace(0.004, 0.045, width)
            flux_thermal[idx] -= severity * np.linspace(0.003, 0.040, width)
            fission_rate[idx] += severity * np.linspace(0.002, 0.026, width)
            capture_rate[idx] -= severity * np.linspace(0.004, 0.045, width)
        elif anomaly_kind == "sensor_drift":
            # Held-out sensor drift: measurements drift without a full neutronic response.
            fuel_temp[idx] += severity * np.linspace(1.0, 28.0, width)
            coolant_temp[idx] += severity * np.linspace(0.8, 12.0, width)
            keff[idx] += severity * np.linspace(0.0001, 0.0008, width)
        else:
            raise ValueError(
                f"Unknown anomaly_kind={anomaly_kind!r}. "
                f"Choose one of: {', '.join(ANOMALY_KINDS)}"
            )

    return pd.DataFrame(
        {
            "time_s": t,
            "fuel_temperature_K": fuel_temp,
            "coolant_temperature_K": coolant_temp,
            "coolant_density_g_cm3": coolant_density,
            "control_rod_position_pct": control_rod,
            "power_norm": power,
            "keff": keff,
            "flux_fast_norm": flux_fast,
            "flux_thermal_norm": flux_thermal,
            "fission_rate_norm": fission_rate,
            "capture_rate_norm": capture_rate,
            "is_anomaly": is_anomaly,
            "anomaly_kind": anomaly_kind if 0 <= anomaly_start < n_steps else "none",
            "anomaly_severity": severity if 0 <= anomaly_start < n_steps else 0.0,
        }
    )


def make_lstm_windows(
    df: pd.DataFrame, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[STATE_FEATURES].to_numpy(dtype=float)
    labels = df["is_anomaly"].to_numpy(dtype=int)
    x, y, anomaly = [], [], []
    for start in range(0, len(df) - lookback - horizon + 1):
        mid = start + lookback
        end = mid + horizon
        x.append(values[start:mid])
        y.append(values[mid:end])
        anomaly.append(int(labels[mid:end].max()))
    return np.asarray(x), np.asarray(y), np.asarray(anomaly)


def persistence_forecast_demo(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    values = df[STATE_FEATURES].to_numpy(dtype=float)
    pred = np.full_like(values, np.nan)
    for t in range(lookback, len(df) - horizon):
        for h in range(1, horizon + 1):
            pred[t + h] = values[t - 1]

    residual = values - pred
    valid_mask = np.isfinite(residual).all(axis=1)
    normal_mask = (df["is_anomaly"].to_numpy() == 0) & valid_mask
    scale = np.nanstd(residual[normal_mask], axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    score = np.full(len(df), np.nan)
    forecast_score = np.mean(np.abs(residual[valid_mask] / scale), axis=1)

    physics_cols = [
        STATE_FEATURES.index("coolant_temperature_K"),
        STATE_FEATURES.index("coolant_density_g_cm3"),
        STATE_FEATURES.index("control_rod_position_pct"),
        STATE_FEATURES.index("power_norm"),
    ]
    fuel_col = STATE_FEATURES.index("fuel_temperature_K")
    design = np.column_stack([np.ones(len(values)), values[:, physics_cols]])
    beta, *_ = np.linalg.lstsq(design[normal_mask], values[normal_mask, fuel_col], rcond=None)
    expected_fuel_temp = design @ beta
    physics_residual = values[:, fuel_col] - expected_fuel_temp
    physics_scale = np.std(physics_residual[normal_mask])
    if physics_scale <= 1.0e-9:
        physics_scale = 1.0
    physics_score = np.abs(physics_residual[valid_mask] / physics_scale)

    score[valid_mask] = 0.45 * forecast_score + 0.55 * physics_score
    threshold = np.nanpercentile(score[normal_mask], 99)

    demo = df[["time_s", "fuel_temperature_K", "keff", "is_anomaly"]].copy()
    demo["fuel_temperature_pred_K"] = pred[:, STATE_FEATURES.index("fuel_temperature_K")]
    demo["keff_pred"] = pred[:, STATE_FEATURES.index("keff")]
    demo["anomaly_score"] = score
    demo["anomaly_threshold"] = threshold
    demo["detected_anomaly"] = (score > threshold).astype(int)
    return demo


def plot_anomaly_demo(demo: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)

    axes[0].plot(demo["time_s"], demo["fuel_temperature_K"], label="Observed/simulated state")
    axes[0].plot(
        demo["time_s"],
        demo["fuel_temperature_pred_K"],
        label="Short-horizon forecast baseline",
        linewidth=1.1,
    )
    anomaly = demo["is_anomaly"].astype(bool)
    if anomaly.any():
        axes[0].fill_between(
            demo["time_s"],
            demo["fuel_temperature_K"].min(),
            demo["fuel_temperature_K"].max(),
            where=anomaly,
            color="#E45756",
            alpha=0.18,
            label="Injected anomaly",
        )
    axes[0].set_ylabel("Fuel temperature (K)")
    axes[0].set_title("LSTM-Ready Reactor-State Forecasting Concept")
    axes[0].legend(loc="upper left")

    axes[1].plot(demo["time_s"], demo["anomaly_score"], color="#4C78A8", label="Prediction-error score")
    axes[1].plot(
        demo["time_s"],
        demo["anomaly_threshold"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="99th percentile normal threshold",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Anomaly score")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lstm_anomaly_demo.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate physics-informed reactor-state sequences for LSTM forecasting."
    )
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--anomaly-start", type=int, default=285)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--anomaly-kind", choices=ANOMALY_KINDS, default="coolant_loss")
    parser.add_argument("--severity", type=float, default=1.0)
    args = parser.parse_args()

    ensure_project_dirs()
    df = simulate_state_series(
        args.steps,
        args.anomaly_start,
        args.seed,
        anomaly_kind=args.anomaly_kind,
        severity=args.severity,
    )
    x, y, anomaly = make_lstm_windows(df, args.lookback, args.horizon)
    demo = persistence_forecast_demo(df, args.lookback, args.horizon)

    series_path = PROCESSED_DATA_DIR / "reactor_state_timeseries.csv"
    windows_path = PROCESSED_DATA_DIR / "state_sequences_lstm_ready.npz"
    demo_path = PROCESSED_DATA_DIR / "state_anomaly_demo.csv"

    df.to_csv(series_path, index=False)
    demo.to_csv(demo_path, index=False)
    np.savez_compressed(
        windows_path,
        X=x,
        y=y,
        anomaly=anomaly,
        feature_names=np.asarray(STATE_FEATURES),
        lookback=np.asarray(args.lookback),
        horizon=np.asarray(args.horizon),
    )
    plot_anomaly_demo(demo)

    print(f"Wrote reactor state series to {series_path}")
    print(f"Wrote LSTM-ready arrays to {windows_path}")
    print(f"X shape: {x.shape}; y shape: {y.shape}; anomaly labels: {anomaly.shape}")
    print(f"Wrote anomaly demo figure to {FIGURE_DIR / 'lstm_anomaly_demo.png'}")


if __name__ == "__main__":
    main()
