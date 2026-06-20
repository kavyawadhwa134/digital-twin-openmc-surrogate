from __future__ import annotations

import argparse
import csv
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from generate_state_sequences import (
    ANOMALY_KINDS,
    STATE_FEATURES,
    make_lstm_windows,
    simulate_state_series,
)
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def make_many_windows(
    seeds: range,
    steps: int,
    lookback: int,
    horizon: int,
    anomaly: bool,
    anomaly_start: int,
    anomaly_kind: str = "coolant_loss",
    severity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, labels, times = [], [], [], []
    for seed in seeds:
        start = anomaly_start if anomaly else -1
        df = simulate_state_series(
            steps,
            anomaly_start=start,
            seed=seed,
            anomaly_kind=anomaly_kind,
            severity=severity if anomaly else 0.0,
        )
        x, y, anomaly_labels = make_lstm_windows(df, lookback, horizon)
        time_axis = df["time_s"].to_numpy()[lookback : lookback + len(x)]
        xs.append(x)
        ys.append(y)
        labels.append(anomaly_labels)
        times.append(time_axis)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(labels),
        np.concatenate(times),
    )


def make_scenario_windows(
    specs: list[dict[str, object]],
    steps: int,
    lookback: int,
    horizon: int,
    anomaly_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blocks = [
        make_many_windows(
            spec["seeds"],
            steps,
            lookback,
            horizon,
            True,
            anomaly_start,
            str(spec["kind"]),
            float(spec["severity"]),
        )
        for spec in specs
    ]
    return tuple(np.concatenate(parts) for parts in zip(*blocks))


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return a
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(a, kernel, mode="same")


def feature_index(name: str) -> int:
    return STATE_FEATURES.index(name)


def physics_residuals(obs: np.ndarray) -> np.ndarray:
    fuel_temp = obs[:, feature_index("fuel_temperature_K")]
    coolant_density = obs[:, feature_index("coolant_density_g_cm3")]
    control_rod = obs[:, feature_index("control_rod_position_pct")]
    power = obs[:, feature_index("power_norm")]
    keff = obs[:, feature_index("keff")]
    flux_fast = obs[:, feature_index("flux_fast_norm")]
    flux_thermal = obs[:, feature_index("flux_thermal_norm")]
    fission_rate = obs[:, feature_index("fission_rate_norm")]
    capture_rate = obs[:, feature_index("capture_rate_norm")]

    expected_reactivity = (
        0.0018 * (55.0 - control_rod)
        + 0.095 * (coolant_density - 0.72)
        - 2.2e-5 * (fuel_temp - 900.0)
    )
    expected_flux_fast = 1.0 + 0.70 * (power - 1.0) + 0.12 * (1.0 - coolant_density / 0.72)
    expected_flux_thermal = 1.0 + 0.55 * (power - 1.0) + 0.26 * (
        coolant_density / 0.72 - 1.0
    )
    expected_fission_rate = power * (1.0 + 0.8 * (keff - 1.0))
    expected_capture_rate = flux_thermal * (1.0 + 0.00035 * (fuel_temp - 900.0))

    return np.column_stack(
        [
            (keff - 1.0) - expected_reactivity,
            flux_fast - expected_flux_fast,
            flux_thermal - expected_flux_thermal,
            fission_rate - expected_fission_rate,
            capture_rate - expected_capture_rate,
        ]
    )


def build_detector_features(
    y_flat: np.ndarray,
    pred_flat: np.ndarray,
    horizon: int,
    n_features: int,
    residual_scale: np.ndarray,
    physics_scale: np.ndarray,
) -> np.ndarray:
    signed_residual_z = (y_flat - pred_flat) / residual_scale
    residual_z = np.abs(signed_residual_z)
    residual_z_seq = residual_z.reshape(-1, horizon, n_features)
    signed_residual_z_seq = signed_residual_z.reshape(-1, horizon, n_features)
    first_z = residual_z_seq[:, 0, :]
    endpoint_z = residual_z_seq[:, -1, :]
    full_mean_by_signal = residual_z_seq.mean(axis=1)
    full_max_by_signal = residual_z_seq.max(axis=1)
    signed_mean_by_signal = signed_residual_z_seq.mean(axis=1)
    signed_slope_by_signal = signed_residual_z_seq[:, -1, :] - signed_residual_z_seq[:, 0, :]
    residual_growth_by_signal = endpoint_z - first_z
    sign_consistency_by_signal = np.abs(signed_mean_by_signal) / (
        full_mean_by_signal + 1.0e-9
    )

    first_step_obs = y_flat.reshape(-1, horizon, n_features)[:, 0, :]
    physics_z = np.abs(physics_residuals(first_step_obs) / physics_scale)

    summary = np.column_stack(
        [
            first_z.mean(axis=1),
            first_z.max(axis=1),
            np.percentile(first_z, 90, axis=1),
            residual_z.mean(axis=1),
            residual_z.max(axis=1),
            np.percentile(residual_z, 95, axis=1),
            endpoint_z.mean(axis=1),
            endpoint_z.max(axis=1),
            np.percentile(endpoint_z, 90, axis=1),
            np.abs(signed_mean_by_signal).mean(axis=1),
            np.abs(signed_slope_by_signal).mean(axis=1),
            sign_consistency_by_signal.mean(axis=1),
            residual_growth_by_signal.mean(axis=1),
            physics_z.mean(axis=1),
            physics_z.max(axis=1),
        ]
    )
    return np.column_stack(
        [
            summary,
            first_z,
            endpoint_z,
            full_mean_by_signal,
            full_max_by_signal,
            signed_mean_by_signal,
            signed_slope_by_signal,
            sign_consistency_by_signal,
            physics_z,
        ]
    )


def add_sensor_noise(
    values: np.ndarray,
    rng: np.random.Generator,
    level: float = 1.0,
) -> np.ndarray:
    noisy = values.copy()
    scales = {
        "fuel_temperature_K": 1.5 * level,
        "coolant_temperature_K": 0.45 * level,
        "coolant_density_g_cm3": 0.0008 * level,
        "control_rod_position_pct": 0.09 * level,
        "power_norm": 0.004 * level,
        "keff": 1.5e-4 * level,
        "flux_fast_norm": 0.004 * level,
        "flux_thermal_norm": 0.004 * level,
        "fission_rate_norm": 0.004 * level,
        "capture_rate_norm": 0.004 * level,
    }
    for idx, feature in enumerate(STATE_FEATURES):
        noisy[..., idx] += rng.normal(0.0, scales[feature], size=noisy[..., idx].shape)
    return noisy


def apply_history_delay(x: np.ndarray, delay_steps: int = 3) -> np.ndarray:
    delayed = x.copy()
    delayed_features = [
        feature_index("fuel_temperature_K"),
        feature_index("coolant_temperature_K"),
        feature_index("flux_fast_norm"),
        feature_index("power_norm"),
    ]
    for idx in delayed_features:
        delayed[:, delay_steps:, idx] = delayed[:, :-delay_steps, idx]
    return delayed


def apply_history_dropout_hold_last(
    x: np.ndarray,
    rng: np.random.Generator,
    dropout_fraction: float = 0.08,
) -> np.ndarray:
    dropped = x.copy()
    monitored = [
        feature_index("fuel_temperature_K"),
        feature_index("coolant_temperature_K"),
        feature_index("power_norm"),
        feature_index("flux_fast_norm"),
    ]
    mask = rng.random((dropped.shape[0], dropped.shape[1], len(monitored))) < dropout_fraction
    for local_idx, feature_idx in enumerate(monitored):
        for t in range(1, dropped.shape[1]):
            missing = mask[:, t, local_idx]
            dropped[missing, t, feature_idx] = dropped[missing, t - 1, feature_idx]
    return dropped


def evaluate_predictions(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive_rate": float(np.mean(predicted[normal_mask]))
        if np.any(normal_mask)
        else float("nan"),
        "true_positive_rate": float(np.mean(predicted[anomaly_mask]))
        if np.any(anomaly_mask)
        else float("nan"),
    }


def tune_probability_alarm(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, int, dict[str, float]]:
    best = None
    for persistence in range(1, 6):
        smoothed = rolling_mean(probabilities, persistence)
        for threshold in np.linspace(0.02, 0.98, 193):
            predicted = smoothed >= threshold
            metrics = evaluate_predictions(labels, predicted)
            if metrics["false_positive_rate"] > 0.005:
                continue
            score = (
                metrics["f1"]
                + 0.04 * metrics["recall"]
                + 0.02 * metrics["precision"]
                - 1.25 * metrics["false_positive_rate"]
            )
            if best is None or score > best[0]:
                best = (score, float(threshold), persistence, metrics)
    if best is None:
        for persistence in range(1, 6):
            smoothed = rolling_mean(probabilities, persistence)
            for threshold in np.linspace(0.02, 0.98, 193):
                predicted = smoothed >= threshold
                metrics = evaluate_predictions(labels, predicted)
                score = metrics["f1"] - metrics["false_positive_rate"]
                if best is None or score > best[0]:
                    best = (score, float(threshold), persistence, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def first_alarm_time(
    time_axis: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    persistence: int,
    anomaly_start: int,
) -> float:
    smoothed = rolling_mean(probabilities, persistence)
    mask = (smoothed >= threshold) & (time_axis >= anomaly_start + 2)
    return float(time_axis[mask][0]) if np.any(mask) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a hybrid reactor-state forecaster and anomaly monitor."
    )
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--anomaly-start", type=int, default=280)
    parser.add_argument("--demo-anomaly-kind", choices=ANOMALY_KINDS, default="coolant_loss")
    parser.add_argument("--demo-severity", type=float, default=0.75)
    parser.add_argument(
        "--name",
        default="state_forecaster",
        help="Artifact name prefix, e.g. state_forecaster_h10 for a 10 s horizon run.",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    calibration_specs = [
        {"kind": "coolant_loss", "severity": 0.45, "seeds": range(300, 304)},
        {"kind": "coolant_loss", "severity": 0.80, "seeds": range(304, 308)},
        {"kind": "control_rod_withdrawal", "severity": 0.50, "seeds": range(308, 312)},
        {"kind": "control_rod_withdrawal", "severity": 0.90, "seeds": range(312, 316)},
        {"kind": "flux_detector_bias", "severity": 0.55, "seeds": range(316, 320)},
        {"kind": "flux_detector_bias", "severity": 1.00, "seeds": range(320, 324)},
    ]
    mixed_heldout_specs = [
        {"kind": "coolant_loss", "severity": 0.60, "seeds": range(400, 405)},
        {"kind": "coolant_loss", "severity": 1.10, "seeds": range(405, 410)},
        {"kind": "control_rod_withdrawal", "severity": 0.70, "seeds": range(410, 415)},
        {"kind": "flux_detector_bias", "severity": 0.75, "seeds": range(415, 420)},
    ]
    unseen_family_specs = [
        {"kind": "coolant_heating", "severity": 0.65, "seeds": range(430, 436)},
        {"kind": "sensor_drift", "severity": 0.85, "seeds": range(436, 442)},
    ]
    weak_stress_specs = [
        {"kind": "coolant_loss", "severity": 0.25, "seeds": range(450, 454)},
        {"kind": "control_rod_withdrawal", "severity": 0.25, "seeds": range(454, 458)},
        {"kind": "flux_detector_bias", "severity": 0.25, "seeds": range(458, 462)},
        {"kind": "coolant_heating", "severity": 0.30, "seeds": range(462, 466)},
        {"kind": "sensor_drift", "severity": 0.35, "seeds": range(466, 470)},
    ]

    x_train, y_train, _, _ = make_many_windows(
        range(100, 181), args.steps, args.lookback, args.horizon, False, args.anomaly_start
    )
    x_val, y_val, val_labels, _ = make_many_windows(
        range(200, 221), args.steps, args.lookback, args.horizon, False, args.anomaly_start
    )
    x_cal_anom, y_cal_anom, cal_anom_labels, _ = make_scenario_windows(
        calibration_specs, args.steps, args.lookback, args.horizon, args.anomaly_start
    )
    x_test_anom, y_test_anom, test_anom_labels, _ = make_scenario_windows(
        mixed_heldout_specs, args.steps, args.lookback, args.horizon, args.anomaly_start
    )
    x_unseen_anom, y_unseen_anom, unseen_anom_labels, _ = make_scenario_windows(
        unseen_family_specs, args.steps, args.lookback, args.horizon, args.anomaly_start
    )
    x_stress_anom, y_stress_anom, stress_anom_labels, _ = make_scenario_windows(
        weak_stress_specs, args.steps, args.lookback, args.horizon, args.anomaly_start
    )
    x_test_norm, y_test_norm, test_norm_labels, _ = make_many_windows(
        range(500, 513), args.steps, args.lookback, args.horizon, False, args.anomaly_start
    )

    n_features = x_train.shape[-1]
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    y_cal_anom_flat = y_cal_anom.reshape(y_cal_anom.shape[0], -1)
    y_test_anom_flat = y_test_anom.reshape(y_test_anom.shape[0], -1)
    y_unseen_anom_flat = y_unseen_anom.reshape(y_unseen_anom.shape[0], -1)
    y_stress_anom_flat = y_stress_anom.reshape(y_stress_anom.shape[0], -1)
    y_test_norm_flat = y_test_norm.reshape(y_test_norm.shape[0], -1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(flatten_windows(x_train))
    y_train_scaled = y_scaler.fit_transform(y_train_flat)
    x_val_scaled = x_scaler.transform(flatten_windows(x_val))
    y_val_scaled = y_scaler.transform(y_val_flat)
    x_cal_anom_scaled = x_scaler.transform(flatten_windows(x_cal_anom))
    x_test_anom_scaled = x_scaler.transform(flatten_windows(x_test_anom))
    x_unseen_anom_scaled = x_scaler.transform(flatten_windows(x_unseen_anom))
    x_stress_anom_scaled = x_scaler.transform(flatten_windows(x_stress_anom))
    x_test_norm_scaled = x_scaler.transform(flatten_windows(x_test_norm))

    forecaster_candidates = [
        {
            "name": "balanced_mlp",
            "hidden_layer_sizes": (224, 144, 72),
            "alpha": 7.5e-5,
            "learning_rate_init": 8.0e-4,
            "random_state": 33,
        },
        {
            "name": "wide_mlp",
            "hidden_layer_sizes": (320, 192, 96),
            "alpha": 5.0e-5,
            "learning_rate_init": 6.0e-4,
            "random_state": 34,
        },
        {
            "name": "regularized_mlp",
            "hidden_layer_sizes": (192, 128, 64),
            "alpha": 1.5e-4,
            "learning_rate_init": 1.0e-3,
            "random_state": 35,
        },
    ]
    t0 = time.perf_counter()
    forecaster_search = []
    candidate_models = []
    for candidate_spec in forecaster_candidates:
        candidate_model = MLPRegressor(
            hidden_layer_sizes=candidate_spec["hidden_layer_sizes"],
            activation="relu",
            alpha=candidate_spec["alpha"],
            batch_size=512,
            learning_rate_init=candidate_spec["learning_rate_init"],
            max_iter=360,
            early_stopping=True,
            n_iter_no_change=28,
            random_state=candidate_spec["random_state"],
        )
        candidate_t0 = time.perf_counter()
        candidate_model.fit(x_train_scaled, y_train_scaled)
        fit_seconds = time.perf_counter() - candidate_t0
        val_pred_scaled = candidate_model.predict(x_val_scaled)
        val_mse_scaled = float(mean_squared_error(y_val_scaled, val_pred_scaled))
        val_pred_scaled_seq = val_pred_scaled.reshape(-1, args.horizon, n_features)
        y_val_scaled_seq = y_val_scaled.reshape(-1, args.horizon, n_features)
        endpoint_mse_scaled = float(
            mean_squared_error(y_val_scaled_seq[:, -1, :], val_pred_scaled_seq[:, -1, :])
        )
        selection_score_scaled = 0.35 * val_mse_scaled + 0.65 * endpoint_mse_scaled
        forecaster_search.append(
            {
                "name": str(candidate_spec["name"]),
                "hidden_layer_sizes": list(candidate_spec["hidden_layer_sizes"]),
                "alpha": float(candidate_spec["alpha"]),
                "learning_rate_init": float(candidate_spec["learning_rate_init"]),
                "fit_seconds": float(fit_seconds),
                "validation_mse_scaled": val_mse_scaled,
                "endpoint_validation_mse_scaled": endpoint_mse_scaled,
                "selection_score_scaled": selection_score_scaled,
                "iterations": int(candidate_model.n_iter_),
            }
        )
        candidate_models.append(
            {
                "name": str(candidate_spec["name"]),
                "model": candidate_model,
                "validation_prediction_scaled": val_pred_scaled,
                "selection_score_scaled": selection_score_scaled,
            }
        )

    candidate_models.sort(key=lambda item: item["selection_score_scaled"])
    selected_models = [candidate_models[0]["model"]]
    selected_forecaster_name = str(candidate_models[0]["name"])
    selected_forecaster_members = [selected_forecaster_name]
    selected_score = float(candidate_models[0]["selection_score_scaled"])

    for ensemble_size in range(2, min(3, len(candidate_models)) + 1):
        members = candidate_models[:ensemble_size]
        ensemble_pred_scaled = np.mean(
            [member["validation_prediction_scaled"] for member in members], axis=0
        )
        ensemble_full_mse = float(mean_squared_error(y_val_scaled, ensemble_pred_scaled))
        ensemble_pred_seq = ensemble_pred_scaled.reshape(-1, args.horizon, n_features)
        ensemble_endpoint_mse = float(
            mean_squared_error(y_val_scaled_seq[:, -1, :], ensemble_pred_seq[:, -1, :])
        )
        ensemble_score = 0.35 * ensemble_full_mse + 0.65 * ensemble_endpoint_mse
        member_names = [str(member["name"]) for member in members]
        forecaster_search.append(
            {
                "name": "ensemble_" + "_".join(member_names),
                "members": member_names,
                "validation_mse_scaled": ensemble_full_mse,
                "endpoint_validation_mse_scaled": ensemble_endpoint_mse,
                "selection_score_scaled": ensemble_score,
            }
        )
        if ensemble_score < selected_score:
            selected_score = ensemble_score
            selected_models = [member["model"] for member in members]
            selected_forecaster_members = member_names
            selected_forecaster_name = "ensemble_" + "_".join(member_names)
    train_seconds = time.perf_counter() - t0

    def predict(x_scaled: np.ndarray) -> np.ndarray:
        pred_scaled = np.mean([forecaster.predict(x_scaled) for forecaster in selected_models], axis=0)
        return y_scaler.inverse_transform(pred_scaled)

    t1 = time.perf_counter()
    y_val_pred = predict(x_val_scaled)
    val_predict_seconds = time.perf_counter() - t1
    y_cal_anom_pred = predict(x_cal_anom_scaled)
    y_test_anom_pred = predict(x_test_anom_scaled)
    y_unseen_anom_pred = predict(x_unseen_anom_scaled)
    y_stress_anom_pred = predict(x_stress_anom_scaled)
    y_test_norm_pred = predict(x_test_norm_scaled)

    residual_val = y_val_flat - y_val_pred
    residual_scale = np.std(residual_val, axis=0)
    residual_scale = np.where(residual_scale > 1.0e-10, residual_scale, 1.0)
    first_step_val = y_val.reshape(-1, args.horizon, n_features)[:, 0, :]
    physics_scale = np.std(physics_residuals(first_step_val), axis=0)
    physics_scale = np.where(physics_scale > 1.0e-10, physics_scale, 1.0)

    detector_rng = np.random.default_rng(9100 + args.horizon)
    x_val_noise = add_sensor_noise(x_val, detector_rng, level=1.0)
    y_val_noise = add_sensor_noise(y_val, detector_rng, level=1.0)
    x_val_delay = apply_history_delay(x_val, delay_steps=3)
    x_val_dropout = apply_history_dropout_hold_last(
        x_val, detector_rng, dropout_fraction=0.08
    )
    x_cal_anom_noise = add_sensor_noise(x_cal_anom, detector_rng, level=0.75)
    y_cal_anom_noise = add_sensor_noise(y_cal_anom, detector_rng, level=0.75)

    y_val_noise_flat = y_val_noise.reshape(y_val_noise.shape[0], -1)
    y_val_noise_pred = predict(x_scaler.transform(flatten_windows(x_val_noise)))
    y_val_delay_pred = predict(x_scaler.transform(flatten_windows(x_val_delay)))
    y_val_dropout_pred = predict(x_scaler.transform(flatten_windows(x_val_dropout)))
    y_cal_anom_noise_flat = y_cal_anom_noise.reshape(y_cal_anom_noise.shape[0], -1)
    y_cal_anom_noise_pred = predict(
        x_scaler.transform(flatten_windows(x_cal_anom_noise))
    )

    cal_x = np.vstack(
        [
            build_detector_features(
                y_val_flat,
                y_val_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
            build_detector_features(
                y_val_noise_flat,
                y_val_noise_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
            build_detector_features(
                y_val_flat,
                y_val_delay_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
            build_detector_features(
                y_val_flat,
                y_val_dropout_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
            build_detector_features(
                y_cal_anom_flat,
                y_cal_anom_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
            build_detector_features(
                y_cal_anom_noise_flat,
                y_cal_anom_noise_pred,
                args.horizon,
                n_features,
                residual_scale,
                physics_scale,
            ),
        ]
    )
    cal_y = np.concatenate(
        [val_labels, val_labels, val_labels, val_labels, cal_anom_labels, cal_anom_labels]
    )

    detector_scaler = StandardScaler()
    cal_x_scaled = detector_scaler.fit_transform(cal_x)
    detector = RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=44,
        n_jobs=-1,
    )
    detector.fit(cal_x_scaled, cal_y)
    cal_prob = detector.predict_proba(cal_x_scaled)[:, 1]
    threshold, persistence, calibration_metrics = tune_probability_alarm(cal_prob, cal_y)

    def detector_probability(y_flat: np.ndarray, y_pred_flat: np.ndarray) -> np.ndarray:
        features = build_detector_features(
            y_flat,
            y_pred_flat,
            args.horizon,
            n_features,
            residual_scale,
            physics_scale,
        )
        return detector.predict_proba(detector_scaler.transform(features))[:, 1]

    normal_prob = detector_probability(y_test_norm_flat, y_test_norm_pred)
    mixed_prob = detector_probability(y_test_anom_flat, y_test_anom_pred)
    unseen_prob = detector_probability(y_unseen_anom_flat, y_unseen_anom_pred)
    stress_prob = detector_probability(y_stress_anom_flat, y_stress_anom_pred)

    normal_pred = rolling_mean(normal_prob, persistence) >= threshold
    mixed_pred = rolling_mean(mixed_prob, persistence) >= threshold
    unseen_pred = rolling_mean(unseen_prob, persistence) >= threshold
    stress_pred = rolling_mean(stress_prob, persistence) >= threshold

    mixed_test_y = np.concatenate([test_norm_labels, test_anom_labels])
    mixed_test_pred = np.concatenate([normal_pred, mixed_pred])
    mixed_test_metrics = evaluate_predictions(mixed_test_y, mixed_test_pred)

    unseen_test_y = np.concatenate([test_norm_labels, unseen_anom_labels])
    unseen_test_pred = np.concatenate([normal_pred, unseen_pred])
    unseen_test_metrics = evaluate_predictions(unseen_test_y, unseen_test_pred)

    stress_test_y = np.concatenate([test_norm_labels, stress_anom_labels])
    stress_test_pred = np.concatenate([normal_pred, stress_pred])
    stress_test_metrics = evaluate_predictions(stress_test_y, stress_test_pred)

    combined_test_y = np.concatenate([test_norm_labels, test_anom_labels, unseen_anom_labels])
    combined_test_pred = np.concatenate([normal_pred, mixed_pred, unseen_pred])
    combined_test_metrics = evaluate_predictions(combined_test_y, combined_test_pred)

    false_positive_rate = float(np.mean(normal_pred))

    y_val_seq = y_val_flat.reshape(-1, args.horizon, n_features)
    y_val_pred_seq = y_val_pred.reshape(-1, args.horizon, n_features)
    feature_metrics = {}
    endpoint_feature_metrics = {}
    for idx, feature in enumerate(STATE_FEATURES):
        truth = y_val_seq[:, :, idx].reshape(-1)
        pred = y_val_pred_seq[:, :, idx].reshape(-1)
        feature_metrics[feature] = {
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        }
        endpoint_truth = y_val_seq[:, -1, idx]
        endpoint_pred = y_val_pred_seq[:, -1, idx]
        endpoint_feature_metrics[feature] = {
            "mae": float(mean_absolute_error(endpoint_truth, endpoint_pred)),
            "rmse": float(np.sqrt(mean_squared_error(endpoint_truth, endpoint_pred))),
        }

    def score_trace(
        anomaly_kind: str,
        severity: float,
        seed: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        df = simulate_state_series(
            args.steps,
            anomaly_start=args.anomaly_start,
            seed=seed,
            anomaly_kind=anomaly_kind,
            severity=severity,
        )
        x, y, labels = make_lstm_windows(df, args.lookback, args.horizon)
        trace_time = df["time_s"].to_numpy()[args.lookback : args.lookback + len(x)]
        y_flat = y.reshape(y.shape[0], -1)
        pred_flat = predict(x_scaler.transform(flatten_windows(x)))
        prob = detector_probability(y_flat, pred_flat)
        pred_seq = pred_flat.reshape(-1, args.horizon, n_features)
        actual = df["is_anomaly"].to_numpy()[args.lookback : args.lookback + len(x)]
        endpoint_actual = df["is_anomaly"].to_numpy()[
            args.lookback + args.horizon - 1 : args.lookback + args.horizon - 1 + len(x)
        ]
        return trace_time, y, pred_seq, prob, labels, actual, endpoint_actual

    def alarm_delay_summary(specs: list[dict[str, object]]) -> dict[str, float | int]:
        delays: list[float] = []
        misses = 0
        for spec in specs:
            for seed in spec["seeds"]:
                trace_time, _, _, prob, _, _, _ = score_trace(
                    str(spec["kind"]), float(spec["severity"]), int(seed)
                )
                alarm = first_alarm_time(
                    trace_time, prob, threshold, persistence, args.anomaly_start
                )
                if np.isfinite(alarm):
                    delays.append(float(alarm - args.anomaly_start))
                else:
                    misses += 1
        if not delays:
            return {
                "events": int(misses),
                "detected_events": 0,
                "missed_events": int(misses),
                "median_alarm_delay_seconds": float("nan"),
                "p90_alarm_delay_seconds": float("nan"),
            }
        return {
            "events": int(len(delays) + misses),
            "detected_events": int(len(delays)),
            "missed_events": int(misses),
            "median_alarm_delay_seconds": float(np.median(delays)),
            "p90_alarm_delay_seconds": float(np.percentile(delays, 90)),
        }

    time_axis, y_demo, pred_demo, demo_prob, labels_demo, actual_anomaly, endpoint_anomaly = score_trace(
        args.demo_anomaly_kind,
        args.demo_severity,
        seed=999,
    )
    demo_prob_smoothed = rolling_mean(demo_prob, persistence)
    alarm_time = first_alarm_time(
        time_axis, demo_prob, threshold, persistence, args.anomaly_start
    )

    fuel_idx = feature_index("fuel_temperature_K")
    keff_idx = feature_index("keff")
    coolant_idx = feature_index("coolant_temperature_K")
    power_idx = feature_index("power_norm")
    flux_idx = feature_index("flux_fast_norm")

    display_step = args.horizon - 1
    decision_time_axis = time_axis - 1.0
    target_time_axis = time_axis + display_step
    forecast_lead_seconds = args.horizon

    observed_flux = y_demo[:, display_step, flux_idx]
    predicted_flux = pred_demo[:, display_step, flux_idx]
    observed_power = y_demo[:, display_step, power_idx]
    predicted_power = pred_demo[:, display_step, power_idx]
    observed_coolant = y_demo[:, display_step, coolant_idx]
    predicted_coolant = pred_demo[:, display_step, coolant_idx]
    observed_fuel = y_demo[:, display_step, fuel_idx]
    predicted_fuel = pred_demo[:, display_step, fuel_idx]
    observed_keff = y_demo[:, display_step, keff_idx]
    predicted_keff = pred_demo[:, display_step, keff_idx]
    observed_reactivity_pcm = (observed_keff - 1.0) * 1.0e5
    predicted_reactivity_pcm = (predicted_keff - 1.0) * 1.0e5

    def describe_specs(specs: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "kind": str(spec["kind"]),
                "severity": float(spec["severity"]),
                "seed_count": len(list(spec["seeds"])),
            }
            for spec in specs
        ]

    mixed_alarm_summary = alarm_delay_summary(mixed_heldout_specs)
    unseen_alarm_summary = alarm_delay_summary(unseen_family_specs)
    stress_alarm_summary = alarm_delay_summary(weak_stress_specs)

    metrics = {
        "model": "sklearn_mlp_sequence_forecaster_plus_hybrid_anomaly_classifier",
        "note": (
            "A trained sequence forecaster predicts normal reactor evolution; a calibrated "
            "hybrid anomaly classifier combines forecast residuals with physics-consistency "
            "residuals. PyTorch/TensorFlow are not installed, so this is not a true LSTM."
        ),
        "validation_design": (
            "Forecaster is trained on normal reactor traces only. Alarm calibration uses "
            "coolant-loss, control-rod-withdrawal, and flux-detector-bias transients; "
            "final validation reports both unseen seeds/severities and held-out anomaly families. "
            "Detector calibration also includes noisy, delayed, and dropout-corrupted normal "
            "windows so ordinary sensor imperfections are less likely to become false alarms."
        ),
        "lookback_seconds": args.lookback,
        "horizon_seconds": args.horizon,
        "forecast_lead_seconds_for_endpoint_metrics": args.horizon,
        "anomaly_start_seconds": args.anomaly_start,
        "demo_anomaly_kind": args.demo_anomaly_kind,
        "demo_anomaly_severity": float(args.demo_severity),
        "selected_forecaster": selected_forecaster_name,
        "selected_forecaster_members": selected_forecaster_members,
        "forecaster_search": forecaster_search,
        "train_windows": int(len(x_train)),
        "normal_validation_windows": int(len(x_val)),
        "calibration_windows": int(len(cal_y)),
        "detector_noise_robust_calibration": {
            "normal_sensor_noise_level": 1.0,
            "normal_delay_steps": 3,
            "normal_dropout_fraction": 0.08,
            "anomaly_sensor_noise_level": 0.75,
        },
        "heldout_mixed_windows": int(len(mixed_test_y)),
        "heldout_unseen_family_windows": int(len(unseen_test_y)),
        "weak_transient_stress_windows": int(len(stress_test_y)),
        "heldout_combined_windows": int(len(combined_test_y)),
        "train_seconds": float(train_seconds),
        "predict_seconds_for_validation_set": float(val_predict_seconds),
        "predict_seconds_per_window": float(val_predict_seconds / len(x_val)),
        "feature_metrics": feature_metrics,
        "trajectory_feature_metrics": feature_metrics,
        "endpoint_feature_metrics": endpoint_feature_metrics,
        "anomaly_threshold": float(threshold),
        "anomaly_persistence_windows": int(persistence),
        "anomaly_precision": combined_test_metrics["precision"],
        "anomaly_recall": combined_test_metrics["recall"],
        "anomaly_f1": combined_test_metrics["f1"],
        "heldout_normal_false_positive_rate": false_positive_rate,
        "heldout_mixed_metrics": mixed_test_metrics,
        "heldout_unseen_family_metrics": unseen_test_metrics,
        "weak_transient_stress_metrics": stress_test_metrics,
        "heldout_combined_metrics": combined_test_metrics,
        "mixed_heldout_alarm_summary": mixed_alarm_summary,
        "unseen_family_alarm_summary": unseen_alarm_summary,
        "weak_transient_stress_alarm_summary": stress_alarm_summary,
        "calibration_metrics": calibration_metrics,
        "calibration_anomaly_specs": describe_specs(calibration_specs),
        "mixed_heldout_anomaly_specs": describe_specs(mixed_heldout_specs),
        "unseen_family_anomaly_specs": describe_specs(unseen_family_specs),
        "weak_transient_stress_specs": describe_specs(weak_stress_specs),
        "demo_alarm_time_seconds": float(alarm_time),
        "alarm_confirmation_delay_seconds": float(alarm_time - args.anomaly_start)
        if np.isfinite(alarm_time)
        else float("nan"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{args.name}_mlp.joblib"
    metrics_path = MODEL_DIR / f"{args.name}_metrics.json"
    joblib.dump(
        {
            "model": selected_models[0],
            "forecaster_models": selected_models,
            "forecaster_mode": "ensemble" if len(selected_models) > 1 else "single",
            "selected_forecaster": selected_forecaster_name,
            "selected_forecaster_members": selected_forecaster_members,
            "detector": detector,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "detector_scaler": detector_scaler,
            "residual_scale": residual_scale,
            "physics_scale": physics_scale,
            "feature_names": STATE_FEATURES,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "anomaly_threshold": threshold,
            "anomaly_persistence_windows": persistence,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    validation_rows = [
        {
            "split": "mixed_heldout",
            **mixed_test_metrics,
            **mixed_alarm_summary,
        },
        {
            "split": "unseen_family",
            **unseen_test_metrics,
            **unseen_alarm_summary,
        },
        {
            "split": "weak_transient_stress",
            **stress_test_metrics,
            **stress_alarm_summary,
        },
    ]
    validation_path = PROCESSED_DATA_DIR / f"{args.name}_validation_summary.csv"
    with validation_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    demo_out = np.column_stack(
        [
            decision_time_axis,
            target_time_axis,
            np.full_like(target_time_axis, forecast_lead_seconds, dtype=float),
            observed_flux,
            predicted_flux,
            observed_power,
            predicted_power,
            observed_coolant,
            predicted_coolant,
            observed_fuel,
            predicted_fuel,
            observed_reactivity_pcm,
            predicted_reactivity_pcm,
            observed_keff,
            predicted_keff,
            demo_prob,
            demo_prob_smoothed,
            actual_anomaly,
            endpoint_anomaly,
            labels_demo,
        ]
    )
    np.savetxt(
        PROCESSED_DATA_DIR / f"{args.name}_demo.csv",
        demo_out,
        delimiter=",",
        header=(
            "decision_time_s,forecast_target_time_s,forecast_lead_seconds,"
            "observed_neutron_flux_norm,predicted_neutron_flux_norm,"
            "observed_reactor_power_norm,predicted_reactor_power_norm,"
            "observed_coolant_outlet_temperature_K,predicted_coolant_outlet_temperature_K,"
            "observed_fuel_temperature_K,predicted_fuel_temperature_K,"
            "observed_reactivity_pcm,predicted_reactivity_pcm,"
            "observed_keff,predicted_keff,anomaly_probability,"
            "smoothed_anomaly_probability,is_anomaly_at_first_forecast_step,"
            "is_anomaly_at_forecast_target,forecast_window_contains_anomaly"
        ),
        comments="",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 12.0), sharex=True)
    panels = [
        ("Neutron flux", observed_flux, predicted_flux, "Fast flux (norm.)"),
        ("Reactor power", observed_power, predicted_power, "Power (norm.)"),
        ("Coolant outlet temperature", observed_coolant, predicted_coolant, "Coolant T (K)"),
        ("Fuel temperature", observed_fuel, predicted_fuel, "Fuel T (K)"),
        ("Reactivity", observed_reactivity_pcm, predicted_reactivity_pcm, "Reactivity (pcm)"),
    ]
    anomaly = endpoint_anomaly.astype(bool)
    for ax, (title, observed, predicted, ylabel) in zip(axes[:5], panels):
        ax.plot(target_time_axis, observed, label="simulated state", linewidth=1.35)
        ax.plot(
            target_time_axis,
            predicted,
            label=f"+{forecast_lead_seconds} s ML forecast",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.axvline(args.anomaly_start, color="#E45756", linestyle=":", linewidth=1.2)
        if np.isfinite(alarm_time):
            ax.axvline(alarm_time, color="#2A9D8F", linestyle="--", linewidth=1.1)
        if anomaly.any():
            ax.fill_between(
                target_time_axis,
                np.nanmin(observed),
                np.nanmax(observed),
                where=anomaly,
                color="#E45756",
                alpha=0.10,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].plot(time_axis, demo_prob, label="hybrid anomaly probability", color="#4C78A8")
    axes[-1].plot(
        time_axis,
        demo_prob_smoothed,
        label=f"{persistence}-window smoothed probability",
        color="#72B7B2",
        linewidth=1.2,
    )
    axes[-1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="calibrated alarm threshold")
    axes[-1].axvline(args.anomaly_start, color="#E45756", linestyle=":", linewidth=1.2, label=f"anomaly begins {args.anomaly_start}s")
    if np.isfinite(alarm_time):
        axes[-1].axvline(
            alarm_time,
            color="#2A9D8F",
            linestyle="--",
            linewidth=1.1,
            label=f"confirmed alarm {alarm_time:.0f}s",
        )
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Alarm probability")
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].legend(loc="upper left", ncol=2)
    fig.suptitle("Hybrid Reactor Digital-Twin Forecasting and Early Anomaly Alarm", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.name}_anomaly_detection.png", dpi=220)
    plt.close(fig)

    split_labels = ["Mixed held-out", "Unseen family", "Weak stress"]
    precision_vals = [row["precision"] for row in validation_rows]
    recall_vals = [row["recall"] for row in validation_rows]
    f1_vals = [row["f1"] for row in validation_rows]
    median_delay = [row["median_alarm_delay_seconds"] for row in validation_rows]
    p90_delay = [row["p90_alarm_delay_seconds"] for row in validation_rows]
    x = np.arange(len(split_labels))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    width = 0.24
    axes[0].bar(x - width, precision_vals, width, label="precision", color="#4C78A8")
    axes[0].bar(x, recall_vals, width, label="recall", color="#F58518")
    axes[0].bar(x + width, f1_vals, width, label="F1", color="#54A24B")
    axes[0].set_ylim(0.85, 1.01)
    axes[0].set_ylabel("Score", fontsize=9)
    axes[0].set_title("Window-level anomaly detection", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_labels, rotation=12, ha="right", fontsize=8)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="lower left", ncol=3, fontsize=8)

    axes[1].bar(x - width / 2, median_delay, width, label="median", color="#B279A2")
    axes[1].bar(x + width / 2, p90_delay, width, label="p90", color="#E45756")
    axes[1].set_ylabel("Alarm delay (s)", fontsize=9)
    axes[1].set_title("Event-level alarm timing", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_labels, rotation=12, ha="right", fontsize=8)
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.name}_validation_summary.png", dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {model_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()
