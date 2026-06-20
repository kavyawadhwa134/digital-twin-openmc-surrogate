from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from generate_state_sequences import (
    STATE_FEATURES,
    make_lstm_windows,
    simulate_state_series,
)
from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from train_state_forecaster import (
    build_detector_features,
    feature_index,
    flatten_windows,
    physics_residuals,
    rolling_mean,
)


KNOWN_FAULT_SPECS = [
    {"kind": "coolant_loss", "severity": 0.60, "seeds": range(600, 606)},
    {"kind": "coolant_loss", "severity": 1.05, "seeds": range(606, 612)},
    {"kind": "control_rod_withdrawal", "severity": 0.65, "seeds": range(612, 618)},
    {"kind": "control_rod_withdrawal", "severity": 1.00, "seeds": range(618, 624)},
    {"kind": "flux_detector_bias", "severity": 0.65, "seeds": range(624, 630)},
    {"kind": "flux_detector_bias", "severity": 1.00, "seeds": range(630, 636)},
]

KNOWN_FAULT_TEST_SPECS = [
    {"kind": "coolant_loss", "severity": 0.72, "seeds": range(700, 706)},
    {"kind": "control_rod_withdrawal", "severity": 0.82, "seeds": range(706, 712)},
    {"kind": "flux_detector_bias", "severity": 0.82, "seeds": range(712, 718)},
]

UNSEEN_FAULT_TEST_SPECS = [
    {"kind": "coolant_heating", "severity": 0.72, "seeds": range(720, 726)},
    {"kind": "sensor_drift", "severity": 0.90, "seeds": range(726, 732)},
]


def parse_horizons(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def feature_indices(names: list[str]) -> list[int]:
    return [feature_index(name) for name in names]


def make_window_block(
    specs: list[dict[str, object]],
    steps: int,
    lookback: int,
    horizon: int,
    anomaly_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, labels, kinds = [], [], [], []
    for spec in specs:
        kind = str(spec["kind"])
        for seed in spec["seeds"]:
            df = simulate_state_series(
                steps,
                anomaly_start=anomaly_start,
                seed=int(seed),
                anomaly_kind=kind,
                severity=float(spec["severity"]),
            )
            x, y, anomaly = make_lstm_windows(df, lookback, horizon)
            xs.append(x)
            ys.append(y)
            labels.append(anomaly)
            kinds.append(np.where(anomaly == 1, kind, "normal"))
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(labels),
        np.concatenate(kinds).astype(str),
    )


def make_normal_windows(
    seeds: range,
    steps: int,
    lookback: int,
    horizon: int,
    anomaly_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, labels = [], [], []
    for seed in seeds:
        df = simulate_state_series(
            steps,
            anomaly_start=-1,
            seed=int(seed),
            anomaly_kind="coolant_loss",
            severity=0.0,
        )
        x, y, anomaly = make_lstm_windows(df, lookback, horizon)
        xs.append(x)
        ys.append(y)
        labels.append(anomaly)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(labels)


def load_bundle(horizon: int) -> dict:
    path = MODEL_DIR / f"state_forecaster_h{horizon}_mlp.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train it with: python scripts/train_state_forecaster.py "
            f"--horizon {horizon} --name state_forecaster_h{horizon}"
        )
    bundle = joblib.load(path)
    if int(bundle["horizon"]) != horizon:
        raise ValueError(f"{path} has horizon={bundle['horizon']}, expected {horizon}")
    return bundle


def predict_bundle(bundle: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = bundle["x_scaler"].transform(flatten_windows(x))
    models = bundle.get("forecaster_models") or [bundle["model"]]
    scaled_members = [model.predict(x_scaled) for model in models]
    member_preds = np.stack(
        [bundle["y_scaler"].inverse_transform(pred) for pred in scaled_members],
        axis=0,
    )
    mean_pred = member_preds.mean(axis=0)
    member_std = member_preds.std(axis=0) if len(models) > 1 else np.zeros_like(mean_pred)
    residual_std = np.asarray(bundle["residual_scale"], dtype=float)[None, :]
    total_std = np.sqrt(member_std**2 + residual_std**2)
    return mean_pred, total_std


def physics_project_sequence(pred_flat: np.ndarray, horizon: int, blend: float = 0.65) -> np.ndarray:
    n_features = len(STATE_FEATURES)
    projected = pred_flat.reshape(-1, horizon, n_features).copy()
    rows = projected.reshape(-1, n_features)

    fuel_temp = rows[:, feature_index("fuel_temperature_K")]
    coolant_density = rows[:, feature_index("coolant_density_g_cm3")]
    control_rod = rows[:, feature_index("control_rod_position_pct")]
    power = rows[:, feature_index("power_norm")]

    expected_reactivity = (
        0.0018 * (55.0 - control_rod)
        + 0.095 * (coolant_density - 0.72)
        - 2.2e-5 * (fuel_temp - 900.0)
    )
    expected_keff = 1.0 + expected_reactivity
    expected_flux_fast = 1.0 + 0.70 * (power - 1.0) + 0.12 * (
        1.0 - coolant_density / 0.72
    )
    expected_flux_thermal = 1.0 + 0.55 * (power - 1.0) + 0.26 * (
        coolant_density / 0.72 - 1.0
    )
    expected_fission_rate = power * (1.0 + 0.8 * (expected_keff - 1.0))
    expected_capture_rate = expected_flux_thermal * (
        1.0 + 0.00035 * (fuel_temp - 900.0)
    )

    replacements = {
        "keff": expected_keff,
        "flux_fast_norm": expected_flux_fast,
        "flux_thermal_norm": expected_flux_thermal,
        "fission_rate_norm": expected_fission_rate,
        "capture_rate_norm": expected_capture_rate,
    }
    for name, expected in replacements.items():
        idx = feature_index(name)
        rows[:, idx] = (1.0 - blend) * rows[:, idx] + blend * expected
    return projected.reshape(pred_flat.shape)


def detector_probability(bundle: dict, y_observed_flat: np.ndarray, y_pred_flat: np.ndarray) -> np.ndarray:
    horizon = int(bundle["horizon"])
    n_features = len(bundle["feature_names"])
    features = build_detector_features(
        y_observed_flat,
        y_pred_flat,
        horizon,
        n_features,
        bundle["residual_scale"],
        bundle["physics_scale"],
    )
    return bundle["detector"].predict_proba(bundle["detector_scaler"].transform(features))[:, 1]


def alarm_predictions(bundle: dict, probabilities: np.ndarray) -> np.ndarray:
    smoothed = rolling_mean(probabilities, int(bundle["anomaly_persistence_windows"]))
    return smoothed >= float(bundle["anomaly_threshold"])


def endpoint_metrics(
    y: np.ndarray,
    pred_flat: np.ndarray,
    pred_std_flat: np.ndarray,
    bundle: dict,
) -> dict[str, object]:
    horizon = int(bundle["horizon"])
    n_features = len(STATE_FEATURES)
    y_seq = y.reshape(-1, horizon, n_features)
    pred_seq = pred_flat.reshape(-1, horizon, n_features)
    std_seq = pred_std_flat.reshape(-1, horizon, n_features)
    truth = y_seq[:, -1, :]
    pred = pred_seq[:, -1, :]
    std = np.maximum(std_seq[:, -1, :], 1.0e-12)
    errors = truth - pred

    feature_report = {}
    for idx, feature in enumerate(STATE_FEATURES):
        feature_report[feature] = {
            "mae": float(mean_absolute_error(truth[:, idx], pred[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(truth[:, idx], pred[:, idx]))),
            "coverage_90": float(np.mean(np.abs(errors[:, idx]) <= 1.645 * std[:, idx])),
            "mean_90_interval_width": float(np.mean(2.0 * 1.645 * std[:, idx])),
        }

    y_std = np.std(truth, axis=0)
    y_std = np.where(y_std > 1.0e-12, y_std, 1.0)
    physics_z = np.abs(physics_residuals(pred) / np.asarray(bundle["physics_scale"])[None, :])

    return {
        "feature_metrics": feature_report,
        "mean_normalized_endpoint_rmse": float(
            np.mean(np.sqrt(np.mean(errors**2, axis=0)) / y_std)
        ),
        "mean_90_coverage": float(np.mean(np.abs(errors) <= 1.645 * std)),
        "mean_90_interval_width_scaled": float(np.mean(2.0 * 1.645 * std / y_std)),
        "prediction_physics_residual_mean_abs_z": float(np.mean(physics_z)),
        "prediction_physics_residual_p95_abs_z": float(np.percentile(physics_z, 95)),
    }


def detection_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    normal = labels == 0
    anomaly = labels == 1
    return {
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_alarm_rate": float(np.mean(predicted[normal])) if np.any(normal) else float("nan"),
        "detection_rate": float(np.mean(predicted[anomaly])) if np.any(anomaly) else float("nan"),
    }


def add_gaussian_noise(values: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
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


def apply_sensor_bias(values: np.ndarray) -> np.ndarray:
    biased = values.copy()
    biased[..., feature_index("fuel_temperature_K")] += 3.0
    biased[..., feature_index("coolant_temperature_K")] += 0.9
    biased[..., feature_index("flux_fast_norm")] *= 1.012
    biased[..., feature_index("flux_thermal_norm")] *= 0.990
    biased[..., feature_index("keff")] += 1.2e-4
    return biased


def apply_delay(x: np.ndarray, delay_steps: int = 3) -> np.ndarray:
    delayed = x.copy()
    delayed_features = feature_indices(
        ["fuel_temperature_K", "coolant_temperature_K", "flux_fast_norm", "power_norm"]
    )
    for idx in delayed_features:
        delayed[:, delay_steps:, idx] = delayed[:, :-delay_steps, idx]
    return delayed


def apply_dropout_hold_last(
    x: np.ndarray,
    rng: np.random.Generator,
    dropout_fraction: float = 0.08,
) -> np.ndarray:
    dropped = x.copy()
    monitored = feature_indices(
        ["fuel_temperature_K", "coolant_temperature_K", "power_norm", "flux_fast_norm"]
    )
    mask = rng.random((dropped.shape[0], dropped.shape[1], len(monitored))) < dropout_fraction
    for local_idx, feature_idx in enumerate(monitored):
        for t in range(1, dropped.shape[1]):
            missing = mask[:, t, local_idx]
            dropped[missing, t, feature_idx] = dropped[missing, t - 1, feature_idx]
    return dropped


def recompute_dependent(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    rows = out.reshape(-1, out.shape[-1])
    fuel_temp = rows[:, feature_index("fuel_temperature_K")]
    coolant_density = rows[:, feature_index("coolant_density_g_cm3")]
    control_rod = rows[:, feature_index("control_rod_position_pct")]
    power = rows[:, feature_index("power_norm")]
    reactivity = (
        0.0018 * (55.0 - control_rod)
        + 0.095 * (coolant_density - 0.72)
        - 2.2e-5 * (fuel_temp - 900.0)
    )
    keff = 1.0 + reactivity
    rows[:, feature_index("keff")] = keff
    rows[:, feature_index("flux_fast_norm")] = 1.0 + 0.70 * (power - 1.0) + 0.12 * (
        1.0 - coolant_density / 0.72
    )
    rows[:, feature_index("flux_thermal_norm")] = 1.0 + 0.55 * (power - 1.0) + 0.26 * (
        coolant_density / 0.72 - 1.0
    )
    rows[:, feature_index("fission_rate_norm")] = power * (1.0 + 0.8 * (keff - 1.0))
    rows[:, feature_index("capture_rate_norm")] = rows[:, feature_index("flux_thermal_norm")] * (
        1.0 + 0.00035 * (fuel_temp - 900.0)
    )
    return out


def transform_ood(values: np.ndarray, scenario: str) -> np.ndarray:
    out = values.copy()
    if scenario == "high_power_maneuver":
        out[..., feature_index("power_norm")] += 0.075
        out[..., feature_index("fuel_temperature_K")] += 16.0
        out[..., feature_index("coolant_temperature_K")] += 4.5
        out[..., feature_index("coolant_density_g_cm3")] -= 0.006
    elif scenario == "low_coolant_density":
        out[..., feature_index("coolant_density_g_cm3")] -= 0.024
        out[..., feature_index("coolant_temperature_K")] += 7.0
        out[..., feature_index("fuel_temperature_K")] += 10.0
    elif scenario == "shifted_control_rods":
        out[..., feature_index("control_rod_position_pct")] -= 6.5
        out[..., feature_index("power_norm")] += 0.045
        out[..., feature_index("fuel_temperature_K")] += 9.0
    elif scenario == "stronger_thermal_feedback":
        fuel = out[..., feature_index("fuel_temperature_K")]
        out[..., feature_index("keff")] -= 1.4e-5 * (fuel - np.mean(fuel))
    else:
        raise ValueError(f"Unknown OOD scenario: {scenario}")
    return recompute_dependent(out)


def evaluate_bundle(
    horizon: int,
    steps: int,
    lookback: int,
    anomaly_start: int,
    rng_seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(rng_seed + horizon)
    bundle = load_bundle(horizon)

    x_norm, y_norm, labels_norm = make_normal_windows(
        range(800, 830), steps, lookback, horizon, anomaly_start
    )
    x_known, y_known, labels_known, kinds_known = make_window_block(
        KNOWN_FAULT_TEST_SPECS, steps, lookback, horizon, anomaly_start
    )
    x_unseen, y_unseen, labels_unseen, kinds_unseen = make_window_block(
        UNSEEN_FAULT_TEST_SPECS, steps, lookback, horizon, anomaly_start
    )

    pred_norm, std_norm = predict_bundle(bundle, x_norm)
    pred_norm_projected = physics_project_sequence(pred_norm, horizon)
    base_metrics = endpoint_metrics(y_norm, pred_norm, std_norm, bundle)
    projected_metrics = endpoint_metrics(y_norm, pred_norm_projected, std_norm, bundle)

    prob_norm = detector_probability(bundle, y_norm.reshape(len(y_norm), -1), pred_norm)
    pred_alarm_norm = alarm_predictions(bundle, prob_norm)
    long_normal_false_alarm = float(np.mean(pred_alarm_norm[labels_norm == 0]))

    horizon_summary = {
        "horizon_seconds": horizon,
        "normal_windows": int(len(x_norm)),
        "fuel_temperature_endpoint_rmse_K": base_metrics["feature_metrics"][
            "fuel_temperature_K"
        ]["rmse"],
        "coolant_temperature_endpoint_rmse_K": base_metrics["feature_metrics"][
            "coolant_temperature_K"
        ]["rmse"],
        "keff_endpoint_rmse_pcm": base_metrics["feature_metrics"]["keff"]["rmse"] * 1.0e5,
        "power_endpoint_rmse_pct": base_metrics["feature_metrics"]["power_norm"]["rmse"]
        * 100.0,
        "fast_flux_endpoint_rmse_pct": base_metrics["feature_metrics"]["flux_fast_norm"][
            "rmse"
        ]
        * 100.0,
        "mean_90_coverage": base_metrics["mean_90_coverage"],
        "mean_normalized_endpoint_rmse": base_metrics["mean_normalized_endpoint_rmse"],
        "prediction_physics_residual_mean_abs_z": base_metrics[
            "prediction_physics_residual_mean_abs_z"
        ],
        "projected_prediction_physics_residual_mean_abs_z": projected_metrics[
            "prediction_physics_residual_mean_abs_z"
        ],
        "projected_mean_normalized_endpoint_rmse": projected_metrics[
            "mean_normalized_endpoint_rmse"
        ],
        "long_normal_false_alarm_rate": long_normal_false_alarm,
        "selected_forecaster": str(bundle.get("selected_forecaster", "unknown")),
        "members": ",".join(map(str, bundle.get("selected_forecaster_members", []))),
    }

    robustness_rows: list[dict[str, object]] = []
    baseline_score = float(base_metrics["mean_normalized_endpoint_rmse"])
    conditions = {
        "baseline": (x_norm, y_norm, y_norm, "clean independent normal traces"),
        "gaussian_sensor_noise_1x": (
            add_gaussian_noise(x_norm, rng, level=1.0),
            y_norm,
            add_gaussian_noise(y_norm, rng, level=1.0),
            "measurement noise applied to state history and observed endpoint",
        ),
        "gaussian_sensor_noise_2x": (
            add_gaussian_noise(x_norm, rng, level=2.0),
            y_norm,
            add_gaussian_noise(y_norm, rng, level=2.0),
            "larger measurement noise stress test",
        ),
        "small_sensor_bias": (
            apply_sensor_bias(x_norm),
            y_norm,
            apply_sensor_bias(y_norm),
            "biased measurements; this should raise some alarm score",
        ),
        "three_second_signal_delay": (
            apply_delay(x_norm, delay_steps=3),
            y_norm,
            y_norm,
            "delayed recent sensor history",
        ),
        "eight_percent_dropout_hold_last": (
            apply_dropout_hold_last(x_norm, rng, dropout_fraction=0.08),
            y_norm,
            y_norm,
            "random missing sensor samples filled by hold-last-value",
        ),
    }
    for scenario in [
        "high_power_maneuver",
        "low_coolant_density",
        "shifted_control_rods",
        "stronger_thermal_feedback",
    ]:
        conditions[f"ood_{scenario}"] = (
            transform_ood(x_norm, scenario),
            transform_ood(y_norm, scenario),
            transform_ood(y_norm, scenario),
            "synthetic out-of-distribution operating-regime transform",
        )

    for name, (x_eval, y_truth, y_observed, description) in conditions.items():
        pred_eval, std_eval = predict_bundle(bundle, x_eval)
        metrics = endpoint_metrics(y_truth, pred_eval, std_eval, bundle)
        prob = detector_probability(bundle, y_observed.reshape(len(y_observed), -1), pred_eval)
        alarm = alarm_predictions(bundle, prob)
        robustness_rows.append(
            {
                "horizon_seconds": horizon,
                "condition": name,
                "description": description,
                "mean_normalized_endpoint_rmse": metrics["mean_normalized_endpoint_rmse"],
                "rmse_multiplier_vs_clean": float(
                    metrics["mean_normalized_endpoint_rmse"] / baseline_score
                )
                if baseline_score > 0
                else float("nan"),
                "fuel_temperature_endpoint_rmse_K": metrics["feature_metrics"][
                    "fuel_temperature_K"
                ]["rmse"],
                "keff_endpoint_rmse_pcm": metrics["feature_metrics"]["keff"]["rmse"]
                * 1.0e5,
                "power_endpoint_rmse_pct": metrics["feature_metrics"]["power_norm"][
                    "rmse"
                ]
                * 100.0,
                "mean_90_coverage": metrics["mean_90_coverage"],
                "normal_alarm_rate": float(np.mean(alarm[labels_norm == 0])),
                "physics_residual_mean_abs_z": metrics[
                    "prediction_physics_residual_mean_abs_z"
                ],
            }
        )

    pred_known, _ = predict_bundle(bundle, x_known)
    pred_unseen, _ = predict_bundle(bundle, x_unseen)
    prob_known = detector_probability(bundle, y_known.reshape(len(y_known), -1), pred_known)
    prob_unseen = detector_probability(bundle, y_unseen.reshape(len(y_unseen), -1), pred_unseen)
    alarm_known = alarm_predictions(bundle, prob_known)
    alarm_unseen = alarm_predictions(bundle, prob_unseen)
    detection_summary = {
        "known_fault_detection": detection_metrics(labels_known, alarm_known),
        "unseen_fault_detection": detection_metrics(labels_unseen, alarm_unseen),
    }

    classification = train_and_evaluate_fault_classifier(
        bundle,
        steps,
        lookback,
        horizon,
        anomaly_start,
    )

    return horizon_summary, robustness_rows, [
        {
            "horizon_seconds": horizon,
            "split": "known_fault_families",
            **detection_summary["known_fault_detection"],
        },
        {
            "horizon_seconds": horizon,
            "split": "unseen_fault_families",
            **detection_summary["unseen_fault_detection"],
        },
    ], classification


def anomaly_feature_matrix(
    bundle: dict,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    kinds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred, _ = predict_bundle(bundle, x)
    features = build_detector_features(
        y.reshape(len(y), -1),
        pred,
        int(bundle["horizon"]),
        len(STATE_FEATURES),
        bundle["residual_scale"],
        bundle["physics_scale"],
    )
    mask = labels == 1
    return features[mask], kinds[mask]


def train_and_evaluate_fault_classifier(
    bundle: dict,
    steps: int,
    lookback: int,
    horizon: int,
    anomaly_start: int,
) -> dict[str, object]:
    x_train, y_train, labels_train, kinds_train = make_window_block(
        KNOWN_FAULT_SPECS, steps, lookback, horizon, anomaly_start
    )
    x_test, y_test, labels_test, kinds_test = make_window_block(
        KNOWN_FAULT_TEST_SPECS, steps, lookback, horizon, anomaly_start
    )
    x_unknown, y_unknown, labels_unknown, kinds_unknown = make_window_block(
        UNSEEN_FAULT_TEST_SPECS, steps, lookback, horizon, anomaly_start
    )

    train_x, train_y = anomaly_feature_matrix(bundle, x_train, y_train, labels_train, kinds_train)
    test_x, test_y = anomaly_feature_matrix(bundle, x_test, y_test, labels_test, kinds_test)
    unknown_x, unknown_y = anomaly_feature_matrix(
        bundle, x_unknown, y_unknown, labels_unknown, kinds_unknown
    )

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    classifier = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=91 + horizon,
        n_jobs=-1,
    )
    classifier.fit(train_x_scaled, train_y)

    train_prob = classifier.predict_proba(train_x_scaled)
    known_accept_threshold = float(np.percentile(np.max(train_prob, axis=1), 5))
    test_prob = classifier.predict_proba(scaler.transform(test_x))
    test_pred = classifier.classes_[np.argmax(test_prob, axis=1)]
    unknown_prob = classifier.predict_proba(scaler.transform(unknown_x))
    unknown_as_unknown = np.max(unknown_prob, axis=1) < known_accept_threshold

    labels_order = ["coolant_loss", "control_rod_withdrawal", "flux_detector_bias"]
    cm = confusion_matrix(test_y, test_pred, labels=labels_order)

    classifier_path = MODEL_DIR / f"state_forecaster_h{horizon}_fault_classifier.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "scaler": scaler,
            "known_accept_threshold": known_accept_threshold,
            "classes": list(classifier.classes_),
            "horizon": horizon,
            "feature_names": STATE_FEATURES,
        },
        classifier_path,
    )

    return {
        "horizon_seconds": horizon,
        "known_fault_classifier_accuracy": float(accuracy_score(test_y, test_pred)),
        "known_fault_classifier_macro_f1": float(f1_score(test_y, test_pred, average="macro")),
        "known_accept_threshold": known_accept_threshold,
        "known_fault_window_count": int(len(test_y)),
        "unknown_fault_window_count": int(len(unknown_y)),
        "unknown_family_rejection_rate": float(np.mean(unknown_as_unknown)),
        "unknown_family_names": sorted(set(map(str, unknown_y))),
        "labels": labels_order,
        "confusion_matrix": cm.astype(int).tolist(),
        "model_path": str(classifier_path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_horizon_summary(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: int(row["horizon_seconds"]))
    horizons = np.array([float(row["horizon_seconds"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    panels = [
        ("Fuel temperature", "fuel_temperature_endpoint_rmse_K", "RMSE (K)", "#4C78A8"),
        ("keff", "keff_endpoint_rmse_pcm", "RMSE (pcm)", "#F58518"),
        ("Reactor power", "power_endpoint_rmse_pct", "RMSE (% norm.)", "#54A24B"),
        ("Fast flux", "fast_flux_endpoint_rmse_pct", "RMSE (% norm.)", "#B279A2"),
    ]
    for ax, (title, key, ylabel, color) in zip(axes.flat, panels):
        values = np.array([float(row[key]) for row in rows])
        ax.plot(horizons, values, marker="o", color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Forecast horizon (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Digital Twin Forecast Accuracy Across Prediction Horizons", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "digital_twin_horizon_degradation.png", dpi=220)
    plt.close(fig)


def plot_robustness(rows: list[dict[str, object]], horizon: int = 10) -> None:
    label_map = {
        "baseline": "Baseline",
        "gaussian_sensor_noise_1x": "1x noise",
        "gaussian_sensor_noise_2x": "2x noise",
        "small_sensor_bias": "Bias",
        "three_second_signal_delay": "3 s delay",
        "eight_percent_dropout_hold_last": "8% dropout",
        "ood_high_power_maneuver": "High-power OOD",
        "ood_low_coolant_density": "Low-density OOD",
        "ood_shifted_control_rods": "Rod-shift OOD",
        "ood_stronger_thermal_feedback": "Feedback OOD",
    }
    order = list(label_map)
    selected_by_condition = {
        str(row["condition"]): row
        for row in rows
        if int(row["horizon_seconds"]) == horizon
    }
    selected = [selected_by_condition[name] for name in order if name in selected_by_condition]
    labels = [label_map[str(row["condition"])] for row in selected]
    multipliers = [float(row["rmse_multiplier_vs_clean"]) for row in selected]
    alarms = [100.0 * float(row["normal_alarm_rate"]) for row in selected]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    y_pos = np.arange(len(selected))
    axes[0].barh(y_pos, multipliers, color="#4C78A8")
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Endpoint error multiplier vs clean")
    axes[0].set_title(f"+{horizon} s forecast robustness")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y_pos, alarms, color="#E45756")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Normal alarm rate (%)")
    axes[1].set_title("False alarms / sensor stress")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "digital_twin_robustness_summary.png", dpi=220)
    plt.close(fig)


def plot_uncertainty_and_classification(
    horizon_rows: list[dict[str, object]],
    classifications: list[dict[str, object]],
) -> None:
    horizon_rows = sorted(horizon_rows, key=lambda row: int(row["horizon_seconds"]))
    horizons = [int(row["horizon_seconds"]) for row in horizon_rows]
    coverage = [100.0 * float(row["mean_90_coverage"]) for row in horizon_rows]
    physics = [float(row["prediction_physics_residual_mean_abs_z"]) for row in horizon_rows]

    h10_cls = min(classifications, key=lambda row: abs(int(row["horizon_seconds"]) - 10))
    cm = np.asarray(h10_cls["confusion_matrix"], dtype=float)
    labels = [label.replace("_", "\n") for label in h10_cls["labels"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.15))
    axes[0].bar([str(h) for h in horizons], coverage, color="#72B7B2")
    axes[0].axhline(90.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("90% interval coverage")
    axes[0].set_xlabel("Horizon (s)")
    axes[0].set_ylabel("Coverage (%)")

    axes[1].plot(horizons, physics, marker="o", color="#F58518", linewidth=1.8)
    axes[1].set_title("Physics residual in forecasts")
    axes[1].set_xlabel("Horizon (s)")
    axes[1].set_ylabel("Mean |z|")
    axes[1].grid(True, alpha=0.25)

    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title("+10 s known-fault classifier")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout(pad=1.05, w_pad=1.6)
    fig.savefig(FIGURE_DIR / "digital_twin_uncertainty_classification.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate digital-twin horizon degradation, robustness, uncertainty, and fault classification."
    )
    parser.add_argument("--horizons", default="10,20,30,60")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--anomaly-start", type=int, default=280)
    parser.add_argument("--rng-seed", type=int, default=20260615)
    args = parser.parse_args()

    ensure_project_dirs()
    horizons = parse_horizons(args.horizons)

    horizon_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    classifications: list[dict[str, object]] = []
    for horizon in horizons:
        horizon_summary, robust, detection, classification = evaluate_bundle(
            horizon,
            args.steps,
            args.lookback,
            args.anomaly_start,
            args.rng_seed,
        )
        horizon_rows.append(horizon_summary)
        robustness_rows.extend(robust)
        detection_rows.extend(detection)
        classifications.append(classification)

    metrics = {
        "evaluation_note": (
            "All robustness results use independent simulated trajectories. Sensor noise, bias, delay, dropout, "
            "and OOD rows are stress transforms of the local physics-informed simulator, not external plant data."
        ),
        "horizons": horizon_rows,
        "robustness": robustness_rows,
        "anomaly_detection": detection_rows,
        "fault_classification": classifications,
    }
    metrics_path = MODEL_DIR / "digital_twin_robustness_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    write_csv(PROCESSED_DATA_DIR / "digital_twin_horizon_summary.csv", horizon_rows)
    write_csv(PROCESSED_DATA_DIR / "digital_twin_robustness_summary.csv", robustness_rows)
    write_csv(PROCESSED_DATA_DIR / "digital_twin_anomaly_detection_summary.csv", detection_rows)
    write_csv(PROCESSED_DATA_DIR / "digital_twin_fault_classification_summary.csv", classifications)

    plot_horizon_summary(horizon_rows)
    plot_robustness(robustness_rows, horizon=10 if 10 in horizons else horizons[0])
    plot_uncertainty_and_classification(horizon_rows, classifications)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {PROCESSED_DATA_DIR / 'digital_twin_horizon_summary.csv'}")
    print(f"Wrote {PROCESSED_DATA_DIR / 'digital_twin_robustness_summary.csv'}")
    print(f"Wrote {FIGURE_DIR / 'digital_twin_horizon_degradation.png'}")
    print(f"Wrote {FIGURE_DIR / 'digital_twin_robustness_summary.png'}")
    print(f"Wrote {FIGURE_DIR / 'digital_twin_uncertainty_classification.png'}")


if __name__ == "__main__":
    main()
