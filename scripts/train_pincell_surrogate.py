from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


CANDIDATE_FEATURES = [
    "fuel_temperature_K",
    "enrichment_wt",
    "moderator_density_g_cm3",
    "moderator_temperature_K",
    "fuel_radius_cm",
    "pin_pitch_cm",
    "cladding_thickness_cm",
    "boron_ppm",
]
ENGINEERED_FEATURES = [
    "fuel_area_cm2",
    "clad_outer_radius_cm",
    "clad_outer_area_cm2",
    "moderator_area_cm2",
    "moderator_to_fuel_area_ratio",
    "pitch_to_fuel_diameter",
    "fuel_temperature_minus_moderator_temperature_K",
    "enrichment_x_moderator_density",
    "boron_x_moderator_density",
]
TARGETS = [
    "keff",
    "fuel_flux",
    "moderator_flux",
    "fission_rate",
    "fuel_capture_rate",
    "moderator_capture_rate",
    "total_capture_rate",
    "power_density_proxy_J_per_source_cm3",
]
STD_COLUMNS = {
    "keff": "keff_std",
    "fuel_flux": "fuel_flux_std",
    "moderator_flux": "moderator_flux_std",
    "fission_rate": "fission_rate_std",
    "fuel_capture_rate": "fuel_capture_rate_std",
    "moderator_capture_rate": "moderator_capture_rate_std",
    "power_density_proxy_J_per_source_cm3": "power_density_proxy_J_per_source_cm3_std",
}


def pcm(delta_keff: np.ndarray) -> np.ndarray:
    return delta_keff * 1.0e5


def relative_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1.0e-30)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def target_label(target: str) -> str:
    labels = {
        "keff": "keff",
        "fuel_flux": "Fuel flux",
        "moderator_flux": "Moderator flux",
        "fission_rate": "Fission rate",
        "fuel_capture_rate": "Fuel capture rate",
        "moderator_capture_rate": "Moderator capture rate",
        "total_capture_rate": "Total capture rate",
        "power_density_proxy_J_per_source_cm3": "Power-density proxy",
    }
    return labels.get(target, target)


def target_std_values(df: pd.DataFrame, target: str) -> np.ndarray | None:
    std_col = std_column_for_target(target)
    if std_col is not None and std_col in df.columns:
        std = df[std_col].to_numpy(dtype=float)
    elif target == "total_capture_rate" and {
        "fuel_capture_rate_std",
        "moderator_capture_rate_std",
    }.issubset(df.columns):
        std = np.sqrt(
            df["fuel_capture_rate_std"].to_numpy(dtype=float) ** 2
            + df["moderator_capture_rate_std"].to_numpy(dtype=float) ** 2
        )
    else:
        return None
    if not np.isfinite(std).all() or np.nanmax(std) <= 0.0:
        return None
    return np.clip(std, 1.0e-30, None)


def openmc_sample_weights(df: pd.DataFrame, target: str) -> np.ndarray | None:
    std = target_std_values(df, target)
    if std is None:
        return None
    low, high = np.percentile(std, [5.0, 95.0])
    std = np.clip(std, max(low, 1.0e-30), max(high, low, 1.0e-30))
    weights = 1.0 / np.square(std)
    weights = weights / np.mean(weights)
    return np.clip(weights, 0.05, 20.0)


def known_noise_alpha(
    train_df: pd.DataFrame,
    target: str,
    min_alpha: float,
) -> np.ndarray | None:
    std = target_std_values(train_df, target)
    if std is None:
        return None
    y_scale = float(np.std(train_df[target].to_numpy(dtype=float)))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        return None
    alpha = np.square(std / y_scale)
    return np.clip(alpha, min_alpha, None)


def label_uncertainty_summary(df: pd.DataFrame, targets: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for target in targets:
        std = target_std_values(df, target)
        if std is None:
            continue
        scale = 1.0e5 if target == "keff" else 1.0
        suffix = "_pcm" if target == "keff" else ""
        summary[target] = {
            f"mean_std{suffix}": float(np.mean(std) * scale),
            f"median_std{suffix}": float(np.median(std) * scale),
            f"p95_std{suffix}": float(np.percentile(std, 95.0) * scale),
        }
    return summary


def build_models(random_state: int, include_gpr: bool = True) -> dict[str, Pipeline]:
    return build_models_for_target(
        random_state=random_state,
        include_gpr=include_gpr,
        train_df=None,
        target=None,
        use_known_noise_gpr=False,
        min_noise_alpha=1.0e-8,
    )


def build_models_for_target(
    random_state: int,
    include_gpr: bool,
    train_df: pd.DataFrame | None,
    target: str | None,
    use_known_noise_gpr: bool,
    min_noise_alpha: float,
) -> dict[str, Pipeline]:
    models = {
        "hgb": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=260,
                        learning_rate=0.04,
                        max_leaf_nodes=22,
                        l2_regularization=1.0e-4,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=2,
                        max_features=0.8,
                        bootstrap=True,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if include_gpr:
        models.update(
            {
        "gpr": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                        + WhiteKernel(noise_level=1.0e-5),
                        normalize_y=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gpr_matern": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                        + WhiteKernel(noise_level=1.0e-5),
                        normalize_y=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
            }
        )
        if use_known_noise_gpr and train_df is not None and target is not None:
            alpha = known_noise_alpha(train_df, target, min_noise_alpha)
            if alpha is not None:
                models.update(
                    {
                        "gpr_noise_rbf": Pipeline(
                            [
                                ("scale", StandardScaler()),
                                (
                                    "model",
                                    GaussianProcessRegressor(
                                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                                        alpha=alpha,
                                        normalize_y=True,
                                        random_state=random_state,
                                    ),
                                ),
                            ]
                        ),
                        "gpr_noise_matern": Pipeline(
                            [
                                ("scale", StandardScaler()),
                                (
                                    "model",
                                    GaussianProcessRegressor(
                                        kernel=ConstantKernel(1.0)
                                        * Matern(length_scale=1.0, nu=1.5),
                                        alpha=alpha,
                                        normalize_y=True,
                                        random_state=random_state,
                                    ),
                                ),
                            ]
                        ),
                    }
                )
    return models


def score_target(
    target: str,
    truth: np.ndarray,
    pred: np.ndarray,
    test_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    result = {
        "r2": float(r2_score(truth, pred)),
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "relative_mae": relative_mae(truth, pred),
    }
    if test_df is not None:
        std = target_std_values(test_df, target)
        if std is not None:
            abs_error = np.abs(pred - truth)
            error_over_std = abs_error / np.clip(std, 1.0e-30, None)
            result["mean_abs_error_over_openmc_std"] = float(np.mean(error_over_std))
            result["median_abs_error_over_openmc_std"] = float(np.median(error_over_std))
            result["percent_within_2_openmc_std"] = float(
                np.mean(error_over_std <= 2.0) * 100.0
            )
            result["percent_within_3_openmc_std"] = float(
                np.mean(error_over_std <= 3.0) * 100.0
            )
    if target == "keff":
        error = pred - truth
        result["mae_pcm"] = float(mean_absolute_error(truth, pred) * 1.0e5)
        result["rmse_pcm"] = float(np.sqrt(mean_squared_error(truth, pred)) * 1.0e5)
        result["max_abs_error_pcm"] = float(np.max(np.abs(pcm(error))))
        abs_error_pcm = np.abs(pcm(error))
        result["percent_within_500_pcm"] = float(np.mean(abs_error_pcm <= 500.0) * 100.0)
        result["percent_within_1000_pcm"] = float(np.mean(abs_error_pcm <= 1000.0) * 100.0)
    return result


def resolve_features(df: pd.DataFrame) -> list[str]:
    features = [
        feature
        for feature in [*CANDIDATE_FEATURES, *ENGINEERED_FEATURES]
        if feature in df.columns
    ]
    if not features:
        raise ValueError("No recognized surrogate input columns found in dataset.")
    return features


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_geometry = {
        "fuel_radius_cm",
        "pin_pitch_cm",
        "cladding_thickness_cm",
    }.issubset(df.columns)
    if required_geometry:
        fuel_radius = df["fuel_radius_cm"]
        clad_outer_radius = df["fuel_radius_cm"] + df["cladding_thickness_cm"]
        df["fuel_area_cm2"] = np.pi * fuel_radius**2
        df["clad_outer_radius_cm"] = clad_outer_radius
        df["clad_outer_area_cm2"] = np.pi * clad_outer_radius**2
        df["moderator_area_cm2"] = df["pin_pitch_cm"] ** 2 - df["clad_outer_area_cm2"]
        df["moderator_to_fuel_area_ratio"] = df["moderator_area_cm2"] / df[
            "fuel_area_cm2"
        ].clip(lower=1.0e-30)
        df["pitch_to_fuel_diameter"] = df["pin_pitch_cm"] / (
            2.0 * fuel_radius
        ).clip(lower=1.0e-30)
    if {"fuel_temperature_K", "moderator_temperature_K"}.issubset(df.columns):
        df["fuel_temperature_minus_moderator_temperature_K"] = (
            df["fuel_temperature_K"] - df["moderator_temperature_K"]
        )
    if {"enrichment_wt", "moderator_density_g_cm3"}.issubset(df.columns):
        df["enrichment_x_moderator_density"] = (
            df["enrichment_wt"] * df["moderator_density_g_cm3"]
        )
    if {"boron_ppm", "moderator_density_g_cm3"}.issubset(df.columns):
        df["boron_x_moderator_density"] = (
            df["boron_ppm"] * df["moderator_density_g_cm3"]
        )
    return df


def predict_with_gp_std(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    estimator = model.named_steps.get("model")
    scaler = model.named_steps.get("scale")
    if isinstance(estimator, GaussianProcessRegressor) and scaler is not None:
        scaled = scaler.transform(x)
        pred, std = estimator.predict(scaled, return_std=True)
        return pred, std
    return model.predict(x), None


def fit_candidate_model(
    model: Pipeline,
    name: str,
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    use_openmc_weights: bool,
) -> None:
    fit_params = {}
    if use_openmc_weights and name in {"hgb", "rf", "extra_trees"}:
        weights = openmc_sample_weights(train_df, target)
        if weights is not None:
            fit_params["model__sample_weight"] = weights
    model.fit(train_df[features], train_df[target], **fit_params)


def model_selection_score(
    target: str,
    metrics: dict[str, float],
    selection_objective: str,
) -> float:
    base_score = metrics["mae_pcm"] if target == "keff" else metrics["relative_mae"]
    if selection_objective != "noise_aware":
        return float(base_score)
    uncertainty_score = metrics.get("mean_abs_error_over_openmc_std")
    if uncertainty_score is None:
        return float(base_score)
    return float(base_score * (1.0 + 0.05 * uncertainty_score))


def make_holdout_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    if "fuel_temperature_K" in df.columns and df["fuel_temperature_K"].nunique() > 1:
        unique_temps = np.sort(df["fuel_temperature_K"].unique())
        if len(unique_temps) <= 20:
            masks["held_out_highest_fuel_temperature"] = df["fuel_temperature_K"] == unique_temps[-1]
        else:
            threshold = df["fuel_temperature_K"].quantile(0.8)
            masks["held_out_hot_fuel_regime"] = df["fuel_temperature_K"] >= threshold
    if "pin_pitch_cm" in df.columns and df["pin_pitch_cm"].nunique() > 1:
        threshold = df["pin_pitch_cm"].quantile(0.8)
        masks["held_out_wide_pitch_geometry"] = df["pin_pitch_cm"] >= threshold
    return {
        name: mask
        for name, mask in masks.items()
        if mask.sum() >= 5 and (~mask).sum() >= 10
    }


def select_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    include_gpr: bool,
    use_openmc_weights: bool,
    use_known_noise_gpr: bool,
    selection_objective: str,
    min_noise_alpha: float,
) -> dict[str, float | str]:
    best_name = None
    best_score = float("inf")
    best_metrics = None
    for name, model in build_models_for_target(
        random_state=random_state,
        include_gpr=include_gpr,
        train_df=train_df,
        target=target,
        use_known_noise_gpr=use_known_noise_gpr,
        min_noise_alpha=min_noise_alpha,
    ).items():
        fit_candidate_model(model, name, train_df, features, target, use_openmc_weights)
        pred = model.predict(test_df[features])
        metrics = score_target(target, test_df[target].to_numpy(), pred, test_df)
        selection_score = model_selection_score(target, metrics, selection_objective)
        if selection_score < best_score:
            best_name = name
            best_score = selection_score
            best_metrics = metrics
    assert best_name is not None and best_metrics is not None
    return {"best_model": best_name, **best_metrics}


def validation_report(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    random_state: int,
    test_size: float,
    include_gpr: bool,
    use_openmc_weights: bool,
    use_known_noise_gpr: bool,
    selection_objective: str,
    min_noise_alpha: float,
) -> dict[str, dict[str, dict[str, float | str]]]:
    report_targets = [
        target
        for target in [
            "keff",
            "fuel_flux",
            "fission_rate",
            "total_capture_rate",
            "power_density_proxy_J_per_source_cm3",
        ]
        if target in targets
    ]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    report = {
        "random_interpolation": {
            target: select_and_score(
                train_df,
                test_df,
                features,
                target,
                random_state,
                include_gpr,
                use_openmc_weights,
                use_known_noise_gpr,
                selection_objective,
                min_noise_alpha,
            )
            for target in report_targets
        }
    }
    for split_name, mask in make_holdout_masks(df).items():
        split_train = df.loc[~mask]
        split_test = df.loc[mask]
        split_include_gpr = include_gpr and len(split_train) <= 800
        report[split_name] = {
            target: select_and_score(
                split_train,
                split_test,
                features,
                target,
                random_state,
                split_include_gpr,
                use_openmc_weights,
                use_known_noise_gpr,
                selection_objective,
                min_noise_alpha,
            )
            for target in report_targets
        }
    return report


def std_column_for_target(target: str) -> str | None:
    return STD_COLUMNS.get(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small pin-cell response surrogate from OpenMC sweep labels."
    )
    parser.add_argument(
        "--dataset",
        default=str(PROCESSED_DATA_DIR / "pincell_sweep_openmc.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=12)
    parser.add_argument("--name", default="pincell", help="Artifact prefix, e.g. pincell or bundle.")
    parser.add_argument(
        "--gpr-max-rows",
        type=int,
        default=800,
        help="Include Gaussian-process candidates only up to this many rows.",
    )
    parser.add_argument(
        "--skip-validation-report",
        action="store_true",
        help="Skip extra held-out temperature/geometry validation fits.",
    )
    parser.add_argument(
        "--disable-openmc-weights",
        action="store_true",
        help="Do not use inverse-variance OpenMC label weights for tree models.",
    )
    parser.add_argument(
        "--disable-noise-aware-gpr",
        action="store_true",
        help="Do not add Gaussian-process candidates that use OpenMC label uncertainties as alpha.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["mae", "noise_aware"],
        default="noise_aware",
        help="Model selection objective. noise_aware lightly penalizes errors larger than OpenMC uncertainty.",
    )
    parser.add_argument(
        "--min-noise-alpha",
        type=float,
        default=1.0e-8,
        help="Minimum normalized per-sample alpha for noise-aware Gaussian processes.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    df = add_engineered_features(pd.read_csv(args.dataset))
    features = resolve_features(df)
    targets = [target for target in TARGETS if target in df.columns]
    if not targets:
        raise ValueError("No recognized surrogate target columns found in dataset.")
    include_gpr = len(df) <= args.gpr_max_rows
    use_openmc_weights = not args.disable_openmc_weights
    use_known_noise_gpr = not args.disable_noise_aware_gpr
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )

    results = {}
    best_models = {}
    best_model_names = {}
    best_predictions = {}
    best_prediction_stds = {}
    total_predict_seconds = 0.0

    for target in targets:
        target_results = {}
        best_name = None
        best_score = float("inf")
        best_pred = None
        best_model = None

        for name, model in build_models_for_target(
            random_state=args.random_state,
            include_gpr=include_gpr,
            train_df=train_df,
            target=target,
            use_known_noise_gpr=use_known_noise_gpr,
            min_noise_alpha=args.min_noise_alpha,
        ).items():
            t0 = time.perf_counter()
            fit_candidate_model(
                model,
                name,
                train_df,
                features,
                target,
                use_openmc_weights,
            )
            train_seconds = time.perf_counter() - t0

            t1 = time.perf_counter()
            pred = model.predict(test_df[features])
            predict_seconds = time.perf_counter() - t1
            metrics_for_target = score_target(
                target,
                test_df[target].to_numpy(),
                pred,
                test_df,
            )
            metrics_for_target["train_seconds"] = float(train_seconds)
            metrics_for_target["predict_seconds"] = float(predict_seconds)
            metrics_for_target["predict_seconds_per_case"] = float(predict_seconds / len(test_df))
            target_results[name] = metrics_for_target

            selection_score = model_selection_score(
                target,
                metrics_for_target,
                args.selection_objective,
            )
            if selection_score < best_score:
                best_name = name
                best_score = selection_score
                best_pred = pred
                best_model = model

        assert best_name is not None and best_pred is not None and best_model is not None
        best_models[target] = best_model
        best_model_names[target] = best_name
        best_pred, best_std = predict_with_gp_std(best_model, test_df[features])
        best_predictions[target] = best_pred
        best_prediction_stds[target] = best_std
        total_predict_seconds += target_results[best_name]["predict_seconds"]
        results[target] = {
            "best_model": best_name,
            "models": target_results,
        }
        joblib.dump(best_model, MODEL_DIR / f"{args.name}_surrogate_{target}_{best_name}.joblib")

    joblib.dump(
        {
            "features": features,
            "targets": targets,
            "best_model_names": best_model_names,
            "models": best_models,
        },
        MODEL_DIR / f"{args.name}_response_surrogate_best.joblib",
    )

    openmc_mean_seconds = float(df["openmc_elapsed_seconds"].replace(0.0, np.nan).mean())
    ml_seconds_per_case = total_predict_seconds / len(test_df)
    speedup = openmc_mean_seconds / ml_seconds_per_case if ml_seconds_per_case > 0 else float("inf")

    comparison = test_df.copy()
    for target in targets:
        comparison[f"ml_{target}"] = best_predictions[target]
        if best_prediction_stds[target] is not None:
            comparison[f"ml_{target}_std"] = best_prediction_stds[target]
        comparison[f"{target}_error"] = comparison[f"ml_{target}"] - comparison[target]
        comparison[f"{target}_abs_error"] = comparison[f"{target}_error"].abs()
        comparison[f"{target}_relative_error"] = comparison[f"{target}_abs_error"] / comparison[
            target
        ].abs().clip(lower=1.0e-30)
        std_col = std_column_for_target(target)
        if std_col is not None and std_col in comparison.columns:
            comparison[f"{target}_error_over_openmc_std"] = comparison[
                f"{target}_abs_error"
            ] / comparison[std_col].abs().clip(lower=1.0e-30)
    if {"fuel_capture_rate_std", "moderator_capture_rate_std"}.issubset(comparison.columns):
        total_capture_std = np.sqrt(
            comparison["fuel_capture_rate_std"] ** 2
            + comparison["moderator_capture_rate_std"] ** 2
        )
        comparison["total_capture_rate_error_over_openmc_std"] = comparison[
            "total_capture_rate_abs_error"
        ] / total_capture_std.clip(lower=1.0e-30)
    comparison["keff_error_pcm"] = pcm(comparison["keff_error"].to_numpy())
    comparison_path = PROCESSED_DATA_DIR / f"{args.name}_surrogate_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    metrics = {
        "best_models": best_model_names,
        "features": features,
        "targets": targets,
        "gpr_candidates_included": include_gpr,
        "noise_aware_training": {
            "openmc_sample_weights_enabled": use_openmc_weights,
            "known_noise_gpr_enabled": use_known_noise_gpr and include_gpr,
            "selection_objective": args.selection_objective,
            "min_noise_alpha": args.min_noise_alpha,
        },
        "label_uncertainty_summary": label_uncertainty_summary(df, targets),
        "validation_report": None
        if args.skip_validation_report
        else validation_report(
            df,
            features,
            targets,
            args.random_state,
            args.test_size,
            include_gpr,
            use_openmc_weights,
            use_known_noise_gpr,
            args.selection_objective,
            args.min_noise_alpha,
        ),
        "target_results": results,
        "summary": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "keff_mae_pcm": results["keff"]["models"][best_model_names["keff"]]["mae_pcm"],
            "keff_rmse_pcm": results["keff"]["models"][best_model_names["keff"]]["rmse_pcm"],
            "keff_max_abs_error_pcm": results["keff"]["models"][best_model_names["keff"]][
                "max_abs_error_pcm"
            ],
            "keff_percent_within_500_pcm": results["keff"]["models"][best_model_names["keff"]][
                "percent_within_500_pcm"
            ],
            "keff_percent_within_1000_pcm": results["keff"]["models"][best_model_names["keff"]][
                "percent_within_1000_pcm"
            ],
            "mean_relative_mae_across_targets": float(
                np.mean(
                    [
                        results[target]["models"][best_model_names[target]]["relative_mae"]
                        for target in targets
                    ]
                )
            ),
        },
        "openmc_mean_elapsed_seconds_per_case": openmc_mean_seconds,
        "ml_predict_seconds_per_case": ml_seconds_per_case,
        "speedup_vs_openmc_pincell": speedup,
        "note": (
            "This is a small OpenMC-label response surrogate for keff, not a replacement "
            "for particle transport or a validated full-core model."
        ),
    }
    metrics_path = MODEL_DIR / f"{args.name}_surrogate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 5.2))
    plt.errorbar(
        comparison["keff"],
        comparison["ml_keff"],
        xerr=comparison["keff_std"],
        fmt="o",
        capsize=3,
        alpha=0.8,
    )
    low = min(comparison["keff"].min(), comparison["ml_keff"].min())
    high = max(comparison["keff"].max(), comparison["ml_keff"].max())
    plt.plot([low, high], [low, high], color="black", linewidth=1.1)
    plt.xlabel("OpenMC keff")
    plt.ylabel("ML surrogate keff")
    plt.title("Pin-Cell Response Surrogate vs OpenMC")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{args.name}_surrogate_vs_openmc.png", dpi=220)
    plt.close()

    plot_targets = [
        target
        for target in [
            "keff",
            "fuel_flux",
            "fission_rate",
            "fuel_capture_rate",
            "total_capture_rate",
            "power_density_proxy_J_per_source_cm3",
        ]
        if target in targets
    ]
    ncols = 3
    nrows = int(np.ceil(len(plot_targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.5 * nrows), squeeze=False)
    for ax, target in zip(axes.ravel(), plot_targets):
        truth = comparison[target]
        pred = comparison[f"ml_{target}"]
        ax.scatter(truth, pred, s=28, alpha=0.82)
        low = min(truth.min(), pred.min())
        high = max(truth.max(), pred.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1.0)
        ax.set_title(target_label(target), fontsize=10)
        ax.set_xlabel("OpenMC")
        ax.set_ylabel("ML")
        if target == "power_density_proxy_J_per_source_cm3":
            ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    for ax in axes.ravel()[len(plot_targets) :]:
        ax.axis("off")
    fig.suptitle("Multi-Output Pin-Cell Surrogate vs OpenMC", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.name}_multioutput_surrogate_vs_openmc.png", dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote comparison rows to {comparison_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
