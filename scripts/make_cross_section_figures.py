from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import openmc
import pandas as pd

from make_pincell_smoke import build_pincell_model
from project_config import DEFAULT_CROSS_SECTIONS, FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR
from train_pincell_surrogate import add_engineered_features


def ensure_openmc_data() -> None:
    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cross_sections or not Path(cross_sections).exists():
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
        openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)


def make_openmc_geometry_cross_section(output_path: Path) -> None:
    ensure_openmc_data()
    model = build_pincell_model(
        fuel_temperature=900.0,
        enrichment=4.0,
        moderator_density=0.82,
        moderator_temperature=590.0,
        fuel_radius=0.4096,
        pin_pitch=1.26,
        cladding_thickness=0.057,
        boron_ppm=600.0,
        batches=8,
        inactive=2,
        particles=300,
    )
    axes = model.plot(
        basis="xy",
        origin=(0.0, 0.0, 0.0),
        width=(1.26, 1.26),
        pixels=(900, 900),
        color_by="material",
        colors={
            model.materials[0]: "firebrick",
            model.materials[1]: "lightskyblue",
            model.materials[2]: "silver",
        },
    )
    axes.set_title("OpenMC Pin-Cell Geometry Cross-Section")
    axes.set_xlabel("x [cm]")
    axes.set_ylabel("y [cm]")
    axes.figure.tight_layout()
    axes.figure.savefig(output_path, dpi=220)
    plt.close(axes.figure)


def load_response_surrogate(model_path: Path) -> tuple[list[str], dict[str, object]]:
    bundle = joblib.load(model_path)
    return bundle["features"], bundle["models"]


def representative_inputs(dataset_path: Path) -> dict[str, float]:
    df = pd.read_csv(dataset_path)
    return {
        "fuel_temperature_K": float(df["fuel_temperature_K"].median()),
        "enrichment_wt": float(df["enrichment_wt"].median()),
        "moderator_density_g_cm3": float(df["moderator_density_g_cm3"].median()),
        "moderator_temperature_K": float(df["moderator_temperature_K"].median()),
        "fuel_radius_cm": float(df["fuel_radius_cm"].median()),
        "pin_pitch_cm": float(df["pin_pitch_cm"].median()),
        "cladding_thickness_cm": float(df["cladding_thickness_cm"].median()),
        "boron_ppm": float(df["boron_ppm"].median()),
    }


def make_surrogate_response_cross_section(
    model_path: Path, dataset_path: Path, output_path: Path
) -> None:
    features, models = load_response_surrogate(model_path)
    base = representative_inputs(dataset_path)

    fuel_temps = np.linspace(600.0, 1200.0, 90)
    moderator_densities = np.linspace(0.65, 0.997, 90)
    tt, dd = np.meshgrid(fuel_temps, moderator_densities)

    rows = []
    for temp, density in zip(tt.ravel(), dd.ravel()):
        row = base.copy()
        row["fuel_temperature_K"] = float(temp)
        row["moderator_density_g_cm3"] = float(density)
        rows.append(row)
    grid = add_engineered_features(pd.DataFrame(rows))

    keff = models["keff"].predict(grid[features]).reshape(tt.shape)
    fuel_flux = models["fuel_flux"].predict(grid[features]).reshape(tt.shape)
    flux_norm = fuel_flux / np.nanmedian(fuel_flux)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    pcm = axes[0].contourf(tt, dd, keff, levels=24, cmap="viridis")
    axes[0].contour(tt, dd, keff, levels=8, colors="white", linewidths=0.45, alpha=0.65)
    axes[0].set_title("ML Surrogate Cross-Section: keff")
    axes[0].set_xlabel("Fuel temperature [K]")
    axes[0].set_ylabel("Moderator density [g/cm3]")
    fig.colorbar(pcm, ax=axes[0], label="Predicted keff")

    flux = axes[1].contourf(tt, dd, flux_norm, levels=24, cmap="magma")
    axes[1].contour(tt, dd, flux_norm, levels=8, colors="white", linewidths=0.45, alpha=0.65)
    axes[1].set_title("ML Surrogate Cross-Section: fuel flux")
    axes[1].set_xlabel("Fuel temperature [K]")
    axes[1].set_ylabel("Moderator density [g/cm3]")
    fig.colorbar(flux, ax=axes[1], label="Predicted flux / median")

    subtitle = (
        f"Fixed at enrichment={base['enrichment_wt']:.2f} wt%, "
        f"pitch={base['pin_pitch_cm']:.3f} cm, "
        f"boron={base['boron_ppm']:.0f} ppm"
    )
    fig.suptitle(subtitle, fontsize=10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_combined_poster_figure(
    openmc_path: Path, surrogate_path: Path, output_path: Path
) -> None:
    openmc_img = mpimg.imread(openmc_path)
    surrogate_img = mpimg.imread(surrogate_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    axes[0].imshow(openmc_img)
    axes[0].axis("off")
    axes[0].set_title("OpenMC simulated model")
    axes[1].imshow(surrogate_img)
    axes[1].axis("off")
    axes[1].set_title("ML surrogate response surface")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create OpenMC geometry and ML response cross-section figures."
    )
    parser.add_argument(
        "--model",
        default=str(
            MODEL_DIR / "pincell_lhs120_highstat_engineered_response_surrogate_best.joblib"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(PROCESSED_DATA_DIR / "pincell_lhs120_highstat_openmc.csv"),
    )
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    openmc_path = FIGURE_DIR / "openmc_pincell_geometry_cross_section.png"
    surrogate_path = FIGURE_DIR / "ml_surrogate_response_cross_section.png"
    combined_path = FIGURE_DIR / "openmc_vs_ml_cross_section_summary.png"

    make_openmc_geometry_cross_section(openmc_path)
    make_surrogate_response_cross_section(Path(args.model), Path(args.dataset), surrogate_path)
    make_combined_poster_figure(openmc_path, surrogate_path, combined_path)

    print(f"Wrote {openmc_path}")
    print(f"Wrote {surrogate_path}")
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
