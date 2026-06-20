from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from itertools import product
from pathlib import Path

import numpy as np
import openmc
import pandas as pd

from make_bundle_model import FUEL_VOLUME_CM3, build_bundle_model
from project_config import DEFAULT_CROSS_SECTIONS, PROCESSED_DATA_DIR, RUNS_DIR, ensure_project_dirs

EV_TO_J = 1.602176634e-19


def parse_elapsed_seconds(stdout: str) -> float | None:
    match = re.search(r"Total time elapsed\s*=\s*([0-9.Ee+-]+)\s+seconds", stdout)
    if not match:
        return None
    return float(match.group(1))


def score_sum(df: pd.DataFrame, score: str) -> tuple[float, float]:
    rows = df[df["score"] == score]
    mean = float(rows["mean"].sum())
    std = float(np.sqrt(np.square(rows["std. dev."]).sum()))
    return mean, std


def score_average(df: pd.DataFrame, score: str) -> tuple[float, float]:
    rows = df[df["score"] == score]
    mean = float(rows["mean"].mean())
    std = float(np.sqrt(np.square(rows["std. dev."]).sum()) / len(rows))
    return mean, std


def run_case(
    case_id: str,
    fuel_temperature: float,
    enrichment: float,
    moderator_density: float,
    batches: int,
    inactive: int,
    particles: int,
    force: bool,
) -> dict[str, float | str]:
    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cross_sections or not Path(cross_sections).exists():
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
        openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)

    run_dir = RUNS_DIR / "bundle_sweep" / case_id
    statepoint = run_dir / f"statepoint.{batches}.h5"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not statepoint.exists() or force:
        model = build_bundle_model(
            fuel_temperature=fuel_temperature,
            enrichment=enrichment,
            moderator_density=moderator_density,
            batches=batches,
            inactive=inactive,
            particles=particles,
            random_seed=29,
        )
        model.export_to_xml(directory=run_dir)
        env = os.environ.copy()
        t0 = time.perf_counter()
        result = subprocess.run(
            ["openmc"],
            cwd=run_dir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        wall_seconds = time.perf_counter() - t0
        openmc_elapsed = parse_elapsed_seconds(result.stdout)
        (run_dir / "openmc_stdout.txt").write_text(result.stdout)
    else:
        wall_seconds = 0.0
        openmc_elapsed = None

    with openmc.StatePoint(statepoint) as sp:
        keff = sp.keff
        fuel_df = sp.get_tally(name="fuel_response").get_pandas_dataframe()
        moderator_df = sp.get_tally(name="moderator_response").get_pandas_dataframe()

        fuel_flux, fuel_flux_std = score_average(fuel_df, "flux")
        fission_rate, fission_rate_std = score_sum(fuel_df, "fission")
        fuel_capture_rate, fuel_capture_rate_std = score_sum(fuel_df, "(n,gamma)")
        kappa_fission_ev, kappa_fission_ev_std = score_sum(fuel_df, "kappa-fission")
        moderator_flux = float(moderator_df.loc[moderator_df["score"] == "flux", "mean"].iloc[0])
        moderator_flux_std = float(
            moderator_df.loc[moderator_df["score"] == "flux", "std. dev."].iloc[0]
        )
        moderator_capture_rate = float(
            moderator_df.loc[moderator_df["score"] == "(n,gamma)", "mean"].iloc[0]
        )
        moderator_capture_rate_std = float(
            moderator_df.loc[moderator_df["score"] == "(n,gamma)", "std. dev."].iloc[0]
        )

    power_density_proxy = kappa_fission_ev * EV_TO_J / FUEL_VOLUME_CM3
    power_density_proxy_std = kappa_fission_ev_std * EV_TO_J / FUEL_VOLUME_CM3

    return {
        "case_id": case_id,
        "fuel_temperature_K": fuel_temperature,
        "enrichment_wt": enrichment,
        "moderator_density_g_cm3": moderator_density,
        "batches": batches,
        "inactive": inactive,
        "particles": particles,
        "bundle_pins": 9,
        "keff": float(keff.nominal_value),
        "keff_std": float(keff.std_dev),
        "fuel_flux": fuel_flux,
        "fuel_flux_std": fuel_flux_std,
        "moderator_flux": moderator_flux,
        "moderator_flux_std": moderator_flux_std,
        "fission_rate": fission_rate,
        "fission_rate_std": fission_rate_std,
        "fuel_capture_rate": fuel_capture_rate,
        "fuel_capture_rate_std": fuel_capture_rate_std,
        "moderator_capture_rate": moderator_capture_rate,
        "moderator_capture_rate_std": moderator_capture_rate_std,
        "total_capture_rate": fuel_capture_rate + moderator_capture_rate,
        "kappa_fission_eV_per_source": kappa_fission_ev,
        "kappa_fission_eV_per_source_std": kappa_fission_ev_std,
        "power_density_proxy_J_per_source_cm3": power_density_proxy,
        "power_density_proxy_J_per_source_cm3_std": power_density_proxy_std,
        "openmc_wall_seconds": wall_seconds,
        "openmc_elapsed_seconds": openmc_elapsed if openmc_elapsed is not None else wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small 3x3 OpenMC fuel-bundle sweep.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batches", type=int, default=18)
    parser.add_argument("--inactive", type=int, default=5)
    parser.add_argument("--particles", type=int, default=1800)
    parser.add_argument("--grid-size", type=int, default=5, help="Default 5 gives 125 cases.")
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "bundle_sweep_openmc.csv"),
    )
    args = parser.parse_args()

    ensure_project_dirs()
    fuel_temperatures = np.linspace(600.0, 1200.0, args.grid_size).round(3).tolist()
    enrichments = np.linspace(2.5, 5.0, args.grid_size).round(3).tolist()
    moderator_densities = np.linspace(0.65, 0.997, args.grid_size).round(4).tolist()

    rows = []
    for i, (temp, enrichment, density) in enumerate(
        product(fuel_temperatures, enrichments, moderator_densities), start=1
    ):
        case_id = f"bundle_case_{i:03d}"
        print(
            f"{case_id}: fuel={temp:.0f}K enrichment={enrichment:.2f}% "
            f"moderator_density={density:.3f} g/cm3"
        )
        rows.append(
            run_case(
                case_id,
                temp,
                enrichment,
                density,
                args.batches,
                args.inactive,
                args.particles,
                args.force,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} OpenMC fuel-bundle cases to {output}")


if __name__ == "__main__":
    main()
