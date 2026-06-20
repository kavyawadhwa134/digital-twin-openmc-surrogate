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

from make_pincell_smoke import build_pincell_model
from project_config import DEFAULT_CROSS_SECTIONS, PROCESSED_DATA_DIR, RUNS_DIR, ensure_project_dirs

EV_TO_J = 1.602176634e-19


def parse_elapsed_seconds(stdout: str) -> float | None:
    match = re.search(r"Total time elapsed\s*=\s*([0-9.Ee+-]+)\s+seconds", stdout)
    if not match:
        return None
    return float(match.group(1))


def run_case(
    case_id: str,
    fuel_temperature: float,
    enrichment: float,
    moderator_density: float,
    moderator_temperature: float,
    fuel_radius: float,
    pin_pitch: float,
    cladding_thickness: float,
    boron_ppm: float,
    batches: int,
    inactive: int,
    particles: int,
    threads: int,
    mpi_ranks: int,
    random_seed: int,
    run_subdir: str,
    force: bool,
) -> dict[str, float | str]:
    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cross_sections or not Path(cross_sections).exists():
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
        openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)

    run_dir = RUNS_DIR / run_subdir / case_id
    statepoint = run_dir / f"statepoint.{batches}.h5"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not statepoint.exists() or force:
        model = build_pincell_model(
            fuel_temperature=fuel_temperature,
            enrichment=enrichment,
            moderator_density=moderator_density,
            moderator_temperature=moderator_temperature,
            fuel_radius=fuel_radius,
            pin_pitch=pin_pitch,
            cladding_thickness=cladding_thickness,
            boron_ppm=boron_ppm,
            batches=batches,
            inactive=inactive,
            particles=particles,
            random_seed=random_seed,
        )
        model.export_to_xml(directory=run_dir)
        env = os.environ.copy()
        command = []
        if mpi_ranks > 1:
            command.extend(["mpiexec", "-n", str(mpi_ranks)])
        command.append("openmc")
        if threads > 1:
            command.extend(["-s", str(threads)])
        t0 = time.perf_counter()
        result = subprocess.run(
            command,
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
        fuel_response = sp.get_tally(name="fuel_response")
        moderator_response = sp.get_tally(name="moderator_response")

        fuel_df = fuel_response.get_pandas_dataframe()
        moderator_df = moderator_response.get_pandas_dataframe()

        fuel_by_score = fuel_df.set_index("score")
        moderator_by_score = moderator_df.set_index("score")

        fuel_flux = float(fuel_by_score.loc["flux", "mean"])
        fuel_flux_std = float(fuel_by_score.loc["flux", "std. dev."])
        fission_rate = float(fuel_by_score.loc["fission", "mean"])
        fission_rate_std = float(fuel_by_score.loc["fission", "std. dev."])
        fuel_capture_rate = float(fuel_by_score.loc["(n,gamma)", "mean"])
        fuel_capture_rate_std = float(fuel_by_score.loc["(n,gamma)", "std. dev."])
        kappa_fission_ev = float(fuel_by_score.loc["kappa-fission", "mean"])
        kappa_fission_ev_std = float(fuel_by_score.loc["kappa-fission", "std. dev."])

        moderator_flux = float(moderator_by_score.loc["flux", "mean"])
        moderator_flux_std = float(moderator_by_score.loc["flux", "std. dev."])
        moderator_capture_rate = float(moderator_by_score.loc["(n,gamma)", "mean"])
        moderator_capture_rate_std = float(moderator_by_score.loc["(n,gamma)", "std. dev."])

    fuel_volume_cm3 = 3.141592653589793 * fuel_radius**2 * 2.0
    power_density_proxy = kappa_fission_ev * EV_TO_J / fuel_volume_cm3
    power_density_proxy_std = kappa_fission_ev_std * EV_TO_J / fuel_volume_cm3

    return {
        "case_id": case_id,
        "fuel_temperature_K": fuel_temperature,
        "enrichment_wt": enrichment,
        "moderator_density_g_cm3": moderator_density,
        "moderator_temperature_K": moderator_temperature,
        "fuel_radius_cm": fuel_radius,
        "pin_pitch_cm": pin_pitch,
        "cladding_thickness_cm": cladding_thickness,
        "boron_ppm": boron_ppm,
        "batches": batches,
        "inactive": inactive,
        "particles": particles,
        "threads": threads,
        "mpi_ranks": mpi_ranks,
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


def latin_hypercube_samples(
    n_cases: int, ranges: dict[str, tuple[float, float]], seed: int
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    names = list(ranges)
    unit = np.empty((n_cases, len(names)))
    for j in range(len(names)):
        unit[:, j] = (np.arange(n_cases) + rng.random(n_cases)) / n_cases
        rng.shuffle(unit[:, j])

    samples = []
    for row in unit:
        sample = {}
        for value, name in zip(row, names):
            low, high = ranges[name]
            sample[name] = float(low + value * (high - low))
        samples.append(sample)
    return samples


def build_samples(args: argparse.Namespace) -> list[dict[str, float]]:
    if args.sample_mode == "grid":
        fuel_temperatures = np.linspace(args.fuel_temp_min, args.fuel_temp_max, args.grid_size)
        enrichments = np.linspace(args.enrichment_min, args.enrichment_max, args.grid_size)
        moderator_densities = np.linspace(
            args.moderator_density_min, args.moderator_density_max, args.grid_size
        )
        samples = []
        for temp, enrichment, density in product(
            fuel_temperatures, enrichments, moderator_densities
        ):
            samples.append(
                {
                    "fuel_temperature_K": float(temp),
                    "enrichment_wt": float(enrichment),
                    "moderator_density_g_cm3": float(density),
                    "moderator_temperature_K": args.moderator_temp_default,
                    "fuel_radius_cm": args.fuel_radius_default,
                    "pin_pitch_cm": args.pin_pitch_default,
                    "cladding_thickness_cm": args.cladding_thickness_default,
                    "boron_ppm": args.boron_ppm_default,
                }
            )
        return samples

    return latin_hypercube_samples(
        args.n_cases,
        {
            "fuel_temperature_K": (args.fuel_temp_min, args.fuel_temp_max),
            "enrichment_wt": (args.enrichment_min, args.enrichment_max),
            "moderator_density_g_cm3": (
                args.moderator_density_min,
                args.moderator_density_max,
            ),
            "moderator_temperature_K": (args.moderator_temp_min, args.moderator_temp_max),
            "fuel_radius_cm": (args.fuel_radius_min, args.fuel_radius_max),
            "pin_pitch_cm": (args.pin_pitch_min, args.pin_pitch_max),
            "cladding_thickness_cm": (
                args.cladding_thickness_min,
                args.cladding_thickness_max,
            ),
            "boron_ppm": (args.boron_ppm_min, args.boron_ppm_max),
        },
        args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small OpenMC pin-cell parameter sweep.")
    parser.add_argument("--force", action="store_true", help="Rerun cases even if statepoints exist.")
    parser.add_argument("--batches", type=int, default=18)
    parser.add_argument("--inactive", type=int, default=5)
    parser.add_argument("--particles", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads passed to openmc -s.")
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=1,
        help="MPI ranks passed as mpiexec -n N openmc. Keep at 1 unless this OpenMC build supports MPI.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-subdir",
        default="pincell_sweep",
        help="Subdirectory under runs/ used for OpenMC case folders.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["grid", "lhs"],
        default="grid",
        help="grid preserves the original 3-D sweep; lhs samples the expanded physics inputs.",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=500,
        help="Number of Latin-hypercube cases when --sample-mode lhs is used.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=6,
        help="Number of values per input dimension. Default 6 gives 216 cases.",
    )
    parser.add_argument("--fuel-temp-min", type=float, default=600.0)
    parser.add_argument("--fuel-temp-max", type=float, default=1200.0)
    parser.add_argument("--enrichment-min", type=float, default=2.5)
    parser.add_argument("--enrichment-max", type=float, default=5.0)
    parser.add_argument("--moderator-density-min", type=float, default=0.65)
    parser.add_argument("--moderator-density-max", type=float, default=0.997)
    parser.add_argument("--moderator-temp-min", type=float, default=560.0)
    parser.add_argument("--moderator-temp-max", type=float, default=620.0)
    parser.add_argument("--moderator-temp-default", type=float, default=293.6)
    parser.add_argument("--fuel-radius-min", type=float, default=0.38)
    parser.add_argument("--fuel-radius-max", type=float, default=0.43)
    parser.add_argument("--fuel-radius-default", type=float, default=0.4096)
    parser.add_argument("--pin-pitch-min", type=float, default=1.20)
    parser.add_argument("--pin-pitch-max", type=float, default=1.34)
    parser.add_argument("--pin-pitch-default", type=float, default=1.26)
    parser.add_argument("--cladding-thickness-min", type=float, default=0.045)
    parser.add_argument("--cladding-thickness-max", type=float, default=0.065)
    parser.add_argument("--cladding-thickness-default", type=float, default=0.057)
    parser.add_argument("--boron-ppm-min", type=float, default=0.0)
    parser.add_argument("--boron-ppm-max", type=float, default=1500.0)
    parser.add_argument("--boron-ppm-default", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "pincell_sweep_openmc.csv"),
    )
    args = parser.parse_args()

    ensure_project_dirs()
    samples = build_samples(args)

    rows = []
    for i, sample in enumerate(samples, start=1):
        prefix = "lhs" if args.sample_mode == "lhs" else "case"
        case_id = f"{prefix}_{i:04d}"
        print(
            f"{case_id}: fuel={sample['fuel_temperature_K']:.0f}K "
            f"enrichment={sample['enrichment_wt']:.2f}% "
            f"mod_density={sample['moderator_density_g_cm3']:.3f} g/cm3 "
            f"mod_temp={sample['moderator_temperature_K']:.1f}K "
            f"pitch={sample['pin_pitch_cm']:.3f}cm"
        )
        rows.append(
            run_case(
                case_id,
                sample["fuel_temperature_K"],
                sample["enrichment_wt"],
                sample["moderator_density_g_cm3"],
                sample["moderator_temperature_K"],
                sample["fuel_radius_cm"],
                sample["pin_pitch_cm"],
                sample["cladding_thickness_cm"],
                sample["boron_ppm"],
                args.batches,
                args.inactive,
                args.particles,
                args.threads,
                args.mpi_ranks,
                args.seed + i,
                args.run_subdir,
                args.force,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} OpenMC pin-cell cases to {output}")


if __name__ == "__main__":
    main()
