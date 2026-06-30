"""Generate 2-group homogenized cross sections (group constants) from 2-D fuel assemblies.

This is the bridge from a single pin-cell to a full reactor core. Routine core analysis does NOT
run continuous-energy Monte Carlo everywhere; it uses the two-step (lattice -> core) method:

  1. LATTICE step (this script): run a heterogeneous 2-D fuel assembly with reflective boundaries
     in OpenMC and tally FLUX-WEIGHTED, assembly-HOMOGENIZED few-group cross sections
     (transport/diffusion coefficient, absorption, nu-fission, fission, scatter matrix, chi), plus
     k-infinity, as a function of the assembly state (fuel T, enrichment, coolant density, boron).
  2. CORE step (downstream): a coarse-mesh nodal diffusion / SP3 solver consumes those group
     constants to compute the full 3-D core power distribution cheaply.

The ML role at core scale is to surrogate THESE group-constant maps (replacing the large
interpolation tables a core simulator normally carries) -- a direct extension of the pin-cell
response surrogate. This script produces that training data.

Output rows: assembly state inputs + 2-group constants (D1,D2, Sa1,Sa2, nuSf1,nuSf2, Sf1,Sf2,
Ss1->1, Ss1->2, Ss2->1, Ss2->2, chi1, chi2) + k_inf and its MC uncertainty.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs as mgxs
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import DEFAULT_CROSS_SECTIONS, PROCESSED_DATA_DIR, RUNS_DIR, ensure_project_dirs

THERMAL_CUTOFF_EV = 0.625
E_MAX_EV = 2.0e7


def build_pin_universe(name, fuel_temp, enrichment, mod_density, mod_temp, boron_ppm,
                       fuel_radius, clad_thickness, is_guide_tube=False):
    water = openmc.Material(name=f"water_{name}")
    water.set_density("g/cm3", mod_density)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    if boron_ppm > 0.0:
        water.add_element("B", boron_ppm * 1.0e-6)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = mod_temp

    clad_outer = fuel_radius + clad_thickness
    cells = []
    mats = [water]
    if is_guide_tube:
        # water-filled guide tube (no fuel)
        cell = openmc.Cell(name=f"gt_{name}", fill=water)
        return openmc.Universe(cells=[cell]), mats

    fuel = openmc.Material(name=f"fuel_{name}")
    fuel.set_density("g/cm3", 10.4)
    fuel.add_element("U", 1.0, enrichment=enrichment)
    fuel.add_element("O", 2.0)
    fuel.temperature = fuel_temp

    clad = openmc.Material(name=f"clad_{name}")
    clad.set_density("g/cm3", 6.55)
    clad.add_element("Zr", 1.0)
    clad.temperature = mod_temp
    mats += [fuel, clad]

    r_fuel = openmc.ZCylinder(r=fuel_radius)
    r_clad = openmc.ZCylinder(r=clad_outer)
    fuel_cell = openmc.Cell(name=f"f_{name}", fill=fuel, region=-r_fuel)
    clad_cell = openmc.Cell(name=f"c_{name}", fill=clad, region=+r_fuel & -r_clad)
    mod_cell = openmc.Cell(name=f"m_{name}", fill=water, region=+r_clad)
    return openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]), mats


def build_assembly_model(fuel_temp, enrichment, mod_density, mod_temp, boron_ppm,
                         fuel_radius, clad_thickness, pin_pitch, lattice_n,
                         batches, inactive, particles, seed):
    pin_u, mats_pin = build_pin_universe(
        "pin", fuel_temp, enrichment, mod_density, mod_temp, boron_ppm,
        fuel_radius, clad_thickness)
    gt_u, mats_gt = build_pin_universe(
        "gt", fuel_temp, enrichment, mod_density, mod_temp, boron_ppm,
        fuel_radius, clad_thickness, is_guide_tube=True)

    # Guide-tube positions (a few symmetric water holes) for a realistic heterogeneous assembly.
    gt_positions = set()
    if lattice_n >= 5:
        c = lattice_n // 2
        offs = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
        span = max(1, lattice_n // 4)
        for dx, dy in offs:
            gt_positions.add((c + dx * span, c + dy * span))

    lat = openmc.RectLattice()
    lat.lower_left = (-lattice_n * pin_pitch / 2, -lattice_n * pin_pitch / 2)
    lat.pitch = (pin_pitch, pin_pitch)
    universes = []
    for j in range(lattice_n):
        row = []
        for i in range(lattice_n):
            row.append(gt_u if (i, j) in gt_positions else pin_u)
        universes.append(row)
    lat.universes = universes

    half = lattice_n * pin_pitch / 2
    xmin = openmc.XPlane(-half, boundary_type="reflective")
    xmax = openmc.XPlane(half, boundary_type="reflective")
    ymin = openmc.YPlane(-half, boundary_type="reflective")
    ymax = openmc.YPlane(half, boundary_type="reflective")
    zmin = openmc.ZPlane(-1.0, boundary_type="reflective")
    zmax = openmc.ZPlane(1.0, boundary_type="reflective")
    root_cell = openmc.Cell(fill=lat, region=+xmin & -xmax & +ymin & -ymax & +zmin & -zmax)
    root_univ = openmc.Universe(cells=[root_cell])
    geometry = openmc.Geometry(root_univ)
    materials = openmc.Materials(geometry.get_all_materials().values())

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles
    settings.seed = seed
    settings.temperature = {"method": "interpolation"}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-half, -half, -1.0], [half, half, 1.0]),
        constraints={"fissionable": True},
    )

    groups = mgxs.EnergyGroups([0.0, THERMAL_CUTOFF_EV, E_MAX_EV])
    lib = mgxs.Library(geometry)
    lib.energy_groups = groups
    lib.mgxs_types = [
        "transport", "absorption", "nu-fission", "fission", "nu-scatter matrix", "chi",
    ]
    lib.domain_type = "universe"
    lib.domains = [root_univ]
    lib.by_nuclide = False
    lib.build_library()
    tallies = openmc.Tallies()
    lib.add_to_tallies_file(tallies, merge=True)

    model = openmc.Model(geometry, materials, settings, tallies)
    return model, lib, root_univ


def extract_group_constants(lib, root_univ):
    def arr(t):
        return np.asarray(lib.get_mgxs(root_univ, t).get_xs())

    transport = arr("transport")           # (2,)
    absorption = arr("absorption")
    nu_fission = arr("nu-fission")
    fission = arr("fission")
    chi = arr("chi")
    scatter = np.asarray(lib.get_mgxs(root_univ, "nu-scatter matrix").get_xs())
    scatter = scatter.reshape(2, 2)        # [g_in, g_out]
    D = 1.0 / (3.0 * np.clip(transport, 1e-30, None))
    # group index 0 = fast (0.625 eV - 20 MeV), 1 = thermal (0 - 0.625 eV) per OpenMC ordering
    return {
        "D1": D[0], "D2": D[1],
        "Sa1": absorption[0], "Sa2": absorption[1],
        "nuSf1": nu_fission[0], "nuSf2": nu_fission[1],
        "Sf1": fission[0], "Sf2": fission[1],
        "Ss1to1": scatter[0, 0], "Ss1to2": scatter[0, 1],
        "Ss2to1": scatter[1, 0], "Ss2to2": scatter[1, 1],
        "chi1": chi[0], "chi2": chi[1],
    }


def run_case(state, lattice_n, batches, inactive, particles, seed, run_subdir, case_id):
    cs = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cs or not Path(cs).exists():
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
        openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)
    run_dir = RUNS_DIR / run_subdir / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model, lib, root_univ = build_assembly_model(
        state["fuel_temperature_K"], state["enrichment_wt"], state["moderator_density_g_cm3"],
        state["moderator_temperature_K"], state["boron_ppm"], state["fuel_radius_cm"],
        state["cladding_thickness_cm"], state["pin_pitch_cm"], lattice_n,
        batches, inactive, particles, seed)
    t0 = time.perf_counter()
    sp_path = model.run(cwd=str(run_dir), output=False)
    elapsed = time.perf_counter() - t0
    with openmc.StatePoint(sp_path) as sp:
        lib.load_from_statepoint(sp)
        keff = sp.keff
        gc = extract_group_constants(lib, root_univ)
    return {
        "case_id": case_id, **state, "lattice_n": lattice_n, "particles": particles,
        "k_inf": float(keff.nominal_value), "k_inf_std": float(keff.std_dev),
        **{k: float(v) for k, v in gc.items()},
        "openmc_elapsed_seconds": elapsed,
    }


def latin_hypercube(n, ranges, seed):
    rng = np.random.default_rng(seed)
    names = list(ranges)
    u = np.empty((n, len(names)))
    for j in range(len(names)):
        u[:, j] = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(u[:, j])
    out = []
    for row in u:
        out.append({nm: float(lo + row[j] * (hi - lo))
                    for j, (nm, (lo, hi)) in enumerate(ranges.items())})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cases", type=int, default=60)
    p.add_argument("--lattice-n", type=int, default=7)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--inactive", type=int, default=10)
    p.add_argument("--particles", type=int, default=2000)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--run-subdir", default="assembly_sweep")
    p.add_argument("--output", default=str(PROCESSED_DATA_DIR / "assembly_groupconst_openmc.csv"))
    args = p.parse_args()

    ensure_project_dirs()
    samples = latin_hypercube(args.n_cases, {
        "fuel_temperature_K": (600.0, 1200.0),
        "enrichment_wt": (2.5, 5.0),
        "moderator_density_g_cm3": (0.65, 0.78),
        "moderator_temperature_K": (560.0, 600.0),
        "boron_ppm": (0.0, 1800.0),
        "fuel_radius_cm": (0.39, 0.42),
        "pin_pitch_cm": (1.24, 1.30),
        "cladding_thickness_cm": (0.05, 0.063),
    }, args.seed)

    rows = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples, 1):
        cid = f"asm_{i:04d}"
        print(f"{cid}: enr={s['enrichment_wt']:.2f}% boron={s['boron_ppm']:.0f}ppm "
              f"fuelT={s['fuel_temperature_K']:.0f}K rho={s['moderator_density_g_cm3']:.3f}", flush=True)
        try:
            rows.append(run_case(s, args.lattice_n, args.batches, args.inactive,
                                 args.particles, args.seed + i, args.run_subdir, cid))
        except Exception as e:
            print(f"  {cid} FAILED: {e}", flush=True)
        # incremental checkpoint so a crash mid-run doesn't lose completed cases
        if rows and i % 25 == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"  [checkpoint] {len(rows)} cases written to {args.output}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} assembly group-constant rows to {args.output}")
    if len(df):
        print(f"k_inf range {df.k_inf.min():.4f}-{df.k_inf.max():.4f}, "
              f"mean k_inf std {df.k_inf_std.mean()*1e5:.0f} pcm")


if __name__ == "__main__":
    main()
