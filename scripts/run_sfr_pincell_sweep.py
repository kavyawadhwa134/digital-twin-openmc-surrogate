"""Generate 2-group homogenized group constants for a SODIUM-COOLED FAST (SFR) pin cell.

Generation IV extension of the LWR pin-cell/assembly pipeline. Same machinery,
fast-reactor physics:

  * MOX fuel (reactor-grade Pu vector on depleted U), no soluble boron
  * HT9-like steel cladding, liquid sodium coolant
  * FAST-appropriate 2-group condensation: boundary at 0.1 MeV (the LWR 0.625 eV
    thermal cutoff is meaningless here; the thermal flux fraction is ~0)

Seven state parameters (LHS): Pu fraction, fuel temperature, sodium density,
coolant/structure temperature, fuel radius, cladding thickness, pin pitch.

Output rows: state inputs + 2-group constants (D1,D2, Sa1,Sa2, nuSf1,nuSf2, Sf1,Sf2,
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
from run_assembly_sweep import latin_hypercube

FAST_CUTOFF_EV = 1.0e5   # 0.1 MeV group boundary (fast-reactor condensation)
E_MAX_EV = 2.0e7

# reactor-grade plutonium isotopic vector (weight fractions of Pu)
PU_VECTOR = {"Pu239": 0.586, "Pu240": 0.240, "Pu241": 0.111, "Pu242": 0.063}


def build_sfr_materials(pu_frac, fuel_temp, na_density, cool_temp):
    """MOX fuel + HT9-like steel cladding + liquid sodium coolant."""
    fuel = openmc.Material(name="mox")
    fuel.set_density("g/cm3", 10.5)
    o_frac = 0.118                      # oxygen weight fraction in (U,Pu)O2
    hm = 1.0 - o_frac                   # heavy-metal weight fraction
    for iso, fr in PU_VECTOR.items():
        fuel.add_nuclide(iso, hm * pu_frac * fr, "wo")
    fuel.add_nuclide("U238", hm * (1.0 - pu_frac) * 0.9975, "wo")
    fuel.add_nuclide("U235", hm * (1.0 - pu_frac) * 0.0025, "wo")   # depleted U
    fuel.add_nuclide("O16", o_frac, "wo")
    fuel.temperature = fuel_temp

    clad = openmc.Material(name="ht9")
    clad.set_density("g/cm3", 7.8)
    clad.add_element("Fe", 0.85, "wo")
    clad.add_element("Cr", 0.12, "wo")
    clad.add_element("Ni", 0.02, "wo")
    clad.add_element("Mo", 0.01, "wo")
    clad.temperature = cool_temp

    cool = openmc.Material(name="sodium")
    cool.set_density("g/cm3", na_density)
    cool.add_element("Na", 1.0)
    cool.temperature = cool_temp
    return fuel, clad, cool


def build_sfr_pin_model(state, batches, inactive, particles, seed):
    fuel, clad, cool = build_sfr_materials(
        state["pu_fraction"], state["fuel_temperature_K"],
        state["sodium_density_g_cm3"], state["coolant_temperature_K"])

    r_f = state["fuel_radius_cm"]
    r_c = r_f + state["cladding_thickness_cm"]
    half = state["pin_pitch_cm"] / 2.0

    rf = openmc.ZCylinder(r=r_f)
    rc = openmc.ZCylinder(r=r_c)
    xmin = openmc.XPlane(-half, boundary_type="reflective")
    xmax = openmc.XPlane(half, boundary_type="reflective")
    ymin = openmc.YPlane(-half, boundary_type="reflective")
    ymax = openmc.YPlane(half, boundary_type="reflective")
    zmin = openmc.ZPlane(-0.5, boundary_type="reflective")
    zmax = openmc.ZPlane(0.5, boundary_type="reflective")
    box = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax

    fuel_cell = openmc.Cell(fill=fuel, region=-rf & +zmin & -zmax)
    clad_cell = openmc.Cell(fill=clad, region=+rf & -rc & +zmin & -zmax)
    cool_cell = openmc.Cell(fill=cool, region=+rc & box)
    root_univ = openmc.Universe(cells=[fuel_cell, clad_cell, cool_cell])
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
        space=openmc.stats.Box([-half, -half, -0.5], [half, half, 0.5]),
        constraints={"fissionable": True})

    groups = mgxs.EnergyGroups([0.0, FAST_CUTOFF_EV, E_MAX_EV])
    lib = mgxs.Library(geometry)
    lib.energy_groups = groups
    lib.mgxs_types = ["transport", "absorption", "nu-fission", "fission",
                      "nu-scatter matrix", "chi"]
    lib.domain_type = "universe"
    lib.domains = [root_univ]
    lib.by_nuclide = False
    lib.build_library()
    tallies = openmc.Tallies()
    lib.add_to_tallies_file(tallies, merge=True)

    return openmc.Model(geometry, materials, settings, tallies), lib, root_univ


def extract_group_constants(lib, root_univ):
    def arr(t):
        return np.asarray(lib.get_mgxs(root_univ, t).get_xs())

    transport = arr("transport")
    absorption = arr("absorption")
    nu_fission = arr("nu-fission")
    fission = arr("fission")
    chi = arr("chi")
    scatter = np.asarray(lib.get_mgxs(root_univ, "nu-scatter matrix").get_xs()).reshape(2, 2)
    D = 1.0 / (3.0 * np.clip(transport, 1e-30, None))
    # group 1 = above 0.1 MeV, group 2 = below 0.1 MeV
    return {
        "D1": D[0], "D2": D[1],
        "Sa1": absorption[0], "Sa2": absorption[1],
        "nuSf1": nu_fission[0], "nuSf2": nu_fission[1],
        "Sf1": fission[0], "Sf2": fission[1],
        "Ss1to1": scatter[0, 0], "Ss1to2": scatter[0, 1],
        "Ss2to1": scatter[1, 0], "Ss2to2": scatter[1, 1],
        "chi1": chi[0], "chi2": chi[1],
    }


def run_case(state, batches, inactive, particles, seed, run_subdir, case_id):
    cs = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cs or not Path(cs).exists():
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
    openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)
    run_dir = RUNS_DIR / run_subdir / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model, lib, root_univ = build_sfr_pin_model(state, batches, inactive, particles, seed)
    t0 = time.perf_counter()
    sp_path = model.run(cwd=str(run_dir), output=False)
    elapsed = time.perf_counter() - t0
    with openmc.StatePoint(sp_path) as sp:
        lib.load_from_statepoint(sp)
        keff = sp.keff
        gc = extract_group_constants(lib, root_univ)
    return {
        "case_id": case_id, **state, "particles": particles,
        "k_inf": float(keff.nominal_value), "k_inf_std": float(keff.std_dev),
        **{k: float(v) for k, v in gc.items()},
        "openmc_elapsed_seconds": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cases", type=int, default=150)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--inactive", type=int, default=12)
    p.add_argument("--particles", type=int, default=4500)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--run-subdir", default="sfr_pincell_sweep")
    p.add_argument("--output", default=str(PROCESSED_DATA_DIR / "sfr_pincell_groupconst.csv"))
    args = p.parse_args()

    ensure_project_dirs()
    # pin_pitch min (0.86) > 2*(fuel_radius max 0.35 + clad max 0.055) = 0.81: no overlap
    samples = latin_hypercube(args.n_cases, {
        "pu_fraction": (0.15, 0.30),
        "fuel_temperature_K": (600.0, 1500.0),
        "sodium_density_g_cm3": (0.75, 0.95),
        "coolant_temperature_K": (600.0, 900.0),
        "fuel_radius_cm": (0.28, 0.35),
        "cladding_thickness_cm": (0.035, 0.055),
        "pin_pitch_cm": (0.86, 1.05),
    }, args.seed)

    rows = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    for i, s in enumerate(samples, 1):
        cid = f"sfr_{i:04d}"
        print(f"{cid}: pu={s['pu_fraction']*100:.1f}% fuelT={s['fuel_temperature_K']:.0f}K "
              f"naRho={s['sodium_density_g_cm3']:.3f} pitch={s['pin_pitch_cm']:.3f}", flush=True)
        try:
            rows.append(run_case(s, args.batches, args.inactive, args.particles,
                                 args.seed + i, args.run_subdir, cid))
        except Exception as e:
            print(f"  {cid} FAILED: {e}", flush=True)
        if rows and i % 25 == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)
            el = time.perf_counter() - t_start
            print(f"  [checkpoint] {len(rows)} cases, {el/60:.0f} min elapsed, "
                  f"~{el/i*(args.n_cases-i)/60:.0f} min remaining", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} SFR pin-cell rows to {args.output}")
    if len(df):
        print(f"k_inf range {df.k_inf.min():.4f}-{df.k_inf.max():.4f}, "
              f"mean k_inf std {df.k_inf_std.mean()*1e5:.0f} pcm, "
              f"mean {df.openmc_elapsed_seconds.mean():.0f} s/case")


if __name__ == "__main__":
    main()
