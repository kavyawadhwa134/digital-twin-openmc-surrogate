"""OpenMC reference for the two-step pipeline validation: a 3x3 assembly COLORSET.

Builds a 3x3 checkerboard of two assembly types (A: high enrichment, B: low enrichment),
each a 7x7 pin lattice with guide tubes (same geometry as run_assembly_sweep.py), with
reflective boundaries, and runs it DIRECTLY in continuous-energy Monte Carlo:

    * k_eff of the colorset
    * 3x3 assembly-wise fission-rate map (the reference power distribution)

It also runs the two single-assembly lattice calculations (types A and B) to extract
REFERENCE 2-group group constants, so the downstream comparison can decompose:

    diffusion(OpenMC GCs)  vs  diffusion(ML GCs)  vs  direct MC
    -> method/homogenization error  vs  surrogate-added error

Output: data/processed/core_colorset_reference.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import openmc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import DEFAULT_CROSS_SECTIONS, PROCESSED_DATA_DIR, RUNS_DIR, ensure_project_dirs
from run_assembly_sweep import build_pin_universe, build_assembly_model, extract_group_constants

# ---- shared state (mid-range, same envelope as the surrogate's training LHS) ----
STATE_COMMON = dict(
    fuel_temperature_K=900.0,
    moderator_density_g_cm3=0.72,
    moderator_temperature_K=580.0,
    boron_ppm=900.0,
    fuel_radius_cm=0.41,
    cladding_thickness_cm=0.057,
    pin_pitch_cm=1.27,
)
ENRICH_A = 4.4   # high-enrichment assembly
ENRICH_B = 2.8   # low-enrichment assembly
LATTICE_N = 7    # pins per assembly side
CORE_N = 3       # assemblies per core side (3x3 checkerboard)

# guide-tube positions inside each 7x7 assembly (same rule as run_assembly_sweep)
def gt_positions(n=LATTICE_N):
    c, span = n // 2, max(1, n // 4)
    return {(c + dx * span, c + dy * span) for dx, dy in
            [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]}


def build_colorset_model(batches, inactive, particles, seed):
    s = STATE_COMMON
    pinA, _ = build_pin_universe("pinA", s["fuel_temperature_K"], ENRICH_A,
                                 s["moderator_density_g_cm3"], s["moderator_temperature_K"],
                                 s["boron_ppm"], s["fuel_radius_cm"], s["cladding_thickness_cm"])
    pinB, _ = build_pin_universe("pinB", s["fuel_temperature_K"], ENRICH_B,
                                 s["moderator_density_g_cm3"], s["moderator_temperature_K"],
                                 s["boron_ppm"], s["fuel_radius_cm"], s["cladding_thickness_cm"])
    gt, _ = build_pin_universe("gt", s["fuel_temperature_K"], ENRICH_A,
                               s["moderator_density_g_cm3"], s["moderator_temperature_K"],
                               s["boron_ppm"], s["fuel_radius_cm"], s["cladding_thickness_cm"],
                               is_guide_tube=True)

    gts = gt_positions()
    npins = CORE_N * LATTICE_N
    pitch = s["pin_pitch_cm"]
    half = npins * pitch / 2.0

    lat = openmc.RectLattice()
    lat.lower_left = (-half, -half)
    lat.pitch = (pitch, pitch)
    universes = []
    for J in range(npins):          # row (y)
        row = []
        for I in range(npins):      # col (x)
            aI, aJ = I // LATTICE_N, J // LATTICE_N          # assembly indices
            li, lj = I % LATTICE_N, J % LATTICE_N            # local pin indices
            if (li, lj) in gts:
                row.append(gt)
            else:
                row.append(pinA if (aI + aJ) % 2 == 0 else pinB)
        universes.append(row)
    lat.universes = universes

    xmin = openmc.XPlane(-half, boundary_type="reflective")
    xmax = openmc.XPlane(half, boundary_type="reflective")
    ymin = openmc.YPlane(-half, boundary_type="reflective")
    ymax = openmc.YPlane(half, boundary_type="reflective")
    zmin = openmc.ZPlane(-1.0, boundary_type="reflective")
    zmax = openmc.ZPlane(1.0, boundary_type="reflective")
    root = openmc.Cell(fill=lat, region=+xmin & -xmax & +ymin & -ymax & +zmin & -zmax)
    geometry = openmc.Geometry(openmc.Universe(cells=[root]))
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
        constraints={"fissionable": True})

    mesh = openmc.RegularMesh()
    mesh.dimension = [CORE_N, CORE_N]
    mesh.lower_left = [-half, -half]
    mesh.upper_right = [half, half]
    mf = openmc.MeshFilter(mesh)
    tal = openmc.Tally(name="asm_fission")
    tal.filters = [mf]
    tal.scores = ["fission", "nu-fission"]
    tallies = openmc.Tallies([tal])

    return openmc.Model(geometry, materials, settings, tallies), half


def main():
    ensure_project_dirs()
    if not os.environ.get("OPENMC_CROSS_SECTIONS"):
        os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
    openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)

    out = {"state_common": STATE_COMMON, "enrich_A": ENRICH_A, "enrich_B": ENRICH_B,
           "lattice_n": LATTICE_N, "core_n": CORE_N,
           "checkerboard": "type A if (aI+aJ)%2==0 else B"}

    # ---- reference single-assembly group constants (types A and B) ----
    for label, enr in (("A", ENRICH_A), ("B", ENRICH_B)):
        print(f"=== reference GC run: assembly {label} (enr={enr}%) ===", flush=True)
        rdir = RUNS_DIR / "core_colorset" / f"gc_{label}"
        rdir.mkdir(parents=True, exist_ok=True)
        model, lib, root_univ = build_assembly_model(
            STATE_COMMON["fuel_temperature_K"], enr,
            STATE_COMMON["moderator_density_g_cm3"], STATE_COMMON["moderator_temperature_K"],
            STATE_COMMON["boron_ppm"], STATE_COMMON["fuel_radius_cm"],
            STATE_COMMON["cladding_thickness_cm"], STATE_COMMON["pin_pitch_cm"],
            LATTICE_N, batches=60, inactive=15, particles=16000, seed=101)
        t0 = time.perf_counter()
        sp_path = model.run(cwd=str(rdir), output=False)
        el = time.perf_counter() - t0
        with openmc.StatePoint(sp_path) as sp:
            lib.load_from_statepoint(sp)
            gc = extract_group_constants(lib, root_univ)
            kinf = float(sp.keff.nominal_value)
            kstd = float(sp.keff.std_dev)
        out[f"gc_{label}"] = {k: float(v) for k, v in gc.items()}
        out[f"gc_{label}"]["k_inf"] = kinf
        out[f"gc_{label}"]["k_inf_std"] = kstd
        out[f"gc_{label}_elapsed_s"] = el
        print(f"  k_inf={kinf:.5f} ({kstd*1e5:.0f} pcm)  [{el:.0f}s]", flush=True)

    # ---- direct MC colorset ----
    print("=== direct MC colorset 3x3 ===", flush=True)
    rdir = RUNS_DIR / "core_colorset" / "colorset"
    rdir.mkdir(parents=True, exist_ok=True)
    model, half = build_colorset_model(batches=100, inactive=25, particles=40000, seed=202)
    t0 = time.perf_counter()
    sp_path = model.run(cwd=str(rdir), output=False)
    elapsed = time.perf_counter() - t0
    with openmc.StatePoint(sp_path) as sp:
        keff = sp.keff
        tal = sp.get_tally(name="asm_fission")
        fis = tal.get_values(scores=["fission"]).reshape(CORE_N, CORE_N)
        fis_std = tal.get_values(scores=["fission"], value="std_dev").reshape(CORE_N, CORE_N)
    power = fis / fis.mean()
    power_rel_std = fis_std / fis
    out["colorset"] = {
        "k_eff": float(keff.nominal_value),
        "k_eff_std": float(keff.std_dev),
        "power_map_mean1": power.tolist(),          # [J][I] row-major from mesh
        "power_rel_std": power_rel_std.tolist(),
        "elapsed_s": elapsed,
        "particles": 40000, "batches": 100, "inactive": 25,
        "half_width_cm": half,
    }
    print(f"  k_eff={keff.nominal_value:.5f} ({keff.std_dev*1e5:.0f} pcm)  [{elapsed:.0f}s]")
    print("  power map (mean=1):")
    for row in power:
        print("   ", "  ".join(f"{v:.4f}" for v in row))

    path = PROCESSED_DATA_DIR / "core_colorset_reference.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}", flush=True)


if __name__ == "__main__":
    main()
