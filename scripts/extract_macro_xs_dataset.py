"""Build a direct macroscopic cross-section dataset from OpenMC HDF5 data.

This is the A100-friendly XS experiment. The older microscopic surrogate predicts
sigma(E, T, nuclide, reaction) and then loops over nuclides/reactions to form a
macroscopic value. That is scientifically useful, but it is a poor speed target.

Here we generate the target that a transport kernel actually accumulates:

    Sigma_r(E, T, material) = sum_i N_i sigma_{i,r}(E, T)

where N_i is an atom density in atom/b-cm and sigma is in barns, so Sigma is
reported in cm^-1. The material definitions below are compact Gen-IV-inspired
homogenized mixtures, not certified benchmark compositions.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openmc.data

from project_config import DEFAULT_CROSS_SECTIONS, GEN_IV_NUCLIDES, PROCESSED_DATA_DIR, TEMPERATURES_K, ensure_project_dirs


NUCLIDES = list(GEN_IV_NUCLIDES.keys())

REACTION_MTS = {
    "elastic": [2],
    "capture": [102],
    "fission": [18],
    "absorption": [102, 18],
    "major": [2, 102, 18],
}

MATERIAL_DEFINITIONS = {
    "lwr_uo2_pin": {
        "atom_density_atom_per_bcm": 0.0730,
        "fractions": {"U235": 0.0150, "U238": 0.3183, "O16": 0.6667},
        "note": "4.5 wt-ish UO2 pin-cell fuel proxy.",
    },
    "sfr_mox_fast": {
        "atom_density_atom_per_bcm": 0.0700,
        "fractions": {"Pu239": 0.055, "U235": 0.010, "U238": 0.260, "O16": 0.650, "Na23": 0.015, "Fe56": 0.010},
        "note": "Fast-spectrum MOX plus small coolant/structure homogenization proxy.",
    },
    "sfr_metal_fast": {
        "atom_density_atom_per_bcm": 0.0600,
        "fractions": {"Pu239": 0.070, "U235": 0.020, "U238": 0.620, "Na23": 0.230, "Fe56": 0.060},
        "note": "Metallic fast-reactor fuel/coolant/steel homogenization proxy.",
    },
    "htgr_triso_graphite": {
        "atom_density_atom_per_bcm": 0.0850,
        "fractions": {"U235": 0.0025, "U238": 0.0475, "O16": 0.1000, "C12": 0.8500},
        "note": "Dilute coated-particle fuel in graphite proxy.",
    },
    "msr_flibe_fuel_salt": {
        "atom_density_atom_per_bcm": 0.0830,
        "fractions": {"U235": 0.012, "U238": 0.110, "Li7": 0.255, "F19": 0.590, "C12": 0.033},
        "note": "Fuel-bearing fluoride salt proxy.",
    },
}


def library_paths(cross_sections_xml: Path) -> dict[str, Path]:
    root = ET.parse(cross_sections_xml).getroot()
    out: dict[str, Path] = {}
    for lib in root.findall("library"):
        if lib.get("type") != "neutron":
            continue
        rel = lib.get("path")
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = cross_sections_xml.parent / path
        for mat in (lib.get("materials") or "").split():
            out[mat] = path
    return out


def parse_temperature_label(label: str) -> int:
    return int(round(float(label.rstrip("K"))))


def normalize_fractions(fractions: dict[str, float]) -> dict[str, float]:
    total = float(sum(fractions.values()))
    if total <= 0.0:
        raise ValueError("Material atom fractions must sum to a positive value.")
    return {n: float(v) / total for n, v in fractions.items()}


def material_number_densities(material: dict) -> dict[str, float]:
    atom_density = float(material["atom_density_atom_per_bcm"])
    fractions = normalize_fractions(material["fractions"])
    return {nuclide: atom_density * frac for nuclide, frac in fractions.items()}


def load_incident_neutron(paths: dict[str, Path], nuclides: list[str]) -> dict[str, openmc.data.IncidentNeutron]:
    cache = {}
    for nuclide in nuclides:
        if nuclide in paths:
            cache[nuclide] = openmc.data.IncidentNeutron.from_hdf5(paths[nuclide])
    return cache


def macro_xs(
    cache: dict[str, openmc.data.IncidentNeutron],
    material: dict,
    reaction: str,
    energies_ev: np.ndarray,
    temperature_k: int,
) -> np.ndarray:
    if reaction not in REACTION_MTS:
        raise KeyError(f"Unknown reaction '{reaction}'. Choose from {sorted(REACTION_MTS)}.")
    sigma = np.zeros_like(energies_ev, dtype=np.float64)
    ndens = material_number_densities(material)
    for nuclide, density in ndens.items():
        if nuclide not in cache:
            continue
        data = cache[nuclide]
        temperature_map = {parse_temperature_label(t): t for t in data.temperatures}
        if temperature_k not in temperature_map:
            continue
        tlab = temperature_map[temperature_k]
        for mt in REACTION_MTS[reaction]:
            rx = data.reactions.get(mt)
            if rx is None or tlab not in rx.xs:
                continue
            sigma += density * rx.xs[tlab](energies_ev)
    return np.maximum(sigma, 1.0e-30)


def material_feature_matrix(material_name: str, n_rows: int) -> np.ndarray:
    material = MATERIAL_DEFINITIONS[material_name]
    fractions = normalize_fractions(material["fractions"])
    return np.stack(
        [np.full(n_rows, fractions.get(nuclide, 0.0), dtype=np.float32) for nuclide in NUCLIDES],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-sections", default=str(DEFAULT_CROSS_SECTIONS))
    parser.add_argument("--temperatures", nargs="+", type=int, default=TEMPERATURES_K)
    parser.add_argument("--materials", nargs="+", default=list(MATERIAL_DEFINITIONS))
    parser.add_argument("--reactions", nargs="+", default=list(REACTION_MTS))
    parser.add_argument("--n-energy", type=int, default=25_000)
    parser.add_argument("--e-min", type=float, default=1.0e-5)
    parser.add_argument("--e-max", type=float, default=2.0e7)
    parser.add_argument("--output", default=str(PROCESSED_DATA_DIR / "macro_xs_dataset.npz"))
    args = parser.parse_args()

    ensure_project_dirs()
    if args.n_energy <= 0:
        raise ValueError("--n-energy must be positive.")

    xml = Path(args.cross_sections).expanduser().resolve()
    if not xml.exists():
        raise FileNotFoundError(xml)
    paths = library_paths(xml)
    cache = load_incident_neutron(paths, NUCLIDES)

    energies = np.logspace(np.log10(args.e_min), np.log10(args.e_max), args.n_energy).astype(np.float64)
    material_ids = {name: i for i, name in enumerate(args.materials)}
    reaction_ids = {name: i for i, name in enumerate(args.reactions)}

    cols: dict[str, list[np.ndarray]] = {
        "energy_eV": [],
        "log10_energy_eV": [],
        "temperature_K": [],
        "atom_density_atom_per_bcm": [],
        "material_id": [],
        "reaction_id": [],
        "macro_xs_cm_inv": [],
        "log10_macro_xs_cm_inv": [],
    }
    comp_cols = {f"frac_{nuclide}": [] for nuclide in NUCLIDES}

    for material_name in args.materials:
        if material_name not in MATERIAL_DEFINITIONS:
            raise KeyError(f"Unknown material '{material_name}'. Choose from {sorted(MATERIAL_DEFINITIONS)}.")
        material = MATERIAL_DEFINITIONS[material_name]
        atom_density = float(material["atom_density_atom_per_bcm"])
        comps = material_feature_matrix(material_name, args.n_energy)
        for temp_k in args.temperatures:
            for reaction in args.reactions:
                values = macro_xs(cache, material, reaction, energies, temp_k)
                n = values.size
                cols["energy_eV"].append(energies)
                cols["log10_energy_eV"].append(np.log10(energies))
                cols["temperature_K"].append(np.full(n, temp_k, dtype=np.float32))
                cols["atom_density_atom_per_bcm"].append(np.full(n, atom_density, dtype=np.float32))
                cols["material_id"].append(np.full(n, material_ids[material_name], dtype=np.int32))
                cols["reaction_id"].append(np.full(n, reaction_ids[reaction], dtype=np.int32))
                cols["macro_xs_cm_inv"].append(values)
                cols["log10_macro_xs_cm_inv"].append(np.log10(values))
                for i, nuclide in enumerate(NUCLIDES):
                    comp_cols[f"frac_{nuclide}"].append(comps[:, i])
                print(f"{material_name:<20} {temp_k:>4}K {reaction:<10}: {n:>7} rows")

    arrays = {}
    for key, parts in {**cols, **comp_cols}.items():
        arr = np.concatenate(parts)
        arrays[key] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "source_library": str(xml),
        "n_rows": int(arrays["energy_eV"].size),
        "nuclides": NUCLIDES,
        "materials": MATERIAL_DEFINITIONS,
        "material_ids": material_ids,
        "reaction_ids": reaction_ids,
        "reaction_mts": REACTION_MTS,
        "temperatures_K": args.temperatures,
        "units": {"macro_xs_cm_inv": "cm^-1", "atom_density_atom_per_bcm": "atom/b-cm"},
        "note": "Homogenized Gen-IV-inspired material proxies for ML benchmarking, not licensed benchmark compositions.",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta['n_rows']:,} rows to {out}")


if __name__ == "__main__":
    main()
