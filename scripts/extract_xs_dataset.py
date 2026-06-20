from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openmc.data
import pandas as pd

from project_config import (
    DEFAULT_CROSS_SECTIONS,
    GEN_IV_NUCLIDES,
    NUCLIDE_PROPERTIES,
    PROCESSED_DATA_DIR,
    REACTIONS,
    TEMPERATURES_K,
    energy_region,
    ensure_project_dirs,
)


def resolve_cross_sections(path_arg: str | None) -> Path:
    candidate = path_arg or os.environ.get("OPENMC_CROSS_SECTIONS") or str(DEFAULT_CROSS_SECTIONS)
    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cross_sections.xml not found: {path}")
    return path


def library_paths(cross_sections_xml: Path) -> dict[str, Path]:
    root = ET.parse(cross_sections_xml).getroot()
    out: dict[str, Path] = {}
    for lib in root.findall("library"):
        if lib.get("type") != "neutron":
            continue
        materials = (lib.get("materials") or "").split()
        rel_path = lib.get("path")
        if not rel_path:
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = cross_sections_xml.parent / path
        for material in materials:
            out[material] = path
    return out


def parse_temperature_label(label: str) -> int:
    return int(round(float(label.rstrip("K"))))


def extract_rows(
    cross_sections_xml: Path,
    nuclides: list[str],
    energy_points: int,
    temperatures: list[int],
) -> pd.DataFrame:
    paths = library_paths(cross_sections_xml)
    energies = np.logspace(-5, np.log10(2.0e7), energy_points)
    rows: list[pd.DataFrame] = []

    for nuclide in nuclides:
        if nuclide not in paths:
            print(f"Skipping {nuclide}: not found in cross-section library")
            continue
        data = openmc.data.IncidentNeutron.from_hdf5(paths[nuclide])
        available_temps = {parse_temperature_label(t): t for t in data.temperatures}

        for temp_k in temperatures:
            if temp_k not in available_temps:
                print(f"Skipping {nuclide} at {temp_k}K: temperature missing")
                continue
            temp_label = available_temps[temp_k]

            for mt, reaction_name in REACTIONS.items():
                if mt not in data.reactions:
                    continue
                rx = data.reactions[mt]
                if temp_label not in rx.xs:
                    continue
                xs = np.asarray(rx.xs[temp_label](energies), dtype=float)
                mask = np.isfinite(xs) & (xs > 0.0)
                if not np.any(mask):
                    continue
                subset = pd.DataFrame(
                    {
                        "energy_eV": energies[mask],
                        "log10_energy_eV": np.log10(energies[mask]),
                        "temperature_K": temp_k,
                        "nuclide": nuclide,
                        "nuclide_Z": NUCLIDE_PROPERTIES.get(nuclide, {}).get("z", -1),
                        "nuclide_A": NUCLIDE_PROPERTIES.get(nuclide, {}).get("a", -1),
                        "nuclide_N": NUCLIDE_PROPERTIES.get(nuclide, {}).get("a", -1)
                        - NUCLIDE_PROPERTIES.get(nuclide, {}).get("z", -1),
                        "nuclide_family": NUCLIDE_PROPERTIES.get(nuclide, {}).get(
                            "family", "other"
                        ),
                        "nuclide_is_actinide": NUCLIDE_PROPERTIES.get(nuclide, {}).get(
                            "is_actinide", 0
                        ),
                        "nuclide_is_fissile": NUCLIDE_PROPERTIES.get(nuclide, {}).get(
                            "is_fissile", 0
                        ),
                        "geniv_relevance": GEN_IV_NUCLIDES.get(nuclide, "other"),
                        "reaction_mt": mt,
                        "reaction_type": reaction_name,
                        "xs_barns": xs[mask],
                        "log10_xs_barns": np.log10(xs[mask]),
                        "energy_region": [energy_region(float(e)) for e in energies[mask]],
                        "source_library": "OpenMC ENDF/B-VIII.0 HDF5",
                    }
                )
                rows.append(subset)
                print(
                    f"{nuclide:>5} {temp_k:>4}K {reaction_name:<8}: "
                    f"{len(subset):>5} rows"
                )

    if not rows:
        raise RuntimeError("No rows extracted. Check nuclides, reactions, and cross-section path.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a supervised ML cross-section dataset from OpenMC HDF5 data."
    )
    parser.add_argument("--cross-sections", default=None, help="Path to cross_sections.xml")
    parser.add_argument(
        "--nuclides",
        nargs="+",
        default=list(GEN_IV_NUCLIDES.keys()),
        help="Nuclides to extract, e.g. U235 U238 Pu239 Na23",
    )
    parser.add_argument("--energy-points", type=int, default=600)
    parser.add_argument("--temperatures", nargs="+", type=int, default=TEMPERATURES_K)
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "xs_dataset.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    cross_sections = resolve_cross_sections(args.cross_sections)
    df = extract_rows(cross_sections, args.nuclides, args.energy_points, args.temperatures)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = (
        df.groupby(["nuclide", "reaction_type", "temperature_K"])
        .size()
        .rename("rows")
        .reset_index()
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {len(df):,} rows to {output}")
    print(f"Wrote extraction summary to {summary_path}")


if __name__ == "__main__":
    main()
