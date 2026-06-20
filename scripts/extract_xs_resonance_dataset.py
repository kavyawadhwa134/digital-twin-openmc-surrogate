"""Extract a resonance-resolved microscopic cross-section dataset on OpenMC's NATIVE energy grid.

The previous extractor (extract_xs_dataset.py) resampled every reaction onto a fixed
1200-point log-spaced grid (~98 points/decade). U-238 capture alone has ~112k native grid
points with ~74k in the 6 eV - 10 keV resonance band; a 1200-point global grid captures
under 0.4% of that structure, so the resonance region the abstract calls "the most demanding"
was never actually represented in the training data.

This extractor instead reads each reaction's native Tabulated1D grid (xs.x, xs.y), which is
dense exactly where the resonances are. Output is a compressed .npz of float32 arrays plus a
JSON metadata sidecar (nuclide/reaction code maps), suitable for a PyTorch surrogate.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openmc.data

from project_config import (
    DEFAULT_CROSS_SECTIONS,
    GEN_IV_NUCLIDES,
    NUCLIDE_PROPERTIES,
    PROCESSED_DATA_DIR,
    REACTIONS,
    TEMPERATURES_K,
    ensure_project_dirs,
)


def library_paths(cross_sections_xml: Path) -> dict[str, Path]:
    root = ET.parse(cross_sections_xml).getroot()
    out: dict[str, Path] = {}
    for lib in root.findall("library"):
        if lib.get("type") != "neutron":
            continue
        rel = lib.get("path")
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = cross_sections_xml.parent / p
        for mat in (lib.get("materials") or "").split():
            out[mat] = p
    return out


def parse_temperature_label(label: str) -> int:
    return int(round(float(label.rstrip("K"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-sections", default=str(DEFAULT_CROSS_SECTIONS))
    parser.add_argument("--nuclides", nargs="+", default=list(GEN_IV_NUCLIDES.keys()))
    parser.add_argument("--temperatures", nargs="+", type=int, default=TEMPERATURES_K)
    parser.add_argument("--e-min", type=float, default=1.0e-5)
    parser.add_argument("--e-max", type=float, default=2.0e7)
    parser.add_argument(
        "--output", default=str(PROCESSED_DATA_DIR / "xs_resonance_dataset.npz")
    )
    args = parser.parse_args()

    ensure_project_dirs()
    xml = Path(args.cross_sections).expanduser().resolve()
    if not xml.exists():
        raise FileNotFoundError(xml)
    paths = library_paths(xml)

    nuclide_ids = {n: i for i, n in enumerate(args.nuclides)}
    reaction_ids = {name: i for i, name in enumerate(sorted(set(REACTIONS.values())))}

    cols = {k: [] for k in (
        "energy_eV", "log10_energy_eV", "temperature_K", "log10_xs_barns", "xs_barns",
        "nuclide_Z", "nuclide_A", "nuclide_N", "is_actinide", "is_fissile",
        "nuclide_id", "reaction_id",
    )}

    for nuclide in args.nuclides:
        if nuclide not in paths:
            print(f"skip {nuclide}: not in library")
            continue
        data = openmc.data.IncidentNeutron.from_hdf5(paths[nuclide])
        avail = {parse_temperature_label(t): t for t in data.temperatures}
        props = NUCLIDE_PROPERTIES.get(nuclide, {})
        for temp_k in args.temperatures:
            if temp_k not in avail:
                continue
            tlab = avail[temp_k]
            for mt, rname in REACTIONS.items():
                if mt not in data.reactions:
                    continue
                rx = data.reactions[mt]
                if tlab not in rx.xs:
                    continue
                xsfun = rx.xs[tlab]
                E = np.asarray(getattr(xsfun, "x", None), dtype=float)
                S = np.asarray(getattr(xsfun, "y", None), dtype=float)
                if E is None or S is None or E.size == 0:
                    continue
                m = (
                    np.isfinite(E) & np.isfinite(S) & (S > 0.0)
                    & (E >= args.e_min) & (E <= args.e_max)
                )
                if not np.any(m):
                    continue
                E, S = E[m], S[m]
                n = E.size
                cols["energy_eV"].append(E)
                cols["log10_energy_eV"].append(np.log10(E))
                cols["temperature_K"].append(np.full(n, temp_k, dtype=float))
                cols["log10_xs_barns"].append(np.log10(S))
                cols["xs_barns"].append(S)
                cols["nuclide_Z"].append(np.full(n, props.get("z", -1), dtype=float))
                cols["nuclide_A"].append(np.full(n, props.get("a", -1), dtype=float))
                cols["nuclide_N"].append(
                    np.full(n, props.get("a", -1) - props.get("z", -1), dtype=float)
                )
                cols["is_actinide"].append(np.full(n, props.get("is_actinide", 0), dtype=float))
                cols["is_fissile"].append(np.full(n, props.get("is_fissile", 0), dtype=float))
                cols["nuclide_id"].append(np.full(n, nuclide_ids[nuclide], dtype=np.int32))
                cols["reaction_id"].append(np.full(n, reaction_ids[rname], dtype=np.int32))
                print(f"{nuclide:>5} {temp_k:>4}K {rname:<8}: {n:>7} native pts")

    if not cols["energy_eV"]:
        raise RuntimeError("No rows extracted.")

    arrays = {}
    for k, parts in cols.items():
        arr = np.concatenate(parts)
        arrays[k] = arr.astype(np.float32) if arr.dtype == np.float64 else arr

    out = Path(args.output)
    np.savez_compressed(out, **arrays)
    meta = {
        "nuclide_ids": nuclide_ids,
        "reaction_ids": reaction_ids,
        "temperatures_K": args.temperatures,
        "n_rows": int(arrays["energy_eV"].size),
        "source_library": "OpenMC ENDF/B-VIII.0 HDF5 native grid",
        "e_min": args.e_min,
        "e_max": args.e_max,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta['n_rows']:,} rows to {out}")
    print(f"nuclides={nuclide_ids}\nreactions={reaction_ids}")


if __name__ == "__main__":
    main()
