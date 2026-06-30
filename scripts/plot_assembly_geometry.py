"""Render the 7x7 heterogeneous fuel assembly geometry used in run_assembly_sweep.py.

Produces a material-colored XY cross section showing the fuel pins (UO2 + Zr clad +
water moderator) and the water-filled guide tubes, with reflective boundaries.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import DEFAULT_CROSS_SECTIONS, FIGURE_DIR, ensure_project_dirs

os.environ["OPENMC_CROSS_SECTIONS"] = str(DEFAULT_CROSS_SECTIONS)
import openmc
openmc.config["cross_sections"] = str(DEFAULT_CROSS_SECTIONS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from run_assembly_sweep import build_assembly_model

# Representative mid-range assembly state (same envelope as the LHS sweep)
STATE = dict(
    fuel_temp=900.0, enrichment=3.75, mod_density=0.72, mod_temp=580.0,
    boron_ppm=900.0, fuel_radius=0.41, clad_thickness=0.057, pin_pitch=1.27,
    lattice_n=7, batches=10, inactive=5, particles=100, seed=1,
)


def main():
    ensure_project_dirs()
    model, lib, root_univ = build_assembly_model(**STATE)
    geom = model.geometry
    mats = geom.get_all_materials().values()

    # Map materials to clear colors by name. OpenMC's plotter wants 0-255 int RGB.
    FUEL_RGB = (199, 51, 46)
    CLAD_RGB = (90, 94, 102)
    WATER_RGB = (170, 210, 242)
    GT_RGB = (39, 110, 140)   # guide-tube water — distinct so the 5 tubes are visible
    colors = {}
    legend = {}
    for m in mats:
        nm = m.name.lower()
        if "fuel" in nm:
            colors[m] = FUEL_RGB; legend["UO₂ fuel (3.75% enr.)"] = tuple(c / 255 for c in FUEL_RGB)
        elif "clad" in nm:
            colors[m] = CLAD_RGB; legend["Zircaloy clad"] = tuple(c / 255 for c in CLAD_RGB)
        elif "gt" in nm:
            colors[m] = GT_RGB; legend["Guide tubes (5 water holes)"] = tuple(c / 255 for c in GT_RGB)
        elif "water" in nm:
            colors[m] = WATER_RGB; legend["Water moderator + 900 ppm boron"] = tuple(c / 255 for c in WATER_RGB)

    half = STATE["lattice_n"] * STATE["pin_pitch"] / 2.0

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    img = root_univ.plot(
        origin=(0.0, 0.0, 0.0),
        width=(2 * half, 2 * half),
        pixels=(1400, 1400),
        basis="xy",
        color_by="material",
        colors=colors,
        axes=ax,
    )
    ax.set_title(
        "OpenMC 7×7 PWR fuel assembly (reflective BC)\n"
        "heterogeneous lattice → 2-group homogenized cross sections",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")

    handles = [mpatches.Patch(color=c, label=l) for l, c in legend.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    out = FIGURE_DIR / "assembly_geometry.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
