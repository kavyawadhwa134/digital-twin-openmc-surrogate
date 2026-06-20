from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_CROSS_SECTIONS = (
    PROJECT_ROOT
    / "nuclear_data"
    / "endfb-viii.0-hdf5"
    / "cross_sections.xml"
)

GEN_IV_NUCLIDES = {
    "U235": "fissile actinide",
    "U238": "fertile actinide / resonance absorption",
    "Pu239": "fast-spectrum fissile actinide",
    "O16": "oxide fuel / coolant constituent",
    "Fe56": "structural material",
    "Na23": "SFR sodium coolant",
    "C12": "HTGR graphite moderator",
    "F19": "MSR fluoride salt constituent",
    "Li7": "MSR lithium salt constituent",
}

NUCLIDE_PROPERTIES = {
    "U235": {"z": 92, "a": 235, "family": "actinide", "is_actinide": 1, "is_fissile": 1},
    "U238": {"z": 92, "a": 238, "family": "actinide", "is_actinide": 1, "is_fissile": 0},
    "Pu239": {"z": 94, "a": 239, "family": "actinide", "is_actinide": 1, "is_fissile": 1},
    "O16": {"z": 8, "a": 16, "family": "oxide", "is_actinide": 0, "is_fissile": 0},
    "Fe56": {"z": 26, "a": 56, "family": "structural", "is_actinide": 0, "is_fissile": 0},
    "Na23": {"z": 11, "a": 23, "family": "coolant", "is_actinide": 0, "is_fissile": 0},
    "C12": {"z": 6, "a": 12, "family": "moderator", "is_actinide": 0, "is_fissile": 0},
    "F19": {"z": 9, "a": 19, "family": "salt", "is_actinide": 0, "is_fissile": 0},
    "Li7": {"z": 3, "a": 7, "family": "salt", "is_actinide": 0, "is_fissile": 0},
}

REACTIONS = {
    2: "elastic",
    18: "fission",
    102: "capture",
}

TEMPERATURES_K = [294, 600, 900, 1200]


def ensure_project_dirs() -> None:
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR, RUNS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def energy_region(energy_ev: float) -> str:
    if energy_ev < 0.625:
        return "thermal"
    if energy_ev < 1.0e5:
        return "epithermal_resonance"
    return "fast"
