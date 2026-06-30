"""Merge assembly_groupconst_openmc.csv (60 cases) + assembly_groupconst_ext140.csv (140 cases)
into assembly_groupconst_200.csv, renaming case IDs so they are unique."""
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import PROCESSED_DATA_DIR

a = pd.read_csv(PROCESSED_DATA_DIR / "assembly_groupconst_openmc.csv")
b = pd.read_csv(PROCESSED_DATA_DIR / "assembly_groupconst_ext140.csv")

# Rename ext cases so IDs don't clash
b["case_id"] = [f"asm_ext_{i+1:04d}" for i in range(len(b))]

combined = pd.concat([a, b], ignore_index=True)
out = PROCESSED_DATA_DIR / "assembly_groupconst_200.csv"
combined.to_csv(out, index=False)
print(f"Combined: {len(a)} + {len(b)} = {len(combined)} cases → {out}")
print(f"k_inf range: {combined.k_inf.min():.4f} – {combined.k_inf.max():.4f}")
print(f"Mean k_inf std: {combined.k_inf_std.mean()*1e5:.0f} pcm")
