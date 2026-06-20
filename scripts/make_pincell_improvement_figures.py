from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_DIR


ROOT = Path("/Users/kavyawadhwa/Documents/Digital Twin")


def load_metrics(name: str) -> dict:
    return json.loads((MODEL_DIR / name).read_text())


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    high = pd.read_csv(PROCESSED_DATA_DIR / "pincell_lhs120_highstat_openmc.csv")
    broad = pd.read_csv(PROCESSED_DATA_DIR / "pincell_lhs500_openmc.csv")
    high_cmp = pd.read_csv(
        PROCESSED_DATA_DIR / "pincell_lhs120_highstat_engineered_surrogate_comparison.csv"
    )
    broad_cmp = pd.read_csv(
        PROCESSED_DATA_DIR / "pincell_lhs500_engineered_surrogate_comparison.csv"
    )
    high_metrics = load_metrics("pincell_lhs120_highstat_engineered_surrogate_metrics.json")
    broad_metrics = load_metrics("pincell_lhs500_engineered_surrogate_metrics.json")

    high_random = high_metrics["validation_report"]["random_interpolation"]["keff"]
    broad_random = broad_metrics["validation_report"]["random_interpolation"]["keff"]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)

    axes[0, 0].hist(high["keff_std"] * 1.0e5, bins=16, alpha=0.78, label="120 high-stat")
    axes[0, 0].hist(broad["keff_std"] * 1.0e5, bins=16, alpha=0.62, label="500 low-stat")
    axes[0, 0].set_title("OpenMC label uncertainty")
    axes[0, 0].set_xlabel("keff standard deviation [pcm]")
    axes[0, 0].set_ylabel("Cases")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].boxplot(
        [
            np.abs(high_cmp["keff_error_pcm"]),
            np.abs(broad_cmp["keff_error_pcm"]),
        ],
        tick_labels=["120 high-stat", "500 low-stat"],
        showfliers=True,
    )
    axes[0, 1].set_title("Surrogate keff absolute error")
    axes[0, 1].set_ylabel("|prediction error| [pcm]")

    groups = ["MAE", "RMSE", "Max"]
    high_vals = [
        high_random["mae_pcm"],
        high_random["rmse_pcm"],
        high_random["max_abs_error_pcm"],
    ]
    broad_vals = [
        broad_random["mae_pcm"],
        broad_random["rmse_pcm"],
        broad_random["max_abs_error_pcm"],
    ]
    x = np.arange(len(groups))
    width = 0.35
    axes[1, 0].bar(x - width / 2, high_vals, width, label="120 high-stat")
    axes[1, 0].bar(x + width / 2, broad_vals, width, label="500 low-stat")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(groups)
    axes[1, 0].set_title("Random interpolation keff error")
    axes[1, 0].set_ylabel("pcm")
    axes[1, 0].legend(frameon=False)

    groups = ["within 500 pcm", "within 1000 pcm"]
    high_vals = [
        high_random["percent_within_500_pcm"],
        high_random["percent_within_1000_pcm"],
    ]
    broad_vals = [
        broad_random["percent_within_500_pcm"],
        broad_random["percent_within_1000_pcm"],
    ]
    x = np.arange(len(groups))
    axes[1, 1].bar(x - width / 2, high_vals, width, label="120 high-stat")
    axes[1, 1].bar(x + width / 2, broad_vals, width, label="500 low-stat")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(groups)
    axes[1, 1].set_ylim(0, 108)
    axes[1, 1].set_title("Tolerance-based accuracy")
    axes[1, 1].set_ylabel("test cases [%]")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "Pin-cell surrogate improvement: high-stat labels beat larger noisy labels",
        fontsize=14,
    )
    output = FIGURE_DIR / "pincell_surrogate_highstat_vs_largedata.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
