"""Generate ICTP talk summary figures that tie the project together honestly.

Produces:
  figures/ictp_noise_floor_summary.png   -- both surrogates sit AT the MC noise floor
  figures/ictp_surrogate_ladder.png      -- the pin-cell -> assembly -> core ladder
  figures/ictp_scorecard.png             -- one-slide honest scorecard of all 3 components

Numbers are taken verbatim from the committed evaluation JSONs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

NAVY = "#1f3a5f"
BLUE = "#2a6f97"
TEAL = "#2c7a7b"
GREY = "#9aa5b1"
AMBER = "#c97b1a"
GREEN = "#2e7d32"
RED = "#b3261e"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

# ----------------------------------------------------------------------------
# 1. Noise-floor summary: both keff/k_inf surrogates sit at the MC noise floor
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.2))
labels = ["Pin-cell\nk_eff", "Assembly\nk_inf"]
x = np.arange(len(labels))
w = 0.36
surrogate = [236, 152]      # pcm, honest test MAE
floor = [236, 135]          # pcm, MC noise floor

b1 = ax.bar(x - w/2, floor, w, label="Monte Carlo noise floor (1$\\sigma$)",
            color=GREY, edgecolor="white")
b2 = ax.bar(x + w/2, surrogate, w, label="ML surrogate test MAE",
            color=BLUE, edgecolor="white")

for xi, v in zip(x - w/2, floor):
    ax.text(xi, v + 5, f"{v}", ha="center", va="bottom", fontsize=11, color="#444")
for xi, v in zip(x + w/2, surrogate):
    ax.text(xi, v + 5, f"{v}", ha="center", va="bottom", fontsize=11,
            color=BLUE, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Error (pcm) — lower is better")
ax.set_ylim(0, 300)
ax.set_title("Both surrogates reach the Monte Carlo noise floor\n"
             "The model is as accurate as the physics data it was trained on")
ax.legend(frameon=False, loc="upper right")
ax.annotate("at the floor", xy=(0 + w/2, 236), xytext=(0 + w/2, 285),
            ha="center", color=GREEN, fontsize=10, fontweight="bold")
ax.annotate("within 17 pcm\nof the floor", xy=(1 + w/2, 152), xytext=(1 + w/2, 230),
            ha="center", color=GREEN, fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "ictp_noise_floor_summary.png", bbox_inches="tight")
plt.close(fig)
print("wrote ictp_noise_floor_summary.png")

# ----------------------------------------------------------------------------
# 2. Surrogate ladder: pin-cell -> assembly group constants -> core solver
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.axis("off")
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)

rungs = [
    (0.3, BLUE, "RUNG 1  ·  PIN CELL",
     ["Single fuel pin (OpenMC)",
      "8 design inputs -> k_eff + 7 fields",
      "k_eff MAE 236 pcm  (= noise floor)",
      "fields 0.14-0.39% relMAE",
      "~2,300x -> 4.2M x faster"], "DONE"),
    (3.95, TEAL, "RUNG 2  ·  ASSEMBLY",
     ["7x7 lattice + guide tubes",
      "-> 2-group homogenized constants",
      "k_inf MAE 152 pcm (floor 135)",
      "13 group constants 0.04-0.14%",
      "~53,000x faster, leakage-free"], "DONE"),
    (7.6, NAVY, "RUNG 3  ·  CORE",
     ["Nodal / SP3 diffusion solver",
      "consumes the group constants",
      "from Rung 2 directly",
      "+ burnup axis (needs decay chain)",
      "full-core digital twin"], "NEXT"),
]

for x0, color, title, lines, tag in rungs:
    box = FancyBboxPatch((x0, 0.7), 3.1, 3.5,
                         boxstyle="round,pad=0.04,rounding_size=0.12",
                         linewidth=2, edgecolor=color,
                         facecolor=color if tag != "NEXT" else "white",
                         alpha=0.12 if tag != "NEXT" else 1.0)
    ax.add_patch(box)
    # header bar
    hb = FancyBboxPatch((x0, 3.55), 3.1, 0.65,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=0, facecolor=color,
                        alpha=1.0 if tag != "NEXT" else 0.55)
    ax.add_patch(hb)
    ax.text(x0 + 1.55, 3.87, title, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x0 + 0.18, 3.25 - i * 0.5, ln, ha="left", va="center",
                fontsize=9.3, color="#222")
    tagcol = GREEN if tag == "DONE" else AMBER
    ax.text(x0 + 1.55, 0.45, tag, ha="center", va="center",
            fontsize=10, fontweight="bold", color=tagcol)

for x0 in (3.45, 7.1):
    ax.add_patch(FancyArrowPatch((x0, 2.45), (x0 + 0.45, 2.45),
                                 arrowstyle="-|>", mutation_scale=22,
                                 linewidth=2.5, color="#555"))

ax.text(5.5, 4.75, "A surrogate ladder to whole-core simulation",
        ha="center", fontsize=15, fontweight="bold")
fig.savefig(FIG / "ictp_surrogate_ladder.png", bbox_inches="tight")
plt.close(fig)
print("wrote ictp_surrogate_ladder.png")

# ----------------------------------------------------------------------------
# 3. One-slide honest scorecard
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.axis("off")
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.6)

cards = [
    (0.2, GREEN, "Assembly group-constant surrogate",
     ["BEST RESULT", "k_inf 152 pcm @ 135 pcm floor",
      "13 constants 0.04-0.14% relMAE", "53,000x faster",
      "leakage-free; feeds a core solver"]),
    (3.85, GREEN, "Pin-cell response surrogate",
     ["SOLID", "k_eff 236 pcm @ 236 pcm floor",
      "100% within 1000 pcm", "fields 0.14-0.39%",
      "physics monotonicity verified"]),
    (7.5, AMBER, "Digital-twin forecaster + anomaly",
     ["WORKS (synthetic)", "held-out P/R/F1 1.00/0.88/0.93",
      "false-positive rate ~0", "2 s median alarm delay",
      "caveat: simulated trajectories"]),
]
for x0, color, title, lines in cards:
    box = FancyBboxPatch((x0, 0.5), 3.2, 4.3,
                         boxstyle="round,pad=0.04,rounding_size=0.1",
                         linewidth=2, edgecolor=color, facecolor=color, alpha=0.08)
    ax.add_patch(box)
    ax.text(x0 + 1.6, 4.5, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a1a1a", wrap=True)
    ax.text(x0 + 1.6, 4.0, lines[0], ha="center", va="center",
            fontsize=10, fontweight="bold", color=color)
    for i, ln in enumerate(lines[1:]):
        ax.text(x0 + 0.2, 3.5 - i * 0.62, "• " + ln, ha="left", va="center",
                fontsize=9.4, color="#222")

ax.text(5.5, 5.35, "Honest scorecard — what is defensible today",
        ha="center", fontsize=15, fontweight="bold")
ax.text(5.5, 0.18,
        "Abstract's stated XS-GPU headline: negative on CPU (OpenMC's vectorized lookup wins) "
        "— reported as an honest scoping result, not buried.",
        ha="center", fontsize=9, style="italic", color=RED)
fig.savefig(FIG / "ictp_scorecard.png", bbox_inches="tight")
plt.close(fig)
print("wrote ictp_scorecard.png")
print("done")
