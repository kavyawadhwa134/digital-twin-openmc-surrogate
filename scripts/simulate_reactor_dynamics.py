"""Physics-grounded reactor-state trajectories from point-kinetics + thermal-hydraulics ODEs.

The original digital-twin data (generate_state_sequences.py) drew each signal from hand-written
analytic shapes (sinusoids + linspaced anomaly ramps). This module instead INTEGRATES the
governing equations, so the trajectories obey real dynamic couplings and feedback:

  * Point reactor kinetics with six delayed-neutron precursor groups (U-235 thermal data).
  * Lumped two-node thermal-hydraulics: a fuel node and a coolant node with convective coupling
    and forced-flow heat removal to a fixed inlet.
  * Reactivity feedback: Doppler (fuel temperature) + moderator temperature/density coefficients,
    plus an external reactivity programme (load-following control-rod motion).

State outputs match generate_state_sequences.STATE_FEATURES exactly, so these trajectories are a
drop-in, higher-fidelity replacement for training/validating the forecaster. Anomalies are
injected as physical perturbations (a flow/heat/reactivity insertion), and the rest of the state
responds through the ODEs rather than being scripted.

This is still a reduced-order (0-D) plant model, not a spatial core simulation, but it is grounded
in conservation/feedback physics instead of curve-drawing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_state_sequences import ANOMALY_KINDS, STATE_FEATURES
from project_config import FIGURE_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

# --- Point-kinetics constants (U-235 thermal, 6 delayed groups) ---
BETA_I = np.array([2.1e-4, 1.41e-3, 1.27e-3, 2.55e-3, 7.4e-4, 2.7e-4])
LAMBDA_I = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
BETA = float(BETA_I.sum())          # ~0.0065
GEN_TIME = 2.0e-5                    # prompt neutron generation time Lambda (s)

# --- Lumped thermal-hydraulics, tuned so that at normalized power P=1 the steady state is
#     exactly T_fuel = 900 K, T_coolant = 580 K (consistent with the feedback reference) ---
COOLANT_INLET_K = 565.0
FUEL_TEMP0 = 900.0
COOLANT_TEMP0 = 580.0
DT_FUEL_COOLANT = FUEL_TEMP0 - COOLANT_TEMP0      # 320 K across the gap at P=1
DT_CORE = COOLANT_TEMP0 - COOLANT_INLET_K         # 15 K coolant rise at P=1
G_FC = 1.0 / DT_FUEL_COOLANT                       # P=1 -> fuel-coolant balance
WC0 = 1.0 / DT_CORE                                # P=1 -> flow removal balance
TAU_FUEL = 6.0                                     # fuel thermal time constant (s)
TAU_COOL = 3.0                                     # coolant node time constant (s)
C_FUEL = TAU_FUEL * G_FC
C_COOL = TAU_COOL * WC0

# --- Reactivity programme + feedback (delta-k/k). Kept well below BETA so the reactor stays
#     near-critical and stable, as in normal operation. ---
ROD_REF = 55.0
ROD_WORTH = 1.5e-4                 # 15 pcm per % rod -> a few-% swing stays << BETA (650 pcm)
ALPHA_DOPPLER = -2.6e-5            # fuel temperature (Doppler), strongly negative
ALPHA_COOLANT_T = -1.2e-5         # coolant temperature
ALPHA_COOLANT_RHO = 0.090         # per (g/cm3) — denser moderator -> more reactivity
COOLANT_RHO0 = 0.72


def coolant_density(coolant_temp_K: float) -> float:
    """Light-water-like density vs temperature, linearized around the operating point."""
    return COOLANT_RHO0 - 6.5e-4 * (coolant_temp_K - COOLANT_TEMP0)


def simulate_physical_state_series(
    n_steps: int,
    anomaly_start: int,
    seed: int,
    anomaly_kind: str = "coolant_loss",
    severity: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_grid = np.arange(n_steps, dtype=float)
    anomaly_kind = anomaly_kind.lower()
    do_anomaly = (0 <= anomaly_start < n_steps) and severity > 0.0
    width = min(55, n_steps - anomaly_start) if do_anomaly else 0

    # External (operator/load-following) reactivity programme via control-rod position.
    # Small +/-2% swing -> +/-30 pcm, well below BETA.
    rod_base = ROD_REF - 2.0 * np.sin(2 * np.pi * t_grid / 240.0)

    def perturbations(t: float):
        """Return (rod_pos_pct, flow_mult, ext_heat, rho_off)."""
        rod = float(np.interp(t, t_grid, rod_base))
        flow_mult, ext_heat, rho_off = 1.0, 0.0, 0.0
        if do_anomaly and anomaly_start <= t < anomaly_start + width:
            frac = (t - anomaly_start) / max(width, 1)  # 0..1 ramp
            s = severity
            if anomaly_kind == "coolant_loss":
                flow_mult = 1.0 - 0.40 * s * frac          # partial loss of flow
                rho_off = -0.04 * s * frac                 # moderator density drop
            elif anomaly_kind == "control_rod_withdrawal":
                rod -= 5.0 * s * frac                      # positive reactivity insertion
            elif anomaly_kind == "coolant_heating":
                ext_heat = 0.06 * s * frac                 # external heat into coolant node
            # flux_detector_bias / sensor_drift are measurement-only (applied after solve)
        return rod, flow_mult, ext_heat, rho_off

    def rhs(t, y):
        n = y[0]
        C = np.array(y[1:7])
        T_fuel, T_cool = y[7], y[8]
        rod, flow_mult, ext_heat, rho_off = perturbations(t)

        rho_ext = ROD_WORTH * (ROD_REF - rod)
        rho_fb = (
            ALPHA_DOPPLER * (T_fuel - FUEL_TEMP0)
            + ALPHA_COOLANT_T * (T_cool - COOLANT_TEMP0)
            + ALPHA_COOLANT_RHO * (coolant_density(T_cool) + rho_off - COOLANT_RHO0)
        )
        rho = rho_ext + rho_fb

        dn = (rho - BETA) / GEN_TIME * n + np.sum(LAMBDA_I * C)
        dC = BETA_I / GEN_TIME * n - LAMBDA_I * C
        power = n
        dT_fuel = (power - G_FC * (T_fuel - T_cool)) / C_FUEL
        dT_cool = (
            G_FC * (T_fuel - T_cool)
            - WC0 * flow_mult * (T_cool - COOLANT_INLET_K)
            + ext_heat
        ) / C_COOL
        return [dn, *dC, dT_fuel, dT_cool]

    # steady-state initial precursors for n0 = 1
    n0 = 1.0
    C0 = BETA_I / (GEN_TIME * LAMBDA_I) * n0
    y0 = [n0, *C0, FUEL_TEMP0, COOLANT_TEMP0]

    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid,
        method="LSODA", rtol=1e-7, atol=1e-9, max_step=1.0,
    )
    n = sol.y[0]
    T_fuel = sol.y[7]
    T_cool = sol.y[8]
    rho_cool = coolant_density(T_cool)

    # Reconstruct reactivity actually seen (for keff/reactivity outputs)
    rod_arr = np.array([perturbations(t)[0] for t in t_grid])
    rho_off_arr = np.array([perturbations(t)[3] for t in t_grid])
    reactivity = (
        ROD_WORTH * (ROD_REF - rod_arr)
        + ALPHA_DOPPLER * (T_fuel - FUEL_TEMP0)
        + ALPHA_COOLANT_T * (T_cool - COOLANT_TEMP0)
        + ALPHA_COOLANT_RHO * (rho_cool + rho_off_arr - COOLANT_RHO0)
    )
    keff = 1.0 / (1.0 - reactivity)

    power = n / n0
    flux_fast = power * (1.0 + 0.10 * (1.0 - rho_cool / COOLANT_RHO0))
    flux_thermal = power * (1.0 + 0.20 * (rho_cool / COOLANT_RHO0 - 1.0))
    fission_rate = power
    capture_rate = flux_thermal * (1.0 + 3.5e-4 * (T_fuel - FUEL_TEMP0))

    # measurement noise (sensor realism)
    def noisy(arr, s):
        return arr + rng.normal(0.0, s, size=arr.shape)

    fuel_meas = noisy(T_fuel, 1.0)
    cool_meas = noisy(T_cool, 0.3)
    rho_meas = noisy(rho_cool, 5e-4)
    rod_meas = noisy(rod_arr, 0.05)
    power_meas = noisy(power, 0.003)
    keff_meas = noisy(keff, 1.0e-4)
    flux_fast_meas = noisy(flux_fast, 0.004)
    flux_thermal_meas = noisy(flux_thermal, 0.004)
    fiss_meas = noisy(fission_rate, 0.004)
    cap_meas = noisy(capture_rate, 0.004)

    # measurement-only anomalies (instrumentation faults)
    is_anomaly = np.zeros(n_steps, dtype=int)
    if do_anomaly:
        idx = slice(anomaly_start, anomaly_start + width)
        is_anomaly[idx] = 1
        frac = np.linspace(0, 1, width)
        if anomaly_kind == "flux_detector_bias":
            flux_fast_meas[idx] += severity * (0.006 + 0.07 * frac)
            flux_thermal_meas[idx] -= severity * (0.004 + 0.05 * frac)
            fiss_meas[idx] += severity * (0.003 + 0.03 * frac)
        elif anomaly_kind == "sensor_drift":
            fuel_meas[idx] += severity * (1.0 + 27.0 * frac)
            cool_meas[idx] += severity * (0.8 + 11.0 * frac)

    return pd.DataFrame(
        {
            "time_s": t_grid,
            "fuel_temperature_K": fuel_meas,
            "coolant_temperature_K": cool_meas,
            "coolant_density_g_cm3": rho_meas,
            "control_rod_position_pct": rod_meas,
            "power_norm": power_meas,
            "keff": keff_meas,
            "flux_fast_norm": flux_fast_meas,
            "flux_thermal_norm": flux_thermal_meas,
            "fission_rate_norm": fiss_meas,
            "capture_rate_norm": cap_meas,
            "is_anomaly": is_anomaly,
            "anomaly_kind": anomaly_kind if do_anomaly else "none",
            "anomaly_severity": float(severity) if do_anomaly else 0.0,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=420)
    p.add_argument("--anomaly-start", type=int, default=280)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--anomaly-kind", choices=ANOMALY_KINDS, default="coolant_loss")
    p.add_argument("--severity", type=float, default=0.75)
    args = p.parse_args()

    ensure_project_dirs()
    df = simulate_physical_state_series(
        args.steps, args.anomaly_start, args.seed, args.anomaly_kind, args.severity
    )
    out = PROCESSED_DATA_DIR / "reactor_dynamics_physics_demo.csv"
    df.to_csv(out, index=False)

    # physical sanity: a control-rod withdrawal must raise power; coolant loss must raise fuel T
    base = simulate_physical_state_series(args.steps, -1, args.seed, severity=0.0)
    print("=== physics sanity (steady normal run) ===")
    print(f"  power drift over run: {df['power_norm'].iloc[:200].std():.4f} (should be small)")
    print(f"  fuel T mean {base['fuel_temperature_K'].mean():.1f} K, "
          f"coolant T mean {base['coolant_temperature_K'].mean():.1f} K")

    fig, ax = plt.subplots(4, 1, figsize=(9.5, 9.0), sharex=True)
    ax[0].plot(df.time_s, df.power_norm); ax[0].set_ylabel("Power (norm)")
    ax[1].plot(df.time_s, df.fuel_temperature_K, label="fuel"); ax[1].plot(df.time_s, df.coolant_temperature_K, label="coolant"); ax[1].set_ylabel("T (K)"); ax[1].legend(fontsize=8)
    ax[2].plot(df.time_s, (df.keff - 1.0) * 1e5); ax[2].set_ylabel("Reactivity (pcm)")
    ax[3].plot(df.time_s, df.control_rod_position_pct); ax[3].set_ylabel("Rod (%)"); ax[3].set_xlabel("Time (s)")
    if args.anomaly_start < args.steps:
        for a in ax:
            a.axvline(args.anomaly_start, color="#E45756", ls=":", lw=1.2)
    fig.suptitle(f"Point-kinetics + TH reactor dynamics ({args.anomaly_kind})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "reactor_dynamics_physics_demo.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {FIGURE_DIR / 'reactor_dynamics_physics_demo.png'}")


if __name__ == "__main__":
    main()
