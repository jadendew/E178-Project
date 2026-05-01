#!/usr/bin/env python3
"""
aero_sweep_to_body_rates.py
----------------------------
Sweeps (alpha, beta, deltaS, deltaD) flight conditions through the same
aero pipeline used in the 6DOF sim, then converts moments into instantaneous
body-axis angular accelerations (p_dot, q_dot, r_dot).

This uses the EXACT same stack as the sim:
  aero.MLPAeroModel  →  aero.aero_forces_moments_body  →  rigid-body EOM

No time-stepping loop — one vectorised pass per sweep point.
Result: a CSV + plots of angular accelerations across the operating envelope.

Usage
-----
    python aero_sweep_to_body_rates.py

    (run from "Controller and Sim" folder, or adjust _SIM_DIR below)

Outputs
-------
    sweep_body_rates.csv            full sweep table with aero coeffs + ang-accel
    sweep_body_rates_alpha.png      p/q/r_dot vs alpha for deltaS sweep
    sweep_body_rates_beta.png       p/q/r_dot vs beta for deltaD sweep
    sweep_body_rates_coeffs.png     Cl/Cm/Cn vs alpha at zero deflection
    sweep_body_rates_qdot_heatmap.png  q_dot as function of alpha x deltaS
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── point at the "Controller and Sim" folder so aero.py / dynamics.py resolve ─
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import aero as _aero
from dynamics import (
    Vehicle, Environment, State,
    euler_to_quat_ypr, dcm_body_to_world,
)


# =============================================================================
# 1.  VEHICLE / ENVIRONMENT  —  copied verbatim from run_sim.py
# =============================================================================

RHO  = 1.225
G    = 9.80665
WIND_WORLD = np.zeros(3)

MASS = 4.0
IXX, IYY, IZZ = 0.8, 0.12, 0.18
IXY, IXZ, IYZ = 0.0, 0.0, 0.0

COM_BODY      = np.array([-0.3, 0.0, 0.0])
AERO_REF_BODY = np.array([-0.3, 0.0, 0.0])

S_REF = 0.476500
B_REF = 2.000000
C_REF = 0.196250

# Trim speed — magnitude of run_sim.py's VEL0_WORLD = [20, 0, -3]
V_TRIM = float(np.linalg.norm([20.0, 0.0, -3.0]))

# Body rates at evaluation point (zero = static / trim linearisation)
# Override to probe non-zero gyroscopic coupling if needed.
OMEGA_TRIM = np.array([0.0, 0.0, 0.0])

# Trim attitude (matches run_sim.py defaults)
YAW_TRIM_DEG   = 0.0
PITCH_TRIM_DEG = 3.0
ROLL_TRIM_DEG  = 0.0


# =============================================================================
# 2.  MLP MODEL PATHS  —  same layout as run_sim.py
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MLP_MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "mlp_aero_state_dictV2.pt")
X_SCALER_PATH  = os.path.join(_SCRIPT_DIR, "..", "xScalerV2.pkl")
Y_SCALER_PATH  = os.path.join(_SCRIPT_DIR, "..", "yScalerV2.pkl")

# AeroConfig — same values as run_sim.py / mlpSimTest.py
AERO_CFG = _aero.AeroConfig(
    sign_cl=1.0, sign_cd=1.0, sign_cy=1.0,
    sign_cll=1.0, sign_cm=1.0, sign_cn=1.0,
    cd0_add=0.01038,
    clp=-0.3546536,
    cmq=-6.2794743,
    cnr=-0.0057922,
    use_static_extrapolation=True,
    max_alpha_deg=35.0,
    max_beta_deg=20.0,
    dcl_dalpha= 4.2721876,  dcl_dbeta=-0.0018362,
    dcd_dalpha= 0.0934618,  dcd_dbeta= 0.0005434,
    dcy_dalpha=-0.0000072,  dcy_dbeta=-0.0952724,
    dcll_dalpha=-0.0000097, dcll_dbeta=-0.0998408,
    dcm_dalpha=-1.2983957,  dcm_dbeta=-0.0027308,
    dcn_dalpha= 0.0000004,  dcn_dbeta= 0.0109517,
)

DELTAS_MAX_DEG = 20.0
DELTAD_MAX_DEG = 10.0


# =============================================================================
# 3.  SWEEP GRID  —  mirrors dataset operating envelope
# =============================================================================

ALPHA_RANGE  = np.arange(-10, 21, 2.5)   # deg
BETA_RANGE   = np.arange(-6,   7, 2)     # deg
DELTAS_RANGE = np.arange(-20, 21, 10)    # deg  symmetric elevon
DELTAD_RANGE = np.arange(-10, 11,  5)    # deg  differential elevon


# =============================================================================
# 4.  INERTIA MATRIX  (built once)
# =============================================================================

I_mat = np.array([
    [ IXX, -IXY, -IXZ],
    [-IXY,  IYY, -IYZ],
    [-IXZ, -IYZ,  IZZ],
], dtype=float)

I_inv = np.linalg.inv(I_mat)


# =============================================================================
# 5.  HELPERS
# =============================================================================

def build_trim_state(alpha_deg: float, beta_deg: float) -> State:
    """
    Quasi-steady State whose body-relative velocity corresponds to
    the requested (alpha, beta) at V_TRIM.
    Uses standard wind-axis conventions (body X fwd, Z down):
        u = V cos(α) cos(β)
        v = V sin(β)
        w = V sin(α) cos(β)
    """
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    u = V_TRIM * np.cos(a) * np.cos(b)
    v = V_TRIM * np.sin(b)
    w = V_TRIM * np.sin(a) * np.cos(b)
    vel_body = np.array([u, v, w])

    q0 = euler_to_quat_ypr(
        yaw_rad=np.deg2rad(YAW_TRIM_DEG),
        pitch_rad=np.deg2rad(PITCH_TRIM_DEG),
        roll_rad=np.deg2rad(ROLL_TRIM_DEG),
    )
    q0 /= np.linalg.norm(q0)
    C_bw = dcm_body_to_world(q0)
    vel_world = C_bw @ vel_body

    return State(
        pos_world=np.zeros(3),
        vel_world=vel_world,
        quat_body_to_world=q0,
        omega_body=OMEGA_TRIM.copy(),
    )


# =============================================================================
# 6.  MAIN SWEEP
# =============================================================================

def run_sweep(aero_model) -> pd.DataFrame:
    """
    For every grid point:
      1. Build quasi-steady State at (alpha, beta)
      2. Call aero_forces_moments_body  [exact same call as dynamics.step_rk4]
      3. ω̇ = I⁻¹ (M_body − ω × Iω)   [exact same EOM as dynamics.step_rk4]
    """
    veh = Vehicle(
        mass=MASS, I_body=I_mat,
        com_body=COM_BODY, aero_ref_body=AERO_REF_BODY,
        S_ref=S_REF, b_ref=B_REF, c_ref=C_REF,
    )
    env = Environment(rho=RHO, g=G, wind_world=WIND_WORLD)

    grid = np.array(
        np.meshgrid(ALPHA_RANGE, BETA_RANGE, DELTAS_RANGE, DELTAD_RANGE,
                    indexing='ij')
    ).reshape(4, -1).T   # (N, 4)

    rows = []
    for (alpha_deg, beta_deg, dS_deg, dD_deg) in grid:
        state = build_trim_state(float(alpha_deg), float(beta_deg))

        # ── identical to the deriv() closure inside dynamics.step_rk4 ─────
        F_b, M_b, coeffs, derived = _aero.aero_forces_moments_body(
            aero_model, veh, env, state, float(dS_deg), float(dD_deg)
        )

        w   = state.omega_body
        H   = I_mat @ w
        w_dot = I_inv @ (M_b - np.cross(w, H))   # [p_dot, q_dot, r_dot]
        # ──────────────────────────────────────────────────────────────────

        rows.append({
            # Inputs
            "alpha_deg":  alpha_deg,
            "beta_deg":   beta_deg,
            "deltaS_deg": dS_deg,
            "deltaD_deg": dD_deg,
            # Aero coefficients (after sign flips, cd0 add, rate damping)
            "CL": coeffs["CL"], "CD": coeffs["CD"], "CY": coeffs["CY"],
            "Cl": coeffs["Cl"], "Cm": coeffs["Cm"], "Cn": coeffs["Cn"],
            # Dimensional body-frame moments (N·m)
            "L_Nm": float(M_b[0]),
            "M_Nm": float(M_b[1]),
            "N_Nm": float(M_b[2]),
            # Angular accelerations (rad/s²)
            "p_dot": float(w_dot[0]),
            "q_dot": float(w_dot[1]),
            "r_dot": float(w_dot[2]),
            # Derived aero quantities
            "V":             derived["V"],
            "qbar":          derived["qbar"],
            "alpha_eff_deg": derived["alpha_tbl_deg"],
            "beta_eff_deg":  derived["beta_tbl_deg"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# 7.  PLOTS
# =============================================================================

def make_plots(df: pd.DataFrame):
    rate_cols   = [("p_dot", "royalblue"), ("q_dot", "firebrick"), ("r_dot", "seagreen")]

    # ── Fig 1: angular accel vs alpha, coloured by deltaS  (β=0, δD=0) ────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(r"Angular Accelerations vs $\alpha$   ($\beta$=0°, $\delta_D$=0°)", fontsize=11)
    sub = df[(df["beta_deg"] == 0) & (df["deltaD_deg"] == 0)].sort_values("alpha_deg")
    cmap = plt.cm.get_cmap("RdBu_r", len(DELTAS_RANGE))
    for ax, (col, _) in zip(axes, rate_cols):
        for j, ds in enumerate(sorted(sub["deltaS_deg"].unique())):
            s = sub[sub["deltaS_deg"] == ds]
            ax.plot(s["alpha_deg"], s[col], color=cmap(j),
                    marker=".", markersize=4, label=f"δS={ds:.0f}°")
        ax.set_xlabel("α (deg)"); ax.set_ylabel(f"{col} (rad/s²)")
        ax.set_title(col); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_alpha.png", dpi=150)
    print("Saved → sweep_body_rates_alpha.png")

    # ── Fig 2: angular accel vs beta, coloured by deltaD  (α=5°, δS=0) ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(r"Angular Accelerations vs $\beta$   ($\alpha$=5°, $\delta_S$=0°)", fontsize=11)
    sub = df[np.isclose(df["alpha_deg"], 5.0) & (df["deltaS_deg"] == 0)].sort_values("beta_deg")
    cmap2 = plt.cm.get_cmap("PiYG", len(DELTAD_RANGE))
    for ax, (col, _) in zip(axes, rate_cols):
        for j, dd in enumerate(sorted(sub["deltaD_deg"].unique())):
            s = sub[sub["deltaD_deg"] == dd]
            ax.plot(s["beta_deg"], s[col], color=cmap2(j),
                    marker=".", markersize=4, label=f"δD={dd:.0f}°")
        ax.set_xlabel("β (deg)"); ax.set_ylabel(f"{col} (rad/s²)")
        ax.set_title(col); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_beta.png", dpi=150)
    print("Saved → sweep_body_rates_beta.png")

    # ── Fig 3: moment coefficients vs alpha at zero deflection ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(r"Moment Coefficients vs $\alpha$   ($\beta$=$\delta_S$=$\delta_D$=0°)", fontsize=11)
    sub = df[(df["beta_deg"] == 0) & (df["deltaS_deg"] == 0) & (df["deltaD_deg"] == 0)].sort_values("alpha_deg")
    for ax, coef in zip(axes, ["Cl", "Cm", "Cn"]):
        ax.plot(sub["alpha_deg"], sub[coef], marker="o", markersize=4)
        ax.set_xlabel("α (deg)"); ax.set_ylabel(coef); ax.set_title(coef); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_coeffs.png", dpi=150)
    print("Saved → sweep_body_rates_coeffs.png")

    # ── Fig 4: q_dot heatmap  alpha × deltaS  (β=0, δD=0) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df["beta_deg"] == 0) & (df["deltaD_deg"] == 0)]
    piv = sub.pivot_table(index="alpha_deg", columns="deltaS_deg", values="q_dot")
    im  = ax.imshow(
        piv.values, aspect="auto", origin="lower",
        extent=[float(DELTAS_RANGE.min()), float(DELTAS_RANGE.max()),
                float(ALPHA_RANGE.min()),  float(ALPHA_RANGE.max())],
        cmap="RdBu_r"
    )
    fig.colorbar(im, ax=ax, label="q̇ (rad/s²)")
    ax.set_xlabel("δS (deg)"); ax.set_ylabel("α (deg)")
    ax.set_title(r"$\dot{q}$ heatmap   ($\beta$=0, $\delta_D$=0)")
    plt.tight_layout()
    plt.savefig("sweep_body_rates_qdot_heatmap.png", dpi=150)
    print("Saved → sweep_body_rates_qdot_heatmap.png")

    plt.close("all")


# =============================================================================
# 8.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Find model files (try V2, V1, no-suffix)
    print("Loading MLPAeroModel …")
    found = False
    for suffix in ["V2", "V1", ""]:
        mp = MLP_MODEL_PATH.replace("V2", suffix)
        xp = X_SCALER_PATH.replace("V2", suffix)
        yp = Y_SCALER_PATH.replace("V2", suffix)
        if all(os.path.isfile(p) for p in [mp, xp, yp]):
            print(f"  Found model files (suffix='{suffix}')")
            found = True
            break
    if not found:
        raise FileNotFoundError(
            "Cannot find MLP model files. Expected next to this script's parent:\n"
            f"  {MLP_MODEL_PATH}\n  {X_SCALER_PATH}\n  {Y_SCALER_PATH}"
        )

    aero_model = _aero.MLPAeroModel(
        model_path=mp, x_scaler_path=xp, y_scaler_path=yp,
        cfg=AERO_CFG, device="cpu",
        max_deltaS_deg=DELTAS_MAX_DEG,
        max_deltaD_deg=DELTAD_MAX_DEG,
    )

    N = len(ALPHA_RANGE) * len(BETA_RANGE) * len(DELTAS_RANGE) * len(DELTAD_RANGE)
    print(f"Running sweep: {N} conditions …")
    df = run_sweep(aero_model)
    print(f"Done.  Shape: {df.shape}")

    out_csv = "sweep_body_rates.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    print("\n── Angular acceleration ranges ──────────────────────────────")
    for col in ["p_dot", "q_dot", "r_dot"]:
        print(f"  {col:6s}:  min={df[col].min(): .4f}  max={df[col].max(): .4f}  "
              f"std={df[col].std():.4f}  rad/s²")

    print("\nGenerating plots …")
    make_plots(df)
    print("All done.")
