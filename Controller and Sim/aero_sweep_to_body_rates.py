# Sweeps (alpha, beta, deltaS, deltaD) flight conditions through the same
# aero pipeline used in the 6DOF sim, then converts moments into instantaneous
# body-axis angular accelerations (p_dot, q_dot, r_dot).
 
# Replaces time-stepping with flight condition variable sweep
# Output: CSV of body dynamics and coefficients derived from flight dynamics variables (alpha/beta/deltaS/deltaD)

#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import aero as _aero
from dynamics import Vehicle, Environment, State, euler_to_quat_ypr, dcm_body_to_world


# --- vehicle / environment (in sync with run_sim.py) ---

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

V_TRIM         = float(np.linalg.norm([20.0, 0.0, -3.0]))
OMEGA_TRIM     = np.array([0.0, 0.0, 0.0])
YAW_TRIM_DEG   = 0.0
PITCH_TRIM_DEG = 3.0
ROLL_TRIM_DEG  = 0.0


# --- model paths ---

_HERE = os.path.dirname(os.path.abspath(__file__))

MLP_MODEL_PATH = os.path.join(_HERE, "..", "mlp_aero_state_dictV2.pt")
X_SCALER_PATH  = os.path.join(_HERE, "..", "xScalerV2.pkl")
Y_SCALER_PATH  = os.path.join(_HERE, "..", "yScalerV2.pkl")

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


# --- sweep grid ---

ALPHA_RANGE  = np.arange(-10, 21, 2.5)
BETA_RANGE   = np.arange(-6,   7, 2)
DELTAS_RANGE = np.arange(-20, 21, 10)
DELTAD_RANGE = np.arange(-10, 11,  5)


# --- inertia ---

I_mat = np.array([
    [ IXX, -IXY, -IXZ],
    [-IXY,  IYY, -IYZ],
    [-IXZ, -IYZ,  IZZ],
], dtype=float)

I_inv = np.linalg.inv(I_mat)


def build_trim_state(alpha_deg, beta_deg):
    # body-frame velocity for requested AoA / sideslip at fixed airspeed
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    vel_body = np.array([
        V_TRIM * np.cos(a) * np.cos(b),
        V_TRIM * np.sin(b),
        V_TRIM * np.sin(a) * np.cos(b),
    ])
    q0 = euler_to_quat_ypr(
        yaw_rad=np.deg2rad(YAW_TRIM_DEG),
        pitch_rad=np.deg2rad(PITCH_TRIM_DEG),
        roll_rad=np.deg2rad(ROLL_TRIM_DEG),
    )
    q0 /= np.linalg.norm(q0)
    vel_world = dcm_body_to_world(q0) @ vel_body
    return State(
        pos_world=np.zeros(3),
        vel_world=vel_world,
        quat_body_to_world=q0,
        omega_body=OMEGA_TRIM.copy(),
    )


def run_sweep(aero_model):
    veh = Vehicle(
        mass=MASS, I_body=I_mat,
        com_body=COM_BODY, aero_ref_body=AERO_REF_BODY,
        S_ref=S_REF, b_ref=B_REF, c_ref=C_REF,
    )
    env = Environment(rho=RHO, g=G, wind_world=WIND_WORLD)

    grid = np.array(
        np.meshgrid(ALPHA_RANGE, BETA_RANGE, DELTAS_RANGE, DELTAD_RANGE, indexing='ij')
    ).reshape(4, -1).T

    rows = []
    for (alpha_deg, beta_deg, dS_deg, dD_deg) in grid:
        state = build_trim_state(float(alpha_deg), float(beta_deg))

        F_b, M_b, coeffs, derived = _aero.aero_forces_moments_body(
            aero_model, veh, env, state, float(dS_deg), float(dD_deg)
        )

        w = state.omega_body
        w_dot = I_inv @ (M_b - np.cross(w, I_mat @ w))

        rows.append({
            "alpha_deg":     alpha_deg,
            "beta_deg":      beta_deg,
            "deltaS_deg":    dS_deg,
            "deltaD_deg":    dD_deg,
            "CL": coeffs["CL"], "CD": coeffs["CD"], "CY": coeffs["CY"],
            "Cl": coeffs["Cl"], "Cm": coeffs["Cm"], "Cn": coeffs["Cn"],
            "L_Nm":          float(M_b[0]),
            "M_Nm":          float(M_b[1]),
            "N_Nm":          float(M_b[2]),
            "p_dot":         float(w_dot[0]),
            "q_dot":         float(w_dot[1]),
            "r_dot":         float(w_dot[2]),
            "V":             derived["V"],
            "qbar":          derived["qbar"],
            "alpha_eff_deg": derived["alpha_tbl_deg"],
            "beta_eff_deg":  derived["beta_tbl_deg"],
        })

    return pd.DataFrame(rows)


def make_plots(df):
    rate_cols = [("p_dot", "royalblue"), ("q_dot", "firebrick"), ("r_dot", "seagreen")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(r"Angular Accelerations vs $\alpha$   ($\beta$=0°, $\delta_D$=0°)", fontsize=11)
    sub  = df[(df["beta_deg"] == 0) & (df["deltaD_deg"] == 0)].sort_values("alpha_deg")
    cmap = plt.cm.get_cmap("RdBu_r", len(DELTAS_RANGE))
    for ax, (col, _) in zip(axes, rate_cols):
        for j, ds in enumerate(sorted(sub["deltaS_deg"].unique())):
            s = sub[sub["deltaS_deg"] == ds]
            ax.plot(s["alpha_deg"], s[col], color=cmap(j), marker=".", markersize=4, label=f"δS={ds:.0f}°")
        ax.set_xlabel("α (deg)"); ax.set_ylabel(f"{col} (rad/s²)")
        ax.set_title(col); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_alpha.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(r"Angular Accelerations vs $\beta$   ($\alpha$=5°, $\delta_S$=0°)", fontsize=11)
    sub  = df[np.isclose(df["alpha_deg"], 5.0) & (df["deltaS_deg"] == 0)].sort_values("beta_deg")
    cmap = plt.cm.get_cmap("PiYG", len(DELTAD_RANGE))
    for ax, (col, _) in zip(axes, rate_cols):
        for j, dd in enumerate(sorted(sub["deltaD_deg"].unique())):
            s = sub[sub["deltaD_deg"] == dd]
            ax.plot(s["beta_deg"], s[col], color=cmap(j), marker=".", markersize=4, label=f"δD={dd:.0f}°")
        ax.set_xlabel("β (deg)"); ax.set_ylabel(f"{col} (rad/s²)")
        ax.set_title(col); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_beta.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(r"Moment Coefficients vs $\alpha$   ($\beta$=$\delta_S$=$\delta_D$=0°)", fontsize=11)
    sub = df[(df["beta_deg"] == 0) & (df["deltaS_deg"] == 0) & (df["deltaD_deg"] == 0)].sort_values("alpha_deg")
    for ax, coef in zip(axes, ["Cl", "Cm", "Cn"]):
        ax.plot(sub["alpha_deg"], sub[coef], marker="o", markersize=4)
        ax.set_xlabel("α (deg)"); ax.set_ylabel(coef); ax.set_title(coef); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sweep_body_rates_coeffs.png", dpi=150)

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

    plt.close("all")


if __name__ == "__main__":
    found = False
    for suffix in ["V2", "V1", ""]:
        mp = MLP_MODEL_PATH.replace("V2", suffix)
        xp = X_SCALER_PATH.replace("V2", suffix)
        yp = Y_SCALER_PATH.replace("V2", suffix)
        if all(os.path.isfile(p) for p in [mp, xp, yp]):
            found = True
            break
    if not found:
        raise FileNotFoundError(f"MLP model files not found, expected at {os.path.dirname(MLP_MODEL_PATH)}")

    aero_model = _aero.MLPAeroModel(
        model_path=mp, x_scaler_path=xp, y_scaler_path=yp,
        cfg=AERO_CFG, device="cpu",
        max_deltaS_deg=DELTAS_MAX_DEG,
        max_deltaD_deg=DELTAD_MAX_DEG,
    )

    N = len(ALPHA_RANGE) * len(BETA_RANGE) * len(DELTAS_RANGE) * len(DELTAD_RANGE)
    print(f"sweeping {N} conditions...")
    df = run_sweep(aero_model)
    df.to_csv("sweep_body_rates.csv", index=False)
    print(f"saved sweep_body_rates.csv  ({df.shape[0]} rows)")

    for col in ["p_dot", "q_dot", "r_dot"]:
        print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}] rad/s²")

    make_plots(df)
    print("done")