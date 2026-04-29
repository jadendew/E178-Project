#!/usr/bin/env python3
"""
run_sim.py

Runner that:
- Loads aero CSV
- Imports controller from controllers/open_loop.py
- Runs 6-DOF simulation
- Writes CSV log
- Computes glide diagnostics + plots
- Optional 3D animation (triad)

All plotting/diagnostics live HERE (per your request).
"""

from __future__ import annotations

# =============================================================================
# USER INPUTS (same style as your current run file)
# =============================================================================

CSV_PATH = r"C:\Users\acey4\Documents\fixedwing optimization zone\vspfiles\control_sweeps\autosweep\aero_autosweep.csv"
OUTPUT_CSV = r"C:\Users\acey4\Documents\fixedwing optimization zone\vspfiles\control_sweeps\autosweep\sim_out.csv"

CONTROLLER_MODULE = "controllers.open_loop" 

# Environment
RHO = 1.225
G = 9.80665
WIND_WORLD = [0.0, 0.0, 0.0]

# Vehicle properties
MASS = 4.0

IXX, IYY, IZZ = 0.8, 0.12, 0.18
IXY, IXZ, IYZ = 0.0, 0.0, 0.0

COM_BODY = [-0.3, 0.0, 0.0]
AERO_REF_POINT_BODY = [-0.3, 0.0, 0.0]

S_REF = 0.476500
B_REF = 2.000000
C_REF = 0.196250

# Controls (open-loop constants; controller reads these)
DELTA_S_DEG = 0.0
# +:pitch up, -:pitch down
DELTA_D_DEG = 0.0
# +:right, -:left


# Aero model options
USE_SUPERPOSITION = True
DEFLECTION_TOL = 1e-6

# Initial state (WORLD NED)
POS0_WORLD = [0.0, 0.0, 0.0]
VEL0_WORLD = [20.0, 0.0, 3.0]  # NED +Z down

HEADING_DEG = 0.0
PITCH_DEG = 3.0
ROLL_DEG = 0.0
OMEGA0_BODY = [0.0, 0.0, 0.0]

# Parasitic drag offset
CD0_ADD = 0.01038

# Rate damping (dimensionless per rad)
CLP = -0.3546536
CMQ = -6.2794743
CNR = -0.0057922

# Simulation
DT = 0.006
T_FINAL = 10.0

# Safety clamps
MAX_SPEED = 250.0
MAX_OMEGA = 50.0
MAX_QBAR = 200000.0
MAX_ALPHA_DEG = 35.0
MAX_BETA_DEG = 20.0

# coefficient sign flips
SIGN_CL = 1.0
SIGN_CD = 1.0
SIGN_CY = 1.0
SIGN_Cl = 1.0
SIGN_Cm = 1.0
SIGN_Cn = 1.0

# Static derivative extrapolation
USE_STATIC_EXTRAPOLATION = True
DCL_DALPHA = 4.2721876
DCL_DBETA  = -0.0018362

DCD_DALPHA = 0.0934618
DCD_DBETA  = 0.0005434

DCY_DALPHA = -0.0000072
DCY_DBETA  = -0.0952724

DCl_DALPHA = -0.0000097
DCl_DBETA  = -0.0998408

DCm_DALPHA = -1.2983957
DCm_DBETA  = -0.0027308

DCn_DALPHA = 0.0000004
DCn_DBETA  = 0.0109517

# Plots / animation
SHOW_PLOTS = True
DO_ANIMATE = True
ANIM_STRIDE = 10
ANIM_AXIS_LEN = 1.0
ANIM_ZOOM = 50.0

# =============================================================================
# END USER INPUTS
# =============================================================================

import os
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from aero import AeroModel
from dynamics import (
    Vehicle, Environment, State,
    euler_to_quat_ypr, dcm_body_to_world,
    wind_angles_from_body_velocity, safe_norm,
    aero_forces_moments_body, step_rk4
)


def build_cfg() -> dict:
    """Pack settings so dynamics can access without globals."""
    return {
        "CSV_PATH": CSV_PATH,
        "OUTPUT_CSV": OUTPUT_CSV,

        "RHO": RHO,
        "G": G,
        "WIND_WORLD": np.array(WIND_WORLD, dtype=float),

        "MASS": MASS,
        "IXX": IXX, "IYY": IYY, "IZZ": IZZ,
        "IXY": IXY, "IXZ": IXZ, "IYZ": IYZ,
        "COM_BODY": np.array(COM_BODY, dtype=float),
        "AERO_REF_POINT_BODY": np.array(AERO_REF_POINT_BODY, dtype=float),
        "S_REF": S_REF, "B_REF": B_REF, "C_REF": C_REF,

        "DELTA_S_DEG": DELTA_S_DEG,
        "DELTA_D_DEG": DELTA_D_DEG,

        "USE_SUPERPOSITION": USE_SUPERPOSITION,
        "DEFLECTION_TOL": DEFLECTION_TOL,

        "CD0_ADD": CD0_ADD,
        "CLP": CLP, "CMQ": CMQ, "CNR": CNR,

        "DT": DT,
        "T_FINAL": T_FINAL,

        "MAX_SPEED": MAX_SPEED,
        "MAX_OMEGA": MAX_OMEGA,
        "MAX_QBAR": MAX_QBAR,
        "MAX_ALPHA_DEG": MAX_ALPHA_DEG,
        "MAX_BETA_DEG": MAX_BETA_DEG,

        "SIGN_CL": SIGN_CL,
        "SIGN_CD": SIGN_CD,
        "SIGN_CY": SIGN_CY,
        "SIGN_Cl": SIGN_Cl,
        "SIGN_Cm": SIGN_Cm,
        "SIGN_Cn": SIGN_Cn,

        "USE_STATIC_EXTRAPOLATION": USE_STATIC_EXTRAPOLATION,
        "DCL_DALPHA": DCL_DALPHA, "DCL_DBETA": DCL_DBETA,
        "DCD_DALPHA": DCD_DALPHA, "DCD_DBETA": DCD_DBETA,
        "DCY_DALPHA": DCY_DALPHA, "DCY_DBETA": DCY_DBETA,
        "DCl_DALPHA": DCl_DALPHA, "DCl_DBETA": DCl_DBETA,
        "DCm_DALPHA": DCm_DALPHA, "DCm_DBETA": DCm_DBETA,
        "DCn_DALPHA": DCn_DALPHA, "DCn_DBETA": DCn_DBETA,
    }


def simulate(cfg: dict, controller_fn) -> pd.DataFrame:
    if not os.path.isfile(cfg["CSV_PATH"]):
        raise FileNotFoundError(cfg["CSV_PATH"])

    df = pd.read_csv(cfg["CSV_PATH"])
    required = {"alpha_deg", "beta_deg", "deltaS_deg", "deltaD_deg", "CL", "CD", "CY", "Cl", "Cm", "Cn"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(required - set(df.columns))}")

    aero = AeroModel(df, tol=cfg["DEFLECTION_TOL"], use_superposition=cfg["USE_SUPERPOSITION"])

    I = np.array([
        [cfg["IXX"], -cfg["IXY"], -cfg["IXZ"]],
        [-cfg["IXY"], cfg["IYY"], -cfg["IYZ"]],
        [-cfg["IXZ"], -cfg["IYZ"], cfg["IZZ"]],
    ], dtype=float)

    veh = Vehicle(
        mass=cfg["MASS"],
        I_body=I,
        com_body=cfg["COM_BODY"],
        aero_ref_body=cfg["AERO_REF_POINT_BODY"],
        S_ref=cfg["S_REF"],
        b_ref=cfg["B_REF"],
        c_ref=cfg["C_REF"],
    )

    env = Environment(
        rho=cfg["RHO"],
        g=cfg["G"],
        wind_world=cfg["WIND_WORLD"],
    )

    yaw = np.deg2rad(HEADING_DEG)
    pitch = np.deg2rad(PITCH_DEG)
    roll = np.deg2rad(ROLL_DEG)
    q0 = euler_to_quat_ypr(yaw=yaw, pitch=pitch, roll=roll)

    state = State(
        pos_world=np.array(POS0_WORLD, dtype=float),
        vel_world=np.array(VEL0_WORLD, dtype=float),
        quat_body_to_world=q0,
        omega_body=np.array(OMEGA0_BODY, dtype=float),
    )

    rows = []
    t = 0.0
    n_steps = int(np.ceil(T_FINAL / DT))

    for i in range(n_steps + 1):
        # state view for controller
        eul = R.from_quat(state.quat_body_to_world).as_euler("ZYX", degrees=True)  # yaw,pitch,roll
        state_view = {
            "yaw_deg": float(eul[0]),
            "pitch_deg": float(eul[1]),
            "roll_deg": float(eul[2]),
            "p": float(state.omega_body[0]),
            "q": float(state.omega_body[1]),
            "r": float(state.omega_body[2]),
            "pos_world": state.pos_world.copy(),
            "vel_world": state.vel_world.copy(),
        }

        # control command
        deltaS_cmd, deltaD_cmd = controller_fn(t, state_view, cfg)

        # forces/moments
        F_b, M_b, coeffs, der = aero_forces_moments_body(
            aero=aero, veh=veh, env=env, state=state,
            deltaS_deg=float(deltaS_cmd), deltaD_deg=float(deltaD_cmd),
            cfg=cfg
        )

        rows.append({
            "t": t,
            "x": state.pos_world[0], "y": state.pos_world[1], "z": state.pos_world[2],
            "vx": state.vel_world[0], "vy": state.vel_world[1], "vz": state.vel_world[2],
            "V": der["V"],
            "alpha_deg": der["alpha_deg"], "beta_deg": der["beta_deg"],
            "alpha_tbl_deg": der["alpha_tbl_deg"], "beta_tbl_deg": der["beta_tbl_deg"],
            "qbar": der["qbar"],

            "deltaS_cmd_deg": float(deltaS_cmd),
            "deltaD_cmd_deg": float(deltaD_cmd),
            "deltaS_eff_deg": der["deltaS_eff_deg"],
            "deltaD_eff_deg": der["deltaD_eff_deg"],

            "yaw_deg": float(eul[0]), "pitch_deg": float(eul[1]), "roll_deg": float(eul[2]),
            "p": state.omega_body[0], "q": state.omega_body[1], "r": state.omega_body[2],

            "CL": coeffs["CL"], "CD": coeffs["CD"], "CY": coeffs["CY"],
            "Cl": coeffs["Cl"], "Cm": coeffs["Cm"], "Cn": coeffs["Cn"],

            "Fax_b": F_b[0], "Fay_b": F_b[1], "Faz_b": F_b[2],
            "L_b": M_b[0], "M_b": M_b[1], "N_b": M_b[2],
        })

        if not (np.isfinite(state.pos_world).all()
                and np.isfinite(state.vel_world).all()
                and np.isfinite(state.quat_body_to_world).all()
                and np.isfinite(state.omega_body).all()):
            print(f"STOP: non-finite state at t={t:.3f}")
            break

        if i < n_steps:
            state, _, _, _, _ = step_rk4(
                state=state, veh=veh, env=env, aero=aero, dt=DT,
                deltaS_deg=float(deltaS_cmd), deltaD_deg=float(deltaD_cmd),
                cfg=cfg
            )
            t += DT

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(out)} rows.")
    return out


# =============================================================================
# Diagnostics + plots (kept here)
# =============================================================================

def make_plots(df: pd.DataFrame):
    t = df["t"].to_numpy()

    def fig3(a, b, c, labels, ylabel, title):
        plt.figure()
        plt.plot(t, a, label=labels[0])
        plt.plot(t, b, label=labels[1])
        plt.plot(t, c, label=labels[2])
        plt.xlabel("t (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()

    plt.figure()
    plt.plot(t, df["CL"], label="CL")
    plt.plot(t, df["CD"], label="CD")
    plt.plot(t, df["CY"], label="CY")
    plt.xlabel("t (s)")
    plt.ylabel("Coeff")
    plt.title("Aero Coefficients Over Time")
    plt.grid(True)
    plt.legend()

    fig3(df["p"], df["q"], df["r"], ["p", "q", "r"], "rad/s", "Body Rates")
    fig3(df["roll_deg"], df["pitch_deg"], df["yaw_deg"], ["roll", "pitch", "yaw"], "deg", "Attitude (Euler ZYX)")
    fig3(df["alpha_deg"], df["beta_deg"], df["V"], ["alpha", "beta", "V"], "deg / m/s", "Angles + Airspeed")

    plt.figure()
    plt.plot(t, -df["z"])
    plt.xlabel("t (s)")
    plt.ylabel("altitude proxy (m) = -z_down")
    plt.title("Altitude Proxy")
    plt.grid(True)


def add_glide_diagnostics_and_plot(df: pd.DataFrame):
    v = df[["vx", "vy", "vz"]].to_numpy(dtype=float)
    V = np.linalg.norm(v, axis=1)
    V = np.maximum(V, 1e-9)
    vhat = (v.T / V).T

    Fb = df[["Fax_b", "Fay_b", "Faz_b"]].to_numpy(dtype=float)

    yaw = np.deg2rad(df["yaw_deg"].to_numpy(dtype=float))
    pitch = np.deg2rad(df["pitch_deg"].to_numpy(dtype=float))
    roll = np.deg2rad(df["roll_deg"].to_numpy(dtype=float))

    Fw = np.zeros_like(Fb)
    for i in range(len(df)):
        Rbw = R.from_euler("ZYX", [yaw[i], pitch[i], roll[i]], degrees=False).as_matrix()
        Fw[i] = Rbw @ Fb[i]

    D = -np.sum(Fw * vhat, axis=1)
    F_parallel = (np.sum(Fw * vhat, axis=1)[:, None]) * vhat
    L = np.linalg.norm(Fw - F_parallel, axis=1)

    LD_eff = L / np.maximum(D, 1e-9)
    LD_table = df["CL"].to_numpy(dtype=float) / np.maximum(df["CD"].to_numpy(dtype=float), 1e-12)

    Vh = np.linalg.norm(v[:, :2], axis=1)
    Vh = np.maximum(Vh, 1e-9)
    gamma = np.arctan2(v[:, 2], Vh)  # positive down (NED)
    glide_ratio_inst = Vh / np.maximum(v[:, 2], 1e-9)

    df["LD_eff"] = LD_eff
    df["LD_table"] = LD_table
    df["gamma_deg"] = np.rad2deg(gamma)
    df["glide_ratio_inst"] = glide_ratio_inst

    t = df["t"].to_numpy(dtype=float)
    plt.figure()
    plt.plot(t, df["LD_eff"], label="L/D from force vector")
    plt.plot(t, df["LD_table"], label="CL/CD from table")
    plt.xlabel("t (s)")
    plt.ylabel("L/D")
    plt.title("L/D consistency check")
    plt.grid(True)
    plt.legend()

    return df


# =============================================================================
# 3D animation (kept here)
# =============================================================================

def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = 0.5 * (x_limits[1] + x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = 0.5 * (y_limits[1] + y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = 0.5 * (z_limits[1] + z_limits[0])

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def animate_3d(df: pd.DataFrame,
               stride: int = 10,
               interval_ms: int = 30,
               trail_len: int = 200,
               axis_len: float = 1.0,
               zoom: float = 50.0,
               show: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    d = df.iloc[::max(1, int(stride))].reset_index(drop=True)

    x = d["x"].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    z = d["z"].to_numpy(dtype=float)

    yaw_deg = d["yaw_deg"].to_numpy(dtype=float)
    pitch_deg = d["pitch_deg"].to_numpy(dtype=float)
    roll_deg = d["roll_deg"].to_numpy(dtype=float)

    # NED -> plot Z-up transform
    T = np.diag([1.0, 1.0, -1.0])
    xp, yp, zp = x, y, -z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Attitude Animation (Z-up plotting frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) [up]")

    base_span = max(
        (np.max(xp) - np.min(xp)),
        (np.max(yp) - np.min(yp)),
        (np.max(zp) - np.min(zp)),
        10.0
    )
    pad = (0.25 * base_span) / max(float(zoom), 1e-6)

    trail, = ax.plot([], [], [], linewidth=1)

    x_line, = ax.plot([], [], [], linewidth=2, color="r")
    y_line, = ax.plot([], [], [], linewidth=2, color="g")
    z_line, = ax.plot([], [], [], linewidth=2, color="b")

    ax.legend(handles=[x_line, y_line, z_line],
              labels=["Body +X", "Body +Y", "Body +Z"], loc="upper left")

    ax.set_xlim(xp[0] - pad, xp[0] + pad)
    ax.set_ylim(yp[0] - pad, yp[0] + pad)
    ax.set_zlim(zp[0] - pad, zp[0] + pad)
    _set_axes_equal(ax)

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    def update(i: int):
        pos_plot = np.array([xp[i], yp[i], zp[i]], dtype=float)

        R_ned = R.from_euler("ZYX", [yaw_deg[i], pitch_deg[i], roll_deg[i]], degrees=True).as_matrix()
        R_plot = T @ R_ned @ T

        x_tip = pos_plot + axis_len * (R_plot @ ex)
        y_tip = pos_plot + axis_len * (R_plot @ ey)
        z_tip = pos_plot + axis_len * (R_plot @ ez)

        x_line.set_data([pos_plot[0], x_tip[0]], [pos_plot[1], x_tip[1]])
        x_line.set_3d_properties([pos_plot[2], x_tip[2]])

        y_line.set_data([pos_plot[0], y_tip[0]], [pos_plot[1], y_tip[1]])
        y_line.set_3d_properties([pos_plot[2], y_tip[2]])

        z_line.set_data([pos_plot[0], z_tip[0]], [pos_plot[1], z_tip[1]])
        z_line.set_3d_properties([pos_plot[2], z_tip[2]])

        i0 = max(0, i - int(trail_len))
        trail.set_data(xp[i0:i+1], yp[i0:i+1])
        trail.set_3d_properties(zp[i0:i+1])

        ax.set_xlim(pos_plot[0] - pad, pos_plot[0] + pad)
        ax.set_ylim(pos_plot[1] - pad, pos_plot[1] + pad)
        ax.set_zlim(pos_plot[2] - pad, pos_plot[2] + pad)
        _set_axes_equal(ax)

        return trail, x_line, y_line, z_line

    ani = FuncAnimation(fig, update, frames=len(d), interval=int(interval_ms), blit=False)

    if show:
        plt.show()

    return ani


def main():
    cfg = build_cfg()

    ctrl_mod = importlib.import_module(CONTROLLER_MODULE)
    controller_fn = ctrl_mod.controller

    df_out = simulate(cfg, controller_fn)

    df_out = add_glide_diagnostics_and_plot(df_out)
    make_plots(df_out)

    ani = None
    if DO_ANIMATE:
        print("Starting animation...")
        ani = animate_3d(
            df_out,
            stride=int(ANIM_STRIDE),
            interval_ms=int(1000 * DT * int(ANIM_STRIDE)),
            axis_len=float(ANIM_AXIS_LEN),
            zoom=float(ANIM_ZOOM),
            show=False,   # <- IMPORTANT: don't show inside animate_3d
        )

    # Show EVERYTHING (plots + animation) in one call:
    if SHOW_PLOTS or DO_ANIMATE:
        plt.show()


if __name__ == "__main__":
    main()
