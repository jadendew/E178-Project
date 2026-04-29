#!/usr/bin/env python3
"""
run_openloop.py

Open-loop 6DOF simulation runner using:
  - aero.py: AeroModel + aero_forces_moments_body (lookup/interp/superposition + forces/moments)
  - dynamics.py: State propagation (RK4)

Keeps physics identical to your monolithic "final" script.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from aero import AeroModel, aero_forces_moments_body
from dynamics import Vehicle, Environment, State, step_rk4, euler_to_quat_ypr


# =============================================================================
# USER INPUTS (same as your final version)
# =============================================================================

CSV_PATH = r"C:\Users\acey4\Documents\fixedwing optimization zone\vspfiles\control_sweeps\autosweep\aero_autosweep.csv"
OUTPUT_CSV = r"C:\Users\acey4\Documents\fixedwing optimization zone\vspfiles\control_sweeps\autosweep\sim_out.csv"

# Environment
RHO = 1.225
G = 9.80665
WIND_WORLD = [0.0, 0.0, 0.0]

# Vehicle
MASS = 4.0
IXX, IYY, IZZ = 0.8, 0.12, 0.18
IXY, IXZ, IYZ = 0.0, 0.0, 0.0

COM_BODY = [-0.3, 0.0, 0.0]
AERO_REF_POINT_BODY = [-0.3, 0.0, 0.0]

S_REF = 0.476500
B_REF = 2.000000
C_REF = 0.196250

# Controls (open-loop constants for now)
DELTA_S_DEG = 0.0   # + pitch up, - pitch down (per your notes)
DELTA_D_DEG = 0.0    # + right, - left (per your notes)

# Aero table handling
USE_SUPERPOSITION = True
DEFLECTION_TOL = 1e-6

# Initial state (WORLD NED)
POS0_WORLD = [0.0, 0.0, 0.0]
VEL0_WORLD = [10.0, 0.0, 20.0]

HEADING_DEG = 0.0
PITCH_DEG = 3.0
ROLL_DEG = 0.0

OMEGA0_BODY = [0.0, 0.0, 0.0]

# CD0 correction
CD0_ADD = 0.01038

# Rate damping (per rad)
CLP = -0.3546536
CMQ = -6.2794743
CNR = -0.0057922

# Simulation
DT = 0.006
T_FINAL = 20.0

# Safety clamps
MAX_SPEED = 250.0
MAX_OMEGA = 50.0
MAX_QBAR = 200000.0
MAX_ALPHA_DEG = 35.0
MAX_BETA_DEG = 20.0

# Sign flips
SIGN_CL = 1.0
SIGN_CD = 1.0
SIGN_CY = 1.0
SIGN_Cl = 1.0
SIGN_Cm = 1.0
SIGN_Cn = 1.0

# Static derivative extrapolation
USE_STATIC_EXTRAPOLATION = True

DCL_DALPHA = 4.2721876
DCL_DBETA = -0.0018362

DCD_DALPHA = 0.0934618
DCD_DBETA = 0.0005434

DCY_DALPHA = -0.0000072
DCY_DBETA = -0.0952724

DCl_DALPHA = -0.0000097
DCl_DBETA = -0.0998408

DCm_DALPHA = -1.2983957
DCm_DBETA = -0.0027308

DCn_DALPHA = 0.0000004
DCn_DBETA = 0.0109517

# Plotting
SHOW_PLOTS = False


# =============================================================================
# Config dict passed into aero/dynamics (keeps behavior identical, but not global)
# =============================================================================

CFG: Dict[str, float] = {
    "CD0_ADD": CD0_ADD,

    "CLP": CLP,
    "CMQ": CMQ,
    "CNR": CNR,

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

    "DCL_DALPHA": DCL_DALPHA,
    "DCL_DBETA": DCL_DBETA,
    "DCD_DALPHA": DCD_DALPHA,
    "DCD_DBETA": DCD_DBETA,
    "DCY_DALPHA": DCY_DALPHA,
    "DCY_DBETA": DCY_DBETA,
    "DCl_DALPHA": DCl_DALPHA,
    "DCl_DBETA": DCl_DBETA,
    "DCm_DALPHA": DCm_DALPHA,
    "DCm_DBETA": DCm_DBETA,
    "DCn_DALPHA": DCn_DALPHA,
    "DCn_DBETA": DCn_DBETA,
}


# =============================================================================
# Simulation
# =============================================================================

def simulate() -> pd.DataFrame:
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    required = {"alpha_deg", "beta_deg", "deltaS_deg", "deltaD_deg", "CL", "CD", "CY", "Cl", "Cm", "Cn"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(required - set(df.columns))}")

    aero = AeroModel(df, tol=DEFLECTION_TOL, use_superposition=USE_SUPERPOSITION)

    I = np.array(
        [
            [IXX, -IXY, -IXZ],
            [-IXY, IYY, -IYZ],
            [-IXZ, -IYZ, IZZ],
        ],
        dtype=float,
    )

    veh = Vehicle(
        mass=MASS,
        I_body=I,
        com_body=np.array(COM_BODY, dtype=float),
        aero_ref_body=np.array(AERO_REF_POINT_BODY, dtype=float),
        S_ref=S_REF,
        b_ref=B_REF,
        c_ref=C_REF,
    )

    env = Environment(
        rho=RHO,
        g=G,
        wind_world=np.array(WIND_WORLD, dtype=float),
    )

    yaw = np.deg2rad(HEADING_DEG)
    pitch = np.deg2rad(PITCH_DEG)
    roll = np.deg2rad(ROLL_DEG)
    q0 = euler_to_quat_ypr(yaw=yaw, pitch=pitch, roll=roll)

    state = State(
        pos_world=np.array(POS0_WORLD, dtype=float),
        vel_world=np.array(VEL0_WORLD, dtype=float),
        quat_body_to_world=q0 / np.linalg.norm(q0),
        omega_body=np.array(OMEGA0_BODY, dtype=float),
    )

    rows = []
    t = 0.0
    n_steps = int(np.ceil(T_FINAL / DT))

    for i in range(n_steps + 1):
        eul = R.from_quat(state.quat_body_to_world).as_euler("ZYX", degrees=True)  # yaw,pitch,roll

        F_b, M_b, coeffs, der = aero_forces_moments_body(
            aero, veh, env, state, DELTA_S_DEG, DELTA_D_DEG, CFG
        )

        rows.append(
            {
                "t": t,
                "x": state.pos_world[0],
                "y": state.pos_world[1],
                "z": state.pos_world[2],
                "vx": state.vel_world[0],
                "vy": state.vel_world[1],
                "vz": state.vel_world[2],
                "V": der["V"],
                "alpha_deg": der["alpha_deg"],
                "beta_deg": der["beta_deg"],
                "alpha_tbl_deg": der["alpha_tbl_deg"],
                "beta_tbl_deg": der["beta_tbl_deg"],
                "qbar": der["qbar"],
                "deltaS_req_deg": der["deltaS_req_deg"],
                "deltaD_req_deg": der["deltaD_req_deg"],
                "deltaS_eff_deg": der["deltaS_eff_deg"],
                "deltaD_eff_deg": der["deltaD_eff_deg"],
                "yaw_deg": eul[0],
                "pitch_deg": eul[1],
                "roll_deg": eul[2],
                "p": state.omega_body[0],
                "q": state.omega_body[1],
                "r": state.omega_body[2],
                "CL": coeffs["CL"],
                "CD": coeffs["CD"],
                "CY": coeffs["CY"],
                "Cl": coeffs["Cl"],
                "Cm": coeffs["Cm"],
                "Cn": coeffs["Cn"],
                "Fax_b": F_b[0],
                "Fay_b": F_b[1],
                "Faz_b": F_b[2],
                "L_b": M_b[0],
                "M_b": M_b[1],
                "N_b": M_b[2],
            }
        )

        # Early stop if state becomes non-finite
        if not (
            np.isfinite(state.pos_world).all()
            and np.isfinite(state.vel_world).all()
            and np.isfinite(state.quat_body_to_world).all()
            and np.isfinite(state.omega_body).all()
        ):
            print(f"STOP: non-finite state at t={t:.3f}")
            break

        # Step
        if i < n_steps:
            state, _, _, _, _ = step_rk4(
                state, veh, env, aero, DT, DELTA_S_DEG, DELTA_D_DEG, CFG
            )
            t += DT

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(out)} rows.")
    return out


# =============================================================================
# Plotting + animation (same as your final)
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

    if SHOW_PLOTS:
        plt.show()


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


def animate_3d(
    df: pd.DataFrame,
    stride: int = 2,
    interval_ms: int = 30,
    trail_len: int = 200,
    axis_len: float = 1.0,
    zoom: float = 1.0,
    show: bool = True,
    save_path: str | None = None,
):
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

    base_span = max((np.max(xp) - np.min(xp)), (np.max(yp) - np.min(yp)), (np.max(zp) - np.min(zp)), 10.0)
    pad = (0.25 * base_span) / max(zoom, 1e-6)

    trail, = ax.plot([], [], [], linewidth=1)
    x_line, = ax.plot([], [], [], linewidth=2, color="r")
    y_line, = ax.plot([], [], [], linewidth=2, color="g")
    z_line, = ax.plot([], [], [], linewidth=2, color="b")

    ax.legend(handles=[x_line, y_line, z_line], labels=["Body +X", "Body +Y", "Body +Z"], loc="upper left")

    ax.set_xlim(xp[0] - pad, xp[0] + pad)
    ax.set_ylim(yp[0] - pad, yp[0] + pad)
    ax.set_zlim(zp[0] - pad, zp[0] + pad)
    _set_axes_equal(ax)

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    def R_body_to_world_ned(i: int) -> np.ndarray:
        return R.from_euler("ZYX", [yaw_deg[i], pitch_deg[i], roll_deg[i]], degrees=True).as_matrix()

    def update(i: int):
        pos_plot = np.array([xp[i], yp[i], zp[i]], dtype=float)

        R_ned = R_body_to_world_ned(i)
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
        trail.set_data(xp[i0 : i + 1], yp[i0 : i + 1])
        trail.set_3d_properties(zp[i0 : i + 1])

        ax.set_xlim(pos_plot[0] - pad, pos_plot[0] + pad)
        ax.set_ylim(pos_plot[1] - pad, pos_plot[1] + pad)
        ax.set_zlim(pos_plot[2] - pad, pos_plot[2] + pad)
        _set_axes_equal(ax)

        return trail, x_line, y_line, z_line

    ani = FuncAnimation(fig, update, frames=len(d), interval=int(interval_ms), blit=False)

    if save_path:
        ani.save(save_path, dpi=150)

    if show:
        plt.show()

    return ani


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

    df["LD_eff"] = L / np.maximum(D, 1e-9)
    df["LD_table"] = df["CL"].to_numpy(dtype=float) / np.maximum(df["CD"].to_numpy(dtype=float), 1e-12)

    Vh = np.linalg.norm(v[:, :2], axis=1)
    Vh = np.maximum(Vh, 1e-9)
    gamma = np.arctan2(v[:, 2], Vh)
    df["gamma_deg"] = np.rad2deg(gamma)
    df["glide_ratio_inst"] = Vh / np.maximum(v[:, 2], 1e-9)

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


if __name__ == "__main__":
    df_out = simulate()
    df_out = add_glide_diagnostics_and_plot(df_out)

    make_plots(df_out)

    stride = 10
    animate_3d(
        df_out,
        stride=stride,
        interval_ms=int(1000 * DT * stride),  # ~1x real time
        axis_len=1.0,
        zoom=50.0,
        show=True,
    )
