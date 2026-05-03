#!/usr/bin/env python3
# run_sim_MPC.py
"""
Runs the 6DOF sim using the fixed-thrust MLP-aero MPC controller.

This is based on your gain-scheduled run_sim file, but removes the cascaded
heading/roll/pitch controller settings and keeps only:
  - sim configuration
  - MLP/table aero model setup
  - fixed thrust setup
  - MPC-specific settings
  - relevant MPC plots
  - 3D animation

Expected project structure:
  project_root/
    run_sim_MPC.py
    dynamicsMPC_dashboard.py
    propulsion.py
    aero.py
    aero_autosweep.csv
    mlp_aero_state_dictV2.pt
    xScalerV2.pkl
    yScalerV2.pkl
    controllers/
      mlp_fixed_thrust_mpc.py
"""

from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dynamicsMPC_dashboard as dynamics
import propulsion
import aero


# =============================================================================
# User inputs
# =============================================================================

CSV_PATH = r"..\aero_autosweep.csv"
OUTPUT_CSV = r"sim_out_mlp_mpc_fixed_thrust_tuned.csv"

CONTROLLER_MODULE = "controllers.mlp_fixed_thrust_mpc"

# Environment
RHO = 1.225
G = 9.80665
WIND_WORLD = [0.0, 5.0, 2.0]  # NED

# Vehicle
MASS = 4.0
IXX, IYY, IZZ = 0.8, 0.12, 0.18
IXY, IXZ, IYZ = 0.0, 0.0, 0.0

COM_BODY = [-0.3, 0.0, 0.0]
AERO_REF_POINT_BODY = [-0.3, 0.0, 0.0]

S_REF = 0.476500
B_REF = 2.000000
C_REF = 0.196250

# Initial state, world NED
POS0_WORLD = [0.0, 0.0, 0.0]
VEL0_WORLD = [20.0, 0.0, -3.0]

HEADING_DEG = 0.0
PITCH_DEG = 0.0
ROLL_DEG = 10.0
OMEGA0_BODY = [0.0, 0.0, 0.0]

# Initial/trim controls
DELTA_S_DEG = 0.0
DELTA_D_DEG = 0.0

# Commands tracked by MPC
HEADING_CMD_DEG = 50.0
PITCH_CMD_DEG = 3.0
ROLL_CMD_DEG = 0.0
BETA_CMD_DEG = 0.0

# Fixed-thrust propulsion
THROTTLE0 = 1.0
THRUST_ENABLED = True
THRUST_MAX_N = 4.0
THRUST_DIR_BODY = [1.0, 0.0, 0.0]
THRUST_POS_BODY = [-0.6, 0.0, 0.0]

# Simulation
DT = 0.006
T_FINAL = 35.0
MAX_SPEED = 250.0
MAX_OMEGA = 50.0

# Actuator limits used by MPC and aero model
DELTAS_MAX_DEG = 20.0
DELTAD_MAX_DEG = 10.0
DELTA_S_RATE_MAX_DEG_S = 120.0
DELTA_D_RATE_MAX_DEG_S = 120.0

# Aero coefficient sign flips
SIGN_CL = 1.0
SIGN_CD = 1.0
SIGN_CY = 1.0
SIGN_Cl = 1.0
SIGN_Cm = 1.0
SIGN_Cn = 1.0

CD0_ADD = 0.01038

# Rate damping derivatives used inside aero.py
CLP = -0.3546536
CMQ = -6.2794743
CNR = -0.0057922

MAX_ALPHA_DEG = 35.0
MAX_BETA_DEG = 20.0

USE_SUPERPOSITION = True
DEFLECTION_TOL = 1e-6

# Static derivative extrapolation
USE_STATIC_EXTRAPOLATION = True
DCL_DALPHA =  4.2721876
DCL_DBETA  = -0.0018362
DCD_DALPHA =  0.0934618
DCD_DBETA  =  0.0005434
DCY_DALPHA = -0.0000072
DCY_DBETA  = -0.0952724
DCl_DALPHA = -0.0000097
DCl_DBETA  = -0.0998408
DCm_DALPHA = -1.2983957
DCm_DBETA  = -0.0027308
DCn_DALPHA =  0.0000004
DCn_DBETA  =  0.0109517

# MLP aero model
USE_MLP_AERO = True
MLP_MODEL_PATH = r"..\mlp_aero_state_dictV2.pt"
X_SCALER_PATH = r"..\xScalerV2.pkl"
Y_SCALER_PATH = r"..\yScalerV2.pkl"

# MPC settings
# Tuned for cleaner all-angle behavior without making the run extremely slow.
# Increase HORIZON / NUM_CANDIDATES later if you want more optimization quality.
MPC_HORIZON = 3
MPC_NUM_CANDIDATES = 25
MPC_CONTROL_PERIOD_S = 0.060       # recompute MPC every 0.06 s, hold between updates
MPC_PRED_DT = 0.060                # prediction step used inside horizon
MPC_SEED = 4

# MPC weights. Increase Q terms for faster tracking; increase R terms for smoother controls.
# Revised after the first MPC run:
#   - roll was under-weighted, so Q_roll is increased
#   - body-rate damping is increased to reduce overshoot/oscillation
#   - beta penalty is increased to reduce sideslip/uncoordinated motion
#   - control-change penalty is increased to reduce surface slamming
MPC_Q_HEADING = 1.5
MPC_Q_PITCH = 3.0
MPC_Q_ROLL = 1.5
MPC_Q_RATES = [0.05, 0.03, 0.05]
MPC_Q_ALPHA = 0.25
MPC_Q_BETA = 1.0
MPC_R_CONTROL = [0.03, 0.03]
MPC_R_DELTA_CONTROL = [0.25, 0.25]
MPC_LOCAL_SIGMA_DEG = [2.0, 1.5]

# Progress dashboard
SHOW_PROGRESS_DASHBOARD = True
DASHBOARD_UPDATE_EVERY_S = 0.25
DASHBOARD_BAR_WIDTH = 32

# Plotting / animation
SHOW_PLOTS = True
DO_ANIMATE = True
ANIM_STRIDE = 10
ANIM_AXIS_LEN = 1.0
ANIM_ZOOM = 50.0
STL_PATH = "lowerqualitymesh.stl"       # set to None to disable STL body
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)


# =============================================================================
# Config builders
# =============================================================================

def build_cfg() -> dynamics.SimConfig:
    I = np.array([
        [IXX, -IXY, -IXZ],
        [-IXY, IYY, -IYZ],
        [-IXZ, -IYZ, IZZ],
    ], dtype=float)

    prop_cfg = propulsion.build_simple_single_thruster(
        enabled=THRUST_ENABLED,
        thrust_max_n=THRUST_MAX_N,
        thrust_dir_body=tuple(THRUST_DIR_BODY),
        thrust_pos_body=tuple(THRUST_POS_BODY),
    )

    cfg = dynamics.SimConfig(
        rho=float(RHO),
        g=float(G),
        wind_world=np.array(WIND_WORLD, dtype=float),
        mass=float(MASS),
        I_body=I,
        com_body=np.array(COM_BODY, dtype=float),
        aero_ref_body=np.array(AERO_REF_POINT_BODY, dtype=float),
        S_ref=float(S_REF),
        b_ref=float(B_REF),
        c_ref=float(C_REF),
        dt=float(DT),
        t_final=float(T_FINAL),
        max_speed=float(MAX_SPEED),
        max_omega=float(MAX_OMEGA),
        pos0_world=np.array(POS0_WORLD, dtype=float),
        vel0_world=np.array(VEL0_WORLD, dtype=float),
        yaw0_deg=float(HEADING_DEG),
        pitch0_deg=float(PITCH_DEG),
        roll0_deg=float(ROLL_DEG),
        omega0_body=np.array(OMEGA0_BODY, dtype=float),
        delta_s_deg=float(DELTA_S_DEG),
        delta_d_deg=float(DELTA_D_DEG),
        throttle0=float(THROTTLE0),
        output_csv=str(OUTPUT_CSV),
        propulsion=prop_cfg,
    )

    # Commands
    cfg.heading_cmd_deg = float(HEADING_CMD_DEG)
    cfg.pitch_cmd_deg = float(PITCH_CMD_DEG)
    cfg.mpc_roll_cmd_deg = float(ROLL_CMD_DEG)
    cfg.mpc_beta_cmd_deg = float(BETA_CMD_DEG)

    # Actuator limits/rate limits for the MPC controller
    cfg.deltaS_max_deg = float(DELTAS_MAX_DEG)
    cfg.deltaD_max_deg = float(DELTAD_MAX_DEG)
    cfg.deltaS_rate_max_deg_s = float(DELTA_S_RATE_MAX_DEG_S)
    cfg.deltaD_rate_max_deg_s = float(DELTA_D_RATE_MAX_DEG_S)

    # MPC settings
    cfg.mpc_horizon = int(MPC_HORIZON)
    cfg.mpc_num_candidates = int(MPC_NUM_CANDIDATES)
    cfg.mpc_control_period_s = float(MPC_CONTROL_PERIOD_S)
    cfg.mpc_pred_dt = float(MPC_PRED_DT)
    cfg.mpc_seed = int(MPC_SEED)

    cfg.mpc_Q_heading = float(MPC_Q_HEADING)
    cfg.mpc_Q_pitch = float(MPC_Q_PITCH)
    cfg.mpc_Q_roll = float(MPC_Q_ROLL)
    cfg.mpc_Q_rates = list(MPC_Q_RATES)
    cfg.mpc_Q_alpha = float(MPC_Q_ALPHA)
    cfg.mpc_Q_beta = float(MPC_Q_BETA)
    cfg.mpc_R_control = list(MPC_R_CONTROL)
    cfg.mpc_R_delta_control = list(MPC_R_DELTA_CONTROL)
    cfg.mpc_local_sigma_deg = list(MPC_LOCAL_SIGMA_DEG)

    # Terminal progress dashboard settings
    cfg.show_progress_dashboard = bool(SHOW_PROGRESS_DASHBOARD)
    cfg.dashboard_update_every_s = float(DASHBOARD_UPDATE_EVERY_S)
    cfg.dashboard_bar_width = int(DASHBOARD_BAR_WIDTH)

    return cfg


def build_aero_model():
    aero_cfg = aero.AeroConfig(
        sign_cl=SIGN_CL, sign_cd=SIGN_CD, sign_cy=SIGN_CY,
        sign_cll=SIGN_Cl, sign_cm=SIGN_Cm, sign_cn=SIGN_Cn,
        cd0_add=CD0_ADD,
        clp=CLP, cmq=CMQ, cnr=CNR,
        use_static_extrapolation=USE_STATIC_EXTRAPOLATION,
        max_alpha_deg=MAX_ALPHA_DEG,
        max_beta_deg=MAX_BETA_DEG,
        dcl_dalpha=DCL_DALPHA, dcl_dbeta=DCL_DBETA,
        dcd_dalpha=DCD_DALPHA, dcd_dbeta=DCD_DBETA,
        dcy_dalpha=DCY_DALPHA, dcy_dbeta=DCY_DBETA,
        dcll_dalpha=DCl_DALPHA, dcll_dbeta=DCl_DBETA,
        dcm_dalpha=DCm_DALPHA, dcm_dbeta=DCm_DBETA,
        dcn_dalpha=DCn_DALPHA, dcn_dbeta=DCn_DBETA,
    )

    if USE_MLP_AERO:
        return aero.MLPAeroModel(
            model_path=MLP_MODEL_PATH,
            x_scaler_path=X_SCALER_PATH,
            y_scaler_path=Y_SCALER_PATH,
            cfg=aero_cfg,
            device="cpu",
            max_deltaS_deg=DELTAS_MAX_DEG,
            max_deltaD_deg=DELTAD_MAX_DEG,
        )

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    return aero.AeroModel(
        df,
        cfg=aero_cfg,
        tol=DEFLECTION_TOL,
        use_superposition=USE_SUPERPOSITION,
    )


# =============================================================================
# Diagnostics and plots
# =============================================================================

def wrap_deg_series(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def merge_mpc_log(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if not hasattr(cfg, "_mpc_log") or len(cfg._mpc_log) == 0:
        return df

    mpc = pd.DataFrame(cfg._mpc_log)
    mpc = mpc.drop_duplicates(subset=["t"], keep="last").sort_values("t")
    out = pd.merge_asof(
        df.sort_values("t"),
        mpc.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=max(float(cfg.dt) * 1.5, 1e-9),
    )
    return out


def coalesce_duplicate_mpc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After merging the dynamics-level MPC columns with the controller-level MPC log,
    pandas may create *_x and *_y duplicate columns. This collapses them back into
    clean names so the plotting functions can use mpc_best_cost, mpc_reused, etc.
    """
    suffix_pairs = []
    for col in list(df.columns):
        if col.endswith("_x") and col[:-2] + "_y" in df.columns:
            suffix_pairs.append(col[:-2])

    for base in suffix_pairs:
        xcol = base + "_x"
        ycol = base + "_y"
        df[base] = df[ycol].combine_first(df[xcol])
        df = df.drop(columns=[xcol, ycol])

    return df


def add_tracking_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df["psi_cmd_deg"] = float(cfg.heading_cmd_deg)
    df["theta_cmd_deg"] = float(cfg.pitch_cmd_deg)
    df["roll_cmd_deg"] = float(getattr(cfg, "mpc_roll_cmd_deg", 0.0))

    df["hdg_err_deg"] = wrap_deg_series(df["psi_cmd_deg"].to_numpy() - df["yaw_deg"].to_numpy())
    df["pitch_err_deg"] = df["theta_cmd_deg"].to_numpy() - df["pitch_deg"].to_numpy()
    df["roll_err_deg"] = wrap_deg_series(df["roll_cmd_deg"].to_numpy() - df["roll_deg"].to_numpy())

    df["p_deg_s"] = np.rad2deg(df["p"].to_numpy())
    df["q_deg_s"] = np.rad2deg(df["q"].to_numpy())
    df["r_deg_s"] = np.rad2deg(df["r"].to_numpy())

    df["altitude_m"] = -df["z"].to_numpy()
    df["LD"] = df["CL"].to_numpy() / np.maximum(df["CD"].to_numpy(), 1e-12)
    return df


def plot_mpc_results(df: pd.DataFrame):
    t = df["t"].to_numpy(dtype=float)

    def fig():
        plt.figure()
        plt.grid(True)

    # Tracking
    fig()
    plt.plot(t, df["yaw_deg"], label="yaw")
    plt.plot(t, df["psi_cmd_deg"], label="yaw command")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("MPC Heading Tracking")
    plt.legend()

    fig()
    plt.plot(t, df["pitch_deg"], label="pitch")
    plt.plot(t, df["theta_cmd_deg"], label="pitch command")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("MPC Pitch Tracking")
    plt.legend()

    fig()
    plt.plot(t, df["roll_deg"], label="roll")
    plt.plot(t, df["roll_cmd_deg"], label="roll command")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("MPC Roll Tracking")
    plt.legend()

    fig()
    plt.plot(t, df["roll_deg"], label="roll")
    plt.plot(t, df["p_deg_s"], label="p roll rate")
    plt.xlabel("t (s)")
    plt.ylabel("deg / deg/s")
    plt.title("MPC Roll Response")
    plt.legend()

    fig()
    plt.plot(t, df["hdg_err_deg"], label="heading error")
    plt.plot(t, df["pitch_err_deg"], label="pitch error")
    plt.plot(t, df["roll_err_deg"], label="roll error")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("MPC Tracking Errors")
    plt.legend()

    # Controls
    fig()
    plt.plot(t, df["deltaS_deg"], label="deltaS")
    plt.plot(t, df["deltaD_deg"], label="deltaD")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("MPC Control Inputs")
    plt.legend()

    fig()
    plt.plot(t, np.abs(df["deltaS_deg"]) / max(float(DELTAS_MAX_DEG), 1e-9), label="|deltaS| / limit")
    plt.plot(t, np.abs(df["deltaD_deg"]) / max(float(DELTAD_MAX_DEG), 1e-9), label="|deltaD| / limit")
    plt.xlabel("t (s)")
    plt.ylabel("fraction of limit")
    plt.title("MPC Surface Saturation Fraction")
    plt.grid(True)
    plt.legend()

    if "mpc_best_cost" in df.columns:
        fig()
        plt.plot(t, df["mpc_best_cost"], label="best horizon cost")
        plt.xlabel("t (s)")
        plt.ylabel("cost")
        plt.title("MPC Best Candidate Cost")
        plt.legend()

    # Rates and aero angles
    fig()
    plt.plot(t, df["p_deg_s"], label="p roll rate")
    plt.plot(t, df["q_deg_s"], label="q pitch rate")
    plt.plot(t, df["r_deg_s"], label="r yaw rate")
    plt.xlabel("t (s)")
    plt.ylabel("deg/s")
    plt.title("Body Rates")
    plt.legend()

    fig()
    plt.plot(t, df["alpha_deg"], label="alpha")
    plt.plot(t, df["beta_deg"], label="beta")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("Airdata Angles")
    plt.legend()

    # Aero coefficients and energy-ish diagnostics
    fig()
    plt.plot(t, df["CL"], label="CL")
    plt.plot(t, df["CD"], label="CD")
    plt.plot(t, df["CY"], label="CY")
    plt.xlabel("t (s)")
    plt.ylabel("coefficient")
    plt.title("MLP Aero Coefficients")
    plt.legend()

    fig()
    plt.plot(t, df["Cl"], label="Cl")
    plt.plot(t, df["Cm"], label="Cm")
    plt.plot(t, df["Cn"], label="Cn")
    plt.xlabel("t (s)")
    plt.ylabel("moment coefficient")
    plt.title("MLP Moment Coefficients")
    plt.legend()

    fig()
    plt.plot(t, df["V"], label="airspeed")
    plt.xlabel("t (s)")
    plt.ylabel("m/s")
    plt.title("Airspeed with Fixed Thrust")
    plt.legend()

    fig()
    plt.plot(t, df["altitude_m"], label="altitude proxy = -z")
    plt.xlabel("t (s)")
    plt.ylabel("m")
    plt.title("Altitude Proxy")
    plt.legend()

    fig()
    plt.plot(t, df["LD"], label="CL/CD")
    plt.xlabel("t (s)")
    plt.ylabel("L/D")
    plt.title("L/D Over Time")
    plt.legend()

    if "T_N" in df.columns:
        fig()
        plt.plot(t, df["T_N"], label="thrust N")
        plt.plot(t, df["throttle"], label="fixed throttle")
        plt.xlabel("t (s)")
        plt.title("Fixed Thrust / Throttle")
        plt.legend()


# =============================================================================
# 3D animation
# =============================================================================

def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_mid = 0.5 * (x_limits[1] + x_limits[0])
    y_mid = 0.5 * (y_limits[1] + y_limits[0])
    z_mid = 0.5 * (z_limits[1] + z_limits[0])
    radius = 0.5 * max([x_range, y_range, z_range, 1e-9])
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def animate_3d(df: pd.DataFrame, stride=10, interval_ms=30, trail_len=200,
               axis_len=1.0, zoom=50.0, stl_path=None, stl_scale=1.0,
               stl_rotation_offset_deg=(0.0, 0.0, 0.0)):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def Rx(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def body_to_ned_rotation(yaw_deg_i, pitch_deg_i, roll_deg_i):
        yaw, pitch, roll = np.deg2rad([yaw_deg_i, pitch_deg_i, roll_deg_i])
        return Rz(yaw) @ Ry(pitch) @ Rx(roll)

    d = df.iloc[::max(1, int(stride))].reset_index(drop=True)
    x = d["x"].to_numpy(float)
    y = d["y"].to_numpy(float)
    z = d["z"].to_numpy(float)
    yaw = d["yaw_deg"].to_numpy(float)
    pitch = d["pitch_deg"].to_numpy(float)
    roll = d["roll_deg"].to_numpy(float)

    T_ned_to_plot = np.diag([1.0, 1.0, -1.0])
    xp, yp, zp = x, y, -z

    stl_vectors_body = None
    if stl_path is not None and os.path.isfile(stl_path):
        from stl import mesh
        stl_mesh = mesh.Mesh.from_file(stl_path)
        stl_vectors_body = stl_mesh.vectors.copy().astype(float)
        max_tris = 1000
        if stl_vectors_body.shape[0] > max_tris:
            keep = np.linspace(0, stl_vectors_body.shape[0] - 1, max_tris).astype(int)
            stl_vectors_body = stl_vectors_body[keep]
            print(f"Downsampled STL to {stl_vectors_body.shape[0]} triangles for animation.")
        stl_vectors_body -= stl_vectors_body.reshape(-1, 3).mean(axis=0)
        stl_vectors_body *= float(stl_scale)
        ro, po, yo = np.deg2rad(stl_rotation_offset_deg)
        R_offset = Rz(yo) @ Ry(po) @ Rx(ro)
        stl_vectors_body = np.einsum("ij,tkj->tki", R_offset, stl_vectors_body)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("MPC 3D Flight Animation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")

    span = max(float(np.ptp(xp)), float(np.ptp(yp)), float(np.ptp(zp)), 10.0)
    pad = (0.25 * span) / max(float(zoom), 1e-6)

    trail, = ax.plot([], [], [], linewidth=1)
    x_line, = ax.plot([], [], [], linewidth=2, label="Body +X")
    y_line, = ax.plot([], [], [], linewidth=2, label="Body +Y")
    z_line, = ax.plot([], [], [], linewidth=2, label="Body +Z")
    ax.legend(loc="upper left")

    stl_poly = None
    if stl_vectors_body is not None:
        stl_poly = Poly3DCollection(stl_vectors_body, alpha=0.7)
        stl_poly.set_edgecolor("k")
        ax.add_collection3d(stl_poly)

    ex_body = np.array([1.0, 0.0, 0.0])
    ey_body = np.array([0.0, 1.0, 0.0])
    ez_body = np.array([0.0, 0.0, -1.0])

    def update(i):
        pos_ned = np.array([x[i], y[i], z[i]], dtype=float)
        pos_plot = T_ned_to_plot @ pos_ned
        R_bw = body_to_ned_rotation(yaw[i], pitch[i], roll[i])

        axes = [
            T_ned_to_plot @ (R_bw @ ex_body),
            T_ned_to_plot @ (R_bw @ ey_body),
            T_ned_to_plot @ (R_bw @ ez_body),
        ]
        lines = [x_line, y_line, z_line]
        for line, axis_vec in zip(lines, axes):
            tip = pos_plot + axis_len * axis_vec
            line.set_data([pos_plot[0], tip[0]], [pos_plot[1], tip[1]])
            line.set_3d_properties([pos_plot[2], tip[2]])

        i0 = max(0, i - int(trail_len))
        trail.set_data(xp[i0:i + 1], yp[i0:i + 1])
        trail.set_3d_properties(zp[i0:i + 1])

        if stl_poly is not None:
            stl_ned = np.einsum("ij,tkj->tki", R_bw, stl_vectors_body)
            stl_plot = np.einsum("ij,tkj->tki", T_ned_to_plot, stl_ned) + pos_plot
            stl_poly.set_verts(stl_plot)

        ax.set_xlim(pos_plot[0] - pad, pos_plot[0] + pad)
        ax.set_ylim(pos_plot[1] - pad, pos_plot[1] + pad)
        ax.set_zlim(pos_plot[2] - pad, pos_plot[2] + pad)
        _set_axes_equal(ax)
        return [trail, x_line, y_line, z_line] + ([] if stl_poly is None else [stl_poly])

    anim = FuncAnimation(
        fig, update, frames=len(d), interval=int(interval_ms),
        blit=False, cache_frame_data=False
    )
    return anim


# =============================================================================
# Main
# =============================================================================

def main():
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cfg = build_cfg()
    aero_model = build_aero_model()

    # This is what lets the MPC use the real MLP aero + physics during prediction.
    cfg._mpc_aero_model = aero_model

    ctrl_mod = importlib.import_module(CONTROLLER_MODULE)
    controller_fn = ctrl_mod.controller

    df_out = dynamics.simulate(cfg, aero_model, controller_fn)
    df_out = merge_mpc_log(df_out, cfg)
    df_out = coalesce_duplicate_mpc_columns(df_out)
    df_out = add_tracking_columns(df_out, cfg)

    # Save merged output with MPC diagnostics too.
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote merged MPC diagnostics to {OUTPUT_CSV}")

    if SHOW_PLOTS:
        plot_mpc_results(df_out)

    ani = None
    if DO_ANIMATE:
        ani = animate_3d(
            df_out,
            stride=int(ANIM_STRIDE),
            interval_ms=int(1000 * DT * int(ANIM_STRIDE)),
            axis_len=float(ANIM_AXIS_LEN),
            zoom=float(ANIM_ZOOM),
            stl_path=STL_PATH,
            stl_scale=STL_SCALE,
            stl_rotation_offset_deg=STL_ROTATION_OFFSET_DEG,
        )

    if SHOW_PLOTS or DO_ANIMATE:
        plt.show()

    return ani


if __name__ == "__main__":
    main()
