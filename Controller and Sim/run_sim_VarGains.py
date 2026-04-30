#!/usr/bin/env python3
# run_sim.py
"""
Runs the 6DOF sim with a chosen controller module.

- All plots/diagnostics/animation live here (per your request).
- Dynamics integration is in dynamics.py
- Propulsion model is in propulsion.py
- Aero model is in aero.py (existing)

IMPORTANT:
To avoid the “animation never opens” issue: we call plt.show() ONLY ONCE at the end.
"""

from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import dynamics
import propulsion
import aero  #aero.py

# =============================================================================
# inputs
# =============================================================================

CSV_PATH = r"..\aero_autosweep.csv"
OUTPUT_CSV = r"sim_out_testE_newmlp.csv"

# Controller module path (must expose controller(t, state, cfg))
#CONTROLLER_MODULE = "controllers.heading_pitch_hold"
CONTROLLER_MODULE = "controllers.heading_pitch_hold_mlp_gain_scheduled"

#open loop: "controllers.open_loop"
#heading 2loop: "controllers.heading_hold"
#heading + pitch: "controllers.heading_pitch_hold"

# Environment
RHO = 1.225
G = 9.80665
WIND_WORLD = [0.0, 0.0, 0.0]  # NED

# Vehicle
MASS = 4.0

IXX, IYY, IZZ = 0.8, 0.12, 0.18
IXY, IXZ, IYZ = 0.0, 0.0, 0.0

COM_BODY = [-0.3, 0.0, 0.0]
AERO_REF_POINT_BODY = [-0.3, 0.0, 0.0]

S_REF = 0.476500
B_REF = 2.000000
C_REF = 0.196250


# Initial state (WORLD NED)
POS0_WORLD = [0.0, 0.0, 0.0]
VEL0_WORLD = [20.0, 0.0, -3.0]  # (x,y,z) with +z down

HEADING_DEG = 0.0
PITCH_DEG = 0.0
ROLL_DEG = 10.0
OMEGA0_BODY = [0.0, 0.0, 0.0]


# ============================================================
# Controls (defaults used by open_loop controller)
# ============================================================

DELTA_S_DEG = 0.0
DELTA_D_DEG = 0.0

# ============================================================
#closed loop commands
# ============================================================

#OG
#HEADING_CMD_DEG = 50.0  # desired heading (deg)
#PITCH_CMD_DEG   = 3.0   # desired pitch (deg)

#A
#HEADING_CMD_DEG = 10.0
#PITCH_CMD_DEG = 0.0

#B
#HEADING_CMD_DEG = 30.0
#PITCH_CMD_DEG = 0.0

#C
#HEADING_CMD_DEG = 25.0
#PITCH_CMD_DEG = 5.0

#D
#HEADING_CMD_DEG = 20.0
#PITCH_CMD_DEG = 3.0

#D2
#HEADING_CMD_DEG = 20.0
#PITCH_CMD_DEG = 3.0


#VEL0_WORLD = [15.0, 0.0, -2.0]
#HEADING_DEG = 0.0
#PITCH_DEG = 0.0
#ROLL_DEG = 10.0
#OMEGA0_BODY = [0.0, 0.0, 0.0]

# E
HEADING_CMD_DEG = 20.0
PITCH_CMD_DEG = 0.0

VEL0_WORLD = [20.0, 0.0, -3.0]
HEADING_DEG = 0.0
PITCH_DEG = 8.0
ROLL_DEG = 20.0
OMEGA0_BODY = [0.0, 0.0, 0.0]

# ============================================================
#heading, roll gains
# ============================================================

HDG_K_PSI = 0.8  #gain for yaw *error*, converts hdg error into des. roll command

ROLL_K_PHI = 0.8    #gain for roll *error*, determines roll 'snappiness'

PHI_MAX_DEG = 30.0  #max roll limit

DELTAD_MAX_DEG = 10.0   #max commanded deflection

ROLL_K_P = 0.15     # gain for roll *rate*, provides roll rate damping

YAW_K_R = 0.0       # gain for yaw *rate*, provides yaw rate damping

# ============================================================
#pitch gains
# ============================================================

PIT_K_THETA = 3.0       # q_cmd (deg/s) per deg pitch error
PIT_K_Q     = 1.0      # deltaS (deg) per (deg/s) pitch-rate error

PIT_K_QD    = 0.0       # extra direct q damping (deg per deg/s), start 0

THETA_MAX_DEG  = 20.0   # clamp pitch command
Q_CMD_MAX_DPS  = 60.0   # clamp q_cmd (deg/s)
DELTAS_MAX_DEG = 20.0   # max deltaS command


# --- Propulsion settings ---

THROTTLE0 = 1.0  # start at 0 for glide, later controller can change
THRUST_ENABLED = True
THRUST_MAX_N =   4.0               # N at throttle=1
THRUST_DIR_BODY = [1.0, 0.0, 0.0]   # +X forward
THRUST_POS_BODY = [-0.6, 0.0, 0.0]  # location in body coords (m)

# Simulation
DT = 0.006
T_FINAL = 35.0

# --- coefficient sign flips ---
SIGN_CL = 1.0
SIGN_CD = 1.0
SIGN_CY = 1.0
SIGN_Cl = 1.0
SIGN_Cm = 1.0
SIGN_Cn = 1.0

# Parasitic drag constant offset (CD0 correction)
CD0_ADD = 0.01038

# Rate damping (dimensionless derivatives per rad)
CLP = -0.3546536
CMQ = -6.2794743
CNR = -0.0057922


#ceilings

MAX_SPEED = 250.0
MAX_OMEGA = 50.0
MAX_ALPHA_DEG = 35.0
MAX_BETA_DEG  = 20.0

# Aero model options
USE_SUPERPOSITION = True
DEFLECTION_TOL = 1e-6


# Static derivative extrapolation (from VSPAERO .stab)

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


# Plotting / animation switches
SHOW_PLOTS = True
DO_ANIMATE = True

ANIM_STRIDE = 10
ANIM_AXIS_LEN = 1.0
ANIM_ZOOM = 50.0

USE_MLP_AERO = True

MLP_MODEL_PATH = r"..\mlp_aero_state_dictV2.pt"
X_SCALER_PATH = r"..\xScalerV2.pkl"
Y_SCALER_PATH = r"..\yScalerV2.pkl"

# ============================================================
# MLP gain scheduling settings
# ============================================================

USE_MLP_GAIN_SCHEDULING = True

# Finite-difference step for derivative estimates
MLP_GS_H_DERIV_DEG = 1.0

# Reference condition for nominal control effectiveness
MLP_GS_REF_ALPHA_DEG = 5.0
MLP_GS_REF_BETA_DEG = 0.0
MLP_GS_REF_DELTAS_DEG = 0.0
MLP_GS_REF_DELTAD_DEG = 0.0

# If effectiveness gets very small, avoid huge gains
MLP_GS_EFFECTIVENESS_EPS = 1e-5

# Clamp scheduled gain scaling
MLP_GS_GAIN_SCALE_MIN = 0.4
MLP_GS_GAIN_SCALE_MAX = 2.5

# Debug printing
MLP_GS_PRINT_DEBUG = False
MLP_GS_PRINT_DT = 1.0


# ============================================================
# Improved MLP gain scheduling / damping settings
# ============================================================

MLP_GS_ROLL_PROP_SCALE_POWER = 0.5
MLP_GS_ROLL_DAMP_SCALE_POWER = 1.0

MLP_GS_PITCH_PROP_SCALE_POWER = 0.5
MLP_GS_PITCH_DAMP_SCALE_POWER = 1.0

MLP_GS_ROLL_DAMP_MULT = 2.0
MLP_GS_PITCH_DAMP_MULT = 2.0
MLP_GS_YAW_DAMP_MULT = 1.5

MLP_GS_DELTAS_RATE_MAX_DPS = 80.0
MLP_GS_DELTAD_RATE_MAX_DPS = 80.0

MLP_GS_COMMAND_SMOOTHING = 0.15

# =============================================================================
# END USER INPUTS
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

    return dynamics.SimConfig(
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


# =============================================================================
# Diagnostics / plots (kept in run_sim)
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

    # Thrust plot if enabled
    if "T_N" in df.columns:
        plt.figure()
        plt.plot(t, df["T_N"], label="Thrust (N)")
        plt.plot(t, df["throttle"], label="Throttle")
        plt.xlabel("t (s)")
        plt.title("Thrust / Throttle")
        plt.grid(True)
        plt.legend()







def add_glide_diagnostics_and_plot(df: pd.DataFrame):
    """
    Adds:
      LD_table = CL/CD
      gamma_deg (positive down, NED)
      glide_ratio_inst = Vh/Vz
    """
    v = df[["vx", "vy", "vz"]].to_numpy(dtype=float)
    Vh = np.linalg.norm(v[:, :2], axis=1)
    Vh = np.maximum(Vh, 1e-9)
    vz = np.maximum(v[:, 2], 1e-9)

    gamma = np.arctan2(v[:, 2], Vh)  # rad, positive down
    df["gamma_deg"] = np.rad2deg(gamma)
    df["glide_ratio_inst"] = Vh / vz
    df["LD_table"] = df["CL"].to_numpy(dtype=float) / np.maximum(df["CD"].to_numpy(dtype=float), 1e-12)

    t = df["t"].to_numpy(dtype=float)
    plt.figure()
    plt.plot(t, df["LD_table"], label="CL/CD (table)")
    plt.xlabel("t (s)")
    plt.ylabel("L/D")
    plt.title("L/D (table)")
    plt.grid(True)
    plt.legend()

    return df

#plotting control stuff

def _wrap_deg_series(a: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def add_heading_tracking_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Adds columns for heading + pitch tracking diagnostics.

    HEADING:
      psi_cmd_deg
      hdg_err_deg (wrapped)
      p_deg_s, r_deg_s  (from p,r rad/s)

    PITCH:
      theta_cmd_deg
      pitch_err_deg
      q_deg_s (from q rad/s)
    """
    # ---------------------------
    # Commands (from cfg)
    # ---------------------------
    psi_cmd = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    theta_cmd = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))

    df["psi_cmd_deg"] = psi_cmd
    df["theta_cmd_deg"] = theta_cmd

    # ---------------------------
    # Errors
    # ---------------------------
    df["hdg_err_deg"] = _wrap_deg_series(
        df["psi_cmd_deg"].to_numpy(dtype=float) - df["yaw_deg"].to_numpy(dtype=float)
    )

    # Pitch error does NOT wrap
    df["pitch_err_deg"] = (
        df["theta_cmd_deg"].to_numpy(dtype=float) - df["pitch_deg"].to_numpy(dtype=float)
    )

    # ---------------------------
    # Rates: your logged p,q,r are rad/s
    # ---------------------------
    if "p" in df.columns:
        df["p_deg_s"] = np.rad2deg(df["p"].to_numpy(dtype=float))
    if "q" in df.columns:
        df["q_deg_s"] = np.rad2deg(df["q"].to_numpy(dtype=float))
    if "r" in df.columns:
        df["r_deg_s"] = np.rad2deg(df["r"].to_numpy(dtype=float))

    return df


def plot_heading_tracking(df: pd.DataFrame):
    """
    High-value plots for heading + pitch tracking:
      1) yaw vs command  + pitch vs command  (two figures)
      2) heading error   + pitch error       (two figures)
      3) roll + roll rate, pitch + pitch rate
      4) deltaD / deltaS / throttle (control effort)
      5) beta (slip proxy) if present
      6) yaw rate if present
    """
    t = df["t"].to_numpy(dtype=float)

    # ---------------------------
    # 1) Yaw vs command
    # ---------------------------
    if "psi_cmd_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["yaw_deg"], label="yaw_deg (psi)")
        plt.plot(t, df["psi_cmd_deg"], label="psi_cmd_deg")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("Heading tracking (yaw vs command)")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 1b) Pitch vs command
    # ---------------------------
    if "theta_cmd_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["pitch_deg"], label="pitch_deg (theta)")
        plt.plot(t, df["theta_cmd_deg"], label="theta_cmd_deg")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("Pitch tracking (pitch vs command)")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 2) Heading error
    # ---------------------------
    if "hdg_err_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["hdg_err_deg"], label="hdg_err_deg (wrapped)")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("Heading error")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 2b) Pitch error
    # ---------------------------
    if "pitch_err_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["pitch_err_deg"], label="pitch_err_deg")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("Pitch error")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 3) Roll response (roll + p)
    # ---------------------------
    plt.figure()
    plt.plot(t, df["roll_deg"], label="roll_deg (phi)")
    if "p_deg_s" in df.columns:
        plt.plot(t, df["p_deg_s"], label="p_deg/s (roll rate)")
    plt.xlabel("t (s)")
    plt.ylabel("deg / deg/s")
    plt.title("Roll response")
    plt.grid(True)
    plt.legend()

    # ---------------------------
    # 3b) Pitch response (pitch + q)
    # ---------------------------
    plt.figure()
    plt.plot(t, df["pitch_deg"], label="pitch_deg (theta)")
    if "q_deg_s" in df.columns:
        plt.plot(t, df["q_deg_s"], label="q_deg/s (pitch rate)")
    plt.xlabel("t (s)")
    plt.ylabel("deg / deg/s")
    plt.title("Pitch response")
    plt.grid(True)
    plt.legend()

    # ---------------------------
    # 4) Control effort
    # ---------------------------
    plt.figure()
    plt.plot(t, df["deltaD_deg"], label="deltaD_deg (diff)")
    plt.plot(t, df["deltaS_deg"], label="deltaS_deg (sym)")
    if "throttle" in df.columns:
        plt.plot(t, df["throttle"], label="throttle")
    plt.xlabel("t (s)")
    plt.ylabel("deg / unit")
    plt.title("Control inputs")
    plt.grid(True)
    plt.legend()

    # ---------------------------
    # 5) Beta (coordination / slip proxy)
    # ---------------------------
    if "beta_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["beta_deg"], label="beta_deg")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("Sideslip (beta)")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 6) Yaw rate
    # ---------------------------
    if "r_deg_s" in df.columns:
        plt.figure()
        plt.plot(t, df["r_deg_s"], label="r_deg/s (yaw rate)")
        plt.xlabel("t (s)")
        plt.ylabel("deg/s")
        plt.title("Yaw rate")
        plt.grid(True)
        plt.legend()


#end plotting control stuff








# --- 3D animation helper (triad) ---
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
               show: bool = False):
    """
    NED -> plot Z-up transform, with consistent rotation transform.
    """
    from matplotlib.animation import FuncAnimation

    d = df.iloc[::max(1, int(stride))].reset_index(drop=True)
    x = d["x"].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    z = d["z"].to_numpy(dtype=float)

    yaw_deg = d["yaw_deg"].to_numpy(dtype=float)
    pitch_deg = d["pitch_deg"].to_numpy(dtype=float)
    roll_deg = d["roll_deg"].to_numpy(dtype=float)

    # NED -> Z-up plotting transform
    T = np.diag([1.0, 1.0, -1.0])
    xp, yp, zp = x, y, -z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Attitude Animation (Z-up plotting frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) [up]")

    base_span = max(
        float(np.max(xp) - np.min(xp)),
        float(np.max(yp) - np.min(yp)),
        float(np.max(zp) - np.min(zp)),
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


# =============================================================================
# Main
# =============================================================================

def main():
    # Ensure this folder is on sys.path so "controllers.*" imports work
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    cfg = build_cfg()

    cfg.heading_cmd_deg = float(HEADING_CMD_DEG)
    cfg.pitch_cmd_deg   = float(PITCH_CMD_DEG)

    #roll/heading stuff
    cfg.hdg_k_psi = float(HDG_K_PSI)
    cfg.roll_k_phi = float(ROLL_K_PHI)
    cfg.roll_k_p = float(ROLL_K_P)
    cfg.phi_max_deg = float(PHI_MAX_DEG)
    cfg.deltaD_max_deg = float(DELTAD_MAX_DEG)
    cfg.yaw_k_r = float(YAW_K_R)

    #pitch stuff
    cfg.pit_k_theta    = float(PIT_K_THETA)
    cfg.pit_k_q        = float(PIT_K_Q)
    cfg.pit_k_qd       = float(PIT_K_QD)
    cfg.theta_max_deg  = float(THETA_MAX_DEG)
    cfg.q_cmd_max_dps  = float(Q_CMD_MAX_DPS)
    cfg.deltaS_max_deg = float(DELTAS_MAX_DEG)

    # ------------------------------------------------------------
    # MLP gain scheduling settings
    # ------------------------------------------------------------
    cfg.use_mlp_gain_scheduling = bool(USE_MLP_GAIN_SCHEDULING)

    cfg.mlp_gs_h_deriv_deg = float(MLP_GS_H_DERIV_DEG)

    cfg.mlp_gs_ref_alpha_deg = float(MLP_GS_REF_ALPHA_DEG)
    cfg.mlp_gs_ref_beta_deg = float(MLP_GS_REF_BETA_DEG)
    cfg.mlp_gs_ref_deltaS_deg = float(MLP_GS_REF_DELTAS_DEG)
    cfg.mlp_gs_ref_deltaD_deg = float(MLP_GS_REF_DELTAD_DEG)

    cfg.mlp_gs_effectiveness_eps = float(MLP_GS_EFFECTIVENESS_EPS)

    cfg.mlp_gs_gain_scale_min = float(MLP_GS_GAIN_SCALE_MIN)
    cfg.mlp_gs_gain_scale_max = float(MLP_GS_GAIN_SCALE_MAX)

    cfg.mlp_gs_print_debug = bool(MLP_GS_PRINT_DEBUG)
    cfg.mlp_gs_print_dt = float(MLP_GS_PRINT_DT)

    cfg.mlp_gs_roll_prop_scale_power = float(MLP_GS_ROLL_PROP_SCALE_POWER)
    cfg.mlp_gs_roll_damp_scale_power = float(MLP_GS_ROLL_DAMP_SCALE_POWER)

    cfg.mlp_gs_pitch_prop_scale_power = float(MLP_GS_PITCH_PROP_SCALE_POWER)
    cfg.mlp_gs_pitch_damp_scale_power = float(MLP_GS_PITCH_DAMP_SCALE_POWER)

    cfg.mlp_gs_roll_damp_mult = float(MLP_GS_ROLL_DAMP_MULT)
    cfg.mlp_gs_pitch_damp_mult = float(MLP_GS_PITCH_DAMP_MULT)
    cfg.mlp_gs_yaw_damp_mult = float(MLP_GS_YAW_DAMP_MULT)

    cfg.mlp_gs_deltaS_rate_max_dps = float(MLP_GS_DELTAS_RATE_MAX_DPS)
    cfg.mlp_gs_deltaD_rate_max_dps = float(MLP_GS_DELTAD_RATE_MAX_DPS)

    cfg.mlp_gs_command_smoothing = float(MLP_GS_COMMAND_SMOOTHING)

    # Make these available to the controller-side alpha/beta clamp
    cfg.max_alpha_deg = float(MAX_ALPHA_DEG)
    cfg.max_beta_deg = float(MAX_BETA_DEG)



    # Load aero table -> build AeroModel
    df = pd.read_csv(CSV_PATH)

    # This must match your existing aero.py implementation signature.
    # If your AeroModel constructor differs, tell me what it is and I’ll adapt it.
    #aero_model = aero.AeroModel(df)


    aero_cfg = aero.AeroConfig(
        # sign flips
        sign_cl=SIGN_CL, sign_cd=SIGN_CD, sign_cy=SIGN_CY,
        sign_cll=SIGN_Cl, sign_cm=SIGN_Cm, sign_cn=SIGN_Cn,

        # cd0
        cd0_add=CD0_ADD,

        # damping
        clp=CLP, cmq=CMQ, cnr=CNR,

        # extrapolation
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
        aero_model = aero.MLPAeroModel(
            model_path=MLP_MODEL_PATH,
            x_scaler_path=X_SCALER_PATH,
            y_scaler_path=Y_SCALER_PATH,
            cfg=aero_cfg,
            device="cpu",
            max_deltaS_deg=DELTAS_MAX_DEG,
            max_deltaD_deg=DELTAD_MAX_DEG,
        )
    else:
        aero_model = aero.AeroModel(
            df,
            cfg=aero_cfg,
            tol=DEFLECTION_TOL,
            use_superposition=USE_SUPERPOSITION
        )

    #aero_model = aero.AeroModel(df, cfg=aero_cfg, tol=DEFLECTION_TOL, use_superposition=USE_SUPERPOSITION)
    # ------------------------------------------------------------
    # Give controller access to aero model for MLP gain scheduling
    # ------------------------------------------------------------
    cfg.aero_model_for_controller = aero_model


    def _estimate_nominal_effectiveness_for_gain_scheduling(aero_model, cfg):
        h = float(cfg.mlp_gs_h_deriv_deg)

        alpha = float(cfg.mlp_gs_ref_alpha_deg)
        beta = float(cfg.mlp_gs_ref_beta_deg)
        dS0 = float(cfg.mlp_gs_ref_deltaS_deg)
        dD0 = float(cfg.mlp_gs_ref_deltaD_deg)

        # dCm/d(deltaS)
        c_sp = aero_model.query(alpha, beta, dS0 + h, dD0)
        c_sm = aero_model.query(alpha, beta, dS0 - h, dD0)
        cm_deltaS_ref = (float(c_sp["Cm"]) - float(c_sm["Cm"])) / (2.0 * h)

        # dCl/d(deltaD)
        c_dp = aero_model.query(alpha, beta, dS0, dD0 + h)
        c_dm = aero_model.query(alpha, beta, dS0, dD0 - h)
        cl_deltaD_ref = (float(c_dp["Cl"]) - float(c_dm["Cl"])) / (2.0 * h)

        return cm_deltaS_ref, cl_deltaD_ref


    cm_deltaS_ref, cl_deltaD_ref = _estimate_nominal_effectiveness_for_gain_scheduling(
        aero_model,
        cfg,
    )

    cfg.mlp_gs_cm_deltaS_ref = float(cm_deltaS_ref)
    cfg.mlp_gs_cl_deltaD_ref = float(cl_deltaD_ref)

    print("\nMLP gain scheduling nominal effectiveness:")
    print(f"  dCm/d(deltaS_deg) = {cfg.mlp_gs_cm_deltaS_ref:+.6f}")
    print(f"  dCl/d(deltaD_deg) = {cfg.mlp_gs_cl_deltaD_ref:+.6f}")

    ctrl_mod = importlib.import_module(CONTROLLER_MODULE)
    controller_fn = ctrl_mod.controller

    df_out = dynamics.simulate(cfg, aero_model, controller_fn)

    # Diagnostics + plots
    df_out = add_glide_diagnostics_and_plot(df_out)

    # Add heading tracking diagnostics (uses cfg.heading_cmd_deg if present)
    df_out = add_heading_tracking_columns(df_out, cfg)

    # Existing plots
    make_plots(df_out)

    # New heading-control plots
    plot_heading_tracking(df_out)





    # Animation
    ani = None
    if DO_ANIMATE:
        print("Starting animation...")
        ani = animate_3d(
            df_out,
            stride=int(ANIM_STRIDE),
            interval_ms=int(1000 * DT * int(ANIM_STRIDE)),
            axis_len=float(ANIM_AXIS_LEN),
            zoom=float(ANIM_ZOOM),
            show=False,
        )

    # SHOW EVERYTHING ONCE
    if SHOW_PLOTS or DO_ANIMATE:
        plt.show()

    # Keep reference (prevents animation GC in some environments)
    return ani


if __name__ == "__main__":
    main()
