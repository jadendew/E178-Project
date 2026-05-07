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

import dynamicsNew as dynamics
import propulsion
import aero  #aero.py

from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# inputs
# =============================================================================

CSV_PATH = r"..\aero_autosweep.csv"
#OUTPUT_CSV = r"sim_out_20cross_withSched4.csv"
OUTPUT_CSV = r"sim_out_basePID_v4.csv"

# Controller module path (must expose controller(t, state, cfg))
CONTROLLER_MODULE = "controllers.heading_pitch_hold_gainScheduledNew"

#open loop: "controllers.open_loop"
#heading 2loop: "controllers.heading_hold"
#heading + pitch: "controllers.heading_pitch_hold"

# Environment
RHO = 1.225
G = 9.80665
WIND_WORLD = [0.0, 5.0, 2.0]  # NED

# Vehicle
MASS = 4 # 4 is base case

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
ROLL_DEG = 0.0
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
HEADING_CMD_DEG = 0.0  # v4: steady heading command, matched to initial yaw
PITCH_CMD_DEG   = 0.0   # v4: base pitch command

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
#HEADING_CMD_DEG = 20.0
#PITCH_CMD_DEG = 0.0

#VEL0_WORLD = [20.0, 0.0, -3.0]
#HEADING_DEG = 0.0
#PITCH_DEG = 8.0
#ROLL_DEG = 20.0
#OMEGA0_BODY = [0.0, 0.0, 0.0]



# ============================================================
#heading, roll gains
# ============================================================

HDG_K_PSI = 0.8  #gain for yaw *error*, converts hdg error into des. roll command

ROLL_K_PHI = 0.8    #gain for roll *error*, determines roll 'snappiness'

PHI_MAX_DEG = 30.0  #max roll limit

DELTAD_MAX_DEG = 10.0   #max commanded deflection

ROLL_K_P = 0.15     # gain for roll *rate*, provides roll rate damping

YAW_K_R = -0.02       # gain for yaw *rate*, provides yaw rate damping

# ============================================================
#pitch gains
# ============================================================

PIT_K_THETA = 3.0       # q_cmd (deg/s) per deg pitch error
PIT_K_Q     = 1.0      # deltaS (deg) per (deg/s) pitch-rate error

PIT_K_QD    = 0.0       # extra direct q damping (deg per deg/s), start 0

THETA_MAX_DEG  = 20.0   # clamp pitch command
Q_CMD_MAX_DPS  = 60.0   # clamp q_cmd (deg/s)
DELTAS_MAX_DEG = 20.0   # max deltaS command

# ============================================================
# gain scheduling settings
# ============================================================

USE_GAIN_SCHEDULING = False # set true for cool things
USE_VELOCITY_SCHEDULING = False

SCHED_V_REF = 20.0
SCHED_V_MIN = 8.0

SCHED_VEL_SCALE_MIN = 0.75
SCHED_VEL_SCALE_MAX = 1.6

DELTA_S_RATE_MAX_DEG_S = 120.0
DELTA_D_RATE_MAX_DEG_S = 120.0
SCHED_VEL_EXP = 0.8

DERIV_FILTER_ALPHA = 0.20
USE_ROLL_DERIVATIVE_DAMPING = False
USE_PITCH_DERIVATIVE_DAMPING = False
USE_YAW_DERIVATIVE_DAMPING = False


# --- Propulsion settings ---

THROTTLE0 = 1.0  # start at 0 for glide, later controller can change
THRUST_ENABLED = True
THRUST_MAX_N =   4.0               # N at throttle=1
THRUST_DIR_BODY = [1.0, 0.0, 0.0]   # +X forward
THRUST_POS_BODY = [-0.6, 0.0, 0.0]  # location in body coords (m)

# Simulation
DT = 0.006
T_FINAL = 15.0

# Time-varying command profile, matched to the MPC v4 case:
#   0-15 s: hold heading steady and command only the pitch-direction sine.
#
# The controller tracks pitch angle, not vertical position directly, so the requested
# 1 m amplitude vertical/pitch-direction sine is converted into an equivalent pitch
# command using the sine-path slope and the nominal forward speed.
USE_COMMAND_PROFILE = True
CMD_SINE_DURATION_S = 15.0
CMD_SINE_FREQ_HZ = 0.5
CMD_SINE_AMPLITUDE_M = 1.0
CMD_SINE_FORWARD_SPEED_MPS = float(np.hypot(VEL0_WORLD[0], VEL0_WORLD[1]))

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
STL_PATH = "lowerqualitymesh.stl"
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)

# Saving outputs
SAVE_PLOTS = True
PLOT_OUTPUT_DIR = "base_PID_plots_v4"
PLOT_DPI = 200

SAVE_ANIMATION = True
ANIMATION_OUTPUT_DIR = "basePID_output_animation_v4"
ANIMATION_FILENAME = "basePID_v4_3d_animation.mp4"  # falls back to .gif if ffmpeg/mp4 saving fails
PITCH_ANIMATION_FILENAME = "basePID_v4_pitch_path_stl_animation.mp4"  # falls back to .gif if ffmpeg/mp4 saving fails
ANIMATION_FPS = 30
ANIMATION_DPI = 140

USE_MLP_AERO = True

MLP_MODEL_PATH = r"..\mlp_aero_state_dictV2.pt"
X_SCALER_PATH = r"..\xScalerV2.pkl"
Y_SCALER_PATH = r"..\yScalerV2.pkl"


# --- chat suggested settings use later
#PIT_K_THETA = 2.0
#PIT_K_Q     = 0.45
#PIT_K_QD    = 0.04

#HDG_K_PSI = 0.7
#ROLL_K_PHI = 0.7
#ROLL_K_P = 0.18
#YAW_K_R = 0.02

#SCHED_V_REF = 20.0
#SCHED_V_MIN = 8.0
#SCHED_VEL_EXP = 1.0

#SCHED_VEL_SCALE_MIN = 0.55
#SCHED_VEL_SCALE_MAX = 1.60

#DELTA_S_RATE_MAX_DEG_S = 80.0
#DELTA_D_RATE_MAX_DEG_S = 80.0
# --- chat suggested settings use later

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


def command_profile_heading_deg(t: float, cfg=None) -> float:
    """
    v4 steady yaw/heading command.

    Heading is held fixed for the entire 15 s run so the response is focused on
    the pitch-direction sine command instead of a yaw maneuver.
    """
    return float(HEADING_CMD_DEG)


def command_profile_pitch_deg(t: float, cfg=None) -> float:
    """
    v4 pitch command for a 15 s, 1 m amplitude, 0.5 Hz pitch-direction sine.

    The controller tracks pitch angle, not vertical position. To represent the
    requested meter-based pitch-direction sine, this converts the vertical sine
    path tangent into an equivalent pitch angle command:

        z_up_cmd = A sin(2*pi*f*t)
        theta_cmd = theta0 + atan2(dz_up_cmd/dt, V_forward)

    where A = CMD_SINE_AMPLITUDE_M.
    """
    theta0 = float(PITCH_CMD_DEG)
    t = float(t)

    if t < 0.0 or t > float(CMD_SINE_DURATION_S):
        return theta0

    amp_m = float(CMD_SINE_AMPLITUDE_M)
    omega = 2.0 * np.pi * float(CMD_SINE_FREQ_HZ)
    v_forward = max(float(CMD_SINE_FORWARD_SPEED_MPS), 1e-6)

    vertical_velocity_cmd = amp_m * omega * np.cos(omega * t)
    pitch_offset_deg = np.rad2deg(np.arctan2(vertical_velocity_cmd, v_forward))
    return theta0 + pitch_offset_deg


def command_profile_position_m(t: float) -> tuple[float, float, float]:
    """
    Diagnostic pitch-direction sine path used for plotting/logging.

    The controller does not directly track this position; it tracks the pitch
    command generated from this sine path slope.

    Returns x_cmd, y_cmd, z_up_cmd in meters.
    """
    t = float(t)
    v_forward = float(CMD_SINE_FORWARD_SPEED_MPS)
    amp_m = float(CMD_SINE_AMPLITUDE_M)

    x_cmd = v_forward * t
    y_cmd = 0.0

    if 0.0 <= t <= float(CMD_SINE_DURATION_S):
        z_up_cmd = amp_m * np.sin(2.0 * np.pi * float(CMD_SINE_FREQ_HZ) * t)
    else:
        z_up_cmd = 0.0

    return x_cmd, y_cmd, z_up_cmd


def make_command_profile_controller(controller_fn, cfg):
    """
    Wrap the existing gain-scheduled controller so the rest of the project does
    not need to be changed. Before every controller call, this updates
    cfg.heading_cmd_deg and cfg.pitch_cmd_deg.
    """
    def controller_with_command_profile(t, state, cfg_in):
        if bool(USE_COMMAND_PROFILE):
            cfg_in.heading_cmd_deg = float(command_profile_heading_deg(t, cfg_in))
            cfg_in.pitch_cmd_deg = float(command_profile_pitch_deg(t, cfg_in))
        return controller_fn(t, state, cfg_in)

    return controller_with_command_profile


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
    # Commands
    # ---------------------------
    if bool(USE_COMMAND_PROFILE):
        t_arr = df["t"].to_numpy(dtype=float)
        df["psi_cmd_deg"] = np.array([command_profile_heading_deg(ti, cfg) for ti in t_arr], dtype=float)
        df["theta_cmd_deg"] = np.array([command_profile_pitch_deg(ti, cfg) for ti in t_arr], dtype=float)
        pos_cmd = np.array([command_profile_position_m(ti) for ti in t_arr], dtype=float)
        df["x_cmd_m"] = pos_cmd[:, 0]
        df["y_cmd_m"] = pos_cmd[:, 1]
        df["z_up_cmd_m"] = pos_cmd[:, 2]
    else:
        psi_cmd = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
        theta_cmd = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))
        df["psi_cmd_deg"] = psi_cmd
        df["theta_cmd_deg"] = theta_cmd

    df["roll_cmd_deg"] = 0.0
    df["altitude_m"] = -df["z"].to_numpy(dtype=float)
    df["x_rel_m"] = df["x"].to_numpy(dtype=float) - float(df["x"].iloc[0])
    df["z_up_rel_m"] = df["altitude_m"].to_numpy(dtype=float) - float(df["altitude_m"].iloc[0])

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

    if "z_up_cmd_m" in df.columns:
        plt.figure()
        plt.plot(t, df["z_up_cmd_m"], label="pitch-direction sine command diagnostic")
        plt.xlabel("t (s)")
        plt.ylabel("z up command (m)")
        plt.title("V4 Pitch-Direction Sine Command Diagnostic")
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

    # ---------------------------
    # 7) Scheduled pitch gains
    # ---------------------------
    if "ctrl_K_THETA" in df.columns and "ctrl_K_Q" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_K_THETA"], label="ctrl_K_THETA")
        plt.plot(t, df["ctrl_K_Q"], label="ctrl_K_Q")
        if "ctrl_K_QD" in df.columns:
            plt.plot(t, df["ctrl_K_QD"], label="ctrl_K_QD")
        plt.xlabel("t (s)")
        plt.ylabel("scheduled gain")
        plt.title("Scheduled Pitch Gains")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 8) Scheduled roll/heading gains
    # ---------------------------
    if "ctrl_K_PSI" in df.columns and "ctrl_K_PHI" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_K_PSI"], label="ctrl_K_PSI")
        plt.plot(t, df["ctrl_K_PHI"], label="ctrl_K_PHI")
        if "ctrl_K_P" in df.columns:
            plt.plot(t, df["ctrl_K_P"], label="ctrl_K_P")
        if "ctrl_K_R" in df.columns:
            plt.plot(t, df["ctrl_K_R"], label="ctrl_K_R")
        plt.xlabel("t (s)")
        plt.ylabel("scheduled gain")
        plt.title("Scheduled Heading/Roll Gains")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 9) Scheduling scale factors
    # ---------------------------
    if "ctrl_vel_scale" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_vel_scale"], label="ctrl_vel_scale")
        if "ctrl_aero_scale" in df.columns:
            plt.plot(t, df["ctrl_aero_scale"], label="ctrl_aero_scale")
        if "ctrl_pitch_err_scale" in df.columns:
            plt.plot(t, df["ctrl_pitch_err_scale"], label="ctrl_pitch_err_scale")
        if "ctrl_roll_err_scale" in df.columns:
            plt.plot(t, df["ctrl_roll_err_scale"], label="ctrl_roll_err_scale")
        if "ctrl_heading_err_scale" in df.columns:
            plt.plot(t, df["ctrl_heading_err_scale"], label="ctrl_heading_err_scale")
        plt.xlabel("t (s)")
        plt.ylabel("scale factor")
        plt.title("Gain Scheduling Scale Factors")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 10) Internal controller commands
    # ---------------------------
    if "ctrl_phi_cmd_deg" in df.columns and "ctrl_q_cmd_deg_s" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_phi_cmd_deg"], label="phi_cmd_deg")
        plt.plot(t, df["ctrl_q_cmd_deg_s"], label="q_cmd_deg_s")
        plt.xlabel("t (s)")
        plt.ylabel("deg / deg/s")
        plt.title("Internal Controller Commands")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 11) Rate limiting effect on deltaS
    # ---------------------------
    if "ctrl_deltaS_unsmoothed_deg" in df.columns and "ctrl_deltaS_final_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_deltaS_unsmoothed_deg"], label="deltaS before rate limit")
        plt.plot(t, df["ctrl_deltaS_final_deg"], label="deltaS after rate limit")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("deltaS Rate Limiting Effect")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # 12) Rate limiting effect on deltaD
    # ---------------------------
    if "ctrl_deltaD_unsmoothed_deg" in df.columns and "ctrl_deltaD_final_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_deltaD_unsmoothed_deg"], label="deltaD before rate limit")
        plt.plot(t, df["ctrl_deltaD_final_deg"], label="deltaD after rate limit")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("deltaD Rate Limiting Effect")
        plt.grid(True)
        plt.legend()
            # ---------------------------
    # Saturation / clamp effect: deltaS
    # ---------------------------
    if "ctrl_deltaS_raw_no_clamp_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_deltaS_raw_no_clamp_deg"], label="deltaS raw no clamp")
        plt.plot(t, df["ctrl_deltaS_unsmoothed_deg"], label="deltaS after clamp")
        plt.plot(t, df["ctrl_deltaS_final_deg"], label="deltaS final after rate limit")
        plt.plot(t, df["deltaS_deg"], "--", label="deltaS sent to sim")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("deltaS Clamp / Rate Limit Effect")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # Saturation / clamp effect: deltaD
    # ---------------------------
    if "ctrl_deltaD_raw_no_clamp_deg" in df.columns:
        plt.figure()
        plt.plot(t, df["ctrl_deltaD_raw_no_clamp_deg"], label="deltaD raw no clamp")
        plt.plot(t, df["ctrl_deltaD_unsmoothed_deg"], label="deltaD after clamp")
        plt.plot(t, df["ctrl_deltaD_final_deg"], label="deltaD final after rate limit")
        plt.plot(t, df["deltaD_deg"], "--", label="deltaD sent to sim")
        plt.xlabel("t (s)")
        plt.ylabel("deg")
        plt.title("deltaD Clamp / Rate Limit Effect")
        plt.grid(True)
        plt.legend()
    # ---------------------------
    # Damping-specific scheduled gains
    # ---------------------------
    damping_gain_cols = [
        "ctrl_K_P",
        "ctrl_K_R",
        "ctrl_K_QD",
    ]

    existing = [c for c in damping_gain_cols if c in df.columns]

    if existing:
        plt.figure()
        for c in existing:
            plt.plot(t, df[c], label=c)
        plt.xlabel("t (s)")
        plt.ylabel("damping gain")
        plt.title("Scheduled Damping Gains")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # Derivative-based damping scale factors
    # ---------------------------
    derivative_damping_cols = [
        "ctrl_roll_accel_damp_scale",
        "ctrl_pitch_accel_damp_scale",
        "ctrl_yaw_accel_damp_scale",
    ]

    existing = [c for c in derivative_damping_cols if c in df.columns]

    if existing:
        plt.figure()
        for c in existing:
            plt.plot(t, df[c], label=c)
        plt.xlabel("t (s)")
        plt.ylabel("scale factor")
        plt.title("Derivative-Based Damping Scale Factors")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # Yaw derivative damping diagnostics
    # ---------------------------
    if "ctrl_r_dot_deg_s2" in df.columns:
        plt.figure()
        if "r_deg_s" in df.columns:
            plt.plot(t, df["r_deg_s"], label="r_deg_s")
        plt.plot(t, df["ctrl_r_dot_deg_s2"], label="r_dot_deg_s2")
        if "ctrl_yaw_accel_damp_scale" in df.columns:
            plt.plot(t, df["ctrl_yaw_accel_damp_scale"], label="yaw accel damp scale")
        if "ctrl_K_R" in df.columns:
            plt.plot(t, df["ctrl_K_R"], label="ctrl_K_R")
        plt.xlabel("t (s)")
        plt.ylabel("rate / derivative / gain")
        plt.title("Yaw Derivative-Based Damping")
        plt.grid(True)
        plt.legend()

    # ---------------------------
    # Pitch derivative damping diagnostics
    # ---------------------------
    if "ctrl_q_dot_deg_s2" in df.columns:
        plt.figure()
        if "q_deg_s" in df.columns:
            plt.plot(t, df["q_deg_s"], label="q_deg_s")
        plt.plot(t, df["ctrl_q_dot_deg_s2"], label="q_dot_deg_s2")
        if "ctrl_pitch_accel_damp_scale" in df.columns:
            plt.plot(t, df["ctrl_pitch_accel_damp_scale"], label="pitch accel damp scale")
        if "ctrl_K_QD" in df.columns:
            plt.plot(t, df["ctrl_K_QD"], label="ctrl_K_QD")
        plt.xlabel("t (s)")
        plt.ylabel("rate / derivative / gain")
        plt.title("Pitch Derivative-Based Damping")
        plt.grid(True)
        plt.legend()


#end plotting control stuff









def save_all_open_figures(output_dir: str, dpi: int = 200):
    """
    Save every currently open matplotlib figure as PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for num in plt.get_fignums():
        fig = plt.figure(num)

        title = ""
        try:
            title = fig.canvas.manager.get_window_title()
        except Exception:
            title = ""

        if not title or title.lower().startswith("figure"):
            axes = fig.get_axes()
            if axes:
                title = axes[0].get_title()

        if not title:
            title = f"figure_{num:02d}"

        safe = "".join(
            ch if ch.isalnum() or ch in (" ", "_", "-") else "_"
            for ch in title
        )
        safe = "_".join(safe.strip().split())

        if not safe:
            safe = f"figure_{num:02d}"

        path = out_dir / f"{num:02d}_{safe}.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        saved_paths.append(path)

    print(f"Saved {len(saved_paths)} plot figure(s) to: {out_dir.resolve()}")

    return saved_paths


def save_animation_file(
    anim,
    output_dir: str,
    filename: str,
    fps: int = 30,
    dpi: int = 140,
):
    """
    Save the matplotlib animation.

    Preferred:
        .mp4 using ffmpeg

    Fallback:
        .gif using pillow
    """
    if anim is None:
        print("No animation object to save.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_path = out_dir / filename
    suffix = requested_path.suffix.lower()

    if suffix not in [".mp4", ".gif"]:
        requested_path = requested_path.with_suffix(".mp4")
        suffix = ".mp4"

    try:
        if suffix == ".gif":
            from matplotlib.animation import PillowWriter

            anim.save(
                requested_path,
                writer=PillowWriter(fps=int(fps)),
                dpi=int(dpi),
            )
        else:
            from matplotlib.animation import FFMpegWriter

            anim.save(
                requested_path,
                writer=FFMpegWriter(fps=int(fps), bitrate=1800),
                dpi=int(dpi),
            )

        print(f"Saved animation to: {requested_path.resolve()}")
        return requested_path

    except Exception as e:
        print(f"Could not save animation as {requested_path.name}: {e}")

        fallback_path = requested_path.with_suffix(".gif")

        try:
            from matplotlib.animation import PillowWriter

            anim.save(
                fallback_path,
                writer=PillowWriter(fps=int(fps)),
                dpi=int(dpi),
            )

            print(f"Saved animation fallback to: {fallback_path.resolve()}")
            return fallback_path

        except Exception as e2:
            print(f"Animation save failed completely: {e2}")
            return None



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
               show: bool = False,
               stl_path: str | None = None,
               stl_scale: float = 1.0,
               stl_offset_body=(0.0, 0.0, 0.0),
               stl_rotation_offset_deg=(0.0, 0.0, 0.0)):
    """
    3D attitude animation with optional animated STL body.

    Required dataframe columns:
        x, y, z, yaw_deg, pitch_deg, roll_deg

    Coordinate convention:
        - Input positions are assumed to be NED.
        - Plotting uses Z-up:
              x_plot = x
              y_plot = y
              z_plot = -z

    STL convention:
        - STL is loaded using numpy-stl:
              from stl import mesh
        - STL vertices are assumed to be in the aircraft/body frame.
        - Body +X is assumed to be forward.
        - Use stl_scale if your STL is in mm but sim is in meters.
        - Use stl_rotation_offset_deg if the STL appears sideways/backwards.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if stl_path is not None:
        from stl import mesh

    # --------------------------------------------------
    # Rotation helpers
    # --------------------------------------------------

    def Rx(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ])

    def Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ])

    def Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def body_to_ned_rotation(yaw_deg_i, pitch_deg_i, roll_deg_i):
        """
        Aerospace Z-Y-X Euler rotation.

        Converts body-frame vectors into NED-frame vectors:
            v_ned = R_body_to_ned @ v_body
        """
        yaw = np.deg2rad(yaw_deg_i)
        pitch = np.deg2rad(pitch_deg_i)
        roll = np.deg2rad(roll_deg_i)

        return Rz(yaw) @ Ry(pitch) @ Rx(roll)

    def stl_alignment_rotation(offset_deg):
        """
        Fixed STL-to-body alignment rotation.

        Input:
            stl_rotation_offset_deg = (roll_offset, pitch_offset, yaw_offset)

        Use this if the STL's native axes do not match your aircraft body axes.
        """
        roll_o, pitch_o, yaw_o = np.deg2rad(offset_deg)

        return Rz(yaw_o) @ Ry(pitch_o) @ Rx(roll_o)

    # --------------------------------------------------
    # Downsample data
    # --------------------------------------------------

    d = df.iloc[::max(1, int(stride))].reset_index(drop=True)

    x = d["x"].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    z = d["z"].to_numpy(dtype=float)

    yaw_deg = d["yaw_deg"].to_numpy(dtype=float)
    pitch_deg = d["pitch_deg"].to_numpy(dtype=float)
    roll_deg = d["roll_deg"].to_numpy(dtype=float)

    # NED -> Z-up plotting transform
    T_ned_to_plot = np.diag([1.0, 1.0, -1.0])

    xp = x
    yp = y
    zp = -z

    # --------------------------------------------------
    # Load STL using numpy-stl
    # --------------------------------------------------

    stl_vectors_body = None
    stl_poly = None

    if stl_path is not None:
        stl_mesh = mesh.Mesh.from_file(stl_path)

        # Same STL triangle format as your example:
        # shape = (number_of_triangles, 3, 3)
        stl_vectors_body = stl_mesh.vectors.copy().astype(float)

        # --------------------------------------------------
        # STL triangle downsampling for faster animation
        # --------------------------------------------------
        max_stl_triangles = 1000  # try 1000, 3000, 5000, or 10000

        num_triangles = stl_vectors_body.shape[0]

        if num_triangles > max_stl_triangles:
            keep_idx = np.linspace(
                0,
                num_triangles - 1,
                max_stl_triangles
            ).astype(int)

            stl_vectors_body = stl_vectors_body[keep_idx]

            print(
                f"Downsampled STL from {num_triangles} triangles "
                f"to {stl_vectors_body.shape[0]} triangles for animation."
            )

        # Center STL around its centroid so rotation is about the vehicle center.
        stl_center = stl_vectors_body.reshape(-1, 3).mean(axis=0)
        stl_vectors_body -= stl_center

        # Scale STL.
        # If STL is in mm and sim is in m, use stl_scale=0.001.
        stl_vectors_body *= float(stl_scale)

        # Apply fixed STL/body alignment correction.
        R_stl_offset = stl_alignment_rotation(stl_rotation_offset_deg)

        stl_vectors_body = np.einsum(
            "ij,tkj->tki",
            R_stl_offset,
            stl_vectors_body
        )

        # Optional fixed offset in body frame.
        stl_offset_body = np.asarray(stl_offset_body, dtype=float)
        stl_vectors_body += stl_offset_body

    # --------------------------------------------------
    # Figure setup
    # --------------------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("3D Attitude Animation with STL")
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

    trail, = ax.plot([], [], [], linewidth=1, color="k")

    x_line, = ax.plot([], [], [], linewidth=2, color="r")
    y_line, = ax.plot([], [], [], linewidth=2, color="g")
    z_line, = ax.plot([], [], [], linewidth=2, color="b")

    ax.legend(
        handles=[x_line, y_line, z_line],
        labels=["Body +X", "Body +Y", "Body +Z"],
        loc="upper left"
    )

    if stl_vectors_body is not None:
        stl_poly = Poly3DCollection(stl_vectors_body, alpha=0.7)
        stl_poly.set_edgecolor("k")
        ax.add_collection3d(stl_poly)

    ax.set_xlim(xp[0] - pad, xp[0] + pad)
    ax.set_ylim(yp[0] - pad, yp[0] + pad)
    ax.set_zlim(zp[0] - pad, zp[0] + pad)

    try:
        _set_axes_equal(ax)
    except NameError:
        pass

    # Body-frame unit vectors
    ex_body = np.array([1.0, 0.0, 0.0])
    ey_body = np.array([0.0, 1.0, 0.0])
    ez_body = np.array([0.0, 0.0, -1.0])

    # --------------------------------------------------
    # Animation update function
    # --------------------------------------------------

    def update(i):
        pos_ned = np.array([x[i], y[i], z[i]], dtype=float)
        pos_plot = T_ned_to_plot @ pos_ned

        R_body_to_ned = body_to_ned_rotation(
            yaw_deg[i],
            pitch_deg[i],
            roll_deg[i]
        )

        # Body axes transformed into Z-up plot frame
        ex_plot = T_ned_to_plot @ (R_body_to_ned @ ex_body)
        ey_plot = T_ned_to_plot @ (R_body_to_ned @ ey_body)
        ez_plot = T_ned_to_plot @ (R_body_to_ned @ ez_body)

        # Update trail
        i0 = max(0, i - int(trail_len))

        trail.set_data(xp[i0:i + 1], yp[i0:i + 1])
        trail.set_3d_properties(zp[i0:i + 1])

        # Update body +X axis
        x_line.set_data(
            [pos_plot[0], pos_plot[0] + axis_len * ex_plot[0]],
            [pos_plot[1], pos_plot[1] + axis_len * ex_plot[1]]
        )
        x_line.set_3d_properties(
            [pos_plot[2], pos_plot[2] + axis_len * ex_plot[2]]
        )

        # Update body +Y axis
        y_line.set_data(
            [pos_plot[0], pos_plot[0] + axis_len * ey_plot[0]],
            [pos_plot[1], pos_plot[1] + axis_len * ey_plot[1]]
        )
        y_line.set_3d_properties(
            [pos_plot[2], pos_plot[2] + axis_len * ey_plot[2]]
        )

        # Update body +Z axis
        z_line.set_data(
            [pos_plot[0], pos_plot[0] + axis_len * ez_plot[0]],
            [pos_plot[1], pos_plot[1] + axis_len * ez_plot[1]]
        )
        z_line.set_3d_properties(
            [pos_plot[2], pos_plot[2] + axis_len * ez_plot[2]]
        )

        # Update STL vehicle body
        if stl_poly is not None:
            # STL/body frame -> NED
            stl_vectors_ned = np.einsum(
                "ij,tkj->tki",
                R_body_to_ned,
                stl_vectors_body
            )

            # NED -> Z-up plotting frame
            stl_vectors_plot = np.einsum(
                "ij,tkj->tki",
                T_ned_to_plot,
                stl_vectors_ned
            )

            # Translate to current vehicle position
            stl_vectors_plot += pos_plot

            # Update STL triangles
            stl_poly.set_verts(stl_vectors_plot)

        # Follow vehicle with view window
        ax.set_xlim(pos_plot[0] - pad, pos_plot[0] + pad)
        ax.set_ylim(pos_plot[1] - pad, pos_plot[1] + pad)
        ax.set_zlim(pos_plot[2] - pad, pos_plot[2] + pad)

        try:
            _set_axes_equal(ax)
        except NameError:
            pass

        artists = [trail, x_line, y_line, z_line]

        if stl_poly is not None:
            artists.append(stl_poly)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(d),
        interval=interval_ms,
        blit=False,
        cache_frame_data=False
    )

    if show:
        plt.show()

    return anim

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



def animate_pitch_path_stl(df: pd.DataFrame, stride=10, interval_ms=30, trail_len=200,
                           zoom=12.0, stl_path=None, stl_scale=1.0,
                           stl_rotation_offset_deg=(0.0, 0.0, 0.0)):
    """
    Second v4 animation matching the MPC v4 style:
      - world-frame side view along y
      - blue dashed commanded pitch trajectory
      - STL aircraft flying along the actual path
      - glider sink removed from the plotted z motion
      - x-window scrolls with the aircraft centered at +/- 25 m
      - z-window scrolls with the aircraft centered at +/- 5 m
    """
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

    t = d["t"].to_numpy(float)
    x = d["x"].to_numpy(float)
    y = d["y"].to_numpy(float)
    z = d["z"].to_numpy(float)
    yaw = d["yaw_deg"].to_numpy(float)
    pitch = d["pitch_deg"].to_numpy(float)
    roll = d["roll_deg"].to_numpy(float)

    psi_cmd = d["psi_cmd_deg"].to_numpy(float) if "psi_cmd_deg" in d.columns else np.full(len(d), float(HEADING_CMD_DEG))
    x_cmd_local = d["x_cmd_m"].to_numpy(float) if "x_cmd_m" in d.columns else float(CMD_SINE_FORWARD_SPEED_MPS) * t
    y_cmd_local = d["y_cmd_m"].to_numpy(float) if "y_cmd_m" in d.columns else np.zeros(len(d), dtype=float)
    z_cmd_up = d["z_up_cmd_m"].to_numpy(float) if "z_up_cmd_m" in d.columns else np.zeros(len(d), dtype=float)

    x0, y0, z0 = float(x[0]), float(y[0]), float(z[0])
    xp = x - x0
    yp = y - y0
    zp_raw = -(z - z0)

    if len(t) >= 2 and (t[-1] - t[0]) > 1e-12:
        sink_rate = (zp_raw[-1] - zp_raw[0]) / (t[-1] - t[0])
        sink_trend = sink_rate * (t - t[0])
    else:
        sink_trend = np.zeros_like(zp_raw)
    zp = zp_raw - sink_trend

    psi_cmd_rad = np.deg2rad(psi_cmd)
    x_cmd_world = x_cmd_local * np.cos(psi_cmd_rad) - y_cmd_local * np.sin(psi_cmd_rad)
    y_cmd_world = x_cmd_local * np.sin(psi_cmd_rad) + y_cmd_local * np.cos(psi_cmd_rad)
    z_cmd_plot = z_cmd_up

    T_ned_to_plot = np.diag([1.0, 1.0, -1.0])

    stl_vectors_body = None
    if stl_path is not None and os.path.isfile(stl_path):
        from stl import mesh
        stl_mesh = mesh.Mesh.from_file(stl_path)
        stl_vectors_body = stl_mesh.vectors.copy().astype(float)
        max_tris = 1000
        if stl_vectors_body.shape[0] > max_tris:
            keep = np.linspace(0, stl_vectors_body.shape[0] - 1, max_tris).astype(int)
            stl_vectors_body = stl_vectors_body[keep]
            print(f"Downsampled STL to {stl_vectors_body.shape[0]} triangles for pitch-path animation.")
        stl_vectors_body -= stl_vectors_body.reshape(-1, 3).mean(axis=0)
        stl_vectors_body *= float(stl_scale)
        ro, po, yo = np.deg2rad(stl_rotation_offset_deg)
        R_offset = Rz(yo) @ Ry(po) @ Rx(ro)
        stl_vectors_body = np.einsum("ij,tkj->tki", R_offset, stl_vectors_body)

    fig = plt.figure(figsize=(18, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Gain-Scheduled V4 Pitch Sine Command Path vs Aircraft Response")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_proj_type("ortho")

    ax.plot(
        x_cmd_world, y_cmd_world, z_cmd_plot,
        color="tab:blue", linestyle="--", linewidth=1.3, alpha=0.35,
        label="commanded path"
    )
    cmd_line, = ax.plot(
        [], [], [],
        color="tab:blue", linestyle="--", linewidth=2.2,
        label="commanded path generated"
    )
    trail, = ax.plot(
        [], [], [],
        color="tab:orange", linewidth=1.5,
        label="actual flight path"
    )
    cmd_point, = ax.plot(
        [], [], [], marker="o", color="tab:blue", linestyle="None",
        label="command point"
    )

    x_line, = ax.plot([], [], [], linewidth=1.0, label="Body +X")
    y_line, = ax.plot([], [], [], linewidth=1.0, label="Body +Y")
    z_line, = ax.plot([], [], [], linewidth=1.0, label="Body +Z")

    stl_poly = None
    if stl_vectors_body is not None:
        stl_poly = Poly3DCollection(stl_vectors_body, alpha=0.7)
        stl_poly.set_edgecolor("k")
        ax.add_collection3d(stl_poly)

    ax.legend(loc="upper left")

    ex_body = np.array([1.0, 0.0, 0.0])
    ey_body = np.array([0.0, 1.0, 0.0])
    ez_body = np.array([0.0, 0.0, -1.0])

    y_all = np.r_[yp, y_cmd_world]
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    y_span = max(y_max - y_min, 1.0)
    y_pad = max(0.30, 0.15 * y_span)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    x_half_window = 25.0
    z_half_window = 5.0
    ax.set_box_aspect((18.0, 1.0, 3.6))
    ax.view_init(elev=0.0, azim=-90.0)

    axis_len = 0.22

    def update(i):
        pos_ned = np.array([xp[i], yp[i], -zp[i]], dtype=float)
        pos_plot = T_ned_to_plot @ pos_ned
        R_bw = body_to_ned_rotation(yaw[i], pitch[i], roll[i])

        axes = [
            T_ned_to_plot @ (R_bw @ ex_body),
            T_ned_to_plot @ (R_bw @ ey_body),
            T_ned_to_plot @ (R_bw @ ez_body),
        ]
        for line, axis_vec in zip([x_line, y_line, z_line], axes):
            tip = pos_plot + axis_len * axis_vec
            line.set_data([pos_plot[0], tip[0]], [pos_plot[1], tip[1]])
            line.set_3d_properties([pos_plot[2], tip[2]])

        i0 = max(0, i - int(trail_len))
        trail.set_data(xp[i0:i + 1], yp[i0:i + 1])
        trail.set_3d_properties(zp[i0:i + 1])

        cmd_line.set_data(x_cmd_world[:i + 1], y_cmd_world[:i + 1])
        cmd_line.set_3d_properties(z_cmd_plot[:i + 1])
        cmd_point.set_data([x_cmd_world[i]], [y_cmd_world[i]])
        cmd_point.set_3d_properties([z_cmd_plot[i]])

        if stl_poly is not None:
            stl_ned = np.einsum("ij,tkj->tki", R_bw, stl_vectors_body)
            stl_plot = np.einsum("ij,tkj->tki", T_ned_to_plot, stl_ned) + pos_plot
            stl_poly.set_verts(stl_plot)

        ax.set_xlim(pos_plot[0] - x_half_window, pos_plot[0] + x_half_window)
        ax.set_zlim(pos_plot[2] - z_half_window, pos_plot[2] + z_half_window)

        return [cmd_line, trail, cmd_point, x_line, y_line, z_line] + ([] if stl_poly is None else [stl_poly])

    anim = FuncAnimation(
        fig, update, frames=len(d), interval=int(interval_ms),
        blit=False, cache_frame_data=False
    )
    return anim



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

    # gain scheduling stuff
    cfg.use_gain_scheduling = bool(USE_GAIN_SCHEDULING)
    cfg.sched_v_ref = float(SCHED_V_REF)
    cfg.sched_v_min = float(SCHED_V_MIN)
    cfg.sched_vel_scale_min = float(SCHED_VEL_SCALE_MIN)
    cfg.sched_vel_scale_max = float(SCHED_VEL_SCALE_MAX)
    cfg.deltaS_rate_max_deg_s = float(DELTA_S_RATE_MAX_DEG_S)
    cfg.deltaD_rate_max_deg_s = float(DELTA_D_RATE_MAX_DEG_S)
    cfg.sched_vel_exp = float(SCHED_VEL_EXP)
    cfg.deriv_filter_alpha = float(DERIV_FILTER_ALPHA)
    cfg.use_roll_derivative_damping = bool(USE_ROLL_DERIVATIVE_DAMPING)
    cfg.use_pitch_derivative_damping = bool(USE_PITCH_DERIVATIVE_DAMPING)
    cfg.use_yaw_derivative_damping = bool(USE_YAW_DERIVATIVE_DAMPING)
    cfg.use_velocity_scheduling = bool(USE_VELOCITY_SCHEDULING)



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


    ctrl_mod = importlib.import_module(CONTROLLER_MODULE)
    controller_fn = ctrl_mod.controller
    controller_fn = make_command_profile_controller(controller_fn, cfg)

    df_out = dynamics.simulate(cfg, aero_model, controller_fn)

    # Diagnostics + plots
    df_out = add_glide_diagnostics_and_plot(df_out)

    # Add heading tracking diagnostics (uses cfg.heading_cmd_deg if present)
    df_out = add_heading_tracking_columns(df_out, cfg)

    # Existing plots
    make_plots(df_out)

    # New heading-control plots
    plot_heading_tracking(df_out)

    sat_cols = [
    "ctrl_q_cmd_saturated",
    "ctrl_phi_cmd_saturated",
    "ctrl_deltaS_saturated",
    "ctrl_deltaD_saturated",
    ]

    print("\nSaturation fractions:")
    for c in sat_cols:
        if c in df_out.columns:
            print(f"{c}: {df_out[c].mean():.3f}")

    # Animations
    ani_3d = None
    ani_pitch = None
    if DO_ANIMATE or SAVE_ANIMATION:
        print("Starting animations...")
        anim_interval_ms = int(1000 * DT * int(ANIM_STRIDE))
        ani_3d = animate_3d(
            df_out,
            stride=int(ANIM_STRIDE),
            interval_ms=anim_interval_ms,
            axis_len=float(ANIM_AXIS_LEN),
            zoom=float(ANIM_ZOOM),
            show=False,
            stl_path=STL_PATH,
            stl_scale=STL_SCALE,
            stl_rotation_offset_deg=STL_ROTATION_OFFSET_DEG
        )

        ani_pitch = animate_pitch_path_stl(
            df_out,
            stride=int(ANIM_STRIDE),
            interval_ms=anim_interval_ms,
            zoom=float(ANIM_ZOOM),
            stl_path=STL_PATH,
            stl_scale=STL_SCALE,
            stl_rotation_offset_deg=STL_ROTATION_OFFSET_DEG,
        )

    # Save plots and animations before plt.show().
    if SAVE_PLOTS:
        save_all_open_figures(PLOT_OUTPUT_DIR, dpi=PLOT_DPI)

    if SAVE_ANIMATION:
        save_animation_file(
            ani_3d,
            output_dir=ANIMATION_OUTPUT_DIR,
            filename=ANIMATION_FILENAME,
            fps=ANIMATION_FPS,
            dpi=ANIMATION_DPI,
        )
        save_animation_file(
            ani_pitch,
            output_dir=ANIMATION_OUTPUT_DIR,
            filename=PITCH_ANIMATION_FILENAME,
            fps=ANIMATION_FPS,
            dpi=ANIMATION_DPI,
        )

    # SHOW EVERYTHING ONCE
    if SHOW_PLOTS or DO_ANIMATE:
        plt.show()

    # Keep references (prevents animation GC in some environments)
    return ani_3d, ani_pitch


if __name__ == "__main__":
    main()
