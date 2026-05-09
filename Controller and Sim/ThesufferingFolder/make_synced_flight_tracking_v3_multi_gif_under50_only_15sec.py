#!/usr/bin/env python3
"""
make_synced_flight_tracking_v3_multi_gif.py

V3 version of the synchronized flight/tracking animation script.

It uses one simulation CSV as the single source of truth, then creates:

1. Combined flight + tracking animation:
      mpc_v3_csv_synced_flight_tracking.mp4
      mpc_v3_csv_synced_flight_tracking.gif

2. Plot-only tracking GIF:
      mpc_v3_tracking_plots_only.gif

3. 3D flight-only GIF:
      mpc_v3_3d_flight_only.gif

4. Downsampled / smaller combined GIF:
      mpc_v3_csv_synced_flight_tracking_small.gif

Default input:
      mpc_output_plots_v3/sim_out_MLP_MPC_v3.csv

If that file is not found, it also tries:
      sim_out_MLP_MPC_v3.csv

Optional:
      lowerqualitymesh.stl
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# User settings
# =============================================================================

CSV_PATH = "mpc_output_plots_v3/sim_out_MLP_MPC_v3.csv"
CSV_FALLBACK_PATH = "sim_out_MLP_MPC_v3.csv"

OUTPUT_DIR = "mpc_output_plots_v3/synced_csv_animation_v3"
OUTPUT_BASENAME = "mpc_v3_csv_synced_flight_tracking"

# Optional STL
STL_PATH = "lowerqualitymesh.stl"
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)
MAX_STL_TRIANGLES = 1000

# Animation timing
FPS = 30
GIF_FPS = 15
DPI_MP4 = 150
DPI_GIF = 110

# Smaller combined GIF settings
SMALL_GIF_FPS = 10
SMALL_GIF_DPI = 72

# Extra optimized GIF exports.
# The script will create additional GIFs that are iteratively reduced until
# each is below this target size, when possible.
TARGET_GIF_SIZE_MB = 49.0

# Only the extra under50/undersampled GIFs use this shorter duration.
# The normal MP4/GIF exports still use the regular SPEEDUP setting.
UNDER50_TARGET_TOTAL_DURATION_SECONDS = 15.0

# CSV time range to animate. Set to None for full CSV.
T_START = 0.0
T_END = None

# Set to 1.0 for real-time. Use 1.5 or 2.0 for shorter exported animations.
SPEEDUP = 1.0

# Layout for combined animation
FIGSIZE = (15.5, 7.8)
LEFT_RIGHT_WIDTH_RATIO = [1.0, 1.18]
SUBPLOT_WSPACE = 0.12
SUBPLOT_HSPACE = 0.24
FIG_TOP = 0.90
FIG_BOTTOM = 0.10
FIG_LEFT = 0.05
FIG_RIGHT = 0.985

# Separate plot-only layout
PLOT_ONLY_FIGSIZE = (9.5, 7.2)

# Separate 3D-only layout
FLIGHT_ONLY_FIGSIZE = (8.5, 7.2)

# Flight view
TRAIL_LEN_SECONDS = 8.0
AXIS_LEN = 1.0
FOLLOW_MODE = True
FOLLOW_RADIUS = 9.0
FIXED_PADDING = 8.0
VIEW_ELEV = 22
VIEW_AZIM = -55

# Tracking plot
SHOW_ERROR_PANEL = True
TRACKING_HISTORY_MODE = "full"  # "full" or "window"
TRACKING_WINDOW_SECONDS = 12.0


# =============================================================================
# Math helpers
# =============================================================================

def wrap_deg(angle):
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def Rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]], dtype=float)


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def body_to_ned_rotation(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def stl_alignment_rotation(offset_deg):
    roll_o, pitch_o, yaw_o = np.deg2rad(offset_deg)
    return Rz(yaw_o) @ Ry(pitch_o) @ Rx(roll_o)


def ned_to_plot(points_ned):
    T = np.diag([1.0, 1.0, -1.0])
    return np.einsum("ij,...j->...i", T, points_ned)


def set_axes_equal_around(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def auto_ylim(*arrays, pad_frac=0.12, min_span=5.0):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -1.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, float(min_span))
    pad = pad_frac * span
    return lo - pad, hi + pad


# =============================================================================
# Data/model loading
# =============================================================================

def resolve_csv_path():
    primary = Path(CSV_PATH)
    fallback = Path(CSV_FALLBACK_PATH)

    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"Could not find CSV at either:\n"
        f"  {primary}\n"
        f"  {fallback}\n"
        f"Edit CSV_PATH at the top of this script if your v3 CSV has a different name."
    )


def load_csv_data():
    csv_path = resolve_csv_path()
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["t", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.sort_values("t").drop_duplicates(subset=["t"], keep="last").reset_index(drop=True)

    if T_START is not None:
        df = df[df["t"] >= float(T_START)].copy()
    if T_END is not None:
        df = df[df["t"] <= float(T_END)].copy()

    df = df.reset_index(drop=True)

    # Command columns with MPC v3 fallback:
    #   0-5 s:   heading = 50 deg, pitch = 3 deg
    #   5-20 s:  heading = 50 deg, pitch follows the v3 sine-derived pitch command
    #   20-30 s: heading = 140 deg, pitch = 3 deg
    #   roll = 0 deg
    if "psi_cmd_deg" not in df.columns:
        t_arr = df["t"].to_numpy(dtype=float)
        df["psi_cmd_deg"] = np.where(t_arr < 20.0, 50.0, 140.0)

    if "theta_cmd_deg" not in df.columns:
        t_arr = df["t"].to_numpy(dtype=float)
        theta0 = 3.0
        sine_start = 5.0
        sine_end = 20.0
        sine_freq_hz = 0.25
        sine_amp_m = 1.0
        forward_speed_mps = float(np.hypot(20.0, 0.0))
        theta_cmd = np.full(len(df), theta0, dtype=float)
        mask = (t_arr >= sine_start) & (t_arr < sine_end)
        tau = t_arr[mask] - sine_start
        omega = 2.0 * np.pi * sine_freq_hz
        vertical_velocity_cmd = sine_amp_m * omega * np.cos(omega * tau)
        pitch_offset_deg = np.rad2deg(np.arctan2(vertical_velocity_cmd, max(forward_speed_mps, 1e-6)))
        theta_cmd[mask] = theta0 + pitch_offset_deg
        df["theta_cmd_deg"] = theta_cmd

    if "roll_cmd_deg" not in df.columns:
        df["roll_cmd_deg"] = 0.0

    df["hdg_err_deg"] = wrap_deg(df["psi_cmd_deg"].to_numpy() - df["yaw_deg"].to_numpy())
    df["pitch_err_deg"] = df["theta_cmd_deg"].to_numpy() - df["pitch_deg"].to_numpy()
    df["roll_err_deg"] = wrap_deg(df["roll_cmd_deg"].to_numpy() - df["roll_deg"].to_numpy())

    return df


def compute_speedup_for_target_duration(df, target_duration_seconds):
    t_csv = df["t"].to_numpy(dtype=float)
    sim_duration = float(t_csv[-1] - t_csv[0]) if len(t_csv) >= 2 else 0.0
    target_duration_seconds = max(float(target_duration_seconds), 1e-6)
    if sim_duration <= 1e-9:
        return 1.0
    return max(sim_duration / target_duration_seconds, 1e-6)


def resample_for_animation(df, fps=None, speedup=None):
    if fps is None:
        fps = FPS
    if speedup is None:
        speedup = SPEEDUP

    t_csv = df["t"].to_numpy(dtype=float)
    t0 = float(t_csv[0])
    t1 = float(t_csv[-1])

    sim_dt_per_frame = float(speedup) / float(fps)
    t_anim = np.arange(t0, t1 + 0.5 * sim_dt_per_frame, sim_dt_per_frame)

    out = {"t": t_anim}
    for col in [
        "x", "y", "z",
        "yaw_deg", "pitch_deg", "roll_deg",
        "psi_cmd_deg", "theta_cmd_deg", "roll_cmd_deg",
        "hdg_err_deg", "pitch_err_deg", "roll_err_deg",
    ]:
        out[col] = np.interp(t_anim, t_csv, df[col].to_numpy(dtype=float))

    return pd.DataFrame(out)


def make_fallback_aircraft_mesh():
    nose = np.array([1.00, 0.00, 0.00])
    tail = np.array([-0.70, 0.00, 0.00])
    left_wing = np.array([-0.15, -0.95, 0.05])
    right_wing = np.array([-0.15, 0.95, 0.05])
    top = np.array([-0.20, 0.00, -0.16])
    bottom = np.array([-0.20, 0.00, 0.16])
    tail_l = np.array([-0.75, -0.25, 0.05])
    tail_r = np.array([-0.75, 0.25, 0.05])

    return np.array([
        [nose, right_wing, top],
        [nose, top, left_wing],
        [nose, bottom, right_wing],
        [nose, left_wing, bottom],
        [tail, top, right_wing],
        [tail, left_wing, top],
        [tail, right_wing, bottom],
        [tail, bottom, left_wing],
        [tail, tail_r, top],
        [tail, top, tail_l],
    ], dtype=float)


def load_body_mesh():
    stl_path = Path(STL_PATH)

    if stl_path.is_file():
        try:
            from stl import mesh
            stl_mesh = mesh.Mesh.from_file(str(stl_path))
            vectors = stl_mesh.vectors.copy().astype(float)

            if vectors.shape[0] > MAX_STL_TRIANGLES:
                keep_idx = np.linspace(0, vectors.shape[0] - 1, MAX_STL_TRIANGLES).astype(int)
                vectors = vectors[keep_idx]
                print(f"Downsampled STL to {vectors.shape[0]} triangles.")

            vectors -= vectors.reshape(-1, 3).mean(axis=0)
            vectors *= float(STL_SCALE)

            R_offset = stl_alignment_rotation(STL_ROTATION_OFFSET_DEG)
            vectors = np.einsum("ij,tkj->tki", R_offset, vectors)

            return vectors

        except Exception as e:
            print(f"Could not load STL: {e}")
            print("Using fallback aircraft mesh.")

    else:
        print(f"STL not found at {stl_path}; using fallback aircraft mesh.")

    return make_fallback_aircraft_mesh()


def transform_mesh_to_plot(mesh_body, row):
    pos_ned = np.array([row["x"], row["y"], row["z"]], dtype=float)
    R_bw = body_to_ned_rotation(row["yaw_deg"], row["pitch_deg"], row["roll_deg"])

    mesh_ned = np.einsum("ij,tkj->tki", R_bw, mesh_body) + pos_ned
    return ned_to_plot(mesh_ned)


# =============================================================================
# Shared frame setup
# =============================================================================

def extract_arrays(df):
    return {
        "t": df["t"].to_numpy(dtype=float),
        "x": df["x"].to_numpy(dtype=float),
        "y": df["y"].to_numpy(dtype=float),
        "z_up": -df["z"].to_numpy(dtype=float),
        "yaw": df["yaw_deg"].to_numpy(dtype=float),
        "pitch": df["pitch_deg"].to_numpy(dtype=float),
        "roll": df["roll_deg"].to_numpy(dtype=float),
        "psi_cmd": df["psi_cmd_deg"].to_numpy(dtype=float),
        "theta_cmd": df["theta_cmd_deg"].to_numpy(dtype=float),
        "roll_cmd": df["roll_cmd_deg"].to_numpy(dtype=float),
        "hdg_err": df["hdg_err_deg"].to_numpy(dtype=float),
        "pitch_err": df["pitch_err_deg"].to_numpy(dtype=float),
        "roll_err": df["roll_err_deg"].to_numpy(dtype=float),
    }


def setup_flight_axis(ax, df, mesh_body, arr, title):
    ax.set_title(title, pad=12, fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    mesh_poly = Poly3DCollection(transform_mesh_to_plot(mesh_body, df.iloc[0]), alpha=0.82)
    mesh_poly.set_facecolor("tab:blue")
    mesh_poly.set_edgecolor("k")
    mesh_poly.set_linewidth(0.15)
    ax.add_collection3d(mesh_poly)

    trail_line, = ax.plot([], [], [], linewidth=1.8, color="k", alpha=0.8)
    pos_dot, = ax.plot([], [], [], "o", color="tab:blue", markersize=4)

    axis_lines = []
    for c in ["r", "g", "b"]:
        line, = ax.plot([], [], [], color=c, linewidth=1.5)
        axis_lines.append(line)

    fixed_xyz = np.column_stack([arr["x"], arr["y"], arr["z_up"]])
    fixed_center = np.mean(fixed_xyz, axis=0)
    fixed_radius = 0.5 * np.max(np.ptp(fixed_xyz, axis=0)) + FIXED_PADDING
    fixed_radius = max(float(fixed_radius), 5.0)

    return mesh_poly, trail_line, pos_dot, axis_lines, fixed_center, fixed_radius


def update_flight(i, df, mesh_body, arr, ax, mesh_poly, trail_line, pos_dot, axis_lines, fixed_center, fixed_radius, fps=FPS):
    mesh_poly.set_verts(transform_mesh_to_plot(mesh_body, df.iloc[i]))

    trail_n = max(1, int(round(TRAIL_LEN_SECONDS * fps / max(SPEEDUP, 1e-9))))
    i0 = max(0, i - trail_n)

    trail_line.set_data(arr["x"][i0:i + 1], arr["y"][i0:i + 1])
    trail_line.set_3d_properties(arr["z_up"][i0:i + 1])

    pos_dot.set_data([arr["x"][i]], [arr["y"][i]])
    pos_dot.set_3d_properties([arr["z_up"][i]])

    pos_plot = np.array([arr["x"][i], arr["y"][i], arr["z_up"][i]], dtype=float)
    R_bw = body_to_ned_rotation(arr["yaw"][i], arr["pitch"][i], arr["roll"][i])

    body_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]

    for line, axis_b in zip(axis_lines, body_axes):
        tip = pos_plot + AXIS_LEN * ned_to_plot(R_bw @ axis_b)
        line.set_data([pos_plot[0], tip[0]], [pos_plot[1], tip[1]])
        line.set_3d_properties([pos_plot[2], tip[2]])

    if FOLLOW_MODE:
        set_axes_equal_around(ax, pos_plot, float(FOLLOW_RADIUS))
    else:
        set_axes_equal_around(ax, fixed_center, fixed_radius)

    return [mesh_poly, trail_line, pos_dot, *axis_lines]


def setup_tracking_axes(ax_track, ax_err, arr):
    ax_track.set_title("Heading, Pitch, and Roll Tracking", pad=8, fontsize=12)
    ax_track.set_ylabel("Angle (deg)")
    ax_track.grid(True)
    ax_track.tick_params(axis="both", labelsize=9)

    h_line, = ax_track.plot([], [], label="Heading / yaw")
    h_cmd_line, = ax_track.plot([], [], "--", label="Heading command")
    p_line, = ax_track.plot([], [], label="Pitch")
    p_cmd_line, = ax_track.plot([], [], "--", label="Pitch command")
    r_line, = ax_track.plot([], [], label="Roll")
    r_cmd_line, = ax_track.plot([], [], "--", label="Roll command")

    h_dot, = ax_track.plot([], [], "o", markersize=4)
    p_dot, = ax_track.plot([], [], "o", markersize=4)
    r_dot, = ax_track.plot([], [], "o", markersize=4)

    ylo, yhi = auto_ylim(
        arr["yaw"], arr["psi_cmd"],
        arr["pitch"], arr["theta_cmd"],
        arr["roll"], arr["roll_cmd"],
        min_span=10.0,
    )
    ax_track.set_ylim(ylo, yhi)
    ax_track.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=8, framealpha=0.9, ncol=2)

    err_lines = None
    if ax_err is not None:
        ax_err.set_title("Tracking Error", pad=10, fontsize=12)
        ax_err.set_xlabel("Simulation time (s)")
        ax_err.set_ylabel("Error (deg)")
        ax_err.grid(True)
        ax_err.tick_params(axis="both", labelsize=9)
        ax_err.axhline(0.0, linewidth=1, color="k", alpha=0.35)

        h_err_line, = ax_err.plot([], [], label="Heading error")
        p_err_line, = ax_err.plot([], [], label="Pitch error")
        r_err_line, = ax_err.plot([], [], label="Roll error")
        err_lines = (h_err_line, p_err_line, r_err_line)

        elo, ehi = auto_ylim(arr["hdg_err"], arr["pitch_err"], arr["roll_err"], min_span=5.0)
        ax_err.set_ylim(elo, ehi)
        ax_err.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=8, framealpha=0.9, ncol=3)

    time_text = ax_track.text(
        0.02,
        0.90,
        "",
        transform=ax_track.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        fontsize=9,
    )

    start_t = float(arr["t"][0])
    final_t = float(arr["t"][-1])

    if TRACKING_HISTORY_MODE.lower() == "full":
        ax_track.set_xlim(start_t, final_t)
        if ax_err is not None:
            ax_err.set_xlim(start_t, final_t)

    lines = (h_line, h_cmd_line, p_line, p_cmd_line, r_line, r_cmd_line, h_dot, p_dot, r_dot)
    return lines, err_lines, time_text


def update_tracking(i, arr, ax_track, ax_err, lines, err_lines, time_text):
    h_line, h_cmd_line, p_line, p_cmd_line, r_line, r_cmd_line, h_dot, p_dot, r_dot = lines

    sl = slice(0, i + 1)
    t = arr["t"]

    h_line.set_data(t[sl], arr["yaw"][sl])
    h_cmd_line.set_data(t[sl], arr["psi_cmd"][sl])
    p_line.set_data(t[sl], arr["pitch"][sl])
    p_cmd_line.set_data(t[sl], arr["theta_cmd"][sl])
    r_line.set_data(t[sl], arr["roll"][sl])
    r_cmd_line.set_data(t[sl], arr["roll_cmd"][sl])

    h_dot.set_data([t[i]], [arr["yaw"][i]])
    p_dot.set_data([t[i]], [arr["pitch"][i]])
    r_dot.set_data([t[i]], [arr["roll"][i]])

    artists = [*lines]

    if err_lines is not None:
        h_err_line, p_err_line, r_err_line = err_lines
        h_err_line.set_data(t[sl], arr["hdg_err"][sl])
        p_err_line.set_data(t[sl], arr["pitch_err"][sl])
        r_err_line.set_data(t[sl], arr["roll_err"][sl])
        artists += [h_err_line, p_err_line, r_err_line]

    if TRACKING_HISTORY_MODE.lower() == "window":
        t_now = t[i]
        window = float(TRACKING_WINDOW_SECONDS)
        lo = max(float(t[0]), t_now - window)
        hi = max(lo + window, t_now)

        if hi > float(t[-1]):
            hi = float(t[-1])
            lo = max(float(t[0]), hi - window)

        ax_track.set_xlim(lo, hi)
        if ax_err is not None:
            ax_err.set_xlim(lo, hi)

    time_text.set_text(
        f"t = {t[i]:.2f} s\n"
        f"heading err = {arr['hdg_err'][i]:+.2f} deg\n"
        f"pitch err = {arr['pitch_err'][i]:+.2f} deg\n"
        f"roll err = {arr['roll_err'][i]:+.2f} deg"
    )
    artists.append(time_text)
    return artists


# =============================================================================
# Animation construction
# =============================================================================

def make_combined_animation(df, figsize=FIGSIZE):
    mesh_body = load_body_mesh()
    arr = extract_arrays(df)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=LEFT_RIGHT_WIDTH_RATIO,
        height_ratios=[2.0, 1.0],
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_track = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1], sharex=ax_track)

    fig.suptitle("Synchronized MPC v3 Flight Animation and Attitude Tracking", fontsize=13, y=0.975)

    flight_objs = setup_flight_axis(ax_3d, df, mesh_body, arr, "MPC v3 Flight Animation")
    tracking_objs = setup_tracking_axes(ax_track, ax_err, arr)

    def update(i):
        artists = []
        artists += update_flight(i, df, mesh_body, arr, ax_3d, *flight_objs)
        artists += update_tracking(i, arr, ax_track, ax_err, *tracking_objs)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    fig.subplots_adjust(
        left=FIG_LEFT,
        right=FIG_RIGHT,
        bottom=FIG_BOTTOM,
        top=FIG_TOP,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )

    return anim, fig


def make_plot_only_animation(df):
    arr = extract_arrays(df)

    fig = plt.figure(figsize=PLOT_ONLY_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.30)
    ax_track = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0], sharex=ax_track)
    fig.suptitle("MPC v3 Attitude Tracking Animation", fontsize=13)

    tracking_objs = setup_tracking_axes(ax_track, ax_err, arr)

    def update(i):
        return update_tracking(i, arr, ax_track, ax_err, *tracking_objs)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return anim, fig


def make_flight_only_animation(df):
    mesh_body = load_body_mesh()
    arr = extract_arrays(df)

    fig = plt.figure(figsize=FLIGHT_ONLY_FIGSIZE, constrained_layout=False)
    ax_3d = fig.add_subplot(111, projection="3d")
    fig.suptitle("MPC v3 3D Flight Animation", fontsize=13)

    flight_objs = setup_flight_axis(ax_3d, df, mesh_body, arr, "3D Flight Only")

    def update(i):
        return update_flight(i, df, mesh_body, arr, ax_3d, *flight_objs)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return anim, fig


# =============================================================================
# Export helpers
# =============================================================================

def save_mp4(anim, path, fps=FPS, dpi=DPI_MP4):
    print(f"Saving MP4: {path}")
    anim.save(path, writer=FFMpegWriter(fps=int(fps), bitrate=2200), dpi=int(dpi))
    print(f"Saved MP4: {path.resolve()}")


def save_gif(anim, path, fps=GIF_FPS, dpi=DPI_GIF):
    print(f"Saving GIF: {path}")
    anim.save(path, writer=PillowWriter(fps=int(fps)), dpi=int(dpi))
    print(f"Saved GIF: {path.resolve()}")


def file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return float("inf")
    return path.stat().st_size / (1024.0 * 1024.0)


def save_gif_under_target_size(anim, path, target_mb=TARGET_GIF_SIZE_MB):
    """
    Save an additional optimized GIF.

    This tries progressively smaller FPS/DPI combinations until the file is just
    under the requested target size. If even the smallest setting is still above
    target, it keeps the smallest version and prints a warning.
    """
    path = Path(path)

    # Ordered from highest quality to lowest. The first one below target wins.
    attempts = [
        (15, 100),
        (12, 95),
        (10, 90),
        (10, 80),
        (8, 75),
        (8, 65),
        (6, 60),
        (5, 55),
        (4, 50),
    ]

    best_path = None
    best_size = float("inf")

    for fps, dpi in attempts:
        temp_path = path.with_name(f"{path.stem}_tmp_fps{fps}_dpi{dpi}{path.suffix}")
        print(f"Trying optimized GIF: {temp_path.name} at {fps} fps, dpi={dpi}")
        try:
            anim.save(temp_path, writer=PillowWriter(fps=int(fps)), dpi=int(dpi))
            size = file_size_mb(temp_path)
            print(f"  size = {size:.2f} MB")

            if size < best_size:
                if best_path is not None and Path(best_path).exists() and Path(best_path) != temp_path:
                    Path(best_path).unlink()
                best_path = temp_path
                best_size = size
            else:
                temp_path.unlink(missing_ok=True)

            if size <= float(target_mb):
                if path.exists():
                    path.unlink()
                temp_path.rename(path)
                print(f"Saved optimized GIF under {target_mb:.1f} MB: {path.resolve()} ({size:.2f} MB)")
                return path

        except Exception as e:
            print(f"  optimized GIF attempt failed at {fps} fps, dpi={dpi}: {e}")
            temp_path.unlink(missing_ok=True)

    if best_path is not None and Path(best_path).exists():
        if path.exists():
            path.unlink()
        Path(best_path).rename(path)
        print(
            f"Saved smallest optimized GIF available: {path.resolve()} "
            f"({best_size:.2f} MB; target was {target_mb:.1f} MB)"
        )
        if best_size > float(target_mb):
            print("WARNING: even the smallest attempted GIF is still above the target size.")
        return path

    print(f"Could not create optimized GIF: {path}")
    return None


def make_small_gif_from_mp4(mp4_path, gif_path, fps=SMALL_GIF_FPS, width=720):
    """
    Make a small combined GIF using ffmpeg when available.
    Falls back to direct matplotlib export if ffmpeg is not installed.
    """
    palette_path = gif_path.with_suffix(".palette.png")
    try:
        print(f"Creating small GIF with ffmpeg: {gif_path}")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_path),
                "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
                str(palette_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_path),
                "-i", str(palette_path),
                "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                str(gif_path),
            ],
            check=True,
        )
        if palette_path.exists():
            palette_path.unlink()
        print(f"Saved small GIF: {gif_path.resolve()}")
        return True
    except Exception as e:
        print(f"ffmpeg small GIF creation failed: {e}")
        if palette_path.exists():
            palette_path.unlink()
        return False


def export_all(df_anim, df_anim_under50):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_mp4 = out_dir / f"{OUTPUT_BASENAME}.mp4"
    combined_gif = out_dir / f"{OUTPUT_BASENAME}.gif"
    combined_small_gif = out_dir / f"{OUTPUT_BASENAME}_small.gif"
    combined_under50_gif = out_dir / f"{OUTPUT_BASENAME}_under50mb.gif"
    plot_only_gif = out_dir / "mpc_v3_tracking_plots_only.gif"
    plot_only_under50_gif = out_dir / "mpc_v3_tracking_plots_only_under50mb.gif"
    flight_only_gif = out_dir / "mpc_v3_3d_flight_only.gif"
    flight_only_under50_gif = out_dir / "mpc_v3_3d_flight_only_under50mb.gif"

    # Combined MP4 + full GIF use the normal/full-duration frame set.
    combined_anim, combined_fig = make_combined_animation(df_anim)
    save_mp4(combined_anim, combined_mp4, fps=FPS, dpi=DPI_MP4)
    save_gif(combined_anim, combined_gif, fps=GIF_FPS, dpi=DPI_GIF)

    # Smaller combined GIF. Prefer ffmpeg conversion from MP4.
    ok = make_small_gif_from_mp4(combined_mp4, combined_small_gif)
    if not ok:
        print("Falling back to direct low-DPI matplotlib small GIF export.")
        save_gif(combined_anim, combined_small_gif, fps=SMALL_GIF_FPS, dpi=SMALL_GIF_DPI)

    plt.close(combined_fig)

    # Additional optimized combined GIF targeted below 50 MB uses the shorter
    # under50/undersampled frame set so it does not take forever.
    combined_under50_anim, combined_under50_fig = make_combined_animation(df_anim_under50)
    save_gif_under_target_size(combined_under50_anim, combined_under50_gif, target_mb=TARGET_GIF_SIZE_MB)
    plt.close(combined_under50_fig)

    # Plot-only GIF. Normal uses full duration; under50 uses shortened frame set.
    plot_anim, plot_fig = make_plot_only_animation(df_anim)
    save_gif(plot_anim, plot_only_gif, fps=GIF_FPS, dpi=DPI_GIF)
    plt.close(plot_fig)

    plot_under50_anim, plot_under50_fig = make_plot_only_animation(df_anim_under50)
    save_gif_under_target_size(plot_under50_anim, plot_only_under50_gif, target_mb=TARGET_GIF_SIZE_MB)
    plt.close(plot_under50_fig)

    # 3D-only GIF. Normal uses full duration; under50 uses shortened frame set.
    flight_anim, flight_fig = make_flight_only_animation(df_anim)
    save_gif(flight_anim, flight_only_gif, fps=GIF_FPS, dpi=DPI_GIF)
    plt.close(flight_fig)

    flight_under50_anim, flight_under50_fig = make_flight_only_animation(df_anim_under50)
    save_gif_under_target_size(flight_under50_anim, flight_only_under50_gif, target_mb=TARGET_GIF_SIZE_MB)
    plt.close(flight_under50_fig)

    return {
        "combined_mp4": combined_mp4,
        "combined_gif": combined_gif,
        "combined_small_gif": combined_small_gif,
        "combined_under50_gif": combined_under50_gif,
        "plot_only_gif": plot_only_gif,
        "plot_only_under50_gif": plot_only_under50_gif,
        "flight_only_gif": flight_only_gif,
        "flight_only_under50_gif": flight_only_under50_gif,
    }


def main():
    df_raw = load_csv_data()

    # Normal outputs keep the normal/full-duration SPEEDUP behavior.
    df_anim = resample_for_animation(df_raw)

    # Only the additional _under50mb GIFs use this shorter undersampled version.
    under50_speedup = compute_speedup_for_target_duration(
        df_raw,
        UNDER50_TARGET_TOTAL_DURATION_SECONDS,
    )
    df_anim_under50 = resample_for_animation(df_raw, speedup=under50_speedup)

    print(f"CSV time range: {df_raw['t'].iloc[0]:.3f} to {df_raw['t'].iloc[-1]:.3f} s")
    print(f"Normal animation frames: {len(df_anim)}")
    print(f"Normal output duration: {len(df_anim) / FPS:.3f} s at {FPS} fps")
    print(f"Normal speedup: {SPEEDUP}x")
    print(f"Under50 animation frames: {len(df_anim_under50)}")
    print(f"Under50 target duration: {UNDER50_TARGET_TOTAL_DURATION_SECONDS:.3f} s")
    print(f"Under50 actual duration: {len(df_anim_under50) / FPS:.3f} s at {FPS} fps")
    print(f"Under50 computed speedup: {under50_speedup:.6g}x")

    outputs = export_all(df_anim, df_anim_under50)

    print("\nSaved outputs:")
    for label, path in outputs.items():
        print(f"  {label}: {path.resolve()}")


if __name__ == "__main__":
    main()
