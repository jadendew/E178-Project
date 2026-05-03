#!/usr/bin/env python3
"""
make_synced_flight_tracking_from_csv.py

Creates a truly synchronized side-by-side animation directly from the simulation CSV.

Why this is better than combining two pre-rendered videos:
    - The CSV is the single source of truth for time.
    - The left flight view and right tracking plot are drawn in the same frame loop.
    - Each video frame corresponds to the exact same simulation time on both sides.
    - No separate GIF/video playback drift or timing mismatch is possible.

Inputs:
    sim_out_MLP_MPC_v2.csv
    optional lowerqualitymesh.stl

Outputs:
    synced_csv_animation/
        mpc_v2_csv_synced_flight_tracking.mp4
        mpc_v2_csv_synced_flight_tracking.gif
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

CSV_PATH = "mpc_output_plots_v2/sim_out_MLP_MPC_v2.csv"

OUTPUT_DIR = "mpc_output_plots_v2/synced_csv_animation"
OUTPUT_BASENAME = "mpc_v2_csv_synced_flight_tracking"

# Optional STL
STL_PATH = "lowerqualitymesh.stl"
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)
MAX_STL_TRIANGLES = 1000

# Animation timing
FPS = 30
DPI_MP4 = 150
DPI_GIF = 110

# CSV time range to animate. Set to None for full CSV.
T_START = 0.0
T_END = None

# Total animation duration.
# Your CSV is ~35 s. If you want real-time, set SPEEDUP = 1.0.
# If you want the web animation shorter, use 1.5 or 2.0.
SPEEDUP = 1.0

# Layout
FIGSIZE = (16, 7.2)
LEFT_RIGHT_WIDTH_RATIO = [1.0, 1.35]

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

# GIF export
EXPORT_GIF = True
GIF_FPS = 15


# =============================================================================
# Math helpers
# =============================================================================

def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


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

def load_csv_data():
    csv_path = Path(CSV_PATH)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

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

    # Command columns with fallbacks.
    if "psi_cmd_deg" not in df.columns:
        df["psi_cmd_deg"] = float(df.get("mpc_yaw_cmd_deg", pd.Series([50.0])).iloc[0])
    if "theta_cmd_deg" not in df.columns:
        df["theta_cmd_deg"] = float(df.get("mpc_pitch_cmd_deg", pd.Series([3.0])).iloc[0])
    if "roll_cmd_deg" not in df.columns:
        df["roll_cmd_deg"] = float(df.get("mpc_roll_cmd_deg", pd.Series([0.0])).iloc[0])

    df["hdg_err_deg"] = wrap_deg(df["psi_cmd_deg"].to_numpy() - df["yaw_deg"].to_numpy())
    df["pitch_err_deg"] = df["theta_cmd_deg"].to_numpy() - df["pitch_deg"].to_numpy()
    df["roll_err_deg"] = wrap_deg(df["roll_cmd_deg"].to_numpy() - df["roll_deg"].to_numpy())

    return df


def resample_for_animation(df):
    """
    Build a frame timeline from CSV time.

    Frame i corresponds to simulation time:
        t_anim[i] = t0 + i * SPEEDUP / FPS

    This means both the flight view and tracking plot are rendered from the same
    exact sim time every frame.
    """
    t_csv = df["t"].to_numpy(dtype=float)
    t0 = float(t_csv[0])
    t1 = float(t_csv[-1])

    sim_dt_per_frame = float(SPEEDUP) / float(FPS)
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
# Animation construction
# =============================================================================

def make_animation(df):
    mesh_body = load_body_mesh()

    t = df["t"].to_numpy(dtype=float)

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z_up = -df["z"].to_numpy(dtype=float)

    yaw = df["yaw_deg"].to_numpy(dtype=float)
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)

    psi_cmd = df["psi_cmd_deg"].to_numpy(dtype=float)
    theta_cmd = df["theta_cmd_deg"].to_numpy(dtype=float)
    roll_cmd = df["roll_cmd_deg"].to_numpy(dtype=float)

    hdg_err = df["hdg_err_deg"].to_numpy(dtype=float)
    pitch_err = df["pitch_err_deg"].to_numpy(dtype=float)
    roll_err = df["roll_err_deg"].to_numpy(dtype=float)

    if SHOW_ERROR_PANEL:
        fig = plt.figure(figsize=FIGSIZE)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=LEFT_RIGHT_WIDTH_RATIO,
            height_ratios=[2.0, 1.0],
            wspace=0.13,
            hspace=0.16,
        )
        ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_track = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 1], sharex=ax_track)
    else:
        fig = plt.figure(figsize=FIGSIZE)
        gs = fig.add_gridspec(1, 2, width_ratios=LEFT_RIGHT_WIDTH_RATIO, wspace=0.13)
        ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_track = fig.add_subplot(gs[0, 1])
        ax_err = None

    fig.suptitle("Synchronized MPC Flight Animation and Attitude Tracking", fontsize=14)

    # 3D flight axis
    ax_3d.set_title("MPC v2 Flight Animation")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z up (m)")
    ax_3d.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    mesh_poly = Poly3DCollection(transform_mesh_to_plot(mesh_body, df.iloc[0]), alpha=0.82)
    mesh_poly.set_facecolor("tab:blue")
    mesh_poly.set_edgecolor("k")
    mesh_poly.set_linewidth(0.15)
    ax_3d.add_collection3d(mesh_poly)

    trail_line, = ax_3d.plot([], [], [], linewidth=1.8, color="k", alpha=0.8)
    pos_dot, = ax_3d.plot([], [], [], "o", color="tab:blue", markersize=4)

    # Body axes
    axis_lines = []
    axis_colors = ["r", "g", "b"]
    for c in axis_colors:
        line, = ax_3d.plot([], [], [], color=c, linewidth=1.5)
        axis_lines.append(line)

    # Tracking axis
    ax_track.set_title("Heading, Pitch, and Roll Tracking")
    ax_track.set_ylabel("Angle (deg)")
    ax_track.grid(True)

    h_line, = ax_track.plot([], [], label="Heading / yaw")
    h_cmd_line, = ax_track.plot([], [], "--", label="Heading command")
    p_line, = ax_track.plot([], [], label="Pitch")
    p_cmd_line, = ax_track.plot([], [], "--", label="Pitch command")
    r_line, = ax_track.plot([], [], label="Roll")
    r_cmd_line, = ax_track.plot([], [], "--", label="Roll command")

    h_dot, = ax_track.plot([], [], "o", markersize=4)
    p_dot, = ax_track.plot([], [], "o", markersize=4)
    r_dot, = ax_track.plot([], [], "o", markersize=4)

    ylo, yhi = auto_ylim(yaw, psi_cmd, pitch, theta_cmd, roll, roll_cmd, min_span=10.0)
    ax_track.set_ylim(ylo, yhi)
    ax_track.legend(loc="upper right", fontsize=8)

    if ax_err is not None:
        ax_err.set_title("Tracking Error")
        ax_err.set_xlabel("Simulation time (s)")
        ax_err.set_ylabel("Error (deg)")
        ax_err.grid(True)
        ax_err.axhline(0.0, linewidth=1, color="k", alpha=0.35)

        h_err_line, = ax_err.plot([], [], label="Heading error")
        p_err_line, = ax_err.plot([], [], label="Pitch error")
        r_err_line, = ax_err.plot([], [], label="Roll error")

        elo, ehi = auto_ylim(hdg_err, pitch_err, roll_err, min_span=5.0)
        ax_err.set_ylim(elo, ehi)
        ax_err.legend(loc="upper right", fontsize=8)
    else:
        h_err_line = p_err_line = r_err_line = None

    time_text = ax_track.text(
        0.02,
        0.94,
        "",
        transform=ax_track.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        fontsize=9,
    )

    final_t = float(t[-1])
    start_t = float(t[0])

    if TRACKING_HISTORY_MODE.lower() == "full":
        ax_track.set_xlim(start_t, final_t)
        if ax_err is not None:
            ax_err.set_xlim(start_t, final_t)

    fixed_xyz = np.column_stack([x, y, z_up])
    fixed_center = np.mean(fixed_xyz, axis=0)
    fixed_radius = 0.5 * np.max(np.ptp(fixed_xyz, axis=0)) + FIXED_PADDING
    fixed_radius = max(float(fixed_radius), 5.0)

    def set_tracking_xlim(i):
        if TRACKING_HISTORY_MODE.lower() != "window":
            return

        t_now = t[i]
        window = float(TRACKING_WINDOW_SECONDS)
        lo = max(start_t, t_now - window)
        hi = max(lo + window, t_now)

        if hi > final_t:
            hi = final_t
            lo = max(start_t, hi - window)

        ax_track.set_xlim(lo, hi)
        if ax_err is not None:
            ax_err.set_xlim(lo, hi)

    def update_body_axes(i):
        pos_ned = np.array([x[i], y[i], -z_up[i]], dtype=float)
        pos_plot = np.array([x[i], y[i], z_up[i]], dtype=float)
        R_bw = body_to_ned_rotation(yaw[i], pitch[i], roll[i])

        body_axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

        for line, axis_b in zip(axis_lines, body_axes):
            tip = pos_plot + AXIS_LEN * ned_to_plot(R_bw @ axis_b)
            line.set_data([pos_plot[0], tip[0]], [pos_plot[1], tip[1]])
            line.set_3d_properties([pos_plot[2], tip[2]])

    def update(i):
        row = df.iloc[i]

        mesh_poly.set_verts(transform_mesh_to_plot(mesh_body, row))

        trail_n = max(1, int(round(TRAIL_LEN_SECONDS * FPS / max(SPEEDUP, 1e-9))))
        i0 = max(0, i - trail_n)

        trail_line.set_data(x[i0:i + 1], y[i0:i + 1])
        trail_line.set_3d_properties(z_up[i0:i + 1])

        pos_dot.set_data([x[i]], [y[i]])
        pos_dot.set_3d_properties([z_up[i]])

        update_body_axes(i)

        if FOLLOW_MODE:
            center = np.array([x[i], y[i], z_up[i]], dtype=float)
            set_axes_equal_around(ax_3d, center, float(FOLLOW_RADIUS))
        else:
            set_axes_equal_around(ax_3d, fixed_center, fixed_radius)

        sl = slice(0, i + 1)

        h_line.set_data(t[sl], yaw[sl])
        h_cmd_line.set_data(t[sl], psi_cmd[sl])
        p_line.set_data(t[sl], pitch[sl])
        p_cmd_line.set_data(t[sl], theta_cmd[sl])
        r_line.set_data(t[sl], roll[sl])
        r_cmd_line.set_data(t[sl], roll_cmd[sl])

        h_dot.set_data([t[i]], [yaw[i]])
        p_dot.set_data([t[i]], [pitch[i]])
        r_dot.set_data([t[i]], [roll[i]])

        if ax_err is not None:
            h_err_line.set_data(t[sl], hdg_err[sl])
            p_err_line.set_data(t[sl], pitch_err[sl])
            r_err_line.set_data(t[sl], roll_err[sl])

        set_tracking_xlim(i)

        time_text.set_text(
            f"t = {t[i]:.2f} s\n"
            f"heading err = {hdg_err[i]:+.2f} deg\n"
            f"pitch err = {pitch_err[i]:+.2f} deg\n"
            f"roll err = {roll_err[i]:+.2f} deg"
        )

        artists = [
            mesh_poly, trail_line, pos_dot,
            *axis_lines,
            h_line, h_cmd_line, p_line, p_cmd_line, r_line, r_cmd_line,
            h_dot, p_dot, r_dot,
            time_text,
        ]

        if ax_err is not None:
            artists += [h_err_line, p_err_line, r_err_line]

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    fig.tight_layout()

    return anim


# =============================================================================
# Export
# =============================================================================

def export_animation(anim):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f"{OUTPUT_BASENAME}.mp4"
    gif_path = out_dir / f"{OUTPUT_BASENAME}.gif"

    print(f"Saving MP4: {mp4_path}")
    anim.save(
        mp4_path,
        writer=FFMpegWriter(fps=int(FPS), bitrate=2200),
        dpi=int(DPI_MP4),
    )
    print(f"Saved MP4: {mp4_path.resolve()}")

    if EXPORT_GIF:
        try:
            print(f"Saving GIF: {gif_path}")
            anim.save(
                gif_path,
                writer=PillowWriter(fps=int(GIF_FPS)),
                dpi=int(DPI_GIF),
            )
            print(f"Saved GIF: {gif_path.resolve()}")
        except Exception as e:
            print(f"Matplotlib GIF export failed: {e}")
            print("Trying ffmpeg conversion from MP4...")
            try:
                palette_path = gif_path.with_suffix(".palette.png")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(mp4_path),
                        "-vf", f"fps={GIF_FPS},scale=1280:-1:flags=lanczos,palettegen",
                        str(palette_path),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(mp4_path),
                        "-i", str(palette_path),
                        "-lavfi", f"fps={GIF_FPS},scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                        str(gif_path),
                    ],
                    check=True,
                )
                if palette_path.exists():
                    palette_path.unlink()
                print(f"Saved GIF: {gif_path.resolve()}")
            except Exception as e2:
                print(f"ffmpeg GIF conversion also failed: {e2}")

    return mp4_path, gif_path


def main():
    df_raw = load_csv_data()
    df_anim = resample_for_animation(df_raw)

    print(f"CSV time range: {df_raw['t'].iloc[0]:.3f} to {df_raw['t'].iloc[-1]:.3f} s")
    print(f"Animation frames: {len(df_anim)}")
    print(f"Output duration: {len(df_anim) / FPS:.3f} s at {FPS} fps")
    print(f"Speedup: {SPEEDUP}x")

    anim = make_animation(df_anim)
    export_animation(anim)


if __name__ == "__main__":
    main()
