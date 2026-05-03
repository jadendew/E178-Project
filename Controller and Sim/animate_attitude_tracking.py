#!/usr/bin/env python3
"""
animate_attitude_tracking.py

Creates live tracking animations for heading, pitch, and roll vs setpoint from
a sim output CSV. Saves both MP4 and GIF.

Required measured columns:
    t, yaw_deg, pitch_deg, roll_deg

Setpoint columns are auto-detected in this order:
    heading: psi_cmd_deg, mpc_yaw_cmd_deg
    pitch:  theta_cmd_deg, mpc_pitch_cmd_deg
    roll:   roll_cmd_deg, mpc_roll_cmd_deg

If a setpoint column is missing, the script uses the constants below.

Output:
    attitude_tracking_animation/
        attitude_tracking_live.mp4
        attitude_tracking_live.gif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# =============================================================================
# User settings
# =============================================================================

CSV_PATH = "mpc_output_plots_v2/sim_out_MLP_MPC_v2.csv"

OUTPUT_DIR = "mpc_output_plots_v2/attitude_tracking_animation"
OUTPUT_BASENAME = "attitude_tracking_live_MPC_v2"

# Fallback setpoints, used only if command columns are not in the CSV.
HEADING_CMD_DEG = 50.0
PITCH_CMD_DEG = 3.0
ROLL_CMD_DEG = 0.0

# Animation settings
FPS = 30
DPI_MP4 = 150
DPI_GIF = 120

# If the CSV has many rows, this keeps the animation smaller/faster.
# Example: STRIDE = 5 means animate every 5th row.
STRIDE = 5

# Set to None to use the full CSV.
# Example: MAX_ANIMATION_SECONDS = 35.0
MAX_ANIMATION_SECONDS = None

# Plot window behavior
FOLLOW_TIME_WINDOW_S = None
# Example:
# FOLLOW_TIME_WINDOW_S = 10.0
# If None, the full time axis is visible from the start.

SHOW_ERROR_PANEL = True


# =============================================================================
# Helpers
# =============================================================================

def wrap_deg(angle_deg):
    """Wrap degrees to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def get_column_or_constant(df, preferred_columns, constant_value):
    for col in preferred_columns:
        if col in df.columns:
            return df[col].to_numpy(dtype=float), col
    return np.full(len(df), float(constant_value)), f"constant {constant_value:g} deg"


def load_tracking_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["t", "yaw_deg", "pitch_deg", "roll_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if MAX_ANIMATION_SECONDS is not None:
        df = df[df["t"] <= float(MAX_ANIMATION_SECONDS)].copy()

    df = df.iloc[::max(1, int(STRIDE))].copy().reset_index(drop=True)

    heading_cmd, heading_source = get_column_or_constant(
        df,
        ["psi_cmd_deg", "mpc_yaw_cmd_deg"],
        HEADING_CMD_DEG,
    )

    pitch_cmd, pitch_source = get_column_or_constant(
        df,
        ["theta_cmd_deg", "mpc_pitch_cmd_deg"],
        PITCH_CMD_DEG,
    )

    roll_cmd, roll_source = get_column_or_constant(
        df,
        ["roll_cmd_deg", "mpc_roll_cmd_deg"],
        ROLL_CMD_DEG,
    )

    out = pd.DataFrame({
        "t": df["t"].to_numpy(dtype=float),
        "heading_deg": df["yaw_deg"].to_numpy(dtype=float),
        "pitch_deg": df["pitch_deg"].to_numpy(dtype=float),
        "roll_deg": df["roll_deg"].to_numpy(dtype=float),
        "heading_cmd_deg": heading_cmd,
        "pitch_cmd_deg": pitch_cmd,
        "roll_cmd_deg": roll_cmd,
    })

    out["heading_err_deg"] = wrap_deg(out["heading_cmd_deg"] - out["heading_deg"])
    out["pitch_err_deg"] = out["pitch_cmd_deg"] - out["pitch_deg"]
    out["roll_err_deg"] = wrap_deg(out["roll_cmd_deg"] - out["roll_deg"])

    print("Command sources:")
    print(f"  heading: {heading_source}")
    print(f"  pitch:   {pitch_source}")
    print(f"  roll:    {roll_source}")

    return out


def auto_ylim(*arrays, pad_frac=0.12, min_span=5.0):
    vals = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return -1.0, 1.0

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, float(min_span))
    pad = pad_frac * span

    return lo - pad, hi + pad


def save_animation(anim, output_dir: str | Path, basename: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = output_dir / f"{basename}.mp4"
    gif_path = output_dir / f"{basename}.gif"

    print(f"Saving MP4: {mp4_path}")
    try:
        anim.save(
            mp4_path,
            writer=FFMpegWriter(fps=int(FPS), bitrate=1800),
            dpi=int(DPI_MP4),
        )
        print(f"Saved MP4: {mp4_path.resolve()}")
    except Exception as e:
        print(f"MP4 save failed: {e}")

    print(f"Saving GIF: {gif_path}")
    try:
        anim.save(
            gif_path,
            writer=PillowWriter(fps=int(FPS)),
            dpi=int(DPI_GIF),
        )
        print(f"Saved GIF: {gif_path.resolve()}")
    except Exception as e:
        print(f"GIF save failed: {e}")


# =============================================================================
# Animation
# =============================================================================

def make_tracking_animation(df: pd.DataFrame):
    t = df["t"].to_numpy(dtype=float)

    heading = df["heading_deg"].to_numpy(dtype=float)
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)

    heading_cmd = df["heading_cmd_deg"].to_numpy(dtype=float)
    pitch_cmd = df["pitch_cmd_deg"].to_numpy(dtype=float)
    roll_cmd = df["roll_cmd_deg"].to_numpy(dtype=float)

    heading_err = df["heading_err_deg"].to_numpy(dtype=float)
    pitch_err = df["pitch_err_deg"].to_numpy(dtype=float)
    roll_err = df["roll_err_deg"].to_numpy(dtype=float)

    if SHOW_ERROR_PANEL:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        ax_track, ax_err = axes
    else:
        fig, ax_track = plt.subplots(figsize=(11, 6))
        ax_err = None

    fig.suptitle("Live Attitude Tracking: Heading, Pitch, Roll")

    # Tracking axis
    ax_track.set_ylabel("Angle (deg)")
    ax_track.grid(True)

    h_line, = ax_track.plot([], [], label="Heading / yaw")
    h_cmd_line, = ax_track.plot([], [], "--", label="Heading command")

    p_line, = ax_track.plot([], [], label="Pitch")
    p_cmd_line, = ax_track.plot([], [], "--", label="Pitch command")

    r_line, = ax_track.plot([], [], label="Roll")
    r_cmd_line, = ax_track.plot([], [], "--", label="Roll command")

    current_marker_h, = ax_track.plot([], [], "o", markersize=5)
    current_marker_p, = ax_track.plot([], [], "o", markersize=5)
    current_marker_r, = ax_track.plot([], [], "o", markersize=5)

    ax_track.legend(loc="upper right")

    ylo, yhi = auto_ylim(
        heading, heading_cmd,
        pitch, pitch_cmd,
        roll, roll_cmd,
        min_span=10.0,
    )
    ax_track.set_ylim(ylo, yhi)

    # Error axis
    if ax_err is not None:
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Error (deg)")
        ax_err.grid(True)

        h_err_line, = ax_err.plot([], [], label="Heading error")
        p_err_line, = ax_err.plot([], [], label="Pitch error")
        r_err_line, = ax_err.plot([], [], label="Roll error")

        ax_err.axhline(0.0, linewidth=1)
        ax_err.legend(loc="upper right")

        elo, ehi = auto_ylim(heading_err, pitch_err, roll_err, min_span=5.0)
        ax_err.set_ylim(elo, ehi)
    else:
        h_err_line = p_err_line = r_err_line = None

    time_text = ax_track.text(
        0.02,
        0.94,
        "",
        transform=ax_track.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    final_t = float(t[-1])
    start_t = float(t[0])

    if FOLLOW_TIME_WINDOW_S is None:
        ax_track.set_xlim(start_t, final_t)
        if ax_err is not None:
            ax_err.set_xlim(start_t, final_t)

    def set_time_window(i):
        if FOLLOW_TIME_WINDOW_S is None:
            return

        t_now = t[i]
        half_window = 0.5 * float(FOLLOW_TIME_WINDOW_S)

        lo = max(start_t, t_now - half_window)
        hi = min(final_t, t_now + half_window)

        if hi - lo < float(FOLLOW_TIME_WINDOW_S):
            if lo <= start_t:
                hi = min(final_t, start_t + float(FOLLOW_TIME_WINDOW_S))
            elif hi >= final_t:
                lo = max(start_t, final_t - float(FOLLOW_TIME_WINDOW_S))

        ax_track.set_xlim(lo, hi)
        if ax_err is not None:
            ax_err.set_xlim(lo, hi)

    def update(i):
        sl = slice(0, i + 1)

        h_line.set_data(t[sl], heading[sl])
        h_cmd_line.set_data(t[sl], heading_cmd[sl])

        p_line.set_data(t[sl], pitch[sl])
        p_cmd_line.set_data(t[sl], pitch_cmd[sl])

        r_line.set_data(t[sl], roll[sl])
        r_cmd_line.set_data(t[sl], roll_cmd[sl])

        current_marker_h.set_data([t[i]], [heading[i]])
        current_marker_p.set_data([t[i]], [pitch[i]])
        current_marker_r.set_data([t[i]], [roll[i]])

        if ax_err is not None:
            h_err_line.set_data(t[sl], heading_err[sl])
            p_err_line.set_data(t[sl], pitch_err[sl])
            r_err_line.set_data(t[sl], roll_err[sl])

        set_time_window(i)

        time_text.set_text(
            f"t = {t[i]:.2f} s\n"
            f"heading err = {heading_err[i]:+.2f} deg\n"
            f"pitch err = {pitch_err[i]:+.2f} deg\n"
            f"roll err = {roll_err[i]:+.2f} deg"
        )

        artists = [
            h_line, h_cmd_line,
            p_line, p_cmd_line,
            r_line, r_cmd_line,
            current_marker_h, current_marker_p, current_marker_r,
            time_text,
        ]

        if ax_err is not None:
            artists += [h_err_line, p_err_line, r_err_line]

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    fig.tight_layout()
    return anim


def main():
    df = load_tracking_data(CSV_PATH)
    anim = make_tracking_animation(df)
    save_animation(anim, OUTPUT_DIR, OUTPUT_BASENAME)
    print("Done.")


if __name__ == "__main__":
    main()
