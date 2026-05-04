#!/usr/bin/env python3
"""
make_controller_comparison_plots_v2.py

Creates the plots needed by the results page:

1. attitude_tracking_horizontal_comparison.png
   - yaw/heading, pitch, and roll in a horizontal row

2. attitude_error_comparison.png
   - heading, pitch, and roll errors in one plot

Also keeps optional position plots if you still want them:
   - position_tracking_comparison.png
   - position_error_comparison.png

Expected CSVs:
    sim_out_MLP_MPC_v2.csv
    sim_out_basePID.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MPC_CSV = "mpc_output_plots_v2/sim_out_MLP_MPC_v2.csv"
PID_CSV = "base_PID_plots/sim_out_basePID.csv"

OUTPUT_DIR = "results_assets"
DPI = 220

HEADING_CMD_DEG = 50.0
PITCH_CMD_DEG = 3.0
ROLL_CMD_DEG = 0.0

POSITION_REFERENCE_MODE = "mpc_final"
T_END = None


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find CSV: {path}")

    df = pd.read_csv(path)
    df = df.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)

    if T_END is not None:
        df = df[df["t"] <= float(T_END)].copy().reset_index(drop=True)

    return df


def get_series(df, candidates, fallback_value=None, required=False):
    for col in candidates:
        if col in df.columns:
            return df[col].to_numpy(dtype=float), col

    if required:
        raise ValueError(f"Missing required columns. Tried: {candidates}")

    if fallback_value is None:
        return None, None

    return np.full(len(df), float(fallback_value)), f"constant {fallback_value:g}"


def common_time(df_a, df_b):
    t0 = max(float(df_a["t"].min()), float(df_b["t"].min()))
    t1 = min(float(df_a["t"].max()), float(df_b["t"].max()))

    if T_END is not None:
        t1 = min(t1, float(T_END))

    dt_a = np.median(np.diff(df_a["t"].to_numpy(dtype=float)))
    dt_b = np.median(np.diff(df_b["t"].to_numpy(dtype=float)))
    dt = max(float(dt_a), float(dt_b))

    return np.arange(t0, t1 + 0.5 * dt, dt)


def interp_array(df, arr, t):
    return np.interp(t, df["t"].to_numpy(dtype=float), np.asarray(arr, dtype=float))


def style_axes(ax):
    ax.grid(True, alpha=0.38)
    ax.set_facecolor("white")


def savefig(fig, name):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def build_data(df, label, t):
    data = {}

    for key, candidates in {
        "x": ["x"],
        "y": ["y"],
        "z": ["z"],
        "yaw": ["yaw_deg"],
        "pitch": ["pitch_deg"],
        "roll": ["roll_deg"],
        "deltaS": ["deltaS_deg", "ctrl_deltaS_final_deg", "mpc_deltaS_cmd_deg"],
        "deltaD": ["deltaD_deg", "ctrl_deltaD_final_deg", "mpc_deltaD_cmd_deg"],
    }.items():
        arr, source = get_series(
            df,
            candidates,
            required=(key in ["x", "y", "z", "yaw", "pitch", "roll"]),
        )
        if arr is not None:
            data[key] = interp_array(df, arr, t)

    heading_cmd, heading_src = get_series(
        df,
        ["psi_cmd_deg", "mpc_yaw_cmd_deg", "heading_cmd_deg"],
        fallback_value=HEADING_CMD_DEG,
    )
    pitch_cmd, pitch_src = get_series(
        df,
        ["theta_cmd_deg", "mpc_pitch_cmd_deg", "pitch_cmd_deg"],
        fallback_value=PITCH_CMD_DEG,
    )
    roll_cmd, roll_src = get_series(
        df,
        ["roll_cmd_deg", "mpc_roll_cmd_deg"],
        fallback_value=ROLL_CMD_DEG,
    )

    data["heading_cmd"] = interp_array(df, heading_cmd, t)
    data["pitch_cmd"] = interp_array(df, pitch_cmd, t)
    data["roll_cmd"] = interp_array(df, roll_cmd, t)

    data["heading_err"] = wrap_deg(data["heading_cmd"] - data["yaw"])
    data["pitch_err"] = data["pitch_cmd"] - data["pitch"]
    data["roll_err"] = wrap_deg(data["roll_cmd"] - data["roll"])

    print(f"{label} command sources:")
    print(f"  heading: {heading_src}")
    print(f"  pitch:   {pitch_src}")
    print(f"  roll:    {roll_src}")

    return data


def plot_attitude_horizontal(t, mpc, pid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)

    specs = [
        ("yaw", "heading_cmd", "Heading / yaw", "deg"),
        ("pitch", "pitch_cmd", "Pitch", "deg"),
        ("roll", "roll_cmd", "Roll", "deg"),
    ]

    for ax, (state_key, cmd_key, title, unit) in zip(axes, specs):
        ax.plot(t, mpc[state_key], label=f"MPC {title}")
        ax.plot(t, pid[state_key], label=f"Base PID {title}")
        ax.plot(t, mpc[cmd_key], "--", label="Command")

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(unit)
        style_axes(ax)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Attitude Tracking Comparison: MPC vs. Base PID", y=1.03)
    savefig(fig, "attitude_tracking_horizontal_comparison.png")


def plot_attitude_error(t, mpc, pid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)

    specs = [
        ("heading_err", "Heading error"),
        ("pitch_err", "Pitch error"),
        ("roll_err", "Roll error"),
    ]

    for ax, (err_key, title) in zip(axes, specs):
        ax.plot(t, mpc[err_key], label=f"MPC {title}")
        ax.plot(t, pid[err_key], label=f"Base PID {title}")
        ax.axhline(0.0, linewidth=1.0, alpha=0.55)

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (deg)")
        style_axes(ax)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Tracking Error Comparison: MPC vs. Base PID", y=1.03)
    savefig(fig, "attitude_error_comparison.png")


def plot_position_tracking(t, mpc, pid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)

    for ax, key, ylabel in zip(axes, ["x", "y", "z"], ["x (m)", "y (m)", "z (m)"]):
        ax.plot(t, mpc[key], label=f"MPC {key}")
        ax.plot(t, pid[key], label=f"Base PID {key}")
        ax.set_title(f"{key} position")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Position History Comparison", y=1.03)
    savefig(fig, "position_tracking_comparison.png")


def get_position_reference(mpc, pid):
    if POSITION_REFERENCE_MODE == "mpc_final":
        return (
            np.full_like(mpc["x"], mpc["x"][-1]),
            np.full_like(mpc["y"], mpc["y"][-1]),
            np.full_like(mpc["z"], mpc["z"][-1]),
            "final MPC position",
        )

    if POSITION_REFERENCE_MODE == "pid_final":
        return (
            np.full_like(pid["x"], pid["x"][-1]),
            np.full_like(pid["y"], pid["y"][-1]),
            np.full_like(pid["z"], pid["z"][-1]),
            "final PID position",
        )

    if POSITION_REFERENCE_MODE == "shared_final_average":
        x_ref = 0.5 * (mpc["x"][-1] + pid["x"][-1])
        y_ref = 0.5 * (mpc["y"][-1] + pid["y"][-1])
        z_ref = 0.5 * (mpc["z"][-1] + pid["z"][-1])
        return (
            np.full_like(mpc["x"], x_ref),
            np.full_like(mpc["y"], y_ref),
            np.full_like(mpc["z"], z_ref),
            "average final position",
        )

    raise ValueError(f"Unknown POSITION_REFERENCE_MODE: {POSITION_REFERENCE_MODE}")


def plot_position_error(t, mpc, pid):
    x_ref, y_ref, z_ref, ref_label = get_position_reference(mpc, pid)

    mpc_err = np.sqrt((mpc["x"] - x_ref)**2 + (mpc["y"] - y_ref)**2 + (mpc["z"] - z_ref)**2)
    pid_err = np.sqrt((pid["x"] - x_ref)**2 + (pid["y"] - y_ref)**2 + (pid["z"] - z_ref)**2)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(t, mpc_err, label="MPC position error")
    ax.plot(t, pid_err, label="Base PID position error")
    ax.set_title(f"Position Error Compared to {ref_label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error magnitude (m)")
    style_axes(ax)
    ax.legend(loc="best")

    savefig(fig, "position_error_comparison.png")


def main():
    mpc_df = load_csv(MPC_CSV)
    pid_df = load_csv(PID_CSV)

    t = common_time(mpc_df, pid_df)

    print(f"Common time range: {t[0]:.3f} to {t[-1]:.3f} s, N={len(t)}")

    mpc = build_data(mpc_df, "MPC", t)
    pid = build_data(pid_df, "Base PID", t)

    plot_attitude_horizontal(t, mpc, pid)
    plot_attitude_error(t, mpc, pid)
    plot_position_tracking(t, mpc, pid)
    plot_position_error(t, mpc, pid)

    print(f"Done. Plots saved in: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
