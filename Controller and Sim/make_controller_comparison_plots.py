#!/usr/bin/env python3
"""
make_controller_comparison_plots.py

Create results-page plots comparing the MLP-MPC controller against the baseline
non-variable-weight cascaded PID controller.

Expected CSVs:
    sim_out_MLP_MPC_v2.csv
    sim_out_basePID.csv

Outputs:
    results_assets/
        heading_tracking_comparison.png
        attitude_tracking_comparison.png
        position_tracking_comparison.png
        position_error_comparison.png
        control_surface_comparison.png

The script is written to tolerate your current CSV format:
    - MPC CSV has psi_cmd_deg / theta_cmd_deg / roll_cmd_deg.
    - Base PID CSV may not have explicit command columns, so fallback constants
      are used for attitude commands.
    - If x_cmd/y_cmd/z_cmd are unavailable, the position error plot uses the
      final MPC position as the reference point so both controllers are compared
      to the same endpoint.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

MPC_CSV = "mpc_output_plots_v2/sim_out_MLP_MPC_v2.csv"
PID_CSV = "base_PID_plots/sim_out_basePID.csv"

OUTPUT_DIR = "results_assets"
DPI = 220

# Fallback commands if a CSV does not include command columns.
HEADING_CMD_DEG = 50.0
PITCH_CMD_DEG = 3.0
ROLL_CMD_DEG = 0.0

# Use this when position command columns are not available.
# Options:
#   "mpc_final" -> compare both controllers to the final MPC position
#   "pid_final" -> compare both controllers to the final PID position
#   "shared_final_average" -> average of final MPC/PID positions
POSITION_REFERENCE_MODE = "mpc_final"

# Optional time limit; set to None for full shared time range.
T_END = None


# =============================================================================
# Helpers
# =============================================================================

def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find CSV: {path}")

    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError(f"{path} does not have a 't' column.")

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


def interp_col(df, col, t):
    return np.interp(t, df["t"].to_numpy(dtype=float), df[col].to_numpy(dtype=float))


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


# =============================================================================
# Plot builders
# =============================================================================

def plot_heading_tracking(t, mpc, pid):
    fig, ax = plt.subplots(figsize=(10, 5.4))

    ax.plot(t, mpc["yaw"], label="MPC heading")
    ax.plot(t, pid["yaw"], label="Base PID heading")
    ax.plot(t, mpc["heading_cmd"], "--", label="Heading command")

    ax.set_title("Heading Tracking: MLP-MPC vs. Base Cascaded PID")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading / yaw (deg)")
    style_axes(ax)
    ax.legend(loc="best")

    savefig(fig, "heading_tracking_comparison.png")


def plot_attitude_tracking(t, mpc, pid):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    items = [
        ("yaw", "heading_cmd", "Heading / yaw (deg)", "Heading"),
        ("pitch", "pitch_cmd", "Pitch (deg)", "Pitch"),
        ("roll", "roll_cmd", "Roll (deg)", "Roll"),
    ]

    for ax, (state_key, cmd_key, ylabel, title) in zip(axes, items):
        ax.plot(t, mpc[state_key], label=f"MPC {title.lower()}")
        ax.plot(t, pid[state_key], label=f"Base PID {title.lower()}")
        ax.plot(t, mpc[cmd_key], "--", label=f"{title} command")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude Tracking Comparison", y=0.995)

    savefig(fig, "attitude_tracking_comparison.png")


def plot_position_tracking(t, mpc, pid):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, key, ylabel in zip(axes, ["x", "y", "z"], ["x (m)", "y (m)", "z (m)"]):
        ax.plot(t, mpc[key], label=f"MPC {key}")
        ax.plot(t, pid[key], label=f"Base PID {key}")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position History: MLP-MPC vs. Base Cascaded PID", y=0.995)

    savefig(fig, "position_tracking_comparison.png")


def get_position_reference(mpc, pid):
    # Use command columns if present in both or either dataset.
    # Otherwise use one common reference endpoint.
    if all(k in mpc for k in ["x_cmd", "y_cmd", "z_cmd"]):
        return mpc["x_cmd"], mpc["y_cmd"], mpc["z_cmd"], "CSV command"

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


def plot_control_surfaces(t, mpc, pid):
    if "deltaS" not in mpc or "deltaD" not in mpc or "deltaS" not in pid or "deltaD" not in pid:
        print("Skipping control surface plot because deltaS/deltaD data is missing.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, mpc["deltaS"], label="MPC δS")
    axes[0].plot(t, pid["deltaS"], label="Base PID δS")
    axes[0].set_ylabel("δS (deg)")
    axes[0].set_title("Control Surface Commands")
    style_axes(axes[0])
    axes[0].legend(loc="best")

    axes[1].plot(t, mpc["deltaD"], label="MPC δD")
    axes[1].plot(t, pid["deltaD"], label="Base PID δD")
    axes[1].set_ylabel("δD (deg)")
    axes[1].set_xlabel("Time (s)")
    style_axes(axes[1])
    axes[1].legend(loc="best")

    savefig(fig, "control_surface_comparison.png")


# =============================================================================
# Main
# =============================================================================

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
        arr, source = get_series(df, candidates, required=(key in ["x", "y", "z", "yaw", "pitch", "roll"]))
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

    for key in ["x_cmd", "y_cmd", "z_cmd"]:
        if key in df.columns:
            data[key] = interp_col(df, key, t)

    print(f"{label} command sources:")
    print(f"  heading: {heading_src}")
    print(f"  pitch:   {pitch_src}")
    print(f"  roll:    {roll_src}")

    return data


def main():
    mpc_df = load_csv(MPC_CSV)
    pid_df = load_csv(PID_CSV)
    t = common_time(mpc_df, pid_df)

    print(f"Common time range: {t[0]:.3f} to {t[-1]:.3f} s, N={len(t)}")

    mpc = build_data(mpc_df, "MPC", t)
    pid = build_data(pid_df, "Base PID", t)

    plot_heading_tracking(t, mpc, pid)
    plot_attitude_tracking(t, mpc, pid)
    plot_position_tracking(t, mpc, pid)
    plot_position_error(t, mpc, pid)
    plot_control_surfaces(t, mpc, pid)

    print(f"Done. Plots saved in: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
