#!/usr/bin/env python3
"""
plot_mpc_pid_comparisons_auto.py

Drop this script into the SAME directory as your MPC and PID/GainSchedule CSV
outputs, then run:

    python plot_mpc_pid_comparisons_auto.py

It will automatically find MPC and PID/GainSchedule CSV files, pair matching
versions when possible, and generate comparison plots, metrics, and a 3D
comparison animation.

Typical expected files:
    sim_out_MLP_MPC_v5.csv
    sim_out_GainSchedule_v5.csv

But the script is intentionally flexible and will also search subfolders.

Outputs:
    mpc_pid_comparison_outputs/
        metrics/
            <case>_metrics.csv
            <case>_summary.txt
        plots/<case>/
            attitude_tracking_horizontal_comparison.png
            attitude_error_comparison.png
            rates_comparison.png
            controls_comparison.png
            aero_angles_comparison.png
            airspeed_altitude_comparison.png
            ground_track_comparison.png
            metric_bar_comparison.png
        animations/<case>/
            <case>_3d_comparison.mp4 and <case>_3d_comparison.gif

Required CSV columns for core plots:
    t, yaw_deg, pitch_deg, roll_deg

Recommended columns:
    psi_cmd_deg, theta_cmd_deg, roll_cmd_deg
    hdg_err_deg, pitch_err_deg, roll_err_deg
    p, q, r
    deltaS_deg, deltaD_deg
    alpha_deg, beta_deg, V
    x, y, z
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# User settings
# =============================================================================

ROOT_DIR = Path(".")
OUTPUT_ROOT = Path("mpc_pid_comparison_outputs_v4")

# Set to None to compare every matching MPC/PID version found.
# Set to "v5" to only compare v5.
VERSION_FILTER = "v4"

# CSV auto-detection settings
SEARCH_RECURSIVELY = True
CSV_GLOB = "**/*.csv" if SEARCH_RECURSIVELY else "*.csv"

# Ignore files/folders created by this script or old comparison scripts.
IGNORE_PATH_PARTS = {
    "mpc_pid_comparison_outputs",
    "comparison_animations",
    "results_assets",
}
IGNORE_FILE_CONTAINS = {
    "comparison_metrics",
    "controller_comparison_metrics",
    "_metrics",
}

# Fallback commands if the CSV does not contain command columns.
# These match the MPC/PID v5 step command.
# Fallback only used if no command/error columns exist in the CSV.
# The uploaded v4 CSVs contain enough columns to reconstruct commands,
# so these should normally not be used.
DEFAULT_HEADING_CMD_1_DEG = 0.0
DEFAULT_HEADING_CMD_2_DEG = 0.0
DEFAULT_PITCH_CMD_DEG = 0.0
DEFAULT_ROLL_CMD_DEG = 0.0
DEFAULT_STEP_TIME_S = 10.0

# Plot controls
DPI = 220
SAVE_PLOTS = True
SHOW_PLOTS = False

MAKE_ATTITUDE_PLOTS = True
MAKE_ERROR_PLOTS = True
MAKE_RATE_PLOTS = True
MAKE_CONTROL_PLOTS = True
MAKE_AERO_AIRSPEED_PLOTS = True
MAKE_GROUND_TRACK_PLOTS = True
MAKE_METRIC_BAR_PLOT = True

# Animation controls
MAKE_COMPARISON_ANIMATION = True
STL_PATH = "lowerqualitymesh.stl"
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)
MAX_STL_TRIANGLES = 1200
ANIM_STRIDE = 10
ANIM_FPS = 30
ANIM_DPI = 140
ANIM_TRAIL_LEN = 250
ANIM_VIEW_ELEV = 22
ANIM_VIEW_AZIM = -55
ANIM_FOLLOW_MODE = "midpoint"  # "midpoint" or "fixed"
ANIM_ZOOM = 42.0


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RunFile:
    path: Path
    controller: str   # "MPC" or "PID"
    version: str      # "v5", "v3", or "unknown"
    case_key: str


# =============================================================================
# General helpers
# =============================================================================

def wrap_deg(angle):
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def rms(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x**2)))


def safe_series(df, col, default=0.0):
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return np.full(len(df), float(default), dtype=float)


def get_first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float), name
    if default is None:
        return None, None
    return np.full(len(df), float(default), dtype=float), f"constant {default:g}"


def interp_to_time(df, arr, t_common):
    return np.interp(
        t_common,
        df["t"].to_numpy(dtype=float),
        np.asarray(arr, dtype=float),
    )


def common_time(df_a, df_b):
    t0 = max(float(df_a["t"].min()), float(df_b["t"].min()))
    t1 = min(float(df_a["t"].max()), float(df_b["t"].max()))
    if t1 <= t0:
        raise ValueError("The two logs do not overlap in time.")

    dt_a = np.median(np.diff(df_a["t"].to_numpy(dtype=float)))
    dt_b = np.median(np.diff(df_b["t"].to_numpy(dtype=float)))
    dt = max(float(dt_a), float(dt_b))
    return np.arange(t0, t1 + 0.5 * dt, dt)


def load_csv(path):
    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError(f"{path} is missing required time column 't'.")
    df = df.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)
    return df


def style_axes(ax):
    ax.grid(True, alpha=0.38)
    ax.set_facecolor("white")


def save_fig(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot: {path}")
    plt.close(fig)


# =============================================================================
# CSV detection and pairing
# =============================================================================

def should_ignore_csv(path: Path) -> bool:
    parts_lower = {p.lower() for p in path.parts}
    if any(part.lower() in parts_lower for part in IGNORE_PATH_PARTS):
        return True

    name_lower = path.name.lower()
    if any(token.lower() in name_lower for token in IGNORE_FILE_CONTAINS):
        return True

    return False


def detect_controller(path: Path) -> str | None:
    s = str(path).lower()

    # Check PID/GainSchedule first because some old files may contain "mlp" but are not MPC.
    if any(token in s for token in ["gainschedule", "gain_schedule", "gain-schedule", "basepid", "base_pid", "pid"]):
        return "PID"

    if any(token in s for token in ["mpc", "mlp_mpc", "mlp-mpc"]):
        return "MPC"

    return None


def detect_version(path: Path) -> str:
    s = str(path).lower()
    matches = re.findall(r"v(\d+)", s)
    if matches:
        return f"v{matches[-1]}"
    return "unknown"


def clean_case_key(path: Path, controller: str, version: str) -> str:
    stem = path.stem.lower()

    # Remove common controller/output words.
    tokens_to_remove = [
        "sim_out", "mlp", "mpc", "gainschedule", "gain_schedule",
        "gain-schedule", "basepid", "base_pid", "pid", controller.lower(),
        version.lower(),
    ]
    key = stem
    for tok in tokens_to_remove:
        key = key.replace(tok, "")

    key = re.sub(r"[_\\-]+", "_", key).strip("_")
    if not key:
        key = version
    return key


def find_run_files(root: Path) -> list[RunFile]:
    run_files: list[RunFile] = []

    for path in root.glob(CSV_GLOB):
        if not path.is_file() or should_ignore_csv(path):
            continue

        controller = detect_controller(path)
        if controller is None:
            continue

        version = detect_version(path)
        if VERSION_FILTER is not None and version != VERSION_FILTER:
            continue

        case_key = clean_case_key(path, controller, version)
        run_files.append(RunFile(path=path, controller=controller, version=version, case_key=case_key))

    return run_files


def score_candidate(run: RunFile) -> tuple[int, int, float]:
    """
    Prefer files in the root directory, then files with version tags, then newer mtime.
    """
    root_preference = 1 if run.path.parent == Path(".") else 0
    version_preference = 1 if run.version != "unknown" else 0
    mtime = run.path.stat().st_mtime
    return (root_preference, version_preference, mtime)


def pair_runs(run_files: list[RunFile]) -> list[tuple[str, RunFile, RunFile]]:
    """
    Pair MPC and PID files by version first. If multiple candidates exist for a
    version, choose the best-scored one. If no version match exists, pair the
    best single MPC with the best single PID.
    """
    mpcs = [r for r in run_files if r.controller == "MPC"]
    pids = [r for r in run_files if r.controller == "PID"]

    if not mpcs:
        raise FileNotFoundError("No MPC CSV files found. Expected a filename/path containing 'MPC'.")
    if not pids:
        raise FileNotFoundError("No PID/GainSchedule CSV files found. Expected a filename/path containing 'PID' or 'GainSchedule'.")

    pairs = []

    versions = sorted(set(r.version for r in mpcs) & set(r.version for r in pids))
    if VERSION_FILTER is not None:
        versions = [v for v in versions if v == VERSION_FILTER]

    for version in versions:
        mpc_best = sorted([r for r in mpcs if r.version == version], key=score_candidate, reverse=True)[0]
        pid_best = sorted([r for r in pids if r.version == version], key=score_candidate, reverse=True)[0]
        case_name = version if version != "unknown" else "comparison"
        pairs.append((case_name, mpc_best, pid_best))

    if not pairs:
        mpc_best = sorted(mpcs, key=score_candidate, reverse=True)[0]
        pid_best = sorted(pids, key=score_candidate, reverse=True)[0]
        case_name = mpc_best.version if mpc_best.version != "unknown" else "comparison"
        pairs.append((case_name, mpc_best, pid_best))

    return pairs


# =============================================================================
# Command-aware data extraction
# =============================================================================

def fallback_heading_command(t):
    t = np.asarray(t, dtype=float)
    return np.where(
        t < float(DEFAULT_STEP_TIME_S),
        float(DEFAULT_HEADING_CMD_1_DEG),
        float(DEFAULT_HEADING_CMD_2_DEG),
    )


def _interp_column(df, col, t_common):
    return interp_to_time(df, df[col].to_numpy(dtype=float), t_common)


def _command_from_error(df, state_col, error_col, t_common, wrapped=False):
    state = df[state_col].to_numpy(dtype=float)
    err = df[error_col].to_numpy(dtype=float)
    cmd = state + err
    if wrapped:
        cmd = wrap_deg(cmd)
    return interp_to_time(df, cmd, t_common)


def build_data(df, label, t_common):
    data = {}

    required = ["yaw_deg", "pitch_deg", "roll_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}")

    for key, candidates in {
        "x": ["x"],
        "y": ["y"],
        "z": ["z"],
        "yaw": ["yaw_deg"],
        "pitch": ["pitch_deg"],
        "roll": ["roll_deg"],
        "deltaS": ["deltaS_deg", "ctrl_deltaS_final_deg", "mpc_deltaS_cmd_deg"],
        "deltaD": ["deltaD_deg", "ctrl_deltaD_final_deg", "mpc_deltaD_cmd_deg"],
        "alpha": ["alpha_deg", "mpc_alpha_deg", "ctrl_alpha_deg"],
        "beta": ["beta_deg", "mpc_beta_deg", "ctrl_beta_deg"],
        "V": ["V", "mpc_V", "ctrl_V"],
        "p": ["p"],
        "q": ["q"],
        "r": ["r"],
    }.items():
        arr, source = get_first_existing(df, candidates, default=None)
        if arr is not None:
            data[key] = interp_to_time(df, arr, t_common)

    # Commands: prefer directly logged command columns. For the v4 BasePID file,
    # command columns are not logged directly, but the controller errors are.
    # Reconstruct those commands with:
    #   heading_cmd = yaw_deg + ctrl_e_psi_deg
    #   pitch_cmd   = pitch_deg + ctrl_e_theta_deg
    #   roll_cmd    = ctrl_phi_cmd_deg, if available
    heading_cmd, heading_src = get_first_existing(
        df,
        ["psi_cmd_deg", "mpc_yaw_cmd_deg", "heading_cmd_deg"],
        default=None,
    )
    if heading_cmd is not None:
        data["heading_cmd"] = interp_to_time(df, heading_cmd, t_common)
    elif "ctrl_e_psi_deg" in df.columns:
        data["heading_cmd"] = _command_from_error(df, "yaw_deg", "ctrl_e_psi_deg", t_common, wrapped=True)
        heading_src = "reconstructed from yaw_deg + ctrl_e_psi_deg"
    elif "hdg_err_deg" in df.columns:
        data["heading_cmd"] = _command_from_error(df, "yaw_deg", "hdg_err_deg", t_common, wrapped=True)
        heading_src = "reconstructed from yaw_deg + hdg_err_deg"
    else:
        data["heading_cmd"] = fallback_heading_command(t_common)
        heading_src = "fallback constant/default heading command"

    pitch_cmd, pitch_src = get_first_existing(
        df,
        ["theta_cmd_deg", "mpc_pitch_cmd_deg", "pitch_cmd_deg"],
        default=None,
    )
    if pitch_cmd is not None:
        data["pitch_cmd"] = interp_to_time(df, pitch_cmd, t_common)
    elif "ctrl_e_theta_deg" in df.columns:
        data["pitch_cmd"] = _command_from_error(df, "pitch_deg", "ctrl_e_theta_deg", t_common, wrapped=False)
        pitch_src = "reconstructed from pitch_deg + ctrl_e_theta_deg"
    elif "pitch_err_deg" in df.columns:
        data["pitch_cmd"] = _command_from_error(df, "pitch_deg", "pitch_err_deg", t_common, wrapped=False)
        pitch_src = "reconstructed from pitch_deg + pitch_err_deg"
    else:
        data["pitch_cmd"] = np.full(len(t_common), float(DEFAULT_PITCH_CMD_DEG), dtype=float)
        pitch_src = "fallback constant/default pitch command"

    roll_cmd, roll_src = get_first_existing(
        df,
        ["roll_cmd_deg", "mpc_roll_cmd_deg", "ctrl_phi_cmd_deg", "phi_cmd_deg"],
        default=None,
    )
    if roll_cmd is not None:
        data["roll_cmd"] = interp_to_time(df, roll_cmd, t_common)
    elif "ctrl_e_phi_deg" in df.columns:
        data["roll_cmd"] = _command_from_error(df, "roll_deg", "ctrl_e_phi_deg", t_common, wrapped=True)
        roll_src = "reconstructed from roll_deg + ctrl_e_phi_deg"
    elif "roll_err_deg" in df.columns:
        data["roll_cmd"] = _command_from_error(df, "roll_deg", "roll_err_deg", t_common, wrapped=True)
        roll_src = "reconstructed from roll_deg + roll_err_deg"
    else:
        data["roll_cmd"] = np.full(len(t_common), float(DEFAULT_ROLL_CMD_DEG), dtype=float)
        roll_src = "fallback constant/default roll command"

    # Prefer logged error columns when present. This avoids small reconstruction
    # differences and uses each controller's own definition of error.
    err, src = get_first_existing(df, ["hdg_err_deg", "mpc_heading_err_deg", "ctrl_e_psi_deg"], default=None)
    data["heading_err"] = interp_to_time(df, err, t_common) if err is not None else wrap_deg(data["heading_cmd"] - data["yaw"])

    err, src = get_first_existing(df, ["pitch_err_deg", "mpc_pitch_err_deg", "ctrl_e_theta_deg"], default=None)
    data["pitch_err"] = interp_to_time(df, err, t_common) if err is not None else data["pitch_cmd"] - data["pitch"]

    err, src = get_first_existing(df, ["roll_err_deg", "mpc_roll_err_deg", "ctrl_e_phi_deg"], default=None)
    data["roll_err"] = interp_to_time(df, err, t_common) if err is not None else wrap_deg(data["roll_cmd"] - data["roll"])

    # Convert rates from rad/s to deg/s when present. If the CSV already has
    # deg/s versions, use those instead.
    for rate in ["p", "q", "r"]:
        direct_col = f"{rate}_deg_s"
        mpc_col = f"mpc_{rate}_deg_s"
        if direct_col in df.columns:
            data[f"{rate}_deg_s"] = _interp_column(df, direct_col, t_common)
        elif mpc_col in df.columns:
            data[f"{rate}_deg_s"] = _interp_column(df, mpc_col, t_common)
        elif rate in data:
            data[f"{rate}_deg_s"] = np.rad2deg(data[rate])

    print(f"{label} command sources:")
    print(f"  heading: {heading_src}")
    print(f"  pitch:   {pitch_src}")
    print(f"  roll:    {roll_src}")

    return data


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(data, label):
    metrics = {
        "controller": label,
        "heading_RMSE_deg": rms(data["heading_err"]),
        "pitch_RMSE_deg": rms(data["pitch_err"]),
        "roll_RMSE_deg": rms(data["roll_err"]),
        "mean_abs_heading_err_deg": float(np.mean(np.abs(data["heading_err"]))),
        "mean_abs_pitch_err_deg": float(np.mean(np.abs(data["pitch_err"]))),
        "mean_abs_roll_err_deg": float(np.mean(np.abs(data["roll_err"]))),
        "max_abs_heading_err_deg": float(np.max(np.abs(data["heading_err"]))),
        "max_abs_pitch_err_deg": float(np.max(np.abs(data["pitch_err"]))),
        "max_abs_roll_err_deg": float(np.max(np.abs(data["roll_err"]))),
    }

    for key in ["p_deg_s", "q_deg_s", "r_deg_s"]:
        if key in data:
            metrics[f"{key}_RMS"] = rms(data[key])
            metrics[f"{key}_max_abs"] = float(np.max(np.abs(data[key])))

    for key in ["deltaS", "deltaD"]:
        if key in data:
            metrics[f"{key}_RMS_deg"] = rms(data[key])
            metrics[f"{key}_max_abs_deg"] = float(np.max(np.abs(data[key])))
            metrics[f"{key}_step_RMS_deg"] = rms(np.diff(data[key])) if len(data[key]) > 1 else 0.0

    for key in ["alpha", "beta"]:
        if key in data:
            metrics[f"{key}_max_abs_deg"] = float(np.max(np.abs(data[key])))
            metrics[f"{key}_RMS_deg"] = rms(data[key])

    if "V" in data:
        metrics["mean_V_m_s"] = float(np.mean(data["V"]))
        metrics["min_V_m_s"] = float(np.min(data["V"]))
        metrics["max_V_m_s"] = float(np.max(data["V"]))

    return metrics


def write_metrics(case_name, mpc, pid):
    out_dir = OUTPUT_ROOT / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        compute_metrics(mpc, "MPC"),
        compute_metrics(pid, "PID/GainSchedule"),
    ]
    metrics_df = pd.DataFrame(rows)

    csv_path = out_dir / f"{case_name}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics: {csv_path}")

    summary_path = out_dir / f"{case_name}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Controller comparison summary: {case_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nPercent change from PID/GainSchedule to MPC:\n")
        f.write("(negative means MPC is lower for that metric)\n\n")

        pid_row = metrics_df[metrics_df["controller"] == "PID/GainSchedule"].iloc[0]
        mpc_row = metrics_df[metrics_df["controller"] == "MPC"].iloc[0]
        for col in metrics_df.columns:
            if col == "controller":
                continue
            pid_val = float(pid_row[col])
            mpc_val = float(mpc_row[col])
            if abs(pid_val) < 1e-12:
                pct = np.nan
            else:
                pct = 100.0 * (mpc_val - pid_val) / abs(pid_val)
            f.write(f"{col:30s}: PID={pid_val:12.5g}, MPC={mpc_val:12.5g}, change={pct:9.2f}%\n")

    print(f"Saved summary: {summary_path}")
    return metrics_df


# =============================================================================
# Plotting
# =============================================================================

def plot_attitude(case_name, t, mpc, pid, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)
    specs = [
        ("yaw", "heading_cmd", "Heading / yaw", "deg"),
        ("pitch", "pitch_cmd", "Pitch", "deg"),
        ("roll", "roll_cmd", "Roll", "deg"),
    ]

    for ax, (state_key, cmd_key, title, unit) in zip(axes, specs):
        ax.plot(t, mpc[state_key], label=f"MPC {title}")
        ax.plot(t, pid[state_key], label=f"PID {title}")
        ax.plot(t, mpc[cmd_key], "--", label="Command")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(unit)
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.suptitle(f"{case_name}: Attitude Tracking Comparison", y=1.03)
    save_fig(fig, out_dir, "attitude_tracking_horizontal_comparison.png")


def plot_errors(case_name, t, mpc, pid, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)
    specs = [
        ("heading_err", "Heading error"),
        ("pitch_err", "Pitch error"),
        ("roll_err", "Roll error"),
    ]

    for ax, (key, title) in zip(axes, specs):
        ax.plot(t, mpc[key], label=f"MPC {title}")
        ax.plot(t, pid[key], label=f"PID {title}")
        ax.axhline(0.0, linewidth=1.0, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (deg)")
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.suptitle(f"{case_name}: Tracking Error Comparison", y=1.03)
    save_fig(fig, out_dir, "attitude_error_comparison.png")


def plot_rates(case_name, t, mpc, pid, out_dir):
    available = [key for key in ["p_deg_s", "q_deg_s", "r_deg_s"] if key in mpc and key in pid]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.7), sharex=True)
    if len(available) == 1:
        axes = [axes]

    titles = {
        "p_deg_s": "Roll rate p",
        "q_deg_s": "Pitch rate q",
        "r_deg_s": "Yaw rate r",
    }

    for ax, key in zip(axes, available):
        ax.plot(t, mpc[key], label=f"MPC {key}")
        ax.plot(t, pid[key], label=f"PID {key}")
        ax.set_title(titles[key])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg/s")
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.suptitle(f"{case_name}: Body Rate Comparison", y=1.03)
    save_fig(fig, out_dir, "rates_comparison.png")


def plot_controls(case_name, t, mpc, pid, out_dir):
    available = [key for key in ["deltaS", "deltaD"] if key in mpc and key in pid]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6.8 * len(available), 4.7), sharex=True)
    if len(available) == 1:
        axes = [axes]

    titles = {
        "deltaS": "Symmetric elevon deltaS",
        "deltaD": "Differential elevon deltaD",
    }

    for ax, key in zip(axes, available):
        ax.plot(t, mpc[key], label=f"MPC {key}")
        ax.plot(t, pid[key], label=f"PID {key}")
        ax.set_title(titles[key])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg")
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.suptitle(f"{case_name}: Control Input Comparison", y=1.03)
    save_fig(fig, out_dir, "controls_comparison.png")


def plot_aero_air(case_name, t, mpc, pid, out_dir):
    keys = [key for key in ["alpha", "beta", "V"] if key in mpc and key in pid]
    if not keys:
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.7), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    labels = {
        "alpha": ("Angle of attack", "deg"),
        "beta": ("Sideslip", "deg"),
        "V": ("Airspeed", "m/s"),
    }

    for ax, key in zip(axes, keys):
        title, ylabel = labels[key]
        ax.plot(t, mpc[key], label=f"MPC {key}")
        ax.plot(t, pid[key], label=f"PID {key}")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.suptitle(f"{case_name}: Airdata Comparison", y=1.03)
    save_fig(fig, out_dir, "aero_angles_airspeed_comparison.png")

    if "z" in mpc and "z" in pid:
        fig, ax = plt.subplots(figsize=(10, 4.7))
        ax.plot(t, -mpc["z"], label="MPC altitude proxy")
        ax.plot(t, -pid["z"], label="PID altitude proxy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("-z (m)")
        ax.set_title(f"{case_name}: Altitude Proxy")
        style_axes(ax)
        ax.legend()
        save_fig(fig, out_dir, "altitude_comparison.png")


def plot_ground_track(case_name, mpc, pid, out_dir):
    if not all(k in mpc for k in ["x", "y"]) or not all(k in pid for k in ["x", "y"]):
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot(mpc["x"], mpc["y"], label="MPC")
    ax.plot(pid["x"], pid["y"], label="PID")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{case_name}: Ground Track")
    ax.axis("equal")
    style_axes(ax)
    ax.legend()
    save_fig(fig, out_dir, "ground_track_comparison.png")


def plot_metric_bars(case_name, metrics_df, out_dir):
    numeric_cols = [c for c in metrics_df.columns if c != "controller"]
    key_metrics = [
        c for c in [
            "heading_RMSE_deg",
            "pitch_RMSE_deg",
            "roll_RMSE_deg",
            "max_abs_heading_err_deg",
            "max_abs_pitch_err_deg",
            "max_abs_roll_err_deg",
            "deltaS_RMS_deg",
            "deltaD_RMS_deg",
            "beta_max_abs_deg",
        ]
        if c in numeric_cols
    ]

    if not key_metrics:
        return

    x = np.arange(len(key_metrics))
    width = 0.36

    mpc_row = metrics_df[metrics_df["controller"] == "MPC"].iloc[0]
    pid_row = metrics_df[metrics_df["controller"] == "PID/GainSchedule"].iloc[0]

    mpc_vals = [float(mpc_row[m]) for m in key_metrics]
    pid_vals = [float(pid_row[m]) for m in key_metrics]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - width / 2, mpc_vals, width, label="MPC")
    ax.bar(x + width / 2, pid_vals, width, label="PID")
    ax.set_xticks(x)
    ax.set_xticklabels(key_metrics, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{case_name}: Key Metric Comparison")
    ax.grid(True, axis="y", alpha=0.38)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "metric_bar_comparison.png")


# =============================================================================
# Animation
# =============================================================================

def Rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def body_to_ned_rotation(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def stl_alignment_rotation(offset_deg):
    roll_o, pitch_o, yaw_o = np.deg2rad(offset_deg)
    return Rz(yaw_o) @ Ry(pitch_o) @ Rx(roll_o)


def make_fallback_aircraft_mesh():
    nose = np.array([1.00, 0.00, 0.00])
    tail = np.array([-0.70, 0.00, 0.00])
    left_wing = np.array([-0.15, -0.95, 0.05])
    right_wing = np.array([-0.15, 0.95, 0.05])
    top = np.array([-0.20, 0.00, -0.16])
    bottom = np.array([-0.20, 0.00, 0.16])
    return np.array([
        [nose, right_wing, top],
        [nose, top, left_wing],
        [nose, bottom, right_wing],
        [nose, left_wing, bottom],
        [tail, top, right_wing],
        [tail, left_wing, top],
        [tail, right_wing, bottom],
        [tail, bottom, left_wing],
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

            vectors -= vectors.reshape(-1, 3).mean(axis=0)
            vectors *= float(STL_SCALE)

            R_offset = stl_alignment_rotation(STL_ROTATION_OFFSET_DEG)
            vectors = np.einsum("ij,tkj->tki", R_offset, vectors)
            return vectors
        except Exception as e:
            print(f"Could not load STL '{stl_path}': {e}. Using fallback mesh.")

    return make_fallback_aircraft_mesh()


def ned_to_plot(points_ned):
    T = np.diag([1.0, 1.0, -1.0])
    return np.einsum("ij,...j->...i", T, points_ned)


def transform_mesh_to_plot(mesh_body, row):
    pos_ned = np.array([row["x"], row["y"], row["z"]], dtype=float)
    R_bw = body_to_ned_rotation(row["yaw_deg"], row["pitch_deg"], row["roll_deg"])
    mesh_ned = np.einsum("ij,tkj->tki", R_bw, mesh_body) + pos_ned
    return ned_to_plot(mesh_ned)


def resample_full_df_for_animation(df, t_common):
    required = ["t", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot animate; missing columns: {missing}")

    out = {"t": t_common}
    t_src = df["t"].to_numpy(dtype=float)
    for col in required[1:]:
        out[col] = np.interp(t_common, t_src, df[col].to_numpy(dtype=float))
    return pd.DataFrame(out)


def set_axes_equal_around(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def make_3d_comparison_animation(case_name, mpc_df, pid_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    t_full = common_time(mpc_df, pid_df)
    t_common = t_full[::max(1, int(ANIM_STRIDE))]
    mpc = resample_full_df_for_animation(mpc_df, t_common)
    pid = resample_full_df_for_animation(pid_df, t_common)

    mesh_body = load_body_mesh()

    mpc_x, mpc_y, mpc_z = mpc["x"].to_numpy(), mpc["y"].to_numpy(), -mpc["z"].to_numpy()
    pid_x, pid_y, pid_z = pid["x"].to_numpy(), pid["y"].to_numpy(), -pid["z"].to_numpy()

    all_xyz = np.column_stack([np.r_[mpc_x, pid_x], np.r_[mpc_y, pid_y], np.r_[mpc_z, pid_z]])
    fixed_center = np.mean(all_xyz, axis=0)
    fixed_radius = 0.5 * np.max(np.ptp(all_xyz, axis=0)) + 8.0
    fixed_radius = max(float(fixed_radius), 5.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{case_name}: MPC vs PID 3D Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    ax.view_init(elev=ANIM_VIEW_ELEV, azim=ANIM_VIEW_AZIM)

    mpc_mesh = Poly3DCollection(transform_mesh_to_plot(mesh_body, mpc.iloc[0]), alpha=0.90)
    mpc_mesh.set_facecolor("tab:blue")
    mpc_mesh.set_edgecolor("k")
    mpc_mesh.set_linewidth(0.2)

    pid_mesh = Poly3DCollection(transform_mesh_to_plot(mesh_body, pid.iloc[0]), alpha=0.32)
    pid_mesh.set_facecolor("0.55")
    pid_mesh.set_edgecolor("k")
    pid_mesh.set_linewidth(0.15)

    ax.add_collection3d(pid_mesh)
    ax.add_collection3d(mpc_mesh)

    mpc_trail, = ax.plot([], [], [], color="tab:blue", linewidth=2.0, label="MPC")
    pid_trail, = ax.plot([], [], [], color="0.45", linewidth=1.6, linestyle="--", alpha=0.55, label="PID/GainSchedule")

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def update(i):
        row_mpc = mpc.iloc[i]
        row_pid = pid.iloc[i]

        mpc_mesh.set_verts(transform_mesh_to_plot(mesh_body, row_mpc))
        pid_mesh.set_verts(transform_mesh_to_plot(mesh_body, row_pid))

        i0 = max(0, i - int(ANIM_TRAIL_LEN))

        mpc_trail.set_data(mpc_x[i0:i + 1], mpc_y[i0:i + 1])
        mpc_trail.set_3d_properties(mpc_z[i0:i + 1])

        pid_trail.set_data(pid_x[i0:i + 1], pid_y[i0:i + 1])
        pid_trail.set_3d_properties(pid_z[i0:i + 1])

        if ANIM_FOLLOW_MODE.lower() == "midpoint":
            center = np.array([
                0.5 * (mpc_x[i] + pid_x[i]),
                0.5 * (mpc_y[i] + pid_y[i]),
                0.5 * (mpc_z[i] + pid_z[i]),
            ], dtype=float)
            sep = np.linalg.norm(np.array([mpc_x[i] - pid_x[i], mpc_y[i] - pid_y[i], mpc_z[i] - pid_z[i]]))
            radius = max(4.0, 0.5 * sep + 10.0)
            radius = radius / max(float(ANIM_ZOOM) / 42.0, 1e-6)
            set_axes_equal_around(ax, center, radius)
        else:
            set_axes_equal_around(ax, fixed_center, fixed_radius)

        time_text.set_text(f"t = {t_common[i]:.2f} s")
        return [mpc_mesh, pid_mesh, mpc_trail, pid_trail, time_text]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t_common),
        interval=1000.0 / ANIM_FPS,
        blit=False,
        cache_frame_data=False,
    )

    mp4_path = out_dir / f"{case_name}_3d_comparison.mp4"
    gif_path = out_dir / f"{case_name}_3d_comparison.gif"

    try:
        print(f"Saving animation MP4: {mp4_path}")
        anim.save(mp4_path, writer=FFMpegWriter(fps=ANIM_FPS, bitrate=1800), dpi=ANIM_DPI)
        print(f"Saved animation: {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed: {e}")

    try:
        print(f"Saving animation GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=ANIM_FPS), dpi=110)
        print(f"Saved animation: {gif_path}")
    except Exception as e:
        print(f"GIF save failed: {e}")

    plt.close(fig)


# =============================================================================
# Main comparison runner
# =============================================================================

def run_comparison(case_name, mpc_run: RunFile, pid_run: RunFile):
    print("\n" + "=" * 80)
    print(f"Comparison case: {case_name}")
    print(f"MPC CSV: {mpc_run.path}")
    print(f"PID CSV: {pid_run.path}")
    print("=" * 80)

    mpc_df = load_csv(mpc_run.path)
    pid_df = load_csv(pid_run.path)

    t = common_time(mpc_df, pid_df)
    print(f"Common time range: {t[0]:.3f} to {t[-1]:.3f} s, N={len(t)}")

    mpc = build_data(mpc_df, "MPC", t)
    pid = build_data(pid_df, "PID/GainSchedule", t)

    plots_dir = OUTPUT_ROOT / "plots" / case_name
    metrics_df = write_metrics(case_name, mpc, pid)

    if MAKE_ATTITUDE_PLOTS:
        plot_attitude(case_name, t, mpc, pid, plots_dir)
    if MAKE_ERROR_PLOTS:
        plot_errors(case_name, t, mpc, pid, plots_dir)
    if MAKE_RATE_PLOTS:
        plot_rates(case_name, t, mpc, pid, plots_dir)
    if MAKE_CONTROL_PLOTS:
        plot_controls(case_name, t, mpc, pid, plots_dir)
    if MAKE_AERO_AIRSPEED_PLOTS:
        plot_aero_air(case_name, t, mpc, pid, plots_dir)
    if MAKE_GROUND_TRACK_PLOTS:
        plot_ground_track(case_name, mpc, pid, plots_dir)
    if MAKE_METRIC_BAR_PLOT:
        plot_metric_bars(case_name, metrics_df, plots_dir)

    if MAKE_COMPARISON_ANIMATION:
        make_3d_comparison_animation(case_name, mpc_df, pid_df, OUTPUT_ROOT / "animations" / case_name)


def main():
    root = ROOT_DIR.resolve()
    os.chdir(root)
    print(f"Searching for CSV files under: {root}")

    run_files = find_run_files(Path("."))
    if not run_files:
        raise FileNotFoundError("No MPC/PID CSV files found.")

    print("\nDetected run CSVs:")
    for run in run_files:
        print(f"  [{run.controller:3s}] {run.version:8s} {run.path}")

    pairs = pair_runs(run_files)

    print("\nPlanned comparisons:")
    for case_name, mpc_run, pid_run in pairs:
        print(f"  {case_name}: {mpc_run.path.name}  vs  {pid_run.path.name}")

    for case_name, mpc_run, pid_run in pairs:
        run_comparison(case_name, mpc_run, pid_run)

    print("\nDone.")
    print(f"All comparison outputs are under: {OUTPUT_ROOT.resolve()}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
