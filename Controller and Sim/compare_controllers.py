#!/usr/bin/env python3
"""
compare_controllers.py

Compares original fixed-gain controller outputs against
MLP gain-scheduled controller outputs.

Expected CSV columns include:
    t
    yaw_deg, pitch_deg, roll_deg
    psi_cmd_deg, theta_cmd_deg
    hdg_err_deg, pitch_err_deg
    p, q, r
    deltaS_deg, deltaD_deg
    beta_deg, alpha_deg, V
    x, y, z

Run:
    python compare_controllers.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User inputs: your actual file naming scheme
# ============================================================

CASES = {
    "testA": {
        "fixed": "sim_out_testA_og.csv",
        "mlp_gs": "sim_out_testA_mlp1.csv",
    },
    "testB": {
        "fixed": "sim_out_testB_og.csv",
        "mlp_gs": "sim_out_testB_mlp1.csv",
    },
    "testC": {
        "fixed": "sim_out_testC_og.csv",
        "mlp_gs": "sim_out_testC_mlp1.csv",
    },
    "testD": {
        "fixed": "sim_out_testD_og.csv",
        "mlp_gs": "sim_out_testD_mlp1.csv",
    },
    "testD2": {
        "fixed": "sim_out_testD2_og.csv",
        "mlp_gs": "sim_out_testD2_mlp1.csv",
    },
    "testE": {
        "fixed": "sim_out_testE_og.csv",
        "mlp_gs": "sim_out_testE_mlp1.csv",
    },
    "testF": {
        "fixed": "sim_out_testF_og.csv",
        "mlp_gs": "sim_out_testF_mlp1.csv",
    },
}

# Match these to the control limits used in run_sim.py.
# These are only used to compute saturation fraction.
DELTAS_MAX_DEG = 20.0
DELTAD_MAX_DEG = 10.0

# Output summary table
METRICS_OUTPUT_CSV = "controller_comparison_metrics_mlp1.csv"

# Toggle plot groups
MAKE_TRACKING_PLOTS = True
MAKE_ERROR_PLOTS = True
MAKE_RATE_PLOTS = True
MAKE_CONTROL_PLOTS = True
MAKE_ENVELOPE_PLOTS = True
MAKE_GROUND_TRACK_PLOTS = True
MAKE_METRIC_BAR_PLOTS = True


# ============================================================
# Utility functions
# ============================================================

def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def wrap_deg(a):
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def safe_col(df, col, default=0.0):
    """
    Return df[col] as ndarray if it exists. Otherwise return a default array.
    """
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def check_file_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find required CSV file: {path}")


def load_csv(path):
    check_file_exists(path)
    return pd.read_csv(path)


def get_heading_error(df):
    """
    Prefer hdg_err_deg if it exists.
    Otherwise compute psi_cmd_deg - yaw_deg if possible.
    """
    if "hdg_err_deg" in df.columns:
        return df["hdg_err_deg"].to_numpy(dtype=float)

    if "psi_cmd_deg" in df.columns and "yaw_deg" in df.columns:
        return wrap_deg(df["psi_cmd_deg"].to_numpy(dtype=float) - df["yaw_deg"].to_numpy(dtype=float))

    return np.zeros(len(df), dtype=float)


def get_pitch_error(df):
    """
    Prefer pitch_err_deg if it exists.
    Otherwise compute theta_cmd_deg - pitch_deg if possible.
    """
    if "pitch_err_deg" in df.columns:
        return df["pitch_err_deg"].to_numpy(dtype=float)

    if "theta_cmd_deg" in df.columns and "pitch_deg" in df.columns:
        return df["theta_cmd_deg"].to_numpy(dtype=float) - df["pitch_deg"].to_numpy(dtype=float)

    return np.zeros(len(df), dtype=float)


def get_rate_deg_s(df, rate_col):
    """
    p, q, r are expected to be in rad/s.
    Converts to deg/s.
    """
    return np.rad2deg(safe_col(df, rate_col))


# ============================================================
# Metrics
# ============================================================

def compute_metrics(df, deltaS_max=DELTAS_MAX_DEG, deltaD_max=DELTAD_MAX_DEG):
    hdg_err = get_heading_error(df)
    pitch_err = get_pitch_error(df)

    p_deg_s = get_rate_deg_s(df, "p")
    q_deg_s = get_rate_deg_s(df, "q")
    r_deg_s = get_rate_deg_s(df, "r")

    deltaS = safe_col(df, "deltaS_deg")
    deltaD = safe_col(df, "deltaD_deg")

    beta = safe_col(df, "beta_deg")
    alpha = safe_col(df, "alpha_deg")
    V = safe_col(df, "V")

    metrics = {
        "heading_RMSE_deg": rms(hdg_err),
        "pitch_RMSE_deg": rms(pitch_err),

        "max_abs_heading_err_deg": float(np.max(np.abs(hdg_err))),
        "max_abs_pitch_err_deg": float(np.max(np.abs(pitch_err))),

        "mean_abs_heading_err_deg": float(np.mean(np.abs(hdg_err))),
        "mean_abs_pitch_err_deg": float(np.mean(np.abs(pitch_err))),

        "max_abs_p_deg_s": float(np.max(np.abs(p_deg_s))),
        "max_abs_q_deg_s": float(np.max(np.abs(q_deg_s))),
        "max_abs_r_deg_s": float(np.max(np.abs(r_deg_s))),

        "rms_p_deg_s": rms(p_deg_s),
        "rms_q_deg_s": rms(q_deg_s),
        "rms_r_deg_s": rms(r_deg_s),

        "deltaS_RMS_deg": rms(deltaS),
        "deltaD_RMS_deg": rms(deltaD),

        "deltaS_max_abs_deg": float(np.max(np.abs(deltaS))),
        "deltaD_max_abs_deg": float(np.max(np.abs(deltaD))),

        "deltaS_sat_fraction": float(np.mean(np.abs(deltaS) > 0.95 * deltaS_max)),
        "deltaD_sat_fraction": float(np.mean(np.abs(deltaD) > 0.95 * deltaD_max)),

        "deltaS_step_RMS_deg": rms(np.diff(deltaS)) if len(deltaS) > 1 else 0.0,
        "deltaD_step_RMS_deg": rms(np.diff(deltaD)) if len(deltaD) > 1 else 0.0,

        "max_abs_beta_deg": float(np.max(np.abs(beta))),
        "max_abs_alpha_deg": float(np.max(np.abs(alpha))),

        "mean_V_m_s": float(np.mean(V)),
        "min_V_m_s": float(np.min(V)),
        "max_V_m_s": float(np.max(V)),
    }

    return metrics


def build_metrics_table(cases):
    rows = []

    for case_name, paths in cases.items():
        for controller_name, csv_path in paths.items():
            df = load_csv(csv_path)
            metrics = compute_metrics(df)

            row = {
                "case": case_name,
                "controller": controller_name,
                "csv": csv_path,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Plotting helpers
# ============================================================

def load_pair(paths):
    fixed = load_csv(paths["fixed"])
    mlp_gs = load_csv(paths["mlp_gs"])
    return fixed, mlp_gs


def plot_tracking_pair(case_name, fixed, mlp_gs):
    if "t" not in fixed.columns or "t" not in mlp_gs.columns:
        return

    # Heading / yaw
    if "yaw_deg" in fixed.columns and "yaw_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["yaw_deg"], label="Original yaw")
        plt.plot(mlp_gs["t"], mlp_gs["yaw_deg"], label="MLP-GS yaw")

        if "psi_cmd_deg" in fixed.columns:
            plt.plot(fixed["t"], fixed["psi_cmd_deg"], "k--", label="Command")

        plt.xlabel("Time [s]")
        plt.ylabel("Yaw / heading [deg]")
        plt.title(f"{case_name}: Heading Tracking")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # Pitch
    if "pitch_deg" in fixed.columns and "pitch_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["pitch_deg"], label="Original pitch")
        plt.plot(mlp_gs["t"], mlp_gs["pitch_deg"], label="MLP-GS pitch")

        if "theta_cmd_deg" in fixed.columns:
            plt.plot(fixed["t"], fixed["theta_cmd_deg"], "k--", label="Command")

        plt.xlabel("Time [s]")
        plt.ylabel("Pitch [deg]")
        plt.title(f"{case_name}: Pitch Tracking")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def plot_error_pair(case_name, fixed, mlp_gs):
    if "t" not in fixed.columns or "t" not in mlp_gs.columns:
        return

    fixed_hdg_err = get_heading_error(fixed)
    mlp_hdg_err = get_heading_error(mlp_gs)

    fixed_pitch_err = get_pitch_error(fixed)
    mlp_pitch_err = get_pitch_error(mlp_gs)

    plt.figure(figsize=(10, 4))
    plt.plot(fixed["t"], fixed_hdg_err, label="Original heading error")
    plt.plot(mlp_gs["t"], mlp_hdg_err, label="MLP-GS heading error")
    plt.xlabel("Time [s]")
    plt.ylabel("Heading error [deg]")
    plt.title(f"{case_name}: Heading Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(fixed["t"], fixed_pitch_err, label="Original pitch error")
    plt.plot(mlp_gs["t"], mlp_pitch_err, label="MLP-GS pitch error")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch error [deg]")
    plt.title(f"{case_name}: Pitch Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_rate_pair(case_name, fixed, mlp_gs):
    if "t" not in fixed.columns or "t" not in mlp_gs.columns:
        return

    rate_specs = [
        ("p", "Roll rate p [deg/s]"),
        ("q", "Pitch rate q [deg/s]"),
        ("r", "Yaw rate r [deg/s]"),
    ]

    for col, ylabel in rate_specs:
        if col in fixed.columns and col in mlp_gs.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(fixed["t"], get_rate_deg_s(fixed, col), label=f"Original {col}")
            plt.plot(mlp_gs["t"], get_rate_deg_s(mlp_gs, col), label=f"MLP-GS {col}")
            plt.xlabel("Time [s]")
            plt.ylabel(ylabel)
            plt.title(f"{case_name}: {ylabel}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()


def plot_control_pair(case_name, fixed, mlp_gs):
    if "t" not in fixed.columns or "t" not in mlp_gs.columns:
        return

    if "deltaS_deg" in fixed.columns and "deltaS_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["deltaS_deg"], label="Original deltaS")
        plt.plot(mlp_gs["t"], mlp_gs["deltaS_deg"], label="MLP-GS deltaS")
        plt.xlabel("Time [s]")
        plt.ylabel("deltaS [deg]")
        plt.title(f"{case_name}: Symmetric Elevon Command")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    if "deltaD_deg" in fixed.columns and "deltaD_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["deltaD_deg"], label="Original deltaD")
        plt.plot(mlp_gs["t"], mlp_gs["deltaD_deg"], label="MLP-GS deltaD")
        plt.xlabel("Time [s]")
        plt.ylabel("deltaD [deg]")
        plt.title(f"{case_name}: Differential Elevon Command")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def plot_envelope_pair(case_name, fixed, mlp_gs):
    if "t" not in fixed.columns or "t" not in mlp_gs.columns:
        return

    if "alpha_deg" in fixed.columns and "alpha_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["alpha_deg"], label="Original alpha")
        plt.plot(mlp_gs["t"], mlp_gs["alpha_deg"], label="MLP-GS alpha")
        plt.xlabel("Time [s]")
        plt.ylabel("alpha [deg]")
        plt.title(f"{case_name}: Angle of Attack")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    if "beta_deg" in fixed.columns and "beta_deg" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["beta_deg"], label="Original beta")
        plt.plot(mlp_gs["t"], mlp_gs["beta_deg"], label="MLP-GS beta")
        plt.xlabel("Time [s]")
        plt.ylabel("beta [deg]")
        plt.title(f"{case_name}: Sideslip")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    if "V" in fixed.columns and "V" in mlp_gs.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(fixed["t"], fixed["V"], label="Original V")
        plt.plot(mlp_gs["t"], mlp_gs["V"], label="MLP-GS V")
        plt.xlabel("Time [s]")
        plt.ylabel("Airspeed [m/s]")
        plt.title(f"{case_name}: Airspeed")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def plot_ground_track_pair(case_name, fixed, mlp_gs):
    # Your run_sim seems to use x, y, z columns.
    if all(c in fixed.columns for c in ["x", "y"]) and all(c in mlp_gs.columns for c in ["x", "y"]):
        plt.figure(figsize=(6, 6))
        plt.plot(fixed["x"], fixed["y"], label="Original")
        plt.plot(mlp_gs["x"], mlp_gs["y"], label="MLP-GS")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"{case_name}: Ground Track")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def plot_all_for_case(case_name, paths):
    fixed, mlp_gs = load_pair(paths)

    if MAKE_TRACKING_PLOTS:
        plot_tracking_pair(case_name, fixed, mlp_gs)

    if MAKE_ERROR_PLOTS:
        plot_error_pair(case_name, fixed, mlp_gs)

    if MAKE_RATE_PLOTS:
        plot_rate_pair(case_name, fixed, mlp_gs)

    if MAKE_CONTROL_PLOTS:
        plot_control_pair(case_name, fixed, mlp_gs)

    if MAKE_ENVELOPE_PLOTS:
        plot_envelope_pair(case_name, fixed, mlp_gs)

    if MAKE_GROUND_TRACK_PLOTS:
        plot_ground_track_pair(case_name, fixed, mlp_gs)


def plot_metric_bars(metrics_df):
    key_metrics = [
        "heading_RMSE_deg",
        "pitch_RMSE_deg",
        "max_abs_p_deg_s",
        "max_abs_q_deg_s",
        "max_abs_r_deg_s",
        "deltaS_RMS_deg",
        "deltaD_RMS_deg",
        "deltaS_sat_fraction",
        "deltaD_sat_fraction",
        "max_abs_beta_deg",
    ]

    cases = metrics_df["case"].unique()
    x = np.arange(len(cases))
    width = 0.35

    for metric in key_metrics:
        plt.figure(figsize=(11, 4))

        fixed_vals = []
        mlp_vals = []

        for case in cases:
            fixed_row = metrics_df[
                (metrics_df["case"] == case) &
                (metrics_df["controller"] == "fixed")
            ]

            mlp_row = metrics_df[
                (metrics_df["case"] == case) &
                (metrics_df["controller"] == "mlp_gs")
            ]

            if len(fixed_row) == 0 or len(mlp_row) == 0:
                fixed_vals.append(np.nan)
                mlp_vals.append(np.nan)
            else:
                fixed_vals.append(fixed_row[metric].iloc[0])
                mlp_vals.append(mlp_row[metric].iloc[0])

        plt.bar(x - width / 2, fixed_vals, width, label="Original")
        plt.bar(x + width / 2, mlp_vals, width, label="MLP-GS")

        plt.xticks(x, cases, rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"Controller Comparison: {metric}")
        plt.grid(True, axis="y")
        plt.legend()
        plt.tight_layout()


def print_improvement_summary(metrics_df):
    """
    Prints percent changes for key metrics.
    Negative percent change means MLP-GS reduced that metric.
    """
    key_metrics = [
        "heading_RMSE_deg",
        "pitch_RMSE_deg",
        "max_abs_p_deg_s",
        "max_abs_q_deg_s",
        "max_abs_r_deg_s",
        "deltaS_RMS_deg",
        "deltaD_RMS_deg",
        "deltaS_sat_fraction",
        "deltaD_sat_fraction",
        "max_abs_beta_deg",
    ]

    print("\nPercent change from Original to MLP-GS:")
    print("(negative means MLP-GS is lower for that metric)\n")

    for case in metrics_df["case"].unique():
        fixed_row = metrics_df[
            (metrics_df["case"] == case) &
            (metrics_df["controller"] == "fixed")
        ]

        mlp_row = metrics_df[
            (metrics_df["case"] == case) &
            (metrics_df["controller"] == "mlp_gs")
        ]

        if len(fixed_row) == 0 or len(mlp_row) == 0:
            continue

        print(f"--- {case} ---")

        for metric in key_metrics:
            fixed_val = float(fixed_row[metric].iloc[0])
            mlp_val = float(mlp_row[metric].iloc[0])

            if abs(fixed_val) < 1e-12:
                pct = np.nan
            else:
                pct = 100.0 * (mlp_val - fixed_val) / abs(fixed_val)

            print(f"{metric:25s}: original={fixed_val:10.4f}, mlp_gs={mlp_val:10.4f}, change={pct:8.2f}%")

        print()


# ============================================================
# Main
# ============================================================

def main():
    print("Loading controller comparison CSVs...")

    # Make sure files exist first, so you catch naming/path issues early.
    for case_name, paths in CASES.items():
        for controller_name, path in paths.items():
            check_file_exists(path)

    metrics_df = build_metrics_table(CASES)

    print("\nController comparison metrics:")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(METRICS_OUTPUT_CSV, index=False)
    print(f"\nSaved metrics table to: {METRICS_OUTPUT_CSV}")

    print_improvement_summary(metrics_df)

    for case_name, paths in CASES.items():
        print(f"Making plots for {case_name}...")
        plot_all_for_case(case_name, paths)

    if MAKE_METRIC_BAR_PLOTS:
        plot_metric_bars(metrics_df)

    plt.show()


if __name__ == "__main__":
    main()