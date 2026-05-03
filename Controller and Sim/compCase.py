#!/usr/bin/env python3
"""
compare_controller_csvs.py

Compares two 6DOF controller output CSV files.

Useful for comparing:
    - fixed gain vs gain scheduled
    - gain scheduled vs derivative damping
    - different yaw damping gains
    - different initial conditions

Outputs:
    1. Printed summary metrics
    2. Comparison plots
    3. Optional CSV summary table

Example:
    python compare_controller_csvs.py fixed_gain.csv gain_sched.csv

Or edit CSV_A and CSV_B below and just run:
    python compare_controller_csvs.py
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User inputs
# ============================================================

# If you do not pass filenames from terminal, these are used.
CSV_A = "sim_out_20cross_noSched4.csv"
CSV_B = "sim_out_20cross_withSched4.csv"

LABEL_A = "No Sched"
LABEL_B = "Sched"

OUTPUT_SUMMARY_CSV = "scheduledcontroller_comparison_summary_20cross_fullComp4.csv"

SHOW_PLOTS = True
SAVE_PLOTS = True
PLOT_DIR = "comparison_plots_20cross_fulltest4"

# Actual actuator limits used in your sim.
# Change these if your run_sim values are different.
DELTAS_LIMIT_DEG = 20.0
DELTAD_LIMIT_DEG = 10.0

# Settling tolerance, in degrees.
PITCH_SETTLING_TOL_DEG = 0.5
HEADING_SETTLING_TOL_DEG = 2.0
ROLL_SETTLING_TOL_DEG = 2.0


# ============================================================
# Helpers
# ============================================================

def wrap_deg(angle_deg):
    """Wrap angle to [-180, 180)."""
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def safe_col(df, col, default=np.nan):
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def mean_abs(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x)))


def max_abs(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.max(np.abs(x)))


def total_variation(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(x))))


def max_rate(time, x):
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)

    n = min(len(time), len(x))
    time = time[:n]
    x = x[:n]

    dt = np.diff(time)
    dx = np.diff(x)

    valid = np.isfinite(dt) & np.isfinite(dx) & (dt > 0)

    if not np.any(valid):
        return np.nan

    return float(np.max(np.abs(dx[valid] / dt[valid])))


def settling_time(time, error, tol):
    """
    Returns first time after which abs(error) stays below tol.
    If never settles, returns NaN.
    """
    time = np.asarray(time, dtype=float)
    error = np.asarray(error, dtype=float)

    n = min(len(time), len(error))
    time = time[:n]
    error = error[:n]

    abs_err = np.abs(error)

    for i in range(n):
        if np.all(abs_err[i:] <= tol):
            return float(time[i])

    return np.nan


def final_mean_error(error, frac=0.10):
    error = np.asarray(error, dtype=float)
    n = len(error)
    if n == 0:
        return np.nan

    n_tail = max(1, int(frac * n))
    return float(np.mean(np.abs(error[-n_tail:])))


def get_command_errors(df):
    """
    Uses command columns if available.
    Falls back to zero command if missing.
    """
    pitch = safe_col(df, "pitch_deg")
    yaw = safe_col(df, "yaw_deg")
    roll = safe_col(df, "roll_deg")

    theta_cmd = safe_col(df, "theta_cmd_deg", default=np.nan)
    psi_cmd = safe_col(df, "psi_cmd_deg", default=np.nan)

    # Fallback if command columns were not added by run_sim.
    if np.all(np.isnan(theta_cmd)):
        theta_cmd = np.zeros(len(df))

    if np.all(np.isnan(psi_cmd)):
        psi_cmd = np.zeros(len(df))

    pitch_err = theta_cmd - pitch
    heading_err = wrap_deg(psi_cmd - yaw)

    # Usually no explicit roll command is logged, but ctrl_phi_cmd_deg may exist.
    if "ctrl_phi_cmd_deg" in df.columns:
        phi_cmd = safe_col(df, "ctrl_phi_cmd_deg")
        roll_err = phi_cmd - roll
    else:
        roll_err = np.full(len(df), np.nan)

    return pitch_err, heading_err, roll_err


def compute_metrics(df, label):
    t = safe_col(df, "t")

    pitch_err, heading_err, roll_err = get_command_errors(df)

    deltaS = safe_col(df, "deltaS_deg")
    deltaD = safe_col(df, "deltaD_deg")

    p = safe_col(df, "p")
    q = safe_col(df, "q")
    r = safe_col(df, "r")

    # Convert rates to deg/s if possible. Your sim logs p,q,r in rad/s.
    p_deg_s = np.rad2deg(p)
    q_deg_s = np.rad2deg(q)
    r_deg_s = np.rad2deg(r)

    metrics = {
        "case": label,

        # Tracking
        "pitch_err_rms_deg": rms(pitch_err),
        "pitch_err_mean_abs_deg": mean_abs(pitch_err),
        "pitch_err_final_mean_deg": final_mean_error(pitch_err),
        "pitch_settling_time_s": settling_time(t, pitch_err, PITCH_SETTLING_TOL_DEG),

        "heading_err_rms_deg": rms(heading_err),
        "heading_err_mean_abs_deg": mean_abs(heading_err),
        "heading_err_final_mean_deg": final_mean_error(heading_err),
        "heading_settling_time_s": settling_time(t, heading_err, HEADING_SETTLING_TOL_DEG),

        "roll_err_rms_deg": rms(roll_err),
        "roll_err_mean_abs_deg": mean_abs(roll_err),
        "roll_settling_time_s": settling_time(t, roll_err, ROLL_SETTLING_TOL_DEG),

        # Body rates
        "max_abs_p_deg_s": max_abs(p_deg_s),
        "max_abs_q_deg_s": max_abs(q_deg_s),
        "max_abs_r_deg_s": max_abs(r_deg_s),
        "r_rms_deg_s": rms(r_deg_s),
        "q_rms_deg_s": rms(q_deg_s),
        "p_rms_deg_s": rms(p_deg_s),

        # Control effort
        "deltaS_rms_deg": rms(deltaS),
        "deltaD_rms_deg": rms(deltaD),
        "deltaS_mean_abs_deg": mean_abs(deltaS),
        "deltaD_mean_abs_deg": mean_abs(deltaD),
        "deltaS_max_abs_deg": max_abs(deltaS),
        "deltaD_max_abs_deg": max_abs(deltaD),

        # Smoothness
        "deltaS_total_variation_deg": total_variation(deltaS),
        "deltaD_total_variation_deg": total_variation(deltaD),
        "deltaS_max_rate_deg_s": max_rate(t, deltaS),
        "deltaD_max_rate_deg_s": max_rate(t, deltaD),

        # Saturation fractions based on limits
        "deltaS_sat_frac": float(np.mean(np.abs(deltaS) >= 0.98 * DELTAS_LIMIT_DEG)),
        "deltaD_sat_frac": float(np.mean(np.abs(deltaD) >= 0.98 * DELTAD_LIMIT_DEG)),
    }

    # Add controller diagnostic saturation flags if present
    for col in [
        "ctrl_q_cmd_saturated",
        "ctrl_phi_cmd_saturated",
        "ctrl_deltaS_saturated",
        "ctrl_deltaD_saturated",
    ]:
        if col in df.columns:
            metrics[f"{col}_mean"] = float(np.nanmean(df[col].to_numpy(dtype=float)))

    # Add scheduled gain summaries if present
    for col in [
        "ctrl_K_PSI",
        "ctrl_K_PHI",
        "ctrl_K_P",
        "ctrl_K_R",
        "ctrl_K_THETA",
        "ctrl_K_Q",
        "ctrl_K_QD",
        "ctrl_vel_scale",
        "ctrl_aero_scale",
        "ctrl_roll_damp_scale",
        "ctrl_pitch_damp_scale",
        "ctrl_yaw_accel_damp_scale",
        "ctrl_roll_accel_damp_scale",
        "ctrl_pitch_accel_damp_scale",
    ]:
        if col in df.columns:
            x = df[col].to_numpy(dtype=float)
            metrics[f"{col}_mean"] = float(np.nanmean(x))
            metrics[f"{col}_min"] = float(np.nanmin(x))
            metrics[f"{col}_max"] = float(np.nanmax(x))

    return metrics


def interp_to_common_time(df_a, df_b, cols):
    """
    Interpolates both datasets to common overlapping time grid.
    Uses df_a time grid clipped to overlap.
    """
    t_a = safe_col(df_a, "t")
    t_b = safe_col(df_b, "t")

    t_min = max(np.nanmin(t_a), np.nanmin(t_b))
    t_max = min(np.nanmax(t_a), np.nanmax(t_b))

    mask = (t_a >= t_min) & (t_a <= t_max)
    t_common = t_a[mask]

    out = {}

    for col in cols:
        if col in df_a.columns and col in df_b.columns:
            a = df_a[col].to_numpy(dtype=float)
            b = df_b[col].to_numpy(dtype=float)

            out[col] = {
                "a": np.interp(t_common, t_a, a),
                "b": np.interp(t_common, t_b, b),
            }

    return t_common, out


def compare_case_differences(df_a, df_b):
    cols = [
        "pitch_deg",
        "yaw_deg",
        "roll_deg",
        "deltaS_deg",
        "deltaD_deg",
        "p",
        "q",
        "r",
        "alpha_deg",
        "beta_deg",
        "V",
        "ctrl_K_R",
        "ctrl_K_QD",
        "ctrl_K_P",
        "ctrl_K_Q",
        "ctrl_K_PHI",
    ]

    t_common, data = interp_to_common_time(df_a, df_b, cols)

    rows = []

    for col, vals in data.items():
        a = vals["a"]
        b = vals["b"]

        # Convert rad/s rate differences to deg/s for readability
        unit = ""
        if col in ["p", "q", "r"]:
            a = np.rad2deg(a)
            b = np.rad2deg(b)
            unit = "deg/s"
        elif "deg" in col:
            unit = "deg"
        else:
            unit = ""

        diff = b - a

        rows.append({
            "variable": col,
            "unit": unit,
            "rms_diff_B_minus_A": rms(diff),
            "mean_abs_diff": mean_abs(diff),
            "max_abs_diff": max_abs(diff),
        })

    return pd.DataFrame(rows)


def save_or_show_plot(name):
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, f"{name}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")


def plot_compare(df_a, df_b, col, label_a, label_b, ylabel=None, title=None, convert=None):
    if col not in df_a.columns or col not in df_b.columns:
        return

    t_a = safe_col(df_a, "t")
    t_b = safe_col(df_b, "t")

    y_a = df_a[col].to_numpy(dtype=float)
    y_b = df_b[col].to_numpy(dtype=float)

    if convert is not None:
        y_a = convert(y_a)
        y_b = convert(y_b)

    plt.figure()
    plt.plot(t_a, y_a, label=label_a)
    plt.plot(t_b, y_b, label=label_b)
    plt.xlabel("t (s)")
    plt.ylabel(ylabel if ylabel else col)
    plt.title(title if title else col)
    plt.grid(True)
    plt.legend()

    save_or_show_plot(col)


def make_comparison_plots(df_a, df_b, label_a, label_b):
    # Attitude
    plot_compare(df_a, df_b, "pitch_deg", label_a, label_b, "deg", "Pitch Comparison")
    plot_compare(df_a, df_b, "yaw_deg", label_a, label_b, "deg", "Yaw / Heading Comparison")
    plot_compare(df_a, df_b, "roll_deg", label_a, label_b, "deg", "Roll Comparison")

    # Tracking errors
    pitch_err_a, heading_err_a, roll_err_a = get_command_errors(df_a)
    pitch_err_b, heading_err_b, roll_err_b = get_command_errors(df_b)

    t_a = safe_col(df_a, "t")
    t_b = safe_col(df_b, "t")

    plt.figure()
    plt.plot(t_a, pitch_err_a, label=f"{label_a} pitch error")
    plt.plot(t_b, pitch_err_b, label=f"{label_b} pitch error")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("Pitch Tracking Error")
    plt.grid(True)
    plt.legend()
    save_or_show_plot("pitch_tracking_error")

    plt.figure()
    plt.plot(t_a, heading_err_a, label=f"{label_a} heading error")
    plt.plot(t_b, heading_err_b, label=f"{label_b} heading error")
    plt.xlabel("t (s)")
    plt.ylabel("deg")
    plt.title("Heading Tracking Error")
    plt.grid(True)
    plt.legend()
    save_or_show_plot("heading_tracking_error")

    # Body rates
    plot_compare(df_a, df_b, "p", label_a, label_b, "deg/s", "Roll Rate p", convert=np.rad2deg)
    plot_compare(df_a, df_b, "q", label_a, label_b, "deg/s", "Pitch Rate q", convert=np.rad2deg)
    plot_compare(df_a, df_b, "r", label_a, label_b, "deg/s", "Yaw Rate r", convert=np.rad2deg)

    # Controls
    plot_compare(df_a, df_b, "deltaS_deg", label_a, label_b, "deg", "deltaS Control")
    plot_compare(df_a, df_b, "deltaD_deg", label_a, label_b, "deg", "deltaD Control")

    # Aero / state
    plot_compare(df_a, df_b, "alpha_deg", label_a, label_b, "deg", "Alpha Comparison")
    plot_compare(df_a, df_b, "beta_deg", label_a, label_b, "deg", "Beta Comparison")
    plot_compare(df_a, df_b, "V", label_a, label_b, "m/s", "Airspeed Comparison")

    # Gain scheduling
    for col in [
        "ctrl_K_PSI",
        "ctrl_K_PHI",
        "ctrl_K_P",
        "ctrl_K_R",
        "ctrl_K_THETA",
        "ctrl_K_Q",
        "ctrl_K_QD",
        "ctrl_vel_scale",
        "ctrl_aero_scale",
        "ctrl_pitch_err_scale",
        "ctrl_roll_err_scale",
        "ctrl_heading_err_scale",
        "ctrl_roll_damp_scale",
        "ctrl_pitch_damp_scale",
        "ctrl_yaw_accel_damp_scale",
        "ctrl_roll_accel_damp_scale",
        "ctrl_pitch_accel_damp_scale",
    ]:
        plot_compare(df_a, df_b, col, label_a, label_b, col, f"{col} Comparison")

    # Derivative damping diagnostics
    plot_compare(df_a, df_b, "ctrl_r_dot_deg_s2", label_a, label_b, "deg/s²", "Yaw Acceleration r_dot")
    plot_compare(df_a, df_b, "ctrl_q_dot_deg_s2", label_a, label_b, "deg/s²", "Pitch Acceleration q_dot")
    plot_compare(df_a, df_b, "ctrl_p_dot_deg_s2", label_a, label_b, "deg/s²", "Roll Acceleration p_dot")


def print_metric_comparison(summary_df):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    print("\n==============================")
    print("Controller Comparison Summary")
    print("==============================\n")

    transposed = summary_df.set_index("case").T
    print(transposed)

    if len(summary_df) == 2:
        a = summary_df.iloc[0]
        b = summary_df.iloc[1]

        print("\n==============================")
        print("Percent Changes: Case B vs Case A")
        print("==============================\n")

        rows = []
        for col in summary_df.columns:
            if col == "case":
                continue

            aval = a[col]
            bval = b[col]

            if pd.isna(aval) or pd.isna(bval):
                continue

            if abs(float(aval)) < 1e-12:
                pct = np.nan
            else:
                pct = 100.0 * (float(bval) - float(aval)) / abs(float(aval))

            rows.append({
                "metric": col,
                "A": aval,
                "B": bval,
                "percent_change_B_vs_A": pct,
            })

        pct_df = pd.DataFrame(rows)

        # Print most useful metrics first
        important = [
            "pitch_err_rms_deg",
            "heading_err_rms_deg",
            "max_abs_r_deg_s",
            "r_rms_deg_s",
            "deltaS_total_variation_deg",
            "deltaD_total_variation_deg",
            "deltaS_max_rate_deg_s",
            "deltaD_max_rate_deg_s",
            "deltaS_sat_frac",
            "deltaD_sat_frac",
        ]

        first = pct_df[pct_df["metric"].isin(important)]
        rest = pct_df[~pct_df["metric"].isin(important)]

        print(pd.concat([first, rest], ignore_index=True).to_string(index=False))


def main():
    if len(sys.argv) >= 3:
        csv_a = sys.argv[1]
        csv_b = sys.argv[2]
    else:
        csv_a = CSV_A
        csv_b = CSV_B

    label_a = os.path.splitext(os.path.basename(csv_a))[0]
    label_b = os.path.splitext(os.path.basename(csv_b))[0]

    if not os.path.isfile(csv_a):
        raise FileNotFoundError(f"Could not find first CSV: {csv_a}")

    if not os.path.isfile(csv_b):
        raise FileNotFoundError(f"Could not find second CSV: {csv_b}")

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    print(f"\nLoaded:")
    print(f"  A: {csv_a}  ({len(df_a)} rows)")
    print(f"  B: {csv_b}  ({len(df_b)} rows)")

    metrics_a = compute_metrics(df_a, label_a)
    metrics_b = compute_metrics(df_b, label_b)

    summary_df = pd.DataFrame([metrics_a, metrics_b])
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print_metric_comparison(summary_df)

    diff_df = compare_case_differences(df_a, df_b)
    diff_csv = "controller_timeseries_difference_summary.csv"
    diff_df.to_csv(diff_csv, index=False)

    print("\n==============================")
    print("Time-Series Difference Summary")
    print("==============================\n")
    print(diff_df.to_string(index=False))

    print(f"\nWrote summary files:")
    print(f"  {OUTPUT_SUMMARY_CSV}")
    print(f"  {diff_csv}")

    if SHOW_PLOTS:
        make_comparison_plots(df_a, df_b, label_a, label_b)
        plt.show()


if __name__ == "__main__":
    main()