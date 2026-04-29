#!/usr/bin/env python3
"""
test_mlp_aero.py

Standalone unit/sanity tests for comparing:
    1. Original table-based AeroModel
    2. New MLP-based MLPAeroModel

Run from the Controller and Sim folder:

    python .\\test_mlp_aero.py

This does NOT run the full 6DOF simulation.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aero


# ============================================================
# User paths
# ============================================================

CSV_PATH = r"..\aero_autosweep.csv"

MLP_MODEL_PATH = r"..\mlp_aero_state_dictV2.pt"
X_SCALER_PATH = r"..\xScalerV2.pkl"
Y_SCALER_PATH = r"..\yScalerV2.pkl"


# ============================================================
# Match run_sim.py aero settings
# ============================================================

SIGN_CL = 1.0
SIGN_CD = 1.0
SIGN_CY = 1.0
SIGN_Cl = 1.0
SIGN_Cm = 1.0
SIGN_Cn = 1.0

CD0_ADD = 0.01038

CLP = -0.3546536
CMQ = -6.2794743
CNR = -0.0057922

MAX_ALPHA_DEG = 35.0
MAX_BETA_DEG = 20.0

USE_SUPERPOSITION = True
DEFLECTION_TOL = 1e-6

DELTAS_MAX_DEG = 20.0
DELTAD_MAX_DEG = 10.0

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


COEF_NAMES = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]


def make_aero_cfg():
    return aero.AeroConfig(
        sign_cl=SIGN_CL,
        sign_cd=SIGN_CD,
        sign_cy=SIGN_CY,
        sign_cll=SIGN_Cl,
        sign_cm=SIGN_Cm,
        sign_cn=SIGN_Cn,

        cd0_add=CD0_ADD,

        clp=CLP,
        cmq=CMQ,
        cnr=CNR,

        use_static_extrapolation=USE_STATIC_EXTRAPOLATION,
        max_alpha_deg=MAX_ALPHA_DEG,
        max_beta_deg=MAX_BETA_DEG,

        dcl_dalpha=DCL_DALPHA,
        dcl_dbeta=DCL_DBETA,
        dcd_dalpha=DCD_DALPHA,
        dcd_dbeta=DCD_DBETA,
        dcy_dalpha=DCY_DALPHA,
        dcy_dbeta=DCY_DBETA,
        dcll_dalpha=DCl_DALPHA,
        dcll_dbeta=DCl_DBETA,
        dcm_dalpha=DCm_DALPHA,
        dcm_dbeta=DCm_DBETA,
        dcn_dalpha=DCn_DALPHA,
        dcn_dbeta=DCn_DBETA,
    )


def coeff_dict_to_array(c):
    return np.array([c[k] for k in COEF_NAMES], dtype=float)


def print_coeff_line(prefix, coeffs):
    vals = " | ".join([f"{k}={coeffs[k]: .6f}" for k in COEF_NAMES])
    print(f"{prefix}: {vals}")


def compare_single_case(table_model, mlp_model, alpha, beta, dS, dD):
    ct = table_model.query(alpha, beta, dS, dD)
    cm = mlp_model.query(alpha, beta, dS, dD)

    at = coeff_dict_to_array(ct)
    am = coeff_dict_to_array(cm)
    diff = am - at

    print("\n" + "=" * 80)
    print(f"Case: alpha={alpha: .2f}, beta={beta: .2f}, deltaS={dS: .2f}, deltaD={dD: .2f}")
    print_coeff_line("TABLE", ct)
    print_coeff_line("MLP  ", cm)

    print("DIFF : " + " | ".join([f"{k}={diff[i]: .6f}" for i, k in enumerate(COEF_NAMES)]))

    if not np.all(np.isfinite(am)):
        print("FAIL: MLP returned NaN or Inf.")

    if cm["CD"] < 0:
        print("FAIL: MLP returned negative CD.")

    if np.max(np.abs(am)) > 10:
        print("WARNING: MLP returned coefficient magnitude > 10. Check scaling/order.")

    return diff


def run_point_comparison(table_model, mlp_model):
    print("\n\n==================== POINT-BY-POINT TABLE VS MLP CHECK ====================")

    test_cases = [
        (0, 0, 0, 0),
        (5, 0, 0, 0),
        (5, 0, 10, 0),
        (5, 0, -10, 0),
        (5, 0, 0, 5),
        (5, 0, 0, -5),
        (10, 5, 10, -5),
        (-5, -5, -10, 5),
    ]

    diffs = []

    for case in test_cases:
        diff = compare_single_case(table_model, mlp_model, *case)
        diffs.append(diff)

    diffs = np.array(diffs)

    print("\nSummary absolute errors over selected test cases:")
    for i, name in enumerate(COEF_NAMES):
        print(
            f"{name}: mean_abs={np.mean(np.abs(diffs[:, i])):.6f}, "
            f"max_abs={np.max(np.abs(diffs[:, i])):.6f}"
        )


def run_deltaS_sweep(mlp_model, alpha=5.0, beta=0.0, deltaD=0.0):
    print("\n\n==================== DELTAS / PITCH CONTROL SIGN CHECK ====================")
    print(f"Fixed alpha={alpha}, beta={beta}, deltaD={deltaD}")

    dS_values = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20], dtype=float)
    rows = []

    for dS in dS_values:
        c = mlp_model.query(alpha, beta, dS, deltaD)
        rows.append([dS, c["CL"], c["CD"], c["Cm"]])
        print(f"dS={dS: 7.2f} | CL={c['CL']: .6f} | CD={c['CD']: .6f} | Cm={c['Cm']: .6f}")

    rows = np.array(rows)
    dCm = rows[-1, 3] - rows[0, 3]

    print(f"\nCm(dS=+20) - Cm(dS=-20) = {dCm:.6f}")
    print("Interpretation: check this sign against what your pitch controller expects.")

    return rows


def run_deltaD_sweep(mlp_model, alpha=5.0, beta=0.0, deltaS=0.0):
    print("\n\n==================== DELTAD / ROLL-YAW CONTROL SIGN CHECK ====================")
    print(f"Fixed alpha={alpha}, beta={beta}, deltaS={deltaS}")

    dD_values = np.array([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10], dtype=float)
    rows = []

    for dD in dD_values:
        c = mlp_model.query(alpha, beta, deltaS, dD)
        rows.append([dD, c["CY"], c["Cl"], c["Cn"]])
        print(f"dD={dD: 7.2f} | CY={c['CY']: .6f} | Cl={c['Cl']: .6f} | Cn={c['Cn']: .6f}")

    rows = np.array(rows)
    dCl = rows[-1, 2] - rows[0, 2]
    dCn = rows[-1, 3] - rows[0, 3]

    print(f"\nCl(dD=+10) - Cl(dD=-10) = {dCl:.6f}")
    print(f"Cn(dD=+10) - Cn(dD=-10) = {dCn:.6f}")
    print("Interpretation: check these signs against what your roll/yaw controller expects.")

    return rows


def run_alpha_sweep(table_model, mlp_model, beta=0.0, deltaS=0.0, deltaD=0.0):
    print("\n\n==================== ALPHA SWEEP CHECK ====================")

    alpha_values = np.linspace(-10, 20, 31)
    table_rows = []
    mlp_rows = []

    for alpha in alpha_values:
        ct = table_model.query(alpha, beta, deltaS, deltaD)
        cm = mlp_model.query(alpha, beta, deltaS, deltaD)

        table_rows.append(coeff_dict_to_array(ct))
        mlp_rows.append(coeff_dict_to_array(cm))

    table_rows = np.array(table_rows)
    mlp_rows = np.array(mlp_rows)

    for i, name in enumerate(COEF_NAMES):
        err = mlp_rows[:, i] - table_rows[:, i]
        print(f"{name}: alpha sweep max_abs_error = {np.max(np.abs(err)):.6f}")

    return alpha_values, table_rows, mlp_rows


def run_beta_sweep(table_model, mlp_model, alpha=5.0, deltaS=0.0, deltaD=0.0):
    print("\n\n==================== BETA SWEEP CHECK ====================")

    beta_values = np.linspace(-10, 10, 21)
    table_rows = []
    mlp_rows = []

    for beta in beta_values:
        ct = table_model.query(alpha, beta, deltaS, deltaD)
        cm = mlp_model.query(alpha, beta, deltaS, deltaD)

        table_rows.append(coeff_dict_to_array(ct))
        mlp_rows.append(coeff_dict_to_array(cm))

    table_rows = np.array(table_rows)
    mlp_rows = np.array(mlp_rows)

    for i, name in enumerate(COEF_NAMES):
        err = mlp_rows[:, i] - table_rows[:, i]
        print(f"{name}: beta sweep max_abs_error = {np.max(np.abs(err)):.6f}")

    return beta_values, table_rows, mlp_rows


def plot_sweep(x, table_rows, mlp_rows, x_label, title_prefix):
    for i, name in enumerate(COEF_NAMES):
        plt.figure()
        plt.plot(x, table_rows[:, i], marker="o", label=f"Table {name}")
        plt.plot(x, mlp_rows[:, i], marker="x", label=f"MLP {name}")
        plt.xlabel(x_label)
        plt.ylabel(name)
        plt.title(f"{title_prefix}: {name}")
        plt.grid(True)
        plt.legend()


def plot_control_sweeps(deltaS_rows, deltaD_rows):
    plt.figure()
    plt.plot(deltaS_rows[:, 0], deltaS_rows[:, 1], marker="o", label="CL")
    plt.plot(deltaS_rows[:, 0], deltaS_rows[:, 2], marker="o", label="CD")
    plt.plot(deltaS_rows[:, 0], deltaS_rows[:, 3], marker="o", label="Cm")
    plt.xlabel("deltaS_deg")
    plt.ylabel("Coefficient")
    plt.title("MLP symmetric elevon sweep")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(deltaD_rows[:, 0], deltaD_rows[:, 1], marker="o", label="CY")
    plt.plot(deltaD_rows[:, 0], deltaD_rows[:, 2], marker="o", label="Cl")
    plt.plot(deltaD_rows[:, 0], deltaD_rows[:, 3], marker="o", label="Cn")
    plt.xlabel("deltaD_deg")
    plt.ylabel("Coefficient")
    plt.title("MLP differential elevon sweep")
    plt.grid(True)
    plt.legend()


def main():
    print("Running MLP aero unit/sanity tests...\n")

    required_files = [CSV_PATH, MLP_MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH]
    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(CSV_PATH)
    cfg = make_aero_cfg()

    print("Building table aero model...")
    table_model = aero.AeroModel(
        df,
        cfg=cfg,
        tol=DEFLECTION_TOL,
        use_superposition=USE_SUPERPOSITION,
    )

    print("Building MLP aero model...")
    mlp_model = aero.MLPAeroModel(
        model_path=MLP_MODEL_PATH,
        x_scaler_path=X_SCALER_PATH,
        y_scaler_path=Y_SCALER_PATH,
        cfg=cfg,
        device="cpu",
        max_deltaS_deg=DELTAS_MAX_DEG,
        max_deltaD_deg=DELTAD_MAX_DEG,
    )

    run_point_comparison(table_model, mlp_model)

    deltaS_rows = run_deltaS_sweep(
        mlp_model,
        alpha=5.0,
        beta=0.0,
        deltaD=0.0,
    )

    deltaD_rows = run_deltaD_sweep(
        mlp_model,
        alpha=5.0,
        beta=0.0,
        deltaS=0.0,
    )

    alpha_values, alpha_table, alpha_mlp = run_alpha_sweep(
        table_model,
        mlp_model,
        beta=0.0,
        deltaS=0.0,
        deltaD=0.0,
    )

    beta_values, beta_table, beta_mlp = run_beta_sweep(
        table_model,
        mlp_model,
        alpha=5.0,
        deltaS=0.0,
        deltaD=0.0,
    )

    plot_sweep(
        alpha_values,
        alpha_table,
        alpha_mlp,
        x_label="alpha_deg",
        title_prefix="Alpha sweep, beta=0, dS=0, dD=0",
    )

    plot_sweep(
        beta_values,
        beta_table,
        beta_mlp,
        x_label="beta_deg",
        title_prefix="Beta sweep, alpha=5, dS=0, dD=0",
    )

    plot_control_sweeps(deltaS_rows, deltaD_rows)

    print("\nDone. Close plots when finished.")
    plt.show()


if __name__ == "__main__":
    main()