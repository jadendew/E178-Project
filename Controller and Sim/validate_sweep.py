#!/usr/bin/env python3
"""
validate_sweep.py  —  physics checks on sweep_body_rates.csv

Usage:
    python validate_sweep.py sweep_body_rates.csv
"""

import sys
import numpy as np
import pandas as pd

# must match aero_sweep_to_body_rates.py
IXX, IYY, IZZ = 0.8, 0.12, 0.18
S_REF  = 0.476500
C_REF  = 0.196250
RHO    = 1.225
V_TRIM = float(np.linalg.norm([20.0, 0.0, -3.0]))
QBAR   = 0.5 * RHO * V_TRIM**2

RESULTS = []


def check(name, passed, warn=False, detail=""):
    tag = "PASS" if passed and not warn else ("WARN" if warn else "FAIL")
    RESULTS.append((tag, name))
    sym = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[tag]
    line = f"  [{tag}] {sym}  {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def load(path):
    df = pd.read_csv(path)
    required = {"alpha_deg", "beta_deg", "deltaS_deg", "deltaD_deg",
                "CL", "CD", "CY", "Cl", "Cm", "Cn",
                "p_dot", "q_dot", "r_dot", "qbar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


def at(df, tol=1e-6, **kwargs):
    mask = np.ones(len(df), dtype=bool)
    for col, val in kwargs.items():
        mask &= np.isclose(df[col].values, val, atol=tol)
    return df[mask].copy()


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def check_finiteness(df):
    print("\n── Finiteness ──────────────────────────────────────")
    for col in ["CL", "CD", "CY", "Cl", "Cm", "Cn", "p_dot", "q_dot", "r_dot"]:
        n_bad = int(np.sum(~np.isfinite(df[col].values)))
        check(f"No NaN/Inf in {col}", n_bad == 0,
              detail=f"{n_bad} non-finite values" if n_bad else "")


def check_cd_positive(df):
    print("\n── CD sign ─────────────────────────────────────────")
    n_neg = int(np.sum(df["CD"].values < 0))
    check("CD >= 0 everywhere", n_neg == 0,
          detail=f"{n_neg} negative values" if n_neg else "")


def check_pitch_stability(df):
    print("\n── Pitch stability ─────────────────────────────────")
    sub = at(df, beta_deg=0, deltaS_deg=0, deltaD_deg=0).sort_values("alpha_deg")
    s_cm   = slope(sub["alpha_deg"].values, sub["Cm"].values)
    s_qdot = slope(sub["alpha_deg"].values, sub["q_dot"].values)
    check("dCm/dα < 0", s_cm < 0,   detail=f"dCm/dα = {s_cm:.4f} /deg")
    check("dq̇/dα < 0",  s_qdot < 0, detail=f"dq̇/dα = {s_qdot:.3f} rad/s²/deg")


def check_delta_s_authority(df):
    print("\n── δS pitch authority ──────────────────────────────")
    sub = at(df, beta_deg=0, deltaD_deg=0)
    sub = sub[np.isclose(sub["alpha_deg"].values, 5.0, atol=1.5)].sort_values("deltaS_deg")
    s_qdot = slope(sub["deltaS_deg"].values, sub["q_dot"].values)
    s_cm   = slope(sub["deltaS_deg"].values, sub["Cm"].values)
    check("dq̇/dδS > 0",  s_qdot > 0, detail=f"dq̇/dδS = {s_qdot:.3f} rad/s²/deg")
    check("dCm/dδS > 0", s_cm > 0,   detail=f"dCm/dδS = {s_cm:.5f} /deg")


def check_lateral(df):
    print("\n── Lateral stability ───────────────────────────────")
    sub = at(df, deltaS_deg=0, deltaD_deg=0)
    sub = sub[np.isclose(sub["alpha_deg"].values, 5.0, atol=1.5)].sort_values("beta_deg")
    s_cl   = slope(sub["beta_deg"].values, sub["Cl"].values)
    s_pdot = slope(sub["beta_deg"].values, sub["p_dot"].values)
    s_cn   = slope(sub["beta_deg"].values, sub["Cn"].values)
    s_rdot = slope(sub["beta_deg"].values, sub["r_dot"].values)
    check("dCl/dβ < 0 (dihedral)",       s_cl < 0,   detail=f"dCl/dβ = {s_cl:.5f} /deg")
    check("dṗ/dβ < 0 consistent",        s_pdot < 0, detail=f"dṗ/dβ = {s_pdot:.4f} rad/s²/deg")
    check("dCn/dβ > 0 (dir. stability)", s_cn > 0,   detail=f"dCn/dβ = {s_cn:.6f} /deg")
    check("dṙ/dβ > 0 consistent",        s_rdot > 0, detail=f"dṙ/dβ = {s_rdot:.4f} rad/s²/deg")


def check_delta_d_authority(df):
    print("\n── δD roll authority ────────────────────────────────")
    sub = at(df, deltaS_deg=0)
    sub = sub[np.isclose(sub["alpha_deg"].values, 5.0, atol=1.5) &
              np.isclose(sub["beta_deg"].values,  0.0, atol=1.0)].sort_values("deltaD_deg")
    s_pdot = slope(sub["deltaD_deg"].values, sub["p_dot"].values)
    s_cl   = slope(sub["deltaD_deg"].values, sub["Cl"].values)
    check("dṗ/dδD > 0",  s_pdot > 0, detail=f"dṗ/dδD = {s_pdot:.4f} rad/s²/deg")
    check("dCl/dδD > 0", s_cl > 0,   detail=f"dCl/dδD = {s_cl:.6f} /deg")


def check_symmetry(df):
    print("\n── Lateral symmetry ────────────────────────────────")
    sub    = at(df, beta_deg=0, deltaD_deg=0)
    max_cl = float(sub["Cl"].abs().max())
    passed = max_cl < 1e-3
    warn   = not passed and max_cl < 5e-3
    label  = "ok" if passed else "marginal" if warn else "too large"
    check("Cl ≈ 0 at β=0, δD=0", passed or warn, warn=warn,
          detail=f"max|Cl| = {max_cl:.2e}  ({label})")


def check_magnitudes(df):
    print("\n── Magnitude sanity ────────────────────────────────")
    sub = at(df, beta_deg=0, deltaS_deg=0, deltaD_deg=0)

    # q_dot back-of-envelope: qbar * S * c * Cm / Iyy
    cm_max       = float(sub["Cm"].abs().max())
    q_dot_boe    = QBAR * S_REF * C_REF * cm_max / IYY
    q_dot_actual = float(df["q_dot"].abs().max())
    ratio        = q_dot_actual / q_dot_boe if q_dot_boe > 0 else np.inf
    passed = 0.3 < ratio < 5.0
    warn   = not passed and 0.1 < ratio < 10.0
    check("q̇ peak within 5x of BofE", passed or warn, warn=warn,
          detail=f"BofE={q_dot_boe:.1f}, actual={q_dot_actual:.1f}, ratio={ratio:.2f}")

    # Cn at zero sideslip should be negligible relative to Cm
    cn_range = float(sub["Cn"].max() - sub["Cn"].min())
    cm_range = float(sub["Cm"].max() - sub["Cm"].min())
    rel      = cn_range / cm_range if cm_range > 0 else np.inf
    passed   = rel < 1e-3
    warn     = not passed and rel < 1e-2
    check("Cn variation at β=0 negligible vs Cm", passed or warn, warn=warn,
          detail=f"ΔCn={cn_range:.2e}, ΔCm={cm_range:.4f}, ratio={rel:.2e}")

    # r_dot at zero lateral inputs should be small relative to p_dot range
    r_dot_max  = float(sub["r_dot"].abs().max())
    pdot_range = float(df["p_dot"].max() - df["p_dot"].min())
    rel        = r_dot_max / pdot_range if pdot_range > 0 else np.inf
    passed     = rel < 0.05
    warn       = not passed and rel < 0.15
    check("ṙ bias at β=δ=0 < 5% of ṗ range", passed or warn, warn=warn,
          detail=f"max|ṙ|={r_dot_max:.4f}, ṗ range={pdot_range:.4f}, rel={rel:.3f}")


def check_trim_null(df):
    print("\n── Trim null reachable ──────────────────────────────")
    sub     = at(df, beta_deg=0, deltaD_deg=0)
    has_pos = bool(np.any(sub["q_dot"].values > 0))
    has_neg = bool(np.any(sub["q_dot"].values < 0))
    check("q̇ crosses zero in α×δS space", has_pos and has_neg,
          detail=f"q̇ range: [{sub['q_dot'].min():.1f}, {sub['q_dot'].max():.1f}] rad/s²")


def check_qdot_ordering(df):
    print("\n── q̇ ordering w.r.t. δS ────────────────────────────")
    sub         = at(df, beta_deg=0, deltaD_deg=0)
    alphas      = sorted(sub["alpha_deg"].unique())
    violations  = 0
    total_pairs = 0
    for a in alphas:
        s = sub[np.isclose(sub["alpha_deg"].values, a)].sort_values("deltaS_deg")
        if len(s) < 2:
            continue
        violations  += int(np.sum(np.diff(s["q_dot"].values) < 0))
        total_pairs += len(s) - 1
    passed = violations == 0
    warn   = not passed and violations <= max(1, total_pairs * 0.05)
    check("q̇ monotonically increases with δS at each α", passed or warn, warn=warn,
          detail=f"{violations}/{total_pairs} ordering violations")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sweep_body_rates.csv"
    print(f"Loading: {path}")
    df = load(path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    check_finiteness(df)
    check_cd_positive(df)
    check_pitch_stability(df)
    check_delta_s_authority(df)
    check_lateral(df)
    check_delta_d_authority(df)
    check_symmetry(df)
    check_magnitudes(df)
    check_trim_null(df)
    check_qdot_ordering(df)

    n_pass = sum(1 for t, _ in RESULTS if t == "PASS")
    n_warn = sum(1 for t, _ in RESULTS if t == "WARN")
    n_fail = sum(1 for t, _ in RESULTS if t == "FAIL")

    print(f"\n{'='*55}")
    print(f"  Results:  {n_pass} PASS   {n_warn} WARN   {n_fail} FAIL  ({len(RESULTS)} total)")
    print(f"{'='*55}")

    if n_fail:
        print("\n  FAILED:")
        for tag, name in RESULTS:
            if tag == "FAIL":
                print(f"    ✗ {name}")
    if n_warn:
        print("\n  Warnings:")
        for tag, name in RESULTS:
            if tag == "WARN":
                print(f"    ~ {name}")

    return n_fail


if __name__ == "__main__":
    sys.exit(main())