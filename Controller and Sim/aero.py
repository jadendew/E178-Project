#!/usr/bin/env python3
"""
aero.py

Aerodynamics model for 6-DOF sim using tabulated coefficients from VSPAERO CSV.

Features:
- (alpha,beta) interpolation on fixed (deltaS,deltaD) slices using RegularGridInterpolator
- Continuous deflection via linear interpolation in deltaS and deltaD
- Optional superposition for separated pitch/roll sweeps:
    C(dS,dD) ≈ C(0,0) + [C(dS,0)-C(0,0)] + [C(0,dD)-C(0,0)]
- Coefficient sign flips
- Optional CD0 constant offset
- Optional static-derivative extrapolation outside alpha/beta table bounds (per rad)
- Optional rate damping: Clp/Cmq/Cnr (dimensionless per rad)

Coordinate conventions (must match dynamics):
- World is NED (X forward, Y right, Z down)
- Body is X forward, Y right, Z down
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
import torch
import joblib


# =============================================================================
# Config
# =============================================================================

@dataclass
class AeroConfig:
    # Sign flips (set -1 if your CSV convention is opposite)
    sign_cl: float = 1.0
    sign_cd: float = 1.0
    sign_cy: float = 1.0
    sign_cll: float = 1.0   # rolling moment coefficient Cl
    sign_cm: float = 1.0
    sign_cn: float = 1.0

    # CD0 offset
    cd0_add: float = 0.0

    # Rate damping (dimensionless derivatives per rad)
    clp: float = 0.0
    cmq: float = 0.0
    cnr: float = 0.0

    # Safety clamp for requested angles (deg)
    max_alpha_deg: float = 35.0
    max_beta_deg: float = 20.0

    # Static derivative extrapolation outside table bounds
    use_static_extrapolation: bool = False

    # Derivatives per rad (must match FINAL sign convention, i.e. after sign flips)
    dcl_dalpha: float = 0.0
    dcl_dbeta: float = 0.0
    dcd_dalpha: float = 0.0
    dcd_dbeta: float = 0.0
    dcy_dalpha: float = 0.0
    dcy_dbeta: float = 0.0
    dcll_dalpha: float = 0.0
    dcll_dbeta: float = 0.0
    dcm_dalpha: float = 0.0
    dcm_dbeta: float = 0.0
    dcn_dalpha: float = 0.0
    dcn_dbeta: float = 0.0


# =============================================================================
# Helpers
# =============================================================================

def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps

def dcm_body_to_world(q_bw: np.ndarray) -> np.ndarray:
    # q_bw is scipy format [x,y,z,w]
    if not np.isfinite(q_bw).all():
        raise ValueError("Quaternion became non-finite (NaN/Inf).")
    n = np.linalg.norm(q_bw)
    if n < 1e-12:
        raise ValueError("Quaternion norm collapsed to ~0.")
    return R.from_quat(q_bw / n).as_matrix()

def wind_angles_from_body_velocity(v_body: np.ndarray) -> Tuple[float, float, float]:
    """
    v_body: air-relative velocity in BODY axes.
    Returns: V, alpha(rad), beta(rad)
    """
    u, v, w = v_body
    V = _safe_norm(v_body)
    alpha = np.arctan2(w, u)
    beta = np.arctan2(v, np.sqrt(u*u + w*w))
    return V, alpha, beta

def build_mlp_model():
    return torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 6)
    )



# =============================================================================
# Slices and deflection interpolation
# =============================================================================

class AeroSlice2D:
    """Interpolates coefficients on a fixed (deltaS, deltaD) slice over (alpha, beta)."""
    def __init__(self, df: pd.DataFrame, deltaS_deg: float, deltaD_deg: float, tol: float = 1e-9):
        sl = df[
            np.isclose(df["deltaS_deg"], deltaS_deg, atol=tol) &
            np.isclose(df["deltaD_deg"], deltaD_deg, atol=tol)
        ].copy()

        if sl.empty:
            raise ValueError(f"No rows for deltaS={deltaS_deg}, deltaD={deltaD_deg}")

        self.deltaS_deg = float(deltaS_deg)
        self.deltaD_deg = float(deltaD_deg)

        self.alphas = np.sort(sl["alpha_deg"].unique())
        self.betas = np.sort(sl["beta_deg"].unique())

        def grid(col: str) -> np.ndarray:
            piv = sl.pivot_table(index="alpha_deg", columns="beta_deg", values=col, aggfunc="mean")
            piv = piv.reindex(index=self.alphas, columns=self.betas)
            if piv.isna().any().any():
                raise ValueError(f"Slice missing grid points for {col}.")
            return piv.to_numpy(dtype=float)

        self.interp = {
            k: RegularGridInterpolator((self.alphas, self.betas), grid(k),
                                       bounds_error=False, fill_value=None)
            for k in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
        }

        self.alpha_min, self.alpha_max = float(self.alphas.min()), float(self.alphas.max())
        self.beta_min, self.beta_max = float(self.betas.min()), float(self.betas.max())

    def query(self, alpha_deg: float, beta_deg: float) -> Dict[str, float]:
        a = float(np.clip(alpha_deg, self.alpha_min, self.alpha_max))
        b = float(np.clip(beta_deg, self.beta_min, self.beta_max))
        pt = np.array([a, b], dtype=float)
        out: Dict[str, float] = {}
        for k in self.interp:
            out[k] = float(np.asarray(self.interp[k](pt)).reshape(-1)[0])
        return out


class AeroModel:
    """
    Coefficient model that can return coeffs for arbitrary (alpha,beta,deltaS,deltaD).

    If use_superposition=True:
        uses only (dS,0), (0,dD), and (0,0) families and superposes increments.

    Otherwise:
        attempts bilinear interpolation in (dS,dD) requiring full grid coverage.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: AeroConfig | None = None,
        tol: float = 1e-9,
        use_superposition: bool = True,
    ):
        self.df = df.copy()
        self.cfg = cfg if cfg is not None else AeroConfig()
        self.tol = float(tol)
        self.use_superposition = bool(use_superposition)

        # global alpha/beta
        self.alpha_min = float(self.df["alpha_deg"].min())
        self.alpha_max = float(self.df["alpha_deg"].max())
        self.beta_min = float(self.df["beta_deg"].min())
        self.beta_max = float(self.df["beta_deg"].max())

        self.deltaS_vals_all = np.sort(self.df["deltaS_deg"].unique().astype(float))
        self.deltaD_vals_all = np.sort(self.df["deltaD_deg"].unique().astype(float))

        # For superposition
        self.deltaS_vals_pitch = np.sort(
            self.df[np.isclose(self.df["deltaD_deg"], 0.0, atol=self.tol)]["deltaS_deg"].unique().astype(float)
        )
        self.deltaD_vals_roll = np.sort(
            self.df[np.isclose(self.df["deltaS_deg"], 0.0, atol=self.tol)]["deltaD_deg"].unique().astype(float)
        )

        self._slice_cache: Dict[Tuple[float, float], AeroSlice2D] = {}

        if self.use_superposition:
            has00 = np.any(
                np.isclose(self.df["deltaS_deg"], 0.0, atol=self.tol) &
                np.isclose(self.df["deltaD_deg"], 0.0, atol=self.tol)
            )
            if not has00:
                raise ValueError("Superposition requires a baseline slice deltaS=0, deltaD=0.")

            if not np.any(np.isclose(self.deltaS_vals_pitch, 0.0, atol=self.tol)):
                raise ValueError("Superposition requires deltaD==0 family to include deltaS=0.")
            if not np.any(np.isclose(self.deltaD_vals_roll, 0.0, atol=self.tol)):
                raise ValueError("Superposition requires deltaS==0 family to include deltaD=0.")

    def _get_slice(self, dS: float, dD: float) -> AeroSlice2D:
        key = (float(dS), float(dD))
        if key not in self._slice_cache:
            self._slice_cache[key] = AeroSlice2D(self.df, dS, dD, tol=self.tol)
        return self._slice_cache[key]

    @staticmethod
    def _bracket(vals: np.ndarray, x: float) -> Tuple[float, float, float]:
        """Return (x0,x1,w) for lerp between x0 and x1."""
        vals = np.asarray(vals, dtype=float)
        x = float(x)

        if x <= float(vals.min()):
            return float(vals.min()), float(vals.min()), 0.0
        if x >= float(vals.max()):
            return float(vals.max()), float(vals.max()), 0.0

        idx = int(np.searchsorted(vals, x))
        x0 = float(vals[idx - 1])
        x1 = float(vals[idx])
        if abs(x1 - x0) < 1e-12:
            return x0, x1, 0.0
        w = float((x - x0) / (x1 - x0))
        return x0, x1, w

    @staticmethod
    def _lerp_coeffs(c0: Dict[str, float], c1: Dict[str, float], w: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in c0:
            out[k] = (1.0 - w) * float(c0[k]) + w * float(c1[k])
        return out

    def clamp_deltas(self, deltaS_deg: float, deltaD_deg: float) -> Tuple[float, float]:
        dS = float(deltaS_deg)
        dD = float(deltaD_deg)
        if self.use_superposition:
            dS = float(np.clip(dS, float(self.deltaS_vals_pitch.min()), float(self.deltaS_vals_pitch.max())))
            dD = float(np.clip(dD, float(self.deltaD_vals_roll.min()),  float(self.deltaD_vals_roll.max())))
        else:
            dS = float(np.clip(dS, float(self.deltaS_vals_all.min()), float(self.deltaS_vals_all.max())))
            dD = float(np.clip(dD, float(self.deltaD_vals_all.min()), float(self.deltaD_vals_all.max())))
        return dS, dD

    def query(self, alpha_deg: float, beta_deg: float, deltaS_deg: float, deltaD_deg: float) -> Dict[str, float]:
        a = float(np.clip(alpha_deg, self.alpha_min, self.alpha_max))
        b = float(np.clip(beta_deg, self.beta_min, self.beta_max))

        dS_req = float(deltaS_deg)
        dD_req = float(deltaD_deg)
        dS, dD = self.clamp_deltas(dS_req, dD_req)

        if self.use_superposition:
            # Baseline
            c00 = self._get_slice(0.0, 0.0).query(a, b)

            # Pitch family (dS,0)
            s0, s1, ws = self._bracket(self.deltaS_vals_pitch, dS)
            cs0 = self._get_slice(s0, 0.0).query(a, b)
            cs1 = self._get_slice(s1, 0.0).query(a, b)
            cs = self._lerp_coeffs(cs0, cs1, ws)

            # Roll family (0,dD)
            d0, d1, wd = self._bracket(self.deltaD_vals_roll, dD)
            cd0 = self._get_slice(0.0, d0).query(a, b)
            cd1 = self._get_slice(0.0, d1).query(a, b)
            cd = self._lerp_coeffs(cd0, cd1, wd)

            out: Dict[str, float] = {}
            for k in c00:
                out[k] = float(c00[k]) + (float(cs[k]) - float(c00[k])) + (float(cd[k]) - float(c00[k]))
            return out

        # Full bilinear grid required
        s0, s1, ws = self._bracket(self.deltaS_vals_all, dS)
        d0, d1, wd = self._bracket(self.deltaD_vals_all, dD)

        c00 = self._get_slice(s0, d0).query(a, b)
        c10 = self._get_slice(s1, d0).query(a, b)
        c01 = self._get_slice(s0, d1).query(a, b)
        c11 = self._get_slice(s1, d1).query(a, b)

        c0 = self._lerp_coeffs(c00, c10, ws)
        c1 = self._lerp_coeffs(c01, c11, ws)
        return self._lerp_coeffs(c0, c1, wd)

#E178 MLP Coefficent Model 
class MLPAeroModel:
    """
    Drop-in aero model using a trained PyTorch MLP.

    Expected MLP input:
        [alpha_deg, beta_deg, deltaS_deg, deltaD_deg]

    Expected MLP output:
        [CL, CD, CY, Cl, Cm, Cn]

    This class is designed to work with the existing
    aero_forces_moments_body(...) function.
    """

    def __init__(
        self,
        model_path: str,
        x_scaler_path: str,
        y_scaler_path: str,
        cfg: AeroConfig | None = None,
        device: str = "cpu",
        max_deltaS_deg: float = 20.0,
        max_deltaD_deg: float = 10.0,
    ):
        self.cfg = cfg if cfg is not None else AeroConfig()
        self.device = torch.device(device)

        # Load trained PyTorch model and sklearn scalers
        self.model = build_mlp_model().to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.eval()

        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

        # Bounds used by the existing aero_forces_moments_body(...)
        self.alpha_min = -float(self.cfg.max_alpha_deg)
        self.alpha_max =  float(self.cfg.max_alpha_deg)
        self.beta_min = -float(self.cfg.max_beta_deg)
        self.beta_max =  float(self.cfg.max_beta_deg)

        self.deltaS_min = -float(max_deltaS_deg)
        self.deltaS_max =  float(max_deltaS_deg)
        self.deltaD_min = -float(max_deltaD_deg)
        self.deltaD_max =  float(max_deltaD_deg)

    def clamp_deltas(self, deltaS_deg: float, deltaD_deg: float):
        """
        Existing aero_forces_moments_body(...) expects this method.
        """
        dS = float(np.clip(deltaS_deg, self.deltaS_min, self.deltaS_max))
        dD = float(np.clip(deltaD_deg, self.deltaD_min, self.deltaD_max))
        return dS, dD

    def query(self, alpha_deg: float, beta_deg: float, deltaS_deg: float, deltaD_deg: float):
        """
        Existing aero_forces_moments_body(...) expects this method.
        Must return a dictionary with:
            CL, CD, CY, Cl, Cm, Cn
        """

        alpha_deg = float(np.clip(alpha_deg, self.alpha_min, self.alpha_max))
        beta_deg = float(np.clip(beta_deg, self.beta_min, self.beta_max))
        deltaS_deg, deltaD_deg = self.clamp_deltas(deltaS_deg, deltaD_deg)

        x = pd.DataFrame(
            [[alpha_deg, beta_deg, deltaS_deg, deltaD_deg]],
            columns=["alpha_deg", "beta_deg", "deltaS_deg", "deltaD_deg"]
        )

        x_scaled = self.x_scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_scaled = self.model(x_tensor).cpu().numpy()

        coeffs = self.y_scaler.inverse_transform(y_scaled)[0]

        return {
            "CL": float(coeffs[0]),
            "CD": float(coeffs[1]),
            "CY": float(coeffs[2]),
            "Cl": float(coeffs[3]),
            "Cm": float(coeffs[4]),
            "Cn": float(coeffs[5]),
        }


# =============================================================================
# Main forces/moments interface (called by dynamics)
# =============================================================================

def aero_forces_moments_body(aero: AeroModel, veh, env, state,
                             deltaS_deg: float, deltaD_deg: float):
    """
    Returns:
      F_body (N), M_body (N*m), coeffs (dict), derived (dict)
    Expects:
      veh has: S_ref, b_ref, c_ref, com_body (3,), aero_ref_body (3,)
      env has: rho, wind_world (3,)
      state has: vel_world (3,), quat_body_to_world (quat x,y,z,w), omega_body (3,)
    """
    cfg = aero.cfg

    C_bw = dcm_body_to_world(state.quat_body_to_world)
    C_wb = C_bw.T

    v_rel_world = state.vel_world - env.wind_world
    v_rel_body = C_wb @ v_rel_world

    V, alpha, beta = wind_angles_from_body_velocity(v_rel_body)
    alpha_deg_raw = float(np.rad2deg(alpha))
    beta_deg_raw = float(np.rad2deg(beta))

    # Requested angles (clamped for safety)
    alpha_deg_req = float(np.clip(alpha_deg_raw, -cfg.max_alpha_deg, cfg.max_alpha_deg))
    beta_deg_req = float(np.clip(beta_deg_raw, -cfg.max_beta_deg, cfg.max_beta_deg))

    # Table angles (clipped to table bounds)
    alpha_deg_tbl = float(np.clip(alpha_deg_req, aero.alpha_min, aero.alpha_max))
    beta_deg_tbl = float(np.clip(beta_deg_req, aero.beta_min, aero.beta_max))

    qbar = 0.5 * float(env.rho) * V * V

    # Deflection handling
    dS_eff, dD_eff = aero.clamp_deltas(deltaS_deg, deltaD_deg)
    coeffs = aero.query(alpha_deg_tbl, beta_deg_tbl, dS_eff, dD_eff)

    # Apply sign flips
    coeffs["CL"] *= cfg.sign_cl
    coeffs["CD"] *= cfg.sign_cd
    coeffs["CY"] *= cfg.sign_cy
    coeffs["Cl"] *= cfg.sign_cll
    coeffs["Cm"] *= cfg.sign_cm
    coeffs["Cn"] *= cfg.sign_cn

    # CD0 correction
    coeffs["CD"] = max(0.0, float(coeffs["CD"]) + float(cfg.cd0_add))

    # Static extrapolation outside table bounds (per rad)
    if cfg.use_static_extrapolation:
        dalpha = np.deg2rad(alpha_deg_req - alpha_deg_tbl)
        dbeta = np.deg2rad(beta_deg_req - beta_deg_tbl)
        if abs(dalpha) > 0.0 or abs(dbeta) > 0.0:
            coeffs["CL"] += cfg.dcl_dalpha * dalpha + cfg.dcl_dbeta * dbeta
            coeffs["CD"] += cfg.dcd_dalpha * dalpha + cfg.dcd_dbeta * dbeta
            coeffs["CY"] += cfg.dcy_dalpha * dalpha + cfg.dcy_dbeta * dbeta
            coeffs["Cl"] += cfg.dcll_dalpha * dalpha + cfg.dcll_dbeta * dbeta
            coeffs["Cm"] += cfg.dcm_dalpha * dalpha + cfg.dcm_dbeta * dbeta
            coeffs["Cn"] += cfg.dcn_dalpha * dalpha + cfg.dcn_dbeta * dbeta

    # Rate damping
    p, q, r = state.omega_body
    fac_b = float(veh.b_ref) / (2.0 * max(V, 1e-3))
    fac_c = float(veh.c_ref) / (2.0 * max(V, 1e-3))
    coeffs["Cl"] += float(cfg.clp) * (p * fac_b)
    coeffs["Cm"] += float(cfg.cmq) * (q * fac_c)
    coeffs["Cn"] += float(cfg.cnr) * (r * fac_b)

    # Robust force construction from v_rel_body direction
    vnorm = float(np.linalg.norm(v_rel_body))
    if vnorm < 1e-9:
        F_body = np.zeros(3, dtype=float)
    else:
        vhat_b = v_rel_body / vnorm
        up_b = np.array([0.0, 0.0, -1.0], dtype=float)  # "up" is -Z (since Z down)

        side_dir_b = np.cross(up_b, vhat_b)
        side_norm = float(np.linalg.norm(side_dir_b))
        if side_norm < 1e-9:
            side_dir_b = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            side_dir_b /= side_norm

        lift_dir_b = np.cross(vhat_b, side_dir_b)
        lift_norm = float(np.linalg.norm(lift_dir_b))
        if lift_norm < 1e-9:
            lift_dir_b = up_b.copy()
        else:
            lift_dir_b /= lift_norm

        drag_dir_b = -vhat_b

        L = float(coeffs["CL"]) * qbar * float(veh.S_ref)
        D = float(coeffs["CD"]) * qbar * float(veh.S_ref)
        Yf = float(coeffs["CY"]) * qbar * float(veh.S_ref)

        F_body = L * lift_dir_b + D * drag_dir_b + Yf * side_dir_b

    # Moments about aero reference point
    M_coeff = qbar * float(veh.S_ref) * np.array(
        [float(veh.b_ref) * float(coeffs["Cl"]),
         float(veh.c_ref) * float(coeffs["Cm"]),
         float(veh.b_ref) * float(coeffs["Cn"])],
        dtype=float
    )

    r_ref_to_com = np.asarray(veh.aero_ref_body, dtype=float) - np.asarray(veh.com_body, dtype=float)
    M_arm = np.cross(r_ref_to_com, F_body)

    M_body = M_coeff + M_arm

    derived = {
        "V": float(V),
        "alpha_deg": float(alpha_deg_req),
        "beta_deg": float(beta_deg_req),
        "alpha_tbl_deg": float(alpha_deg_tbl),
        "beta_tbl_deg": float(beta_deg_tbl),
        "qbar": float(qbar),
        "deltaS_req_deg": float(deltaS_deg),
        "deltaD_req_deg": float(deltaD_deg),
        "deltaS_eff_deg": float(dS_eff),
        "deltaD_eff_deg": float(dD_eff),
    }
    return F_body, M_body, coeffs, derived
