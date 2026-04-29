"""
dynamics.py

Physics + integration only (no plotting).

Contains:
- State/Vehicle/Environment dataclasses
- Quaternion helpers
- Wind angles from body velocity
- Aero force/moment evaluation (uses AeroModel from aero.py)
- RK4 integrator step

IMPORTANT conventions:
- WORLD: NED (X forward, Y right, Z down)
- BODY:  X forward, Y right, Z down
"""

from __future__ import annotations

from dataclasses import dataclass


from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from aero import AeroModel


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Vehicle:
    mass: float
    I_body: np.ndarray
    com_body: np.ndarray
    aero_ref_body: np.ndarray
    S_ref: float
    b_ref: float
    c_ref: float

@dataclass
class Environment:
    rho: float
    g: float
    wind_world: np.ndarray

@dataclass
class State:
    pos_world: np.ndarray
    vel_world: np.ndarray
    quat_body_to_world: np.ndarray  # scipy quat [x,y,z,w]
    omega_body: np.ndarray          # [p,q,r] rad/s


# =============================================================================
# Math helpers
# =============================================================================

def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps

def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / safe_norm(q)

def omega_matrix(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([
        [0.0,   wz,  -wy,  wx],
        [-wz,  0.0,   wx,  wy],
        [wy,  -wx,  0.0,  wz],
        [-wx, -wy, -wz,  0.0]
    ], dtype=float)

def dcm_body_to_world(q_bw: np.ndarray) -> np.ndarray:
    if not np.isfinite(q_bw).all():
        raise ValueError("Quaternion became non-finite (NaN/Inf).")
    n = np.linalg.norm(q_bw)
    if n < 1e-12:
        raise ValueError("Quaternion norm collapsed to ~0.")
    return R.from_quat(q_bw / n).as_matrix()

def euler_to_quat_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return R.from_euler("ZYX", [yaw, pitch, roll], degrees=False).as_quat()

def wind_angles_from_body_velocity(v_body: np.ndarray) -> Tuple[float, float, float]:
    u, v, w = v_body
    V = safe_norm(v_body)
    alpha = np.arctan2(w, u)
    beta  = np.arctan2(v, np.sqrt(u*u + w*w))
    return V, alpha, beta


# =============================================================================
# Aero forces + moments
# =============================================================================

def aero_forces_moments_body(
    aero: AeroModel,
    veh: Vehicle,
    env: Environment,
    state: State,
    deltaS_deg: float,
    deltaD_deg: float,
    cfg: Dict
):
    """
    Returns:
      F_body (N), M_body (N*m), coeffs (dict), derived (dict)
    """
    # Settings from cfg
    MAX_ALPHA_DEG = float(cfg["MAX_ALPHA_DEG"])
    MAX_BETA_DEG  = float(cfg["MAX_BETA_DEG"])
    MAX_QBAR      = float(cfg["MAX_QBAR"])

    SIGN_CL = float(cfg["SIGN_CL"])
    SIGN_CD = float(cfg["SIGN_CD"])
    SIGN_CY = float(cfg["SIGN_CY"])
    SIGN_Cl = float(cfg["SIGN_Cl"])
    SIGN_Cm = float(cfg["SIGN_Cm"])
    SIGN_Cn = float(cfg["SIGN_Cn"])

    CD0_ADD = float(cfg["CD0_ADD"])

    USE_STATIC_EXTRAPOLATION = bool(cfg["USE_STATIC_EXTRAPOLATION"])
    DCL_DALPHA = float(cfg["DCL_DALPHA"]); DCL_DBETA = float(cfg["DCL_DBETA"])
    DCD_DALPHA = float(cfg["DCD_DALPHA"]); DCD_DBETA = float(cfg["DCD_DBETA"])
    DCY_DALPHA = float(cfg["DCY_DALPHA"]); DCY_DBETA = float(cfg["DCY_DBETA"])
    DCl_DALPHA = float(cfg["DCl_DALPHA"]); DCl_DBETA = float(cfg["DCl_DBETA"])
    DCm_DALPHA = float(cfg["DCm_DALPHA"]); DCm_DBETA = float(cfg["DCm_DBETA"])
    DCn_DALPHA = float(cfg["DCn_DALPHA"]); DCn_DBETA = float(cfg["DCn_DBETA"])

    CLP = float(cfg["CLP"])
    CMQ = float(cfg["CMQ"])
    CNR = float(cfg["CNR"])

    # Kinematics
    C_bw = dcm_body_to_world(state.quat_body_to_world)
    C_wb = C_bw.T

    v_rel_world = state.vel_world - env.wind_world
    v_rel_body = C_wb @ v_rel_world

    V, alpha, beta = wind_angles_from_body_velocity(v_rel_body)
    alpha_deg_raw = float(np.rad2deg(alpha))
    beta_deg_raw  = float(np.rad2deg(beta))

    # Requested angles (clamped)
    alpha_deg_req = float(np.clip(alpha_deg_raw, -MAX_ALPHA_DEG, MAX_ALPHA_DEG))
    beta_deg_req  = float(np.clip(beta_deg_raw,  -MAX_BETA_DEG,  MAX_BETA_DEG))

    # Table query angles (clamped to table bounds)
    alpha_deg_tbl = float(np.clip(alpha_deg_req, aero.alpha_min, aero.alpha_max))
    beta_deg_tbl  = float(np.clip(beta_deg_req,  aero.beta_min,  aero.beta_max))

    # qbar
    qbar = 0.5 * env.rho * V * V
    qbar = float(min(qbar, MAX_QBAR))

    # Deflection handling (continuous / superposed)
    dS_eff, dD_eff = aero.clamp_deltas(deltaS_deg, deltaD_deg)

    coeffs = aero.query(alpha_deg=alpha_deg_tbl, beta_deg=beta_deg_tbl,
                        deltaS_deg=dS_eff, deltaD_deg=dD_eff)

    # Sign flips
    coeffs["CL"] *= SIGN_CL
    coeffs["CD"] *= SIGN_CD
    coeffs["CY"] *= SIGN_CY
    coeffs["Cl"] *= SIGN_Cl
    coeffs["Cm"] *= SIGN_Cm
    coeffs["Cn"] *= SIGN_Cn

    # CD0 correction
    coeffs["CD"] = max(0.0, float(coeffs["CD"]) + CD0_ADD)

    # Static extrapolation (outside table in alpha/beta, but still within safety clamp)
    if USE_STATIC_EXTRAPOLATION:
        dalpha = np.deg2rad(alpha_deg_req - alpha_deg_tbl)
        dbeta  = np.deg2rad(beta_deg_req  - beta_deg_tbl)
        if abs(dalpha) > 0.0 or abs(dbeta) > 0.0:
            coeffs["CL"] += DCL_DALPHA * dalpha + DCL_DBETA * dbeta
            coeffs["CD"] += DCD_DALPHA * dalpha + DCD_DBETA * dbeta
            coeffs["CY"] += DCY_DALPHA * dalpha + DCY_DBETA * dbeta
            coeffs["Cl"] += DCl_DALPHA * dalpha + DCl_DBETA * dbeta
            coeffs["Cm"] += DCm_DALPHA * dalpha + DCm_DBETA * dbeta
            coeffs["Cn"] += DCn_DALPHA * dalpha + DCn_DBETA * dbeta

    # Rate damping
    p, q, r = state.omega_body
    fac_b = veh.b_ref / (2.0 * max(V, 1e-3))
    fac_c = veh.c_ref / (2.0 * max(V, 1e-3))
    coeffs["Cl"] += CLP * (p * fac_b)
    coeffs["Cm"] += CMQ * (q * fac_c)
    coeffs["Cn"] += CNR * (r * fac_b)

    # =============================================================================
    # Robust force construction from v_rel_body direction (consistent L/D)
    # =============================================================================
    vnorm = float(np.linalg.norm(v_rel_body))
    if vnorm < 1e-9:
        F_body = np.zeros(3, dtype=float)
    else:
        vhat_b = v_rel_body / vnorm

        # BODY: +Z down => up is -Z
        up_b = np.array([0.0, 0.0, -1.0], dtype=float)

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

        L = float(coeffs["CL"]) * qbar * veh.S_ref
        D = float(coeffs["CD"]) * qbar * veh.S_ref
        Yf = float(coeffs["CY"]) * qbar * veh.S_ref

        F_body = L * lift_dir_b + D * drag_dir_b + Yf * side_dir_b

    # Moments about aero reference point
    M_coeff = qbar * veh.S_ref * np.array(
        [veh.b_ref * float(coeffs["Cl"]),
         veh.c_ref * float(coeffs["Cm"]),
         veh.b_ref * float(coeffs["Cn"])],
        dtype=float
    )

    # Arm moment if ref != COM
    r_ref_to_com = veh.aero_ref_body - veh.com_body
    M_arm = np.cross(r_ref_to_com, F_body)

    M_body = M_coeff + M_arm

    derived = {
        "V": float(V),
        "alpha_deg": alpha_deg_req,
        "beta_deg": beta_deg_req,
        "alpha_tbl_deg": alpha_deg_tbl,
        "beta_tbl_deg": beta_deg_tbl,
        "qbar": float(qbar),
        "deltaS_req_deg": float(deltaS_deg),
        "deltaD_req_deg": float(deltaD_deg),
        "deltaS_eff_deg": float(dS_eff),
        "deltaD_eff_deg": float(dD_eff),
    }
    return F_body, M_body, coeffs, derived


# =============================================================================
# Integrator
# =============================================================================

def step_rk4(
    state: State,
    veh: Vehicle,
    env: Environment,
    aero: AeroModel,
    dt: float,
    deltaS_deg: float,
    deltaD_deg: float,
    cfg: Dict
):
    MAX_SPEED = float(cfg["MAX_SPEED"])
    MAX_OMEGA = float(cfg["MAX_OMEGA"])

    def deriv(s: State):
        F_b, M_b, coeffs, der = aero_forces_moments_body(aero, veh, env, s, deltaS_deg, deltaD_deg, cfg)

        C_bw = dcm_body_to_world(s.quat_body_to_world)
        F_aero_w = C_bw @ F_b

        # Gravity in NED: +g in +Z (down)
        F_grav_w = np.array([0.0, 0.0, veh.mass * env.g], dtype=float)

        a_w = (F_aero_w + F_grav_w) / veh.mass

        I = veh.I_body
        w = s.omega_body
        H = I @ w
        w_dot = np.linalg.solve(I, (M_b - np.cross(w, H)))

        q = s.quat_body_to_world
        q_dot = 0.5 * omega_matrix(w) @ q

        return a_w, w_dot, q_dot, coeffs, der, F_b, M_b

    a1, w1, q1, c1, d1, F1, M1 = deriv(state)

    def mk(base: State, dp, dv, dq, dw):
        qn = base.quat_body_to_world + dq
        return State(
            pos_world=base.pos_world + dp,
            vel_world=base.vel_world + dv,
            quat_body_to_world=quat_normalize(qn),
            omega_body=base.omega_body + dw
        )

    s2 = mk(state, dp=0.5*dt*state.vel_world, dv=0.5*dt*a1, dq=0.5*dt*q1, dw=0.5*dt*w1)
    a2, w2, q2, _, _, _, _ = deriv(s2)

    s3 = mk(state, dp=0.5*dt*(state.vel_world + 0.5*dt*a1), dv=0.5*dt*a2, dq=0.5*dt*q2, dw=0.5*dt*w2)
    a3, w3, q3, _, _, _, _ = deriv(s3)

    s4 = mk(state, dp=dt*(state.vel_world + 0.5*dt*a2), dv=dt*a3, dq=dt*q3, dw=dt*w3)
    a4, w4, q4, _, _, _, _ = deriv(s4)

    pos = state.pos_world + (dt/6.0)*(state.vel_world + 2*(state.vel_world+0.5*dt*a1) + 2*(state.vel_world+0.5*dt*a2) + (state.vel_world+dt*a3))
    vel = state.vel_world + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)
    omg = state.omega_body + (dt/6.0)*(w1 + 2*w2 + 2*w3 + w4)
    quat = quat_normalize(state.quat_body_to_world + (dt/6.0)*(q1 + 2*q2 + 2*q3 + q4))

    # Safety clamps
    V = safe_norm(vel)
    if V > MAX_SPEED:
        vel = vel * (MAX_SPEED / V)
    wmag = safe_norm(omg)
    if wmag > MAX_OMEGA:
        omg = omg * (MAX_OMEGA / wmag)

    return State(pos_world=pos, vel_world=vel, quat_body_to_world=quat, omega_body=omg), c1, d1, F1, M1
