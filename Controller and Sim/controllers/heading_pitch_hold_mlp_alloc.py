# controllers/heading_pitch_hold_mlp_alloc.py

"""
Heading + pitch controller with MLP-based control allocation.

This keeps the same outer/inner loop logic as heading_pitch_hold.py,
but instead of returning the raw deltaS/deltaD commands directly,
it uses the MLP aero model to allocate deltaS/deltaD that produce
the desired moment coefficients.

Returns:
  (deltaS_deg, deltaD_deg, throttle)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import aero
from controllers.mlp_allocator import MLPControlAllocator


_allocator = None


def wrap_deg(angle_deg: float) -> float:
    """Wrap to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def get_alpha_beta_deg(state, cfg):
    """
    Compute alpha and beta the same way aero.py does.

    Assumes:
      state.vel_world
      state.quat_body_to_world
      cfg.wind_world
    """
    C_bw = aero.dcm_body_to_world(state.quat_body_to_world)
    C_wb = C_bw.T

    wind_world = np.asarray(getattr(cfg, "wind_world", np.zeros(3)), dtype=float)

    v_rel_world = np.asarray(state.vel_world, dtype=float) - wind_world
    v_rel_body = C_wb @ v_rel_world

    V, alpha_rad, beta_rad = aero.wind_angles_from_body_velocity(v_rel_body)

    return float(np.rad2deg(alpha_rad)), float(np.rad2deg(beta_rad)), float(V)


def controller(t, state, cfg):
    global _allocator

    # ------------------------------------------------------------
    # Need aero model from run_sim.py
    # ------------------------------------------------------------
    if not hasattr(cfg, "aero_model_for_allocator"):
        raise AttributeError(
            "cfg.aero_model_for_allocator is missing. "
            "In run_sim.py, after creating aero_model, add:\n"
            "    cfg.aero_model_for_allocator = aero_model"
        )

    if _allocator is None:
        _allocator = MLPControlAllocator(
            aero_model=cfg.aero_model_for_allocator,
            deltaS_max_deg=float(getattr(cfg, "deltaS_max_deg", 20.0)),
            deltaD_max_deg=float(getattr(cfg, "deltaD_max_deg", 10.0)),
            elevon_max_deg=float(getattr(cfg, "mlp_alloc_elevon_max_deg", 25.0)),
            w_cl=float(getattr(cfg, "mlp_alloc_w_cl", 5.0)),
            w_cm=float(getattr(cfg, "mlp_alloc_w_cm", 10.0)),
            w_cn=float(getattr(cfg, "mlp_alloc_w_cn", 1.0)),
            w_effort=float(getattr(cfg, "mlp_alloc_w_effort", 0.0005)),
            w_smooth=float(getattr(cfg, "mlp_alloc_w_smooth", 0.02)),
            w_elevon_limit=float(getattr(cfg, "mlp_alloc_w_elevon_limit", 100.0)),
            maxiter=int(getattr(cfg, "mlp_alloc_maxiter", 20)),
        )

    # ------------------------------------------------------------
    # Trim/defaults
    # ------------------------------------------------------------
    deltaS_trim = float(getattr(cfg, "delta_s_deg", 0.0))
    throttle_trim = float(getattr(cfg, "throttle0", 0.0))

    # ------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------
    psi_cmd_deg = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    theta_cmd_deg = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))

    # ------------------------------------------------------------
    # HEADING/ROLL gains
    # ------------------------------------------------------------
    K_PSI = float(getattr(cfg, "hdg_k_psi", 0.8))
    K_PHI = float(getattr(cfg, "roll_k_phi", 1.2))
    K_P   = float(getattr(cfg, "roll_k_p", 0.15))
    K_R   = float(getattr(cfg, "yaw_k_r", 0.0))

    PHI_MAX_DEG    = float(getattr(cfg, "phi_max_deg", 30.0))
    DELTAD_MAX_DEG = float(getattr(cfg, "deltaD_max_deg", 20.0))

    # ------------------------------------------------------------
    # PITCH gains
    # ------------------------------------------------------------
    K_THETA = float(getattr(cfg, "pit_k_theta", 0.8))
    K_Q     = float(getattr(cfg, "pit_k_q", 0.06))
    K_QD    = float(getattr(cfg, "pit_k_qd", 0.0))

    THETA_MAX_DEG  = float(getattr(cfg, "theta_max_deg", 20.0))
    QS_CMD_MAX_DPS = float(getattr(cfg, "q_cmd_max_dps", 60.0))
    DELTAS_MAX_DEG = float(getattr(cfg, "deltaS_max_deg", 20.0))

    # ------------------------------------------------------------
    # Attitude and body rates
    # ------------------------------------------------------------
    yaw_deg, pitch_deg, roll_deg = R.from_quat(
        state.quat_body_to_world
    ).as_euler("ZYX", degrees=True)

    p_deg_s = float(np.rad2deg(float(state.omega_body[0])))
    q_deg_s = float(np.rad2deg(float(state.omega_body[1])))
    r_deg_s = float(np.rad2deg(float(state.omega_body[2])))

    # ============================================================
    # Same virtual-control commands as heading_pitch_hold.py
    # ============================================================

    # HEADING -> BANK -> virtual deltaD
    e_psi_deg = wrap_deg(psi_cmd_deg - float(yaw_deg))
    phi_cmd_deg = clamp(K_PSI * e_psi_deg, -PHI_MAX_DEG, PHI_MAX_DEG)

    e_phi_deg = float(phi_cmd_deg - float(roll_deg))
    deltaD_virtual = (K_PHI * e_phi_deg) - (K_P * p_deg_s) - (K_R * r_deg_s)
    deltaD_virtual = clamp(deltaD_virtual, -DELTAD_MAX_DEG, DELTAD_MAX_DEG)

    # Preserve your existing convention:
    # the original controller returned -deltaD_cmd.
    deltaD_virtual = -deltaD_virtual

    # PITCH -> q_cmd -> virtual deltaS
    theta_cmd_deg = clamp(theta_cmd_deg, -THETA_MAX_DEG, THETA_MAX_DEG)

    e_theta_deg = float(theta_cmd_deg - float(pitch_deg))

    q_cmd_deg_s = clamp(K_THETA * e_theta_deg, -QS_CMD_MAX_DPS, QS_CMD_MAX_DPS)

    deltaS_delta = (K_Q * (q_cmd_deg_s - q_deg_s)) - (K_QD * q_deg_s)
    deltaS_virtual = float(deltaS_trim + deltaS_delta)
    deltaS_virtual = clamp(deltaS_virtual, -DELTAS_MAX_DEG, DELTAS_MAX_DEG)

    # ------------------------------------------------------------
    # Compute current alpha/beta
    # ------------------------------------------------------------
    alpha_deg, beta_deg, V = get_alpha_beta_deg(state, cfg)

    # Clamp alpha/beta to aero model safety range if present
    alpha_deg = clamp(
        alpha_deg,
        -float(getattr(cfg, "max_alpha_deg", 35.0)),
        float(getattr(cfg, "max_alpha_deg", 35.0)),
    )
    beta_deg = clamp(
        beta_deg,
        -float(getattr(cfg, "max_beta_deg", 20.0)),
        float(getattr(cfg, "max_beta_deg", 20.0)),
    )

    # ------------------------------------------------------------
    # Convert virtual control command into desired moment coefficients
    # ------------------------------------------------------------
    target_coeffs = cfg.aero_model_for_allocator.query(
        alpha_deg,
        beta_deg,
        deltaS_virtual,
        deltaD_virtual,
    )

    Cl_cmd = float(target_coeffs["Cl"])
    Cm_cmd = float(target_coeffs["Cm"])
    Cn_cmd = float(target_coeffs["Cn"])

    # ------------------------------------------------------------
    # MLP-based allocation
    # ------------------------------------------------------------
    deltaS_alloc, deltaD_alloc = _allocator.allocate(
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        Cl_cmd=Cl_cmd,
        Cm_cmd=Cm_cmd,
        Cn_cmd=Cn_cmd,
        initial_guess=np.array([deltaS_virtual, deltaD_virtual], dtype=float),
    )

    throttle = float(np.clip(throttle_trim, 0.0, 1.0))

    return (float(deltaS_alloc), float(deltaD_alloc), throttle)