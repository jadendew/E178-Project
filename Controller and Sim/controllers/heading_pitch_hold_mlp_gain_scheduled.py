 # controllers/heading_pitch_hold_mlp_gain_scheduled.py
"""
Heading + pitch tracking controller with MLP-informed gain scheduling.

Same basic architecture as heading_pitch_hold.py:

  HEADING outer loop:
    heading error -> bank command

  ROLL inner loop:
    bank error + roll-rate damping -> deltaD

  PITCH outer loop:
    pitch error -> q_cmd

  PITCH RATE inner loop:
    q_cmd - q -> deltaS

New part:
  The MLP aero model is used online to estimate local control effectiveness:
      dCm/d(deltaS)
      dCl/d(deltaD)

  These derivatives are used to schedule pitch and roll gains.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import aero

_prev_deltaS_cmd = 0.0
_prev_deltaD_cmd = 0.0
_first_call = True

def wrap_deg(angle_deg: float) -> float:
    """Wrap to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def get_alpha_beta_deg(state, cfg):
    """
    Compute alpha and beta the same way aero.py does.

    Uses:
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


def estimate_control_effectiveness(aero_model, alpha_deg, beta_deg, deltaS_trim, deltaD_trim, h_deg=1.0):
    """
    Estimate local control derivatives using central differences.

    Returns:
      Cm_deltaS = dCm / d(deltaS_deg)
      Cl_deltaD = dCl / d(deltaD_deg)
    """

    h = float(h_deg)

    # Pitch/elevator effectiveness
    c_s_plus = aero_model.query(
        alpha_deg,
        beta_deg,
        deltaS_trim + h,
        deltaD_trim,
    )

    c_s_minus = aero_model.query(
        alpha_deg,
        beta_deg,
        deltaS_trim - h,
        deltaD_trim,
    )

    Cm_deltaS = (float(c_s_plus["Cm"]) - float(c_s_minus["Cm"])) / (2.0 * h)

    # Roll/differential elevon effectiveness
    c_d_plus = aero_model.query(
        alpha_deg,
        beta_deg,
        deltaS_trim,
        deltaD_trim + h,
    )

    c_d_minus = aero_model.query(
        alpha_deg,
        beta_deg,
        deltaS_trim,
        deltaD_trim - h,
    )

    Cl_deltaD = (float(c_d_plus["Cl"]) - float(c_d_minus["Cl"])) / (2.0 * h)

    return Cm_deltaS, Cl_deltaD


def controller(t, state, cfg):
    global _prev_deltaS_cmd, _prev_deltaD_cmd, _first_call
    # ------------------------------------------------------------
    # Need aero model from run_sim.py
    # ------------------------------------------------------------
    if not hasattr(cfg, "aero_model_for_controller"):
        raise AttributeError(
            "cfg.aero_model_for_controller is missing. "
            "In run_sim.py, after creating aero_model, add:\n"
            "    cfg.aero_model_for_controller = aero_model"
        )

    aero_model = cfg.aero_model_for_controller

    # ------------------------------------------------------------
    # Trim/defaults from cfg
    # ------------------------------------------------------------
    deltaS_trim = float(getattr(cfg, "delta_s_deg", 0.0))
    deltaD_trim = float(getattr(cfg, "delta_d_deg", 0.0))
    throttle_trim = float(getattr(cfg, "throttle0", 0.0))

    # ------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------
    psi_cmd_deg = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    theta_cmd_deg = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))

    # ------------------------------------------------------------
    # Base HEADING/ROLL gains
    # ------------------------------------------------------------
    K_PSI = float(getattr(cfg, "hdg_k_psi", 0.8))
    K_PHI_BASE = float(getattr(cfg, "roll_k_phi", 1.2))
    K_P_BASE = float(getattr(cfg, "roll_k_p", 0.15))
    K_R_BASE = float(getattr(cfg, "yaw_k_r", 0.0))

    PHI_MAX_DEG = float(getattr(cfg, "phi_max_deg", 30.0))
    DELTAD_MAX_DEG = float(getattr(cfg, "deltaD_max_deg", 20.0))

    # ------------------------------------------------------------
    # Base PITCH gains
    # ------------------------------------------------------------
    K_THETA = float(getattr(cfg, "pit_k_theta", 0.8))
    K_Q_BASE = float(getattr(cfg, "pit_k_q", 0.06))
    K_QD_BASE = float(getattr(cfg, "pit_k_qd", 0.0))

    THETA_MAX_DEG = float(getattr(cfg, "theta_max_deg", 20.0))
    QS_CMD_MAX_DPS = float(getattr(cfg, "q_cmd_max_dps", 60.0))
    DELTAS_MAX_DEG = float(getattr(cfg, "deltaS_max_deg", 20.0))

    # ------------------------------------------------------------
    # Gain-scheduling settings
    # ------------------------------------------------------------
    USE_MLP_GAIN_SCHEDULING = bool(getattr(cfg, "use_mlp_gain_scheduling", True))

    H_DERIV_DEG = float(getattr(cfg, "mlp_gs_h_deriv_deg", 1.0))
    EFFECTIVENESS_EPS = float(getattr(cfg, "mlp_gs_effectiveness_eps", 1e-5))

    # Reference derivatives at nominal condition
    CM_DELTAS_REF = float(getattr(cfg, "mlp_gs_cm_deltaS_ref", 0.0))
    CL_DELTAD_REF = float(getattr(cfg, "mlp_gs_cl_deltaD_ref", 0.0))

    # Clamp scheduled gain multiplier
    GAIN_SCALE_MIN = float(getattr(cfg, "mlp_gs_gain_scale_min", 0.4))
    GAIN_SCALE_MAX = float(getattr(cfg, "mlp_gs_gain_scale_max", 2.5))

    # Optional: print gain-scheduling diagnostics occasionally
    PRINT_DEBUG = bool(getattr(cfg, "mlp_gs_print_debug", False))
    PRINT_DT = float(getattr(cfg, "mlp_gs_print_dt", 1.0))

    # ------------------------------------------------------------
    # Measure attitude from quaternion
    # scipy "ZYX" returns [yaw, pitch, roll]
    # ------------------------------------------------------------
    yaw_deg, pitch_deg, roll_deg = R.from_quat(
        state.quat_body_to_world
    ).as_euler("ZYX", degrees=True)

    # Body rates rad/s -> deg/s
    p_deg_s = float(np.rad2deg(float(state.omega_body[0])))
    q_deg_s = float(np.rad2deg(float(state.omega_body[1])))
    r_deg_s = float(np.rad2deg(float(state.omega_body[2])))

    # ------------------------------------------------------------
    # Current alpha/beta
    # ------------------------------------------------------------
    alpha_deg, beta_deg, V = get_alpha_beta_deg(state, cfg)

    max_alpha_deg = float(getattr(cfg, "max_alpha_deg", 35.0))
    max_beta_deg = float(getattr(cfg, "max_beta_deg", 20.0))

    alpha_deg = clamp(alpha_deg, -max_alpha_deg, max_alpha_deg)
    beta_deg = clamp(beta_deg, -max_beta_deg, max_beta_deg)

    # ------------------------------------------------------------
    # MLP gain scheduling
    # ------------------------------------------------------------
    K_PHI = K_PHI_BASE
    K_P = K_P_BASE
    K_R = K_R_BASE
    K_Q = K_Q_BASE
    K_QD = K_QD_BASE

    if USE_MLP_GAIN_SCHEDULING:
        Cm_deltaS, Cl_deltaD = estimate_control_effectiveness(
            aero_model=aero_model,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            deltaS_trim=deltaS_trim,
            deltaD_trim=deltaD_trim,
            h_deg=H_DERIV_DEG,
        )

        # If no reference values were provided, use current condition as fallback.
        # Better: set references in run_sim.py at nominal alpha=5, beta=0.
        if abs(CM_DELTAS_REF) < EFFECTIVENESS_EPS:
            CM_DELTAS_REF = Cm_deltaS

        if abs(CL_DELTAD_REF) < EFFECTIVENESS_EPS:
            CL_DELTAD_REF = Cl_deltaD

        # Gain scale = nominal effectiveness / current effectiveness.
        # If current surface effectiveness is larger, gain scale decreases.
        # If current surface effectiveness is smaller, gain scale increases.
        pitch_scale = abs(CM_DELTAS_REF) / max(abs(Cm_deltaS), EFFECTIVENESS_EPS)
        roll_scale = abs(CL_DELTAD_REF) / max(abs(Cl_deltaD), EFFECTIVENESS_EPS)

        pitch_scale = clamp(pitch_scale, GAIN_SCALE_MIN, GAIN_SCALE_MAX)
        roll_scale = clamp(roll_scale, GAIN_SCALE_MIN, GAIN_SCALE_MAX)

        # ------------------------------------------------------------
        # Improved MLP gain scheduling:
        #   - proportional gains get mild effectiveness scaling
        #   - rate damping gains get stronger scaling to reduce oscillation
        # ------------------------------------------------------------

        ROLL_PROP_SCALE_POWER = float(getattr(cfg, "mlp_gs_roll_prop_scale_power", 0.5))
        ROLL_DAMP_SCALE_POWER = float(getattr(cfg, "mlp_gs_roll_damp_scale_power", 1.0))

        PITCH_PROP_SCALE_POWER = float(getattr(cfg, "mlp_gs_pitch_prop_scale_power", 0.5))
        PITCH_DAMP_SCALE_POWER = float(getattr(cfg, "mlp_gs_pitch_damp_scale_power", 1.0))

        ROLL_DAMP_MULT = float(getattr(cfg, "mlp_gs_roll_damp_mult", 2.0))
        PITCH_DAMP_MULT = float(getattr(cfg, "mlp_gs_pitch_damp_mult", 2.0))
        YAW_DAMP_MULT = float(getattr(cfg, "mlp_gs_yaw_damp_mult", 1.5))

        roll_prop_scale = roll_scale ** ROLL_PROP_SCALE_POWER
        roll_damp_scale = roll_scale ** ROLL_DAMP_SCALE_POWER

        pitch_prop_scale = pitch_scale ** PITCH_PROP_SCALE_POWER
        pitch_damp_scale = pitch_scale ** PITCH_DAMP_SCALE_POWER

        K_PHI = K_PHI_BASE * roll_prop_scale
        K_P   = K_P_BASE   * roll_damp_scale * ROLL_DAMP_MULT
        K_R   = K_R_BASE   * roll_damp_scale * YAW_DAMP_MULT

        K_Q   = K_Q_BASE   * pitch_prop_scale
        K_QD  = K_QD_BASE  * pitch_damp_scale * PITCH_DAMP_MULT

        if PRINT_DEBUG and (t % PRINT_DT) < float(getattr(cfg, "dt", 0.01)):
            print(
                f"[MLP-GS] t={t:6.2f} "
                f"alpha={alpha_deg:7.2f} beta={beta_deg:7.2f} "
                f"Cm_dS={Cm_deltaS:+.6f} Cl_dD={Cl_deltaD:+.6f} "
                f"pitch_scale={pitch_scale:.3f} roll_scale={roll_scale:.3f}"
            )

    # ============================================================
    # HEADING -> BANK -> deltaD
    # ============================================================
    e_psi_deg = wrap_deg(psi_cmd_deg - float(yaw_deg))
    phi_cmd_deg = clamp(K_PSI * e_psi_deg, -PHI_MAX_DEG, PHI_MAX_DEG)

    e_phi_deg = float(phi_cmd_deg - float(roll_deg))

    deltaD_cmd = (K_PHI * e_phi_deg) - (K_P * p_deg_s) - (K_R * r_deg_s)
    deltaD_cmd = clamp(deltaD_cmd, -DELTAD_MAX_DEG, DELTAD_MAX_DEG)

    # ============================================================
    # PITCH -> q_cmd -> deltaS
    # ============================================================
    theta_cmd_deg = clamp(theta_cmd_deg, -THETA_MAX_DEG, THETA_MAX_DEG)

    e_theta_deg = float(theta_cmd_deg - float(pitch_deg))

    q_cmd_deg_s = clamp(K_THETA * e_theta_deg, -QS_CMD_MAX_DPS, QS_CMD_MAX_DPS)

    deltaS_delta = (K_Q * (q_cmd_deg_s - q_deg_s)) - (K_QD * q_deg_s)
    deltaS_cmd = float(deltaS_trim + deltaS_delta)
    deltaS_cmd = clamp(deltaS_cmd, -DELTAS_MAX_DEG, DELTAS_MAX_DEG)

    throttle = float(np.clip(throttle_trim, 0.0, 1.0))

    # Preserve original convention from heading_pitch_hold.py:
    # original returns (deltaS_cmd, -deltaD_cmd, throttle)
    # ------------------------------------------------------------
    # Command smoothing and rate limiting
    # ------------------------------------------------------------

    # Preserve original convention from heading_pitch_hold.py:
    # original returns -deltaD_cmd.
    deltaD_out = -float(deltaD_cmd)
    deltaS_out = float(deltaS_cmd)

    # ------------------------------------------------------------
    # Initial recovery bypass
    # ------------------------------------------------------------
    rate_limit_start_time = float(getattr(cfg, "mlp_gs_rate_limit_start_time", 0.25))

    if t < rate_limit_start_time:
        # During the first transient, allow the controller to respond freely.
        # This prevents delayed pitch correction from creating huge q spikes.
        _prev_deltaS_cmd = deltaS_out
        _prev_deltaD_cmd = deltaD_out
        _first_call = False

        return (float(deltaS_out), float(deltaD_out), throttle)

    if _first_call:
        _prev_deltaS_cmd = deltaS_out
        _prev_deltaD_cmd = deltaD_out
        _first_call = False

    dt = float(getattr(cfg, "dt", 0.006))

    deltaS_rate_max_dps = float(getattr(cfg, "mlp_gs_deltaS_rate_max_dps", 80.0))
    deltaD_rate_max_dps = float(getattr(cfg, "mlp_gs_deltaD_rate_max_dps", 80.0))

    max_deltaS_step = deltaS_rate_max_dps * dt
    max_deltaD_step = deltaD_rate_max_dps * dt

    # Rate limit
    deltaS_out = clamp(
        deltaS_out,
        _prev_deltaS_cmd - max_deltaS_step,
        _prev_deltaS_cmd + max_deltaS_step,
    )

    deltaD_out = clamp(
        deltaD_out,
        _prev_deltaD_cmd - max_deltaD_step,
        _prev_deltaD_cmd + max_deltaD_step,
    )

    # Optional low-pass smoothing
    smooth = float(getattr(cfg, "mlp_gs_command_smoothing", 0.15))
    smooth = clamp(smooth, 0.0, 1.0)

    deltaS_out = (1.0 - smooth) * _prev_deltaS_cmd + smooth * deltaS_out
    deltaD_out = (1.0 - smooth) * _prev_deltaD_cmd + smooth * deltaD_out

    _prev_deltaS_cmd = deltaS_out
    _prev_deltaD_cmd = deltaD_out

    return (float(deltaS_out), float(deltaD_out), throttle)