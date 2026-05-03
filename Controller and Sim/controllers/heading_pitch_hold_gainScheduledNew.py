# controllers/heading_pitch_hold.py
"""
Closed-loop heading + pitch tracking controller.

Returns:
  (deltaS_deg, deltaD_deg, throttle)

Architecture:
  HEADING (outer):
    heading error (psi_cmd - psi) -> phi_cmd (bank command)
  ROLL (inner):
    (phi_cmd - phi) + roll-rate damping -> deltaD_cmd

  PITCH (outer):
    pitch error (theta_cmd - theta) -> q_cmd (pitch rate command)
  PITCH RATE (inner):
    (q_cmd - q) -> deltaS_cmd

Notes:
- This tracks *heading* (yaw angle from attitude), not ground-track/course.
- Pitch control assumes deltaS primarily affects pitch moment (Cm) as expected.
- If pitch moves the wrong way, flip the sign of pitch gains or deltaS output.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def wrap_deg(angle_deg: float) -> float:
    """Wrap to [-180, 180)."""
    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))

def smooth_scale_by_abs_value(
    x,
    x_low,
    x_high,
    scale_high,
    scale_low,
):
    """
    Smoothly maps abs(x) to a scale.

    For abs(x) <= x_low:
        scale ~= scale_high

    For abs(x) >= x_high:
        scale ~= scale_low

    Between x_low and x_high:
        smooth transition.

    This avoids hard jumps in gain scheduling.
    """
    ax = abs(float(x))

    if x_high <= x_low:
        return float(scale_low)

    s = (ax - x_low) / (x_high - x_low)
    s = clamp(s, 0.0, 1.0)

    # smoothstep: 0 -> 1 with zero slope at ends
    s = s * s * (3.0 - 2.0 * s)

    return float(scale_high + (scale_low - scale_high) * s)

def update_filtered_rate_derivatives(cfg, p_deg_s, q_deg_s, r_deg_s, dt):
    """
    Estimate angular accelerations from body rates and low-pass filter them.

    Inputs:
        p_deg_s, q_deg_s, r_deg_s in deg/s
        dt in seconds

    Returns:
        p_dot_filt, q_dot_filt, r_dot_filt in deg/s^2

    Purpose:
        Allows the scheduler to increase damping when angular rates
        are changing rapidly, not only when rates are already large.
    """

    dt = max(float(dt), 1e-9)

    # First call: initialize previous values and return zero derivatives
    if not hasattr(cfg, "_prev_p_deg_s"):
        cfg._prev_p_deg_s = float(p_deg_s)
        cfg._prev_q_deg_s = float(q_deg_s)
        cfg._prev_r_deg_s = float(r_deg_s)

        cfg._filt_p_dot_deg_s2 = 0.0
        cfg._filt_q_dot_deg_s2 = 0.0
        cfg._filt_r_dot_deg_s2 = 0.0

        return 0.0, 0.0, 0.0

    # Raw finite-difference derivatives
    p_dot_raw = (float(p_deg_s) - float(cfg._prev_p_deg_s)) / dt
    q_dot_raw = (float(q_deg_s) - float(cfg._prev_q_deg_s)) / dt
    r_dot_raw = (float(r_deg_s) - float(cfg._prev_r_deg_s)) / dt

    # Low-pass filter
    alpha = float(getattr(cfg, "deriv_filter_alpha", 0.20))
    alpha = clamp(alpha, 0.01, 1.0)

    cfg._filt_p_dot_deg_s2 = (
        alpha * p_dot_raw + (1.0 - alpha) * float(cfg._filt_p_dot_deg_s2)
    )
    cfg._filt_q_dot_deg_s2 = (
        alpha * q_dot_raw + (1.0 - alpha) * float(cfg._filt_q_dot_deg_s2)
    )
    cfg._filt_r_dot_deg_s2 = (
        alpha * r_dot_raw + (1.0 - alpha) * float(cfg._filt_r_dot_deg_s2)
    )

    # Store current rates for next timestep
    cfg._prev_p_deg_s = float(p_deg_s)
    cfg._prev_q_deg_s = float(q_deg_s)
    cfg._prev_r_deg_s = float(r_deg_s)

    return (
        float(cfg._filt_p_dot_deg_s2),
        float(cfg._filt_q_dot_deg_s2),
        float(cfg._filt_r_dot_deg_s2),
    )

# may need to delete this if it causes timing problems ----------------------------
def rate_limit_cmd(cmd, prev_cmd, max_rate_deg_s, dt):
    max_step = max_rate_deg_s * dt
    return clamp(cmd, prev_cmd - max_step, prev_cmd + max_step)
# may need to delete this if it causes timing problems ------------------------

def compute_airdata_from_state(state, cfg):
    """
    Computes approximate airspeed, alpha, and beta from the current sim state.

    Assumptions:
    - state.vel_world is velocity in world/NED frame.
    - state.quat_body_to_world maps body frame to world frame.
    - cfg.wind_world is wind velocity in world frame.
    """

    vel_world = np.asarray(state.vel_world, dtype=float)
    wind_world = np.asarray(getattr(cfg, "wind_world", np.zeros(3)), dtype=float)

    v_air_world = vel_world - wind_world

    # Convert world velocity to body frame
    rot_body_to_world = R.from_quat(state.quat_body_to_world)
    v_air_body = rot_body_to_world.inv().apply(v_air_world)

    u = float(v_air_body[0])
    v = float(v_air_body[1])
    w = float(v_air_body[2])

    V = float(np.linalg.norm(v_air_body))

    if V < 1e-9:
        return 0.0, 0.0, 0.0

    # Common aircraft convention:
    # alpha = atan2(w, u)
    # beta  = asin(v / V)
    alpha_deg = float(np.rad2deg(np.arctan2(w, max(abs(u), 1e-9))))
    beta_deg = float(np.rad2deg(np.arcsin(np.clip(v / V, -1.0, 1.0))))

    return V, alpha_deg, beta_deg

def schedule_gains(
    cfg,
    V,
    alpha_deg,
    beta_deg,
    e_psi_deg,
    e_phi_deg,
    e_theta_deg,
    p_deg_s,
    q_deg_s,
    r_deg_s,
):
    """
    Smooth gain schedule for the cascaded heading/pitch controller.

    Keeps:
      heading -> roll -> deltaD
      pitch -> q_cmd -> deltaS

    Improvements over the first version:
      - Smooth transitions instead of step changes.
      - Allows mild gain boost near trim.
      - Uses gentler velocity scheduling.
      - Avoids over-aggressive pitch-rate gain scaling.
    """

    # ------------------------------------------------------------
    # Base gains from cfg
    # ------------------------------------------------------------
    K_PSI_BASE = float(getattr(cfg, "hdg_k_psi", 0.8))
    K_PHI_BASE = float(getattr(cfg, "roll_k_phi", 1.2))
    K_P_BASE   = float(getattr(cfg, "roll_k_p", 0.15))
    K_R_BASE   = float(getattr(cfg, "yaw_k_r", 0.0))

    K_THETA_BASE = float(getattr(cfg, "pit_k_theta", 0.8))
    K_Q_BASE     = float(getattr(cfg, "pit_k_q", 0.06))
    K_QD_BASE    = float(getattr(cfg, "pit_k_qd", 0.0))

    if not bool(getattr(cfg, "use_gain_scheduling", True)):
        return {
            "K_PSI": K_PSI_BASE,
            "K_PHI": K_PHI_BASE,
            "K_P": K_P_BASE,
            "K_R": K_R_BASE,
            "K_THETA": K_THETA_BASE,
            "K_Q": K_Q_BASE,
            "K_QD": K_QD_BASE,
            "vel_scale": 1.0,
            "aero_scale": 1.0,
            "pitch_err_scale": 1.0,
            "roll_err_scale": 1.0,
            "heading_err_scale": 1.0,
            "roll_damp_scale": 1.0,
            "pitch_damp_scale": 1.0,
        }

    # ------------------------------------------------------------
    # Velocity scheduling
    #
    # If disabled, vel_scale = 1.0 and speed does not affect gains.
    # This is useful when the aircraft operates over a narrow speed range.
    # ------------------------------------------------------------
    if bool(getattr(cfg, "use_velocity_scheduling", True)):
        V_ref = float(getattr(cfg, "sched_v_ref", 20.0))
        V_min = float(getattr(cfg, "sched_v_min", 8.0))
        V_eff = max(float(V), V_min)

        vel_exp = float(getattr(cfg, "sched_vel_exp", 1.0))

        vel_scale = (V_ref / V_eff) ** vel_exp
        vel_scale = clamp(
            vel_scale,
            float(getattr(cfg, "sched_vel_scale_min", 0.55)),
            float(getattr(cfg, "sched_vel_scale_max", 1.60)),
        )
    else:
        vel_scale = 1.0
    # ------------------------------------------------------------
    # Smooth error-based scheduling
    # ------------------------------------------------------------
    pitch_err_scale = smooth_scale_by_abs_value(
        e_theta_deg,
        x_low=0.75,
        x_high=8.0,
        scale_high=0.9,
        scale_low=1.3,
    )

    roll_err_scale = smooth_scale_by_abs_value(
        e_phi_deg,
        x_low=2.0,
        x_high=25.0,
        scale_high=0.9,
        scale_low=1.3,
    )

    heading_err_scale = smooth_scale_by_abs_value(
        e_psi_deg,
        x_low=5.0,
        x_high=45.0,
        scale_high=0.8,
        scale_low=1.35,
    )

    # ------------------------------------------------------------
    # Smooth nonlinear aero protection
    # ------------------------------------------------------------
    alpha_scale = smooth_scale_by_abs_value(
        alpha_deg,
        x_low=3.0,
        x_high=18.0,
        scale_high=1.05,
        scale_low=0.80,
    )

    beta_scale = smooth_scale_by_abs_value(
        beta_deg,
        x_low=2.0,
        x_high=14.0,
        scale_high=1.05,
        scale_low=0.80,
    )

    aero_scale = min(alpha_scale, beta_scale)

    # ------------------------------------------------------------
    # Rate damping schedule
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Rate-based damping schedule
    #
    # These increase damping when angular rates are large.
    # They do NOT reduce tracking gains.
    # ------------------------------------------------------------
    roll_damp_scale = smooth_scale_by_abs_value(
        p_deg_s,
        x_low=5.0,
        x_high=45.0,
        scale_high=1.00,
        scale_low=2.200,
    )

    pitch_damp_scale = smooth_scale_by_abs_value(
        q_deg_s,
        x_low=5.0,
        x_high=45.0,
        scale_high=1.00,
        scale_low=2.40,
    )

    yaw_damp_scale = smooth_scale_by_abs_value(
        r_deg_s,
        x_low=4.0,
        x_high=35.0,
        scale_high=1.00,
        scale_low=1.50,
    )

    # ------------------------------------------------------------
    # Final scheduled gains
    #
    # Tracking gains keep a minimum authority so the controller
    # can still move toward the setpoint even when far away.
    #
    # Damping gains are allowed to grow more strongly.
    # ------------------------------------------------------------

    # Outer heading loop: heading error -> commanded roll
    K_PSI = K_PSI_BASE * heading_err_scale
    K_PSI = clamp(K_PSI, 0.65 * K_PSI_BASE, 1.60 * K_PSI_BASE)

    # Outer pitch loop: pitch error -> commanded pitch rate
    K_THETA = K_THETA_BASE * pitch_err_scale * aero_scale
    K_THETA = clamp(K_THETA, 0.65 * K_THETA_BASE, 1.80 * K_THETA_BASE)

    # Inner roll loop: roll error -> deltaD
    roll_inner_scale = vel_scale * roll_err_scale * aero_scale
    roll_inner_scale = clamp(roll_inner_scale, 0.65, 2.25)
    K_PHI = K_PHI_BASE * roll_inner_scale

    # Inner pitch loop: pitch-rate error -> deltaS
    # Keep this from dropping too low so pitch still tracks.
    pitch_inner_scale = vel_scale * pitch_err_scale * aero_scale
    pitch_inner_scale = clamp(pitch_inner_scale, 0.65, 2.25)
    K_Q = K_Q_BASE * pitch_inner_scale

    # Roll damping: p-rate damping
    roll_damp_total = max(0.75, vel_scale) * roll_damp_scale
    roll_damp_total = clamp(roll_damp_total, 0.75, 3.00)
    K_P = K_P_BASE * roll_damp_total

    # Yaw damping: r-rate damping
    yaw_damp_total = max(0.75, vel_scale) * yaw_damp_scale
    yaw_damp_total = clamp(yaw_damp_total, 0.75, 3.00)
    K_R = K_R_BASE * yaw_damp_total

    # Direct pitch damping: q-rate damping
    pitch_damp_total = max(0.75, vel_scale) * pitch_damp_scale
    pitch_damp_total = clamp(pitch_damp_total, 0.75, 3.50)
    K_QD = K_QD_BASE * pitch_damp_total

    return {
        "K_PSI": K_PSI,
        "K_PHI": K_PHI,
        "K_P": K_P,
        "K_R": K_R,
        "K_THETA": K_THETA,
        "K_Q": K_Q,
        "K_QD": K_QD,
        "vel_scale": vel_scale,
        "aero_scale": aero_scale,
        "pitch_err_scale": pitch_err_scale,
        "roll_err_scale": roll_err_scale,
        "heading_err_scale": heading_err_scale,
        "roll_damp_scale": roll_damp_scale,
        "pitch_damp_scale": pitch_damp_scale,
    }

def controller(t, state, cfg):
    # ------------------------------------------------------------
    # Trim/defaults from cfg (matches your style)
    # ------------------------------------------------------------
    deltaS_trim = float(getattr(cfg, "delta_s_deg", 0.0))
    throttle_trim = float(getattr(cfg, "throttle0", 0.0))

    # ------------------------------------------------------------
    # Commands (you can set these in run_sim via cfg.<field>)
    # ------------------------------------------------------------
    psi_cmd_deg = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    theta_cmd_deg = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))

    # ------------------------------------------------------------
    # HEADING/ROLL gains (same as your heading_hold.py)
    # ------------------------------------------------------------

    PHI_MAX_DEG    = float(getattr(cfg, "phi_max_deg", 30.0))
    DELTAD_MAX_DEG = float(getattr(cfg, "deltaD_max_deg", 20.0))

    # ------------------------------------------------------------
    # PITCH gains (new)
    #
    # Outer pitch loop: theta_err -> q_cmd
    # Inner pitch-rate loop: (q_cmd - q) -> deltaS_cmd
    #
    # Units (keep intuitive):
    #   q_cmd_deg_s = K_THETA * theta_err_deg
    #   deltaS_deg  = K_Q * (q_cmd_deg_s - q_deg_s)
    # ------------------------------------------------------------
    # Optional: direct pitch-rate damping term using q (like a PD)
    # deltaS += -K_QD * q_deg_s

    THETA_MAX_DEG  = float(getattr(cfg, "theta_max_deg", 20.0))   # pitch command clamp
    QS_CMD_MAX_DPS = float(getattr(cfg, "q_cmd_max_dps", 60.0))   # max pitch rate cmd (deg/s)
    DELTAS_MAX_DEG = float(getattr(cfg, "deltaS_max_deg", 20.0))  # actuator limit

    # ------------------------------------------------------------
    # Measure attitude from quaternion
    # scipy "ZYX" returns [yaw, pitch, roll]
    # ------------------------------------------------------------
    yaw_deg, pitch_deg, roll_deg = R.from_quat(state.quat_body_to_world).as_euler("ZYX", degrees=True)

    # Body rates (rad/s -> deg/s)
    p_deg_s = float(np.rad2deg(float(state.omega_body[0])))
    q_deg_s = float(np.rad2deg(float(state.omega_body[1])))
    r_deg_s = float(np.rad2deg(float(state.omega_body[2])))

    # ------------------------------------------------------------
    # Estimate angular accelerations for derivative-based damping
    # ------------------------------------------------------------
    dt = float(getattr(cfg, "dt", 0.01))

    p_dot_deg_s2, q_dot_deg_s2, r_dot_deg_s2 = update_filtered_rate_derivatives(
        cfg=cfg,
        p_deg_s=p_deg_s,
        q_deg_s=q_deg_s,
        r_deg_s=r_deg_s,
        dt=dt,
    )

    cfg._ctrl_p_dot_deg_s2 = float(p_dot_deg_s2)
    cfg._ctrl_q_dot_deg_s2 = float(q_dot_deg_s2)
    cfg._ctrl_r_dot_deg_s2 = float(r_dot_deg_s2)

    # ============================================================
    # HEADING preliminary
    # ============================================================
    e_psi_deg = wrap_deg(psi_cmd_deg - float(yaw_deg))

    K_PSI_PRE = float(getattr(cfg, "hdg_k_psi", 0.8))
    phi_cmd_pre_deg = clamp(K_PSI_PRE * e_psi_deg, -PHI_MAX_DEG, PHI_MAX_DEG)
    e_phi_deg = float(phi_cmd_pre_deg - float(roll_deg))

    # Pitch error
    theta_cmd_deg = clamp(theta_cmd_deg, -THETA_MAX_DEG, THETA_MAX_DEG)
    e_theta_deg = float(theta_cmd_deg - float(pitch_deg))

    # Airdata
    V, alpha_deg, beta_deg = compute_airdata_from_state(state, cfg)

    # Schedule gains
    gains = schedule_gains(
        cfg,
        V,
        alpha_deg,
        beta_deg,
        e_psi_deg,
        e_phi_deg,
        e_theta_deg,
        p_deg_s,
        q_deg_s,
        r_deg_s,
    )

    K_PSI = gains["K_PSI"]
    K_PHI = gains["K_PHI"]
    K_P = gains["K_P"]
    K_R = gains["K_R"]
    K_THETA = gains["K_THETA"]
    K_Q = gains["K_Q"]
    K_QD = gains["K_QD"]

    # ------------------------------------------------------------
    # Optional axis-specific derivative-based damping boost
    #
    # Key idea:
    #   Keep derivative damping where it helps.
    #   Disable or weaken it where it hurts.
    # ------------------------------------------------------------

    use_roll_deriv = bool(getattr(cfg, "use_roll_derivative_damping", False))
    use_pitch_deriv = bool(getattr(cfg, "use_pitch_derivative_damping", True))
    use_yaw_deriv = bool(getattr(cfg, "use_yaw_derivative_damping", False))

    if use_roll_deriv:
        roll_accel_damp_scale = smooth_scale_by_abs_value(
            p_dot_deg_s2,
            x_low=80.0,
            x_high=700.0,
            scale_high=1.00,
            scale_low=1.60,
        )
    else:
        roll_accel_damp_scale = 1.0

    if use_pitch_deriv:
        pitch_accel_damp_scale = smooth_scale_by_abs_value(
            q_dot_deg_s2,
            x_low=40.0,
            x_high=400.0,
            scale_high=1.00,
            scale_low=2.25,
        )
    else:
        pitch_accel_damp_scale = 1.0

    if use_yaw_deriv:
        yaw_accel_damp_scale = smooth_scale_by_abs_value(
            r_dot_deg_s2,
            x_low=100.0,
            x_high=900.0,
            scale_high=1.00,
            scale_low=1.50,
        )
    else:
        yaw_accel_damp_scale = 1.0

    # Apply only to damping gains
    K_P *= roll_accel_damp_scale
    K_QD *= pitch_accel_damp_scale
    K_R *= yaw_accel_damp_scale

    # Safety clamps
    K_P_BASE_SAFE = abs(float(getattr(cfg, "roll_k_p", 0.18)))
    K_QD_BASE_SAFE = abs(float(getattr(cfg, "pit_k_qd", 0.05)))
    K_R_BASE_SAFE = abs(float(getattr(cfg, "yaw_k_r", 0.04)))

    K_P = clamp(K_P, 0.0, 4.0 * max(K_P_BASE_SAFE, 1e-6))
    K_QD = clamp(K_QD, 0.0, 6.0 * max(K_QD_BASE_SAFE, 1e-6))

    # Keep yaw damping magnitude more conservative
    K_R = clamp(
        K_R,
        -3.0 * max(K_R_BASE_SAFE, 1e-6),
        3.0 * max(K_R_BASE_SAFE, 1e-6),
    )

    cfg._ctrl_use_roll_derivative_damping = float(use_roll_deriv)
    cfg._ctrl_use_pitch_derivative_damping = float(use_pitch_deriv)
    cfg._ctrl_use_yaw_derivative_damping = float(use_yaw_deriv)

    cfg._ctrl_roll_accel_damp_scale = float(roll_accel_damp_scale)
    cfg._ctrl_pitch_accel_damp_scale = float(pitch_accel_damp_scale)
    cfg._ctrl_yaw_accel_damp_scale = float(yaw_accel_damp_scale)
    # ------------------------------------------------------------
    # Store controller diagnostics on cfg so dynamics.py can log them
    # ------------------------------------------------------------
    cfg._ctrl_K_PSI = float(K_PSI)
    cfg._ctrl_K_PHI = float(K_PHI)
    cfg._ctrl_K_P = float(K_P)
    cfg._ctrl_K_R = float(K_R)

    cfg._ctrl_K_THETA = float(K_THETA)
    cfg._ctrl_K_Q = float(K_Q)
    cfg._ctrl_K_QD = float(K_QD)

    cfg._ctrl_V = float(V)
    cfg._ctrl_alpha_deg = float(alpha_deg)
    cfg._ctrl_beta_deg = float(beta_deg)

    cfg._ctrl_vel_scale = float(gains.get("vel_scale", np.nan))
    cfg._ctrl_aero_scale = float(gains.get("aero_scale", np.nan))
    cfg._ctrl_pitch_err_scale = float(gains.get("pitch_err_scale", np.nan))
    cfg._ctrl_roll_err_scale = float(gains.get("roll_err_scale", np.nan))
    cfg._ctrl_heading_err_scale = float(gains.get("heading_err_scale", np.nan))
    cfg._ctrl_roll_damp_scale = float(gains.get("roll_damp_scale", np.nan))
    cfg._ctrl_pitch_damp_scale = float(gains.get("pitch_damp_scale", np.nan))

    # ------------------------------------------------------------
    # Store preliminary controller diagnostics on cfg
    # These variables exist at this point in the controller.
    # ------------------------------------------------------------
    cfg._ctrl_e_psi_deg = float(e_psi_deg)
    cfg._ctrl_e_phi_deg_pre = float(e_phi_deg)
    cfg._ctrl_e_theta_deg = float(e_theta_deg)

    # ============================================================
    # HEADING -> BANK -> deltaD
    # ============================================================
    phi_cmd_deg = clamp(K_PSI * e_psi_deg, -PHI_MAX_DEG, PHI_MAX_DEG)
    e_phi_deg = float(phi_cmd_deg - float(roll_deg))

    deltaD_cmd = (K_PHI * e_phi_deg) - (K_P * p_deg_s) - (K_R * r_deg_s)
    deltaD_cmd = clamp(deltaD_cmd, -DELTAD_MAX_DEG, DELTAD_MAX_DEG)

    # ============================================================
    # PITCH -> q_cmd -> deltaS
    # ============================================================
    q_cmd_deg_s = clamp(K_THETA * e_theta_deg, -QS_CMD_MAX_DPS, QS_CMD_MAX_DPS)

    deltaS_delta = (K_Q * (q_cmd_deg_s - q_deg_s)) - (K_QD * q_deg_s)
    deltaS_cmd = float(deltaS_trim + deltaS_delta)
    deltaS_cmd = clamp(deltaS_cmd, -DELTAS_MAX_DEG, DELTAS_MAX_DEG)

    # ------------------------------------------------------------
    # Saturation diagnostics
    # ------------------------------------------------------------
    q_cmd_raw_deg_s = K_THETA * e_theta_deg
    deltaS_raw_no_clamp = float(deltaS_trim + deltaS_delta)
    deltaD_raw_no_clamp = (K_PHI * e_phi_deg) - (K_P * p_deg_s) - (K_R * r_deg_s)
    phi_cmd_raw_deg = K_PSI * e_psi_deg

    cfg._ctrl_q_cmd_raw_deg_s = float(q_cmd_raw_deg_s)
    cfg._ctrl_phi_cmd_raw_deg = float(phi_cmd_raw_deg)

    cfg._ctrl_deltaS_raw_no_clamp_deg = float(deltaS_raw_no_clamp)
    cfg._ctrl_deltaD_raw_no_clamp_deg = float(deltaD_raw_no_clamp)

    cfg._ctrl_q_cmd_saturated = float(abs(q_cmd_raw_deg_s) >= 0.98 * QS_CMD_MAX_DPS)
    cfg._ctrl_phi_cmd_saturated = float(abs(phi_cmd_raw_deg) >= 0.98 * PHI_MAX_DEG)
    cfg._ctrl_deltaS_saturated = float(abs(deltaS_raw_no_clamp) >= 0.98 * DELTAS_MAX_DEG)
    cfg._ctrl_deltaD_saturated = float(abs(deltaD_raw_no_clamp) >= 0.98 * DELTAD_MAX_DEG)

    # ------------------------------------------------------------
    # Store command diagnostics BEFORE rate limiting
    # These now exist because phi_cmd_deg, deltaD_cmd,
    # q_cmd_deg_s, and deltaS_cmd have already been computed.
    # ------------------------------------------------------------
    cfg._ctrl_phi_cmd_deg = float(phi_cmd_deg)
    cfg._ctrl_e_phi_deg = float(e_phi_deg)
    cfg._ctrl_q_cmd_deg_s = float(q_cmd_deg_s)

    cfg._ctrl_deltaS_unsmoothed_deg = float(deltaS_cmd)
    cfg._ctrl_deltaD_unsmoothed_deg = float(deltaD_cmd)

    # Throttle constant for now
    throttle = float(np.clip(throttle_trim, 0.0, 1.0))

    # IMPORTANT:
    # You already discovered deltaD needed a sign flip to match your sim convention.
    # Keep that here too:
    # ------- may need to get rid of if rate limiting causes problems --------
    dt = float(getattr(cfg, "dt", 0.01))

    deltaS_rate_max = float(getattr(cfg, "deltaS_rate_max_deg_s", 120.0))
    deltaD_rate_max = float(getattr(cfg, "deltaD_rate_max_deg_s", 120.0))

    prev_deltaS = float(getattr(cfg, "_prev_deltaS_cmd", deltaS_trim))
    prev_deltaD = float(getattr(cfg, "_prev_deltaD_cmd", 0.0))

    deltaS_cmd = rate_limit_cmd(deltaS_cmd, prev_deltaS, deltaS_rate_max, dt)
    deltaD_cmd = rate_limit_cmd(deltaD_cmd, prev_deltaD, deltaD_rate_max, dt)

    cfg._prev_deltaS_cmd = float(deltaS_cmd)
    cfg._prev_deltaD_cmd = float(deltaD_cmd)
    # ------------------------------------------------------------
    # Store final command diagnostics AFTER rate limiting
    # ------------------------------------------------------------
    cfg._ctrl_deltaS_final_deg = float(deltaS_cmd)
    cfg._ctrl_deltaD_final_deg = float(deltaD_cmd)
    # ------- Madse need to delete if rate limiting causes problems --------
    return (float(deltaS_cmd), float(-deltaD_cmd), throttle)
