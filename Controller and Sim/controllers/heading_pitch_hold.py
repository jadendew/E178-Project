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
    K_PSI = float(getattr(cfg, "hdg_k_psi", 0.8))     # deg bank / deg heading error
    K_PHI = float(getattr(cfg, "roll_k_phi", 1.2))    # deg deltaD / deg roll error
    K_P   = float(getattr(cfg, "roll_k_p", 0.15))     # deg deltaD / (deg/s) roll rate damping
    K_R   = float(getattr(cfg, "yaw_k_r", 0.0))       # deg deltaD / (deg/s) yaw-rate damping

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
    K_THETA = float(getattr(cfg, "pit_k_theta", 0.8))     # (deg/s) / deg
    K_Q     = float(getattr(cfg, "pit_k_q", 0.06))        # deg deltaS / (deg/s)

    # Optional: direct pitch-rate damping term using q (like a PD)
    # deltaS += -K_QD * q_deg_s
    K_QD    = float(getattr(cfg, "pit_k_qd", 0.0))        # start 0.0; try 0.01~0.10 if needed

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
    # Clamp pitch command itself (prevents crazy requests)
    theta_cmd_deg = clamp(theta_cmd_deg, -THETA_MAX_DEG, THETA_MAX_DEG)

    e_theta_deg = float(theta_cmd_deg - float(pitch_deg))

    # Outer: pitch error -> desired pitch rate (deg/s)
    q_cmd_deg_s = clamp(K_THETA * e_theta_deg, -QS_CMD_MAX_DPS, QS_CMD_MAX_DPS)

    # Inner: pitch rate tracking -> deltaS command (deg)
    # If your sign convention is opposite (pitch goes wrong way), flip sign on K_Q or on final return.
    deltaS_delta = (K_Q * (q_cmd_deg_s - q_deg_s)) - (K_QD * q_deg_s)
    deltaS_cmd = float(deltaS_trim + deltaS_delta)
    deltaS_cmd = clamp(deltaS_cmd, -DELTAS_MAX_DEG, DELTAS_MAX_DEG)

    # Throttle constant for now
    throttle = float(np.clip(throttle_trim, 0.0, 1.0))

    # IMPORTANT:
    # You already discovered deltaD needed a sign flip to match your sim convention.
    # Keep that here too:
    return (float(deltaS_cmd), float(-deltaD_cmd), throttle)
