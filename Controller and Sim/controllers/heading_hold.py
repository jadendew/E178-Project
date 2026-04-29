# controllers/heading_hold.py
"""
Closed-loop heading tracking controller.

Returns:
  (deltaS_deg, deltaD_deg, throttle)

Architecture:
  heading error (psi_cmd - psi) -> phi_cmd (bank command)
  roll loop: (phi_cmd - phi) + roll-rate damping -> deltaD

Notes:
- This tracks *heading* (yaw angle from attitude), not ground-track/course.
- If the aircraft turns the wrong way, flip the sign of K_PSI or K_PHI.
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
    """
    Inputs:
      t: time (s)
      state: dynamics.State (pos, vel, quat_body_to_world [x,y,z,w], omega_body [rad/s])
      cfg: dynamics.SimConfig (plus any optional fields you attach in run_sim)

    Output:
      (deltaS_deg, deltaD_deg, throttle)
    """

    # ------------------------------------------------------------
    # Defaults from cfg (same pattern as open_loop)
    # ------------------------------------------------------------
    deltaS_trim = float(getattr(cfg, "delta_s_deg", 0.0))
    throttle_trim = float(getattr(cfg, "throttle0", 0.0))

    # ------------------------------------------------------------
    # Heading command
    # - If you add cfg.heading_cmd_deg in run_sim, it will use it.
    # - Otherwise, default to initial heading yaw0_deg.
    # ------------------------------------------------------------
    psi_cmd_deg = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))

    # ------------------------------------------------------------
    # Gains (you can override by attaching these to cfg in run_sim)
    # Units are in DEG and DEG/s to keep it intuitive:
    #   phi_cmd_deg = K_PSI * heading_err_deg
    #   deltaD_deg  = K_PHI * roll_err_deg - K_P * p_deg_s
    # ------------------------------------------------------------
    K_PSI = float(getattr(cfg, "hdg_k_psi", 0.8))     # deg bank / deg heading error
    K_PHI = float(getattr(cfg, "roll_k_phi", 1.2))    # deg deltaD / deg roll error
    K_P   = float(getattr(cfg, "roll_k_p", 0.15))     # deg deltaD / (deg/s) roll rate damping

    # Limits
    PHI_MAX_DEG   = float(getattr(cfg, "phi_max_deg", 30.0))     # bank command limit
    DELTAD_MAX_DEG = float(getattr(cfg, "deltaD_max_deg", 20.0)) # actuator limit

    # Optional: add a small yaw-rate damper term using body r (rad/s)
    # deltaD += -K_R * r_deg_s
    K_R = float(getattr(cfg, "yaw_k_r", 0.0))  # start at 0.0; try 0.02~0.10 if needed

    # ------------------------------------------------------------
    # Measure attitude (yaw, roll) from quaternion
    # ------------------------------------------------------------
    yaw_deg, pitch_deg, roll_deg = R.from_quat(state.quat_body_to_world).as_euler("ZYX", degrees=True)

    # Heading error (wrap!)
    e_psi_deg = wrap_deg(psi_cmd_deg - float(yaw_deg))

    # Outer loop: heading -> bank command
    phi_cmd_deg = clamp(K_PSI * e_psi_deg, -PHI_MAX_DEG, PHI_MAX_DEG)

    # Inner loop: roll -> deltaD
    e_phi_deg = float(phi_cmd_deg - float(roll_deg))

    p_rad_s = float(state.omega_body[0])  # roll rate
    r_rad_s = float(state.omega_body[2])  # yaw rate
    p_deg_s = float(np.rad2deg(p_rad_s))
    r_deg_s = float(np.rad2deg(r_rad_s))

    deltaD_cmd = (K_PHI * e_phi_deg) - (K_P * p_deg_s) - (K_R * r_deg_s)
    deltaD_cmd = clamp(deltaD_cmd, -DELTAD_MAX_DEG, DELTAD_MAX_DEG)

    # Throttle: keep constant for now (controller can change later)
    throttle = float(np.clip(throttle_trim, 0.0, 1.0))

    return (deltaS_trim, float(-deltaD_cmd), throttle)
