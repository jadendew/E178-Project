# controllers/mlp_fixed_thrust_mpc.py
"""
Fixed-thrust MLP-aero MPC controller with warm-start + smaller soft deadband + moderated heading-to-roll + predicted heading-rate damping for dynamicsMPC_dashboard.py.

This controller DOES NOT replace your aero/physics. Instead it uses your existing
MLP aero model through dynamicsNew.step_rk4(...) during the MPC rollout.

Required setup in run_sim_MPC.py:
    cfg._mpc_aero_model = aero_model

Controller output accepted by dynamicsMPC_dashboard.py:
    dict with deltaS_deg, deltaD_deg, throttle

Main commands:
    cfg.heading_cmd_deg  -> yaw/heading command in deg
    cfg.pitch_cmd_deg    -> pitch command in deg
    cfg.mpc_roll_cmd_deg -> optional roll command, default 0 deg

Control inputs optimized by MPC:
    deltaS_deg, deltaD_deg

Thrust:
    fixed at cfg.throttle0. The MPC does not optimize throttle.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import dynamicsMPC_dashboard as dynamics


# =============================================================================
# Small helpers
# =============================================================================

def wrap_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))

def soft_deadband(error: float, deadband: float, inside_scale: float = 0.10) -> float:
    """
    Soft deadband for tracking errors.

    Hard deadband makes errors inside the band equal zero, which can leave a
    nonzero trim error. This soft version still penalizes inside-band error,
    but at reduced strength. Outside the band, the cost grows normally and
    continuously.
    """
    error = float(error)
    deadband = abs(float(deadband))
    inside_scale = float(np.clip(float(inside_scale), 0.0, 1.0))
    abs_e = abs(error)

    if deadband <= 1.0e-12:
        return error

    if abs_e <= deadband:
        return float(inside_scale * error)

    return float(np.sign(error) * (inside_scale * deadband + (abs_e - deadband)))


def steady_state_check(state, cfg) -> tuple[bool, dict]:
    """Return whether the aircraft is close enough to target to use hold/deadband logic."""
    yaw_deg, pitch_deg, roll_deg = euler_ypr_deg(state)
    p_deg_s, q_deg_s, r_deg_s = np.rad2deg(state.omega_body)
    psi_cmd, theta_cmd, roll_cmd = get_command_targets(state, cfg)

    e_psi = abs(wrap_deg(psi_cmd - yaw_deg))
    e_theta = abs(float(theta_cmd - pitch_deg))
    e_phi = abs(wrap_deg(roll_cmd - roll_deg))

    steady_now = (
        e_psi <= float(getattr(cfg, "mpc_steady_heading_err_deg", 1.0))
        and e_theta <= float(getattr(cfg, "mpc_steady_pitch_err_deg", 0.5))
        and e_phi <= float(getattr(cfg, "mpc_steady_roll_err_deg", 2.0))
        and abs(float(p_deg_s)) <= float(getattr(cfg, "mpc_steady_p_rate_deg_s", 5.0))
        and abs(float(q_deg_s)) <= float(getattr(cfg, "mpc_steady_q_rate_deg_s", 5.0))
        and abs(float(r_deg_s)) <= float(getattr(cfg, "mpc_steady_r_rate_deg_s", 5.0))
    )

    info = {
        "e_psi_abs": float(e_psi),
        "e_theta_abs": float(e_theta),
        "e_phi_abs": float(e_phi),
        "p_deg_s": float(p_deg_s),
        "q_deg_s": float(q_deg_s),
        "r_deg_s": float(r_deg_s),
    }
    return bool(steady_now), info


def rate_limit_cmd(cmd: float, prev_cmd: float, max_rate_deg_s: float, dt: float) -> float:
    max_step = abs(float(max_rate_deg_s)) * float(dt)
    return clamp(cmd, prev_cmd - max_step, prev_cmd + max_step)


def lowpass_cmd(raw_cmd: float, prev_filtered_cmd: float, tau_s: float, dt_s: float) -> tuple[float, float]:
    """
    First-order low-pass filter for actuator commands.

    y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    alpha = dt / (tau + dt)

    Larger tau_s gives smoother but slower actuator commands.
    Returns filtered command and alpha.
    """
    tau_s = max(float(tau_s), 1.0e-9)
    dt_s = max(float(dt_s), 1.0e-9)
    alpha = dt_s / (tau_s + dt_s)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    y = float(prev_filtered_cmd) + alpha * (float(raw_cmd) - float(prev_filtered_cmd))
    return float(y), alpha


def euler_ypr_deg(state) -> tuple[float, float, float]:
    """Return yaw, pitch, roll in deg from state.quat_body_to_world."""
    yaw_deg, pitch_deg, roll_deg = R.from_quat(state.quat_body_to_world).as_euler(
        "ZYX", degrees=True
    )
    return float(yaw_deg), float(pitch_deg), float(roll_deg)


def airdata_from_state(state, cfg) -> tuple[float, float, float]:
    """Compute V, alpha_deg, beta_deg from the current state and cfg.wind_world."""
    vel_world = np.asarray(state.vel_world, dtype=float)
    wind_world = np.asarray(getattr(cfg, "wind_world", np.zeros(3)), dtype=float)
    v_air_world = vel_world - wind_world

    rot_body_to_world = R.from_quat(state.quat_body_to_world)
    v_air_body = rot_body_to_world.inv().apply(v_air_world)

    u, v, w = [float(x) for x in v_air_body]
    V = float(np.linalg.norm(v_air_body))

    if V < 1e-9:
        return 0.0, 0.0, 0.0

    alpha_deg = float(np.rad2deg(np.arctan2(w, u)))
    beta_deg = float(np.rad2deg(np.arcsin(np.clip(v / V, -1.0, 1.0))))
    return V, alpha_deg, beta_deg


def make_vehicle_and_env(cfg):
    veh = dynamics.Vehicle(
        mass=float(cfg.mass),
        I_body=np.asarray(cfg.I_body, dtype=float),
        com_body=np.asarray(cfg.com_body, dtype=float),
        aero_ref_body=np.asarray(cfg.aero_ref_body, dtype=float),
        S_ref=float(cfg.S_ref),
        b_ref=float(cfg.b_ref),
        c_ref=float(cfg.c_ref),
    )
    env = dynamics.Environment(
        rho=float(cfg.rho),
        g=float(cfg.g),
        wind_world=np.asarray(cfg.wind_world, dtype=float),
    )
    return veh, env




def get_command_targets(state, cfg):
    """
    Returns yaw, pitch, and roll commands for the MPC.

    If cfg.mpc_use_dynamic_roll_cmd is True, roll command is generated from
    heading error so the vehicle is allowed to bank during large heading changes.
    As heading error goes to zero, this naturally returns roll_cmd back to the
    base roll command, usually 0 deg.
    """
    yaw_deg, pitch_deg, roll_deg = euler_ypr_deg(state)

    psi_cmd = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    theta_cmd = float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", 0.0)))

    base_roll_cmd = float(getattr(cfg, "mpc_roll_cmd_deg", 0.0))

    if bool(getattr(cfg, "mpc_use_dynamic_roll_cmd", False)):
        e_psi = wrap_deg(psi_cmd - yaw_deg)
        k_phi = float(getattr(cfg, "mpc_heading_to_roll_gain", 0.25))
        phi_max = float(getattr(cfg, "mpc_roll_cmd_max_deg", 25.0))
        fade_start = abs(float(getattr(cfg, "mpc_heading_to_roll_deadband_deg", 2.0)))
        fade_full = abs(float(getattr(cfg, "mpc_heading_to_roll_fade_full_deg", 10.0)))
        fade_full = max(fade_full, fade_start + 1.0e-6)

        # Fade out heading-to-roll coupling near the target instead of using roll
        # for tiny steady-state heading cleanup. This preserves the large banked
        # turn while reducing the yaw/roll limit cycle near the command.
        abs_e = abs(e_psi)
        if abs_e <= fade_start:
            roll_blend = 0.0
        elif abs_e >= fade_full:
            roll_blend = 1.0
        else:
            roll_blend = (abs_e - fade_start) / (fade_full - fade_start)

        e_psi_for_roll = roll_blend * e_psi
        roll_cmd = float(np.clip(base_roll_cmd + k_phi * e_psi_for_roll, -phi_max, phi_max))

        cfg._mpc_heading_to_roll_blend = float(roll_blend)
        cfg._mpc_e_psi_for_roll_deg = float(e_psi_for_roll)
    else:
        roll_cmd = base_roll_cmd
        cfg._mpc_heading_to_roll_blend = 0.0
        cfg._mpc_e_psi_for_roll_deg = 0.0

    return psi_cmd, theta_cmd, roll_cmd


# =============================================================================
# MPC cost and rollout
# =============================================================================

def tracking_cost(state, control_vec, prev_control_vec, cfg) -> float:
    """State/control cost for one prediction node."""
    yaw_deg, pitch_deg, roll_deg = euler_ypr_deg(state)
    p, q, r = [float(x) for x in state.omega_body]
    p_deg_s, q_deg_s, r_deg_s = np.rad2deg([p, q, r])
    V, alpha_deg, beta_deg = airdata_from_state(state, cfg)

    psi_cmd, theta_cmd, roll_cmd = get_command_targets(state, cfg)
    alpha_cmd = float(getattr(cfg, "mpc_alpha_cmd_deg", alpha_deg))
    beta_cmd = float(getattr(cfg, "mpc_beta_cmd_deg", 0.0))

    e_psi = wrap_deg(psi_cmd - yaw_deg)
    e_theta = float(theta_cmd - pitch_deg)
    e_phi = wrap_deg(roll_cmd - roll_deg)
    e_alpha = float(alpha_cmd - alpha_deg)
    e_beta = float(beta_cmd - beta_deg)

    # Soft deadbands reduce near-trim hunting without completely accepting
    # nonzero steady-state error. Inside the band, errors still count, just less.
    inside_scale = float(getattr(cfg, "mpc_deadband_inside_scale", 0.10))
    e_psi = soft_deadband(e_psi, float(getattr(cfg, "mpc_heading_deadband_deg", 0.50)), inside_scale)
    e_theta = soft_deadband(e_theta, float(getattr(cfg, "mpc_pitch_deadband_deg", 0.25)), inside_scale)
    e_phi = soft_deadband(e_phi, float(getattr(cfg, "mpc_roll_deadband_deg", 0.50)), inside_scale)
    e_alpha = soft_deadband(e_alpha, float(getattr(cfg, "mpc_alpha_deadband_deg", 0.25)), inside_scale)
    e_beta = soft_deadband(e_beta, float(getattr(cfg, "mpc_beta_deadband_deg", 0.25)), inside_scale)

    Q_HEADING = float(getattr(cfg, "mpc_Q_heading", 1.0))
    Q_PITCH = float(getattr(cfg, "mpc_Q_pitch", 3.0))
    Q_ROLL = float(getattr(cfg, "mpc_Q_roll", 0.25))
    Q_RATES = np.asarray(getattr(cfg, "mpc_Q_rates", [0.012, 0.020, 0.012]), dtype=float)
    Q_ALPHA = float(getattr(cfg, "mpc_Q_alpha", 0.15))
    Q_BETA = float(getattr(cfg, "mpc_Q_beta", 0.50))
    Q_ALT = float(getattr(cfg, "mpc_Q_altitude", 0.0))
    Q_SPEED = float(getattr(cfg, "mpc_Q_speed", 0.0))

    R_U = np.asarray(getattr(cfg, "mpc_R_control", [0.012, 0.012]), dtype=float)
    R_DU = np.asarray(getattr(cfg, "mpc_R_delta_control", [0.10, 0.10]), dtype=float)

    deltaS_deg, deltaD_deg = np.asarray(control_vec, dtype=float)
    prev_deltaS_deg, prev_deltaD_deg = np.asarray(prev_control_vec, dtype=float)

    cost = 0.0
    cost += Q_HEADING * e_psi**2
    cost += Q_PITCH * e_theta**2
    cost += Q_ROLL * e_phi**2
    cost += Q_ALPHA * e_alpha**2
    cost += Q_BETA * e_beta**2
    cost += float(np.sum(Q_RATES * np.array([p_deg_s, q_deg_s, r_deg_s])**2))

    # Optional altitude and speed terms. Use with care because fixed thrust limits authority.
    if Q_ALT > 0.0 and hasattr(cfg, "altitude_cmd_m"):
        altitude_m = -float(state.pos_world[2])  # NED z down -> altitude proxy
        cost += Q_ALT * (float(cfg.altitude_cmd_m) - altitude_m) ** 2

    if Q_SPEED > 0.0 and hasattr(cfg, "speed_cmd_mps"):
        cost += Q_SPEED * (float(cfg.speed_cmd_mps) - V) ** 2

    u = np.array([deltaS_deg, deltaD_deg], dtype=float)
    du = np.array([deltaS_deg - prev_deltaS_deg, deltaD_deg - prev_deltaD_deg], dtype=float)
    trim_u = np.array([
        float(getattr(cfg, "mpc_trim_deltaS_deg", getattr(cfg, "delta_s_deg", 0.0))),
        float(getattr(cfg, "mpc_trim_deltaD_deg", getattr(cfg, "delta_d_deg", 0.0))),
    ], dtype=float)
    cost += float(np.sum(R_U * (u - trim_u)**2))
    cost += float(np.sum(R_DU * du**2))

    return float(cost)


def rollout_sequence_cost(state, sequence, cfg, veh, env, aero_model) -> float:
    """Roll out one candidate sequence using the real dynamics integrator."""
    sim_state = state
    prev_u = np.asarray(getattr(cfg, "_mpc_prev_u", [cfg.delta_s_deg, cfg.delta_d_deg]), dtype=float)
    fixed_throttle = float(np.clip(getattr(cfg, "throttle0", 1.0), 0.0, 1.0))
    pred_dt = float(getattr(cfg, "mpc_pred_dt", max(float(cfg.dt), 0.02)))

    total = 0.0
    prev_yaw_deg = euler_ypr_deg(sim_state)[0]
    Q_YAW_RATE_PRED = float(getattr(cfg, "mpc_Q_yaw_rate_pred", 0.0))

    for k, u in enumerate(sequence):
        deltaS_deg, deltaD_deg = [float(x) for x in u]
        ctrl = dynamics.Control(deltaS_deg=deltaS_deg, deltaD_deg=deltaD_deg, throttle=fixed_throttle)

        sim_state, _, _, _ = dynamics.step_rk4(
            sim_state, veh, env, cfg, aero_model, ctrl, pred_dt
        )

        yaw_deg = euler_ypr_deg(sim_state)[0]
        yaw_rate_pred_deg_s = wrap_deg(yaw_deg - prev_yaw_deg) / max(pred_dt, 1.0e-9)
        prev_yaw_deg = yaw_deg

        # Later horizon nodes get slightly more weight so the optimizer cares about where it is going.
        node_weight = 1.0 + 0.08 * float(k)
        total += node_weight * tracking_cost(sim_state, u, prev_u, cfg)

        # Directly damp predicted heading-rate. Body yaw-rate r is useful, but in a
        # banked flying-wing turn, heading-rate is not exactly the same quantity.
        # This helps reduce late slow heading wander without shutting off heading authority.
        if Q_YAW_RATE_PRED > 0.0:
            total += node_weight * Q_YAW_RATE_PRED * yaw_rate_pred_deg_s**2

        prev_u = np.asarray(u, dtype=float)

        # Safety: immediately reject sequences that drive crazy states.
        if not (
            np.isfinite(sim_state.pos_world).all()
            and np.isfinite(sim_state.vel_world).all()
            and np.isfinite(sim_state.quat_body_to_world).all()
            and np.isfinite(sim_state.omega_body).all()
        ):
            return 1.0e30

    # Terminal cost: make the final predicted heading matter more than small
    # intermediate dithering. This hardens heading tracking without needing to
    # over-penalize every node in the horizon.
    yaw_T, pitch_T, roll_T = euler_ypr_deg(sim_state)
    psi_cmd_T, theta_cmd_T, roll_cmd_T = get_command_targets(sim_state, cfg)
    e_psi_T = wrap_deg(psi_cmd_T - yaw_T)
    e_theta_T = float(theta_cmd_T - pitch_T)
    e_phi_T = wrap_deg(roll_cmd_T - roll_T)

    total += float(getattr(cfg, "mpc_Q_terminal_heading", 0.0)) * e_psi_T**2
    total += float(getattr(cfg, "mpc_Q_terminal_pitch", 0.0)) * e_theta_T**2
    total += float(getattr(cfg, "mpc_Q_terminal_roll", 0.0)) * e_phi_T**2

    return float(total)


def get_warm_start_sequence(cfg, last_u: np.ndarray) -> np.ndarray:
    """
    Shift the previous best sequence forward by one step.

    Previous best: [u0, u1, u2, ...]
    New warm start: [u1, u2, ..., u_last]
    """
    H = int(getattr(cfg, "mpc_horizon", 5))
    last_u = np.asarray(last_u, dtype=float)

    prev_best = getattr(cfg, "_mpc_prev_best_sequence", None)
    if prev_best is None:
        return np.tile(last_u, (H, 1))

    prev_best = np.asarray(prev_best, dtype=float)
    if prev_best.shape != (H, 2):
        return np.tile(last_u, (H, 1))

    shifted = np.zeros_like(prev_best)
    shifted[:-1] = prev_best[1:]
    shifted[-1] = prev_best[-1]
    return shifted


def sample_sequences(cfg, rng, last_u: np.ndarray) -> list[np.ndarray]:
    """Create warm-started candidate control sequences for random-shooting MPC."""
    H = int(getattr(cfg, "mpc_horizon", 5))
    N = int(getattr(cfg, "mpc_num_candidates", 60))

    dS_max = float(getattr(cfg, "deltaS_max_deg", 20.0))
    dD_max = float(getattr(cfg, "deltaD_max_deg", 10.0))
    u_min = np.array([-dS_max, -dD_max], dtype=float)
    u_max = np.array([ dS_max,  dD_max], dtype=float)

    sequences: list[np.ndarray] = []
    warm = np.clip(get_warm_start_sequence(cfg, last_u), u_min, u_max)

    # Always include deterministic candidates first.
    sequences.append(warm.copy())
    sequences.append(np.tile(last_u, (H, 1)))
    trim_u = np.array([float(getattr(cfg, "delta_s_deg", 0.0)), float(getattr(cfg, "delta_d_deg", 0.0))])
    sequences.append(np.tile(trim_u, (H, 1)))

    # Most candidates are local perturbations around the warm-start sequence.
    # This is the main anti-jitter change: the optimizer searches near its last solution
    # instead of rediscovering a new random solution every control update.
    local_sigma = np.asarray(getattr(cfg, "mpc_local_sigma_deg", [2.0, 2.0]), dtype=float)
    warm_sigma = np.asarray(getattr(cfg, "mpc_warm_sigma_deg", [0.8, 0.8]), dtype=float)

    n_warm = max(1, int(float(getattr(cfg, "mpc_warm_start_fraction", 0.75)) * N))
    n_local = max(1, int(0.15 * N))
    n_global = max(1, N - n_warm - n_local)

    for _ in range(n_warm):
        seq = warm + rng.normal(0.0, warm_sigma, size=(H, 2))
        for k in range(1, H):
            seq[k] = 0.75 * seq[k - 1] + 0.25 * seq[k]
        sequences.append(np.clip(seq, u_min, u_max))

    # A few local samples around the last actual command.
    for _ in range(n_local):
        seq = last_u + rng.normal(0.0, local_sigma, size=(H, 2))
        for k in range(1, H):
            seq[k] = 0.65 * seq[k - 1] + 0.35 * seq[k]
        sequences.append(np.clip(seq, u_min, u_max))

    # A few global samples remain to escape local minima.
    for _ in range(n_global):
        seq = rng.uniform(u_min, u_max, size=(H, 2))
        for k in range(1, H):
            seq[k] = 0.50 * seq[k - 1] + 0.50 * seq[k]
        sequences.append(np.clip(seq, u_min, u_max))

    return sequences


# =============================================================================
# Public controller entry point
# =============================================================================

def controller(t, state, cfg):
    """MPC controller called by dynamicsMPC_dashboard.simulate(t, state, cfg)."""
    fixed_throttle = float(np.clip(getattr(cfg, "throttle0", 1.0), 0.0, 1.0))

    if not hasattr(cfg, "_mpc_aero_model"):
        raise RuntimeError(
            "MPC controller needs cfg._mpc_aero_model. Set cfg._mpc_aero_model = aero_model in run_sim_MPC.py."
        )

    # Only recompute every mpc_control_period_s. Between updates, hold the previous command.
    dt = float(getattr(cfg, "dt", 0.01))
    control_period_s = float(getattr(cfg, "mpc_control_period_s", max(dt, 0.03)))
    last_update_t = float(getattr(cfg, "_mpc_last_update_t", -1.0e9))
    last_u = np.asarray(getattr(cfg, "_mpc_prev_u", [cfg.delta_s_deg, cfg.delta_d_deg]), dtype=float)

    if (float(t) - last_update_t) < (control_period_s - 0.5 * dt):
        deltaS_cmd, deltaD_cmd = [float(x) for x in last_u]
        _write_diagnostics(
            t, state, cfg, deltaS_cmd, deltaD_cmd, np.nan,
            reused=True, raw_deltaS_cmd=deltaS_cmd, raw_deltaD_cmd=deltaD_cmd
        )
        return {"deltaS_deg": deltaS_cmd, "deltaD_deg": deltaD_cmd, "throttle": fixed_throttle}

    rng = getattr(cfg, "_mpc_rng", None)
    if rng is None:
        rng = np.random.default_rng(int(getattr(cfg, "mpc_seed", 4)))
        cfg._mpc_rng = rng

    veh, env = make_vehicle_and_env(cfg)
    aero_model = cfg._mpc_aero_model

    # Baseline candidate: hold the current command. This gives the hold-rule a real cost reference.
    H = int(getattr(cfg, "mpc_horizon", 4))
    hold_seq = np.tile(last_u, (H, 1))
    hold_cost = rollout_sequence_cost(state, hold_seq, cfg, veh, env, aero_model)

    best_cost = hold_cost
    best_seq = hold_seq.copy()

    for seq in sample_sequences(cfg, rng, last_u):
        c = rollout_sequence_cost(state, seq, cfg, veh, env, aero_model)
        if c < best_cost:
            best_cost = c
            best_seq = seq

    # Save the best full sequence before any output deadband/rate limit.
    cfg._mpc_prev_best_sequence = np.asarray(best_seq, dtype=float).copy()

    deltaS_cmd, deltaD_cmd = [float(x) for x in best_seq[0]]

    # Final safety: actuator rate limits at the real MPC update rate.
    dS_rate = float(getattr(cfg, "deltaS_rate_max_deg_s", 120.0))
    dD_rate = float(getattr(cfg, "deltaD_rate_max_deg_s", 120.0))
    deltaS_rate_limited = rate_limit_cmd(deltaS_cmd, last_u[0], dS_rate, control_period_s)
    deltaD_rate_limited = rate_limit_cmd(deltaD_cmd, last_u[1], dD_rate, control_period_s)
    raw_u = np.array([deltaS_rate_limited, deltaD_rate_limited], dtype=float)

    # Near steady state, do not update for tiny command changes or tiny cost improvements.
    steady_now, steady_info = steady_state_check(state, cfg)
    cmd_change = float(np.linalg.norm(raw_u - last_u))
    min_cmd_change = float(getattr(cfg, "mpc_hold_cmd_change_deadband_deg", 0.20))
    min_improvement_frac = float(getattr(cfg, "mpc_hold_min_improvement_frac", 0.02))
    improvement_frac = float((hold_cost - best_cost) / max(abs(hold_cost), 1.0e-9))

    yaw_now, _, _ = euler_ypr_deg(state)
    psi_cmd_now = float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", 0.0)))
    heading_err_abs_now = abs(wrap_deg(psi_cmd_now - yaw_now))
    hold_heading_gate = heading_err_abs_now <= float(getattr(cfg, "mpc_hold_max_heading_err_deg", 0.75))

    hold_due_to_small_change = bool(steady_now and hold_heading_gate and cmd_change < min_cmd_change)
    hold_due_to_low_improvement = bool(steady_now and hold_heading_gate and improvement_frac < min_improvement_frac)

    if bool(getattr(cfg, "mpc_use_steady_hold_rule", True)) and (hold_due_to_small_change or hold_due_to_low_improvement):
        filtered_u = last_u.copy()
        hold_rule_active = 1
    else:
        filtered_u = raw_u.copy()
        hold_rule_active = 0

    # Optional output filter is intentionally off by default in this version.
    # Keeping this block available lets you enable a very weak filter later, but the
    # recommended anti-jitter mechanisms are warm-start + cost deadband + hold rule.
    filter_enabled = bool(getattr(cfg, "mpc_use_output_filter", False))
    alpha = 1.0
    tau = 0.0
    if filter_enabled and steady_now:
        prev_filt_u = np.asarray(getattr(cfg, "_mpc_prev_filtered_u", last_u), dtype=float)
        tau = float(getattr(cfg, "mpc_output_filter_tau_steady_s", 0.08))
        dS_filtered, alpha = lowpass_cmd(filtered_u[0], prev_filt_u[0], tau, control_period_s)
        dD_filtered, _ = lowpass_cmd(filtered_u[1], prev_filt_u[1], tau, control_period_s)
        filtered_u = np.array([dS_filtered, dD_filtered], dtype=float)

    # Keep commands inside actuator limits after all anti-jitter logic.
    dS_max = float(getattr(cfg, "deltaS_max_deg", 20.0))
    dD_max = float(getattr(cfg, "deltaD_max_deg", 10.0))
    filtered_u[0] = clamp(filtered_u[0], -dS_max, dS_max)
    filtered_u[1] = clamp(filtered_u[1], -dD_max, dD_max)

    deltaS_cmd, deltaD_cmd = [float(x) for x in filtered_u]

    cfg._mpc_prev_u = np.array([deltaS_cmd, deltaD_cmd], dtype=float)
    cfg._mpc_prev_filtered_u = np.array([deltaS_cmd, deltaD_cmd], dtype=float)
    cfg._mpc_last_raw_u = raw_u.copy()
    cfg._mpc_filter_active = int(bool(filter_enabled))
    cfg._mpc_filter_steady = int(bool(steady_now))
    cfg._mpc_filter_alpha = float(alpha)
    cfg._mpc_filter_tau_s = float(tau)
    cfg._mpc_hold_rule_active = int(hold_rule_active)
    cfg._mpc_cmd_change_norm_deg = float(cmd_change)
    cfg._mpc_improvement_frac = float(improvement_frac)
    cfg._mpc_hold_cost = float(hold_cost)
    cfg._mpc_hold_heading_gate = int(bool(hold_heading_gate))
    cfg._mpc_heading_err_abs_for_hold_deg = float(heading_err_abs_now)
    cfg._mpc_last_update_t = float(t)

    _write_diagnostics(
        t, state, cfg, deltaS_cmd, deltaD_cmd, best_cost, reused=False,
        raw_deltaS_cmd=float(raw_u[0]), raw_deltaD_cmd=float(raw_u[1])
    )

    return {"deltaS_deg": deltaS_cmd, "deltaD_deg": deltaD_cmd, "throttle": fixed_throttle}


def _write_diagnostics(t, state, cfg, deltaS_cmd, deltaD_cmd, best_cost, reused=False, raw_deltaS_cmd=None, raw_deltaD_cmd=None):
    """Write diagnostics both to cfg fields and cfg._mpc_log for run_sim_MPC plots."""
    yaw_deg, pitch_deg, roll_deg = euler_ypr_deg(state)
    p, q, r = [float(x) for x in state.omega_body]
    p_deg_s, q_deg_s, r_deg_s = np.rad2deg([p, q, r])
    V, alpha_deg, beta_deg = airdata_from_state(state, cfg)

    psi_cmd, theta_cmd, roll_cmd = get_command_targets(state, cfg)

    e_psi = wrap_deg(psi_cmd - yaw_deg)
    e_theta = float(theta_cmd - pitch_deg)
    e_phi = wrap_deg(roll_cmd - roll_deg)

    # Fill legacy ctrl_* fields that dynamicsNew already logs.
    cfg._ctrl_e_psi_deg = float(e_psi)
    cfg._ctrl_e_theta_deg = float(e_theta)
    cfg._ctrl_e_phi_deg = float(e_phi)
    cfg._ctrl_phi_cmd_deg = float(roll_cmd)
    cfg._mpc_heading_to_roll_deadband_deg = float(getattr(cfg, "mpc_heading_to_roll_deadband_deg", 1.0))
    cfg._ctrl_q_cmd_deg_s = 0.0
    cfg._ctrl_deltaS_unsmoothed_deg = float(deltaS_cmd)
    cfg._ctrl_deltaD_unsmoothed_deg = float(deltaD_cmd)
    cfg._ctrl_deltaS_final_deg = float(deltaS_cmd)
    cfg._ctrl_deltaD_final_deg = float(deltaD_cmd)
    cfg._ctrl_V = float(V)
    cfg._ctrl_alpha_deg = float(alpha_deg)
    cfg._ctrl_beta_deg = float(beta_deg)

    # No scheduled gains in MPC; set to NaN rather than pretending.
    cfg._ctrl_K_PSI = np.nan
    cfg._ctrl_K_PHI = np.nan
    cfg._ctrl_K_P = np.nan
    cfg._ctrl_K_R = np.nan
    cfg._ctrl_K_THETA = np.nan
    cfg._ctrl_K_Q = np.nan
    cfg._ctrl_K_QD = np.nan

    if raw_deltaS_cmd is None:
        raw_deltaS_cmd = deltaS_cmd
    if raw_deltaD_cmd is None:
        raw_deltaD_cmd = deltaD_cmd

    if not hasattr(cfg, "_mpc_log"):
        cfg._mpc_log = []

    cfg._mpc_log.append({
        "t": float(t),
        "mpc_best_cost": float(best_cost) if np.isfinite(best_cost) else np.nan,
        "mpc_reused": int(bool(reused)),
        "mpc_deltaS_cmd_deg": float(deltaS_cmd),
        "mpc_deltaD_cmd_deg": float(deltaD_cmd),
        "mpc_deltaS_raw_cmd_deg": float(raw_deltaS_cmd),
        "mpc_deltaD_raw_cmd_deg": float(raw_deltaD_cmd),
        "mpc_filter_active": float(getattr(cfg, "_mpc_filter_active", np.nan)),
        "mpc_filter_steady": float(getattr(cfg, "_mpc_filter_steady", np.nan)),
        "mpc_filter_alpha": float(getattr(cfg, "_mpc_filter_alpha", np.nan)),
        "mpc_filter_tau_s": float(getattr(cfg, "_mpc_filter_tau_s", np.nan)),
        "mpc_hold_rule_active": float(getattr(cfg, "_mpc_hold_rule_active", np.nan)),
        "mpc_cmd_change_norm_deg": float(getattr(cfg, "_mpc_cmd_change_norm_deg", np.nan)),
        "mpc_improvement_frac": float(getattr(cfg, "_mpc_improvement_frac", np.nan)),
        "mpc_hold_cost": float(getattr(cfg, "_mpc_hold_cost", np.nan)),
        "mpc_hold_heading_gate": float(getattr(cfg, "_mpc_hold_heading_gate", np.nan)),
        "mpc_heading_err_abs_for_hold_deg": float(getattr(cfg, "_mpc_heading_err_abs_for_hold_deg", np.nan)),
        "mpc_heading_err_deg": float(e_psi),
        "mpc_pitch_err_deg": float(e_theta),
        "mpc_roll_err_deg": float(e_phi),
        "mpc_p_deg_s": float(p_deg_s),
        "mpc_q_deg_s": float(q_deg_s),
        "mpc_r_deg_s": float(r_deg_s),
        "mpc_V": float(V),
        "mpc_alpha_deg": float(alpha_deg),
        "mpc_beta_deg": float(beta_deg),
        "mpc_yaw_cmd_deg": float(psi_cmd),
        "mpc_pitch_cmd_deg": float(theta_cmd),
        "mpc_roll_cmd_deg": float(roll_cmd),
        "mpc_heading_to_roll_blend": float(getattr(cfg, "_mpc_heading_to_roll_blend", np.nan)),
        "mpc_e_psi_for_roll_deg": float(getattr(cfg, "_mpc_e_psi_for_roll_deg", np.nan)),
        "mpc_deadband_inside_scale": float(getattr(cfg, "mpc_deadband_inside_scale", np.nan)),
    })
