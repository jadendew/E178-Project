# dynamicsMPC_dashboard.py
"""
6DOF rigid-body dynamics integrator for NED world frame.

MPC-focused version of dynamicsNew.py.

Changes from dynamicsNew.py:
- Keeps the same physics, dataclasses, controller interface, and RK4 integration.
- Removes old cascaded/gain-scheduling diagnostic columns from the output log.
- Adds MPC-specific diagnostic columns pulled from cfg._mpc_* and cfg._ctrl_* fields.
- Keeps dict controller outputs compatible: deltaS_deg, deltaD_deg, throttle.

Original base behavior:

WORLD: NED (X forward, Y right, Z down)
BODY:  X forward, Y right, Z down

Adds:
- Thrust force/moment term (from propulsion.py)
- Controller-driven inputs (deltaS, deltaD, throttle)
- Logs both aero and thrust contributions

Speed fix implemented:
- NO extra RK4 call for logging (previously dt=0 pass + real pass).
  We now do ONE RK4 step per timestep, and log using the k1 (current-state) outputs
  returned by step_rk4().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Local modules
import propulsion  # propulsion.py
import aero        # aero.py (must provide AeroModel + aero_forces_moments_body)


# =============================================================================
# Dataclasses
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
    quat_body_to_world: np.ndarray
    omega_body: np.ndarray


@dataclass
class Control:
    deltaS_deg: float
    deltaD_deg: float
    throttle: float  # 0..1


@dataclass
class SimConfig:
    # Environment
    rho: float
    g: float
    wind_world: np.ndarray

    # Vehicle
    mass: float
    I_body: np.ndarray
    com_body: np.ndarray
    aero_ref_body: np.ndarray
    S_ref: float
    b_ref: float
    c_ref: float

    # Simulation settings
    dt: float
    t_final: float
    max_speed: float
    max_omega: float

    # Initial condition
    pos0_world: np.ndarray
    vel0_world: np.ndarray
    yaw0_deg: float
    pitch0_deg: float
    roll0_deg: float
    omega0_body: np.ndarray

    # Controls (defaults for open-loop controllers to read)
    delta_s_deg: float
    delta_d_deg: float
    throttle0: float

    # Output
    output_csv: str

    # Propulsion
    propulsion: propulsion.PropulsionConfig


# =============================================================================
# Helpers
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
    n = float(np.linalg.norm(q_bw))
    if n < 1e-12:
        raise ValueError("Quaternion norm collapsed to ~0.")
    return R.from_quat(q_bw / n).as_matrix()


def euler_to_quat_ypr(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    return R.from_euler("ZYX", [yaw_rad, pitch_rad, roll_rad], degrees=False).as_quat()


def _normalize_controller_output(u: Any, cfg: SimConfig) -> Control:
    """
    Accepts:
      - Control dataclass
      - (deltaS, deltaD)
      - (deltaS, deltaD, throttle)
      - dict with keys: deltaS_deg, deltaD_deg, throttle
    """
    if isinstance(u, Control):
        dS, dD, thr = u.deltaS_deg, u.deltaD_deg, u.throttle
    elif isinstance(u, dict):
        dS = float(u.get("deltaS_deg", cfg.delta_s_deg))
        dD = float(u.get("deltaD_deg", cfg.delta_d_deg))
        thr = float(u.get("throttle", cfg.throttle0))
    elif isinstance(u, (tuple, list)) and len(u) == 2:
        dS, dD = float(u[0]), float(u[1])
        thr = float(cfg.throttle0)
    elif isinstance(u, (tuple, list)) and len(u) >= 3:
        dS, dD, thr = float(u[0]), float(u[1]), float(u[2])
    else:
        dS, dD, thr = float(cfg.delta_s_deg), float(cfg.delta_d_deg), float(cfg.throttle0)

    thr = float(np.clip(thr, 0.0, 1.0))
    return Control(deltaS_deg=float(dS), deltaD_deg=float(dD), throttle=thr)


# =============================================================================
# Integrator
# =============================================================================

def step_rk4(
    state: State,
    veh: Vehicle,
    env: Environment,
    cfg: SimConfig,
    aero_model: Any,
    ctrl: Control,
    dt: float
) -> Tuple[State, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    RK4 step. Returns:
      next_state,
      derived (V, alpha, beta, etc)   <-- from k1 (current state)
      coeffs (CL,CD,...)             <-- from k1
      extras (thrust + totals)       <-- from k1

    NOTE: This returns k1 data (current state), which is exactly what you want to log.
    """

    def deriv(s: State):
        # --- AERO (single signature) ---
        F_aero_b, M_aero_b, coeffs, derived = aero.aero_forces_moments_body(
            aero_model, veh, env, s, ctrl.deltaS_deg, ctrl.deltaD_deg
        )

        # --- THRUST ---
        F_thr_b, M_thr_b, T_total = propulsion.thrust_force_moment_body(
            cfg.propulsion, ctrl.throttle, veh.com_body
        )

        # Total body forces/moments (excluding gravity, applied in world)
        F_tot_b = F_aero_b + F_thr_b
        M_tot_b = M_aero_b + M_thr_b

        # Convert body force -> world
        C_bw = dcm_body_to_world(s.quat_body_to_world)
        F_tot_w = C_bw @ F_tot_b

        # Gravity in NED is +g in +Z (down)
        F_grav_w = np.array([0.0, 0.0, veh.mass * env.g], dtype=float)

        a_w = (F_tot_w + F_grav_w) / veh.mass

        # Angular accel in body
        I = veh.I_body
        w = s.omega_body
        H = I @ w
        w_dot = np.linalg.solve(I, (M_tot_b - np.cross(w, H)))

        # Quaternion kinematics
        q = s.quat_body_to_world
        q_dot = 0.5 * omega_matrix(w) @ q

        extras = {
            "T_N": float(T_total),
            "Ftx_b": float(F_thr_b[0]), "Fty_b": float(F_thr_b[1]), "Ftz_b": float(F_thr_b[2]),
            "Lthr_b": float(M_thr_b[0]), "Mthr_b": float(M_thr_b[1]), "Nthr_b": float(M_thr_b[2]),
            "Ftotx_b": float(F_tot_b[0]), "Ftoty_b": float(F_tot_b[1]), "Ftotz_b": float(F_tot_b[2]),
            "Ltot_b": float(M_tot_b[0]), "Mtot_b": float(M_tot_b[1]), "Ntot_b": float(M_tot_b[2]),
        }

        return a_w, w_dot, q_dot, coeffs, derived, extras

    # k1 at current state (these are what we'll log)
    a1, w1, q1, c1, d1, ex1 = deriv(state)

    def mk(base: State, dp, dv, dq, dw):
        qn = base.quat_body_to_world + dq
        return State(
            pos_world=base.pos_world + dp,
            vel_world=base.vel_world + dv,
            quat_body_to_world=quat_normalize(qn),
            omega_body=base.omega_body + dw
        )

    # k2
    s2 = mk(state, dp=0.5*dt*state.vel_world, dv=0.5*dt*a1, dq=0.5*dt*q1, dw=0.5*dt*w1)
    a2, w2, q2, _, _, _ = deriv(s2)

    # k3
    s3 = mk(state, dp=0.5*dt*(state.vel_world + 0.5*dt*a1), dv=0.5*dt*a2, dq=0.5*dt*q2, dw=0.5*dt*w2)
    a3, w3, q3, _, _, _ = deriv(s3)

    # k4
    s4 = mk(state, dp=dt*(state.vel_world + 0.5*dt*a2), dv=dt*a3, dq=dt*q3, dw=dt*w3)
    a4, w4, q4, _, _, _ = deriv(s4)

    # Integrate
    pos = state.pos_world + (dt/6.0)*(
        state.vel_world
        + 2*(state.vel_world + 0.5*dt*a1)
        + 2*(state.vel_world + 0.5*dt*a2)
        + (state.vel_world + dt*a3)
    )
    vel = state.vel_world + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)
    omg = state.omega_body + (dt/6.0)*(w1 + 2*w2 + 2*w3 + w4)
    quat = quat_normalize(state.quat_body_to_world + (dt/6.0)*(q1 + 2*q2 + 2*q3 + q4))

    # Safety clamps
    V = safe_norm(vel)
    if V > cfg.max_speed:
        vel = vel * (cfg.max_speed / V)
    wmag = safe_norm(omg)
    if wmag > cfg.max_omega:
        omg = omg * (cfg.max_omega / wmag)

    next_state = State(pos_world=pos, vel_world=vel, quat_body_to_world=quat, omega_body=omg)

    # Return k1 info for logging (current-state forces/coeffs/etc)
    return next_state, d1, c1, ex1


# =============================================================================
# Terminal progress dashboard helpers
# =============================================================================

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress_dashboard(
    *,
    i: int,
    n_steps: int,
    t_sim: float,
    t_final: float,
    start_wall: float,
    bar_width: int = 32,
    best_cost: float | None = None,
    control_update: float | None = None,
    force_newline: bool = False,
) -> None:
    """
    Single-line terminal dashboard with progress, wall time, ETA, speed, and MPC status.

    This intentionally avoids external packages like tqdm so the file stays drop-in.
    """
    completed = min(max(i, 0), max(n_steps, 1))
    frac = completed / max(n_steps, 1)
    filled = int(round(bar_width * frac))
    bar = "#" * filled + "-" * (bar_width - filled)

    wall_elapsed = max(time.perf_counter() - start_wall, 1e-12)
    steps_per_s = completed / wall_elapsed if completed > 0 else 0.0
    eta_s = (n_steps - completed) / steps_per_s if steps_per_s > 1e-12 else float("nan")

    if best_cost is None or not np.isfinite(best_cost):
        cost_str = "cost=--"
    else:
        cost_str = f"cost={best_cost:.3g}"

    if control_update is None or not np.isfinite(control_update):
        update_str = "mpc=--"
    else:
        update_str = "mpc=solve" if float(control_update) > 0.5 else "mpc=hold"

    msg = (
        f"\r[{bar}] {100.0 * frac:6.2f}% | "
        f"sim {t_sim:7.3f}/{t_final:.3f}s | "
        f"wall {_format_duration(wall_elapsed)} | "
        f"ETA {_format_duration(eta_s)} | "
        f"{steps_per_s:6.1f} steps/s | "
        f"{update_str} | {cost_str}"
    )

    # Pad a little so shorter later strings clear old characters.
    sys.stdout.write(msg + " " * 8)
    if force_newline:
        sys.stdout.write("\n")
    sys.stdout.flush()


# =============================================================================
# Top-level simulation
# =============================================================================

def simulate(
    cfg: SimConfig,
    aero_model: Any,
    controller_fn: Callable[[float, State, SimConfig], Any],
) -> pd.DataFrame:
    """
    Runs the simulation and returns a dataframe log.

    Speed fix:
      - ONE rk4 call per step; log from the returned k1-derived outputs.
    """
    veh = Vehicle(
        mass=cfg.mass,
        I_body=cfg.I_body,
        com_body=cfg.com_body,
        aero_ref_body=cfg.aero_ref_body,
        S_ref=cfg.S_ref,
        b_ref=cfg.b_ref,
        c_ref=cfg.c_ref,
    )

    env = Environment(
        rho=cfg.rho,
        g=cfg.g,
        wind_world=cfg.wind_world,
    )

    q0 = euler_to_quat_ypr(
        yaw_rad=np.deg2rad(cfg.yaw0_deg),
        pitch_rad=np.deg2rad(cfg.pitch0_deg),
        roll_rad=np.deg2rad(cfg.roll0_deg),
    )

    state = State(
        pos_world=np.array(cfg.pos0_world, dtype=float),
        vel_world=np.array(cfg.vel0_world, dtype=float),
        quat_body_to_world=quat_normalize(q0),
        omega_body=np.array(cfg.omega0_body, dtype=float),
    )

    dt = float(cfg.dt)
    n_steps = int(np.ceil(cfg.t_final / dt))
    t = 0.0

    rows = []

    show_dashboard = bool(getattr(cfg, "show_progress_dashboard", True))
    dashboard_update_every_s = float(getattr(cfg, "dashboard_update_every_s", 0.25))
    dashboard_bar_width = int(getattr(cfg, "dashboard_bar_width", 32))
    dashboard_start_wall = time.perf_counter()
    dashboard_last_wall = 0.0

    if show_dashboard:
        print(
            f"Starting MPC sim: t_final={cfg.t_final:.3f}s, "
            f"dt={dt:.6f}s, steps={n_steps}, "
            f"horizon={getattr(cfg, 'mpc_horizon', 'NA')}, "
            f"candidates={getattr(cfg, 'mpc_num_candidates', 'NA')}"
        )
        _print_progress_dashboard(
            i=0,
            n_steps=n_steps,
            t_sim=0.0,
            t_final=float(cfg.t_final),
            start_wall=dashboard_start_wall,
            bar_width=dashboard_bar_width,
            best_cost=None,
            control_update=None,
        )

    # We will log n_steps+1 rows: at t=0, dt, ..., n_steps*dt (approx >= t_final)
    for i in range(n_steps + 1):
        # Controller -> Control
        u_raw = controller_fn(t, state, cfg)
        ctrl = _normalize_controller_output(u_raw, cfg)

        # If the MPC controller keeps a list-based diagnostic log, mirror the latest
        # entry onto cfg attributes so each dynamics row can log it directly.
        if hasattr(cfg, "_mpc_log") and len(getattr(cfg, "_mpc_log")) > 0:
            latest_mpc = cfg._mpc_log[-1]
            for _k, _v in latest_mpc.items():
                if _k != "t":
                    setattr(cfg, f"_{_k}", _v)
            # Convenience flag: 1 when the MPC solved a new optimization,
            # 0 when it reused the previous command between MPC update periods.
            if "mpc_reused" in latest_mpc:
                cfg._mpc_control_update = 0.0 if bool(latest_mpc["mpc_reused"]) else 1.0

        # Euler for logging
        eul = R.from_quat(state.quat_body_to_world).as_euler("ZYX", degrees=True)  # yaw,pitch,roll

        # Step (or just evaluate one last time at final row without advancing)
        if i < n_steps:
            next_state, derived, coeffs, extras = step_rk4(state, veh, env, cfg, aero_model, ctrl, dt)
        else:
            # Final row: do a *single* step_rk4 with dt=0 would reintroduce the RK4 cost.
            # Instead: do a tiny dt (1e-12) to get derived/coeffs/extras; state advance is negligible.
            # (Still RK4, but only once total at the very end.)
            next_state, derived, coeffs, extras = step_rk4(state, veh, env, cfg, aero_model, ctrl, 1e-12)

        # Pull what we can from derived
        V = float(derived.get("V", np.linalg.norm(state.vel_world)))
        alpha_deg = float(derived.get("alpha_deg", np.nan))
        beta_deg = float(derived.get("beta_deg", np.nan))
        qbar = float(derived.get("qbar", np.nan))

        rows.append({
            "t": t,
            "x": state.pos_world[0], "y": state.pos_world[1], "z": state.pos_world[2],
            "vx": state.vel_world[0], "vy": state.vel_world[1], "vz": state.vel_world[2],

            "V": V, "alpha_deg": alpha_deg, "beta_deg": beta_deg, "qbar": qbar,

            "yaw_deg": eul[0], "pitch_deg": eul[1], "roll_deg": eul[2],
            "p": state.omega_body[0], "q": state.omega_body[1], "r": state.omega_body[2],

            "deltaS_deg": ctrl.deltaS_deg,
            "deltaD_deg": ctrl.deltaD_deg,
            "throttle": ctrl.throttle,

            "CL": float(coeffs.get("CL", np.nan)),
            "CD": float(coeffs.get("CD", np.nan)),
            "CY": float(coeffs.get("CY", np.nan)),
            "Cl": float(coeffs.get("Cl", np.nan)),
            "Cm": float(coeffs.get("Cm", np.nan)),
            "Cn": float(coeffs.get("Cn", np.nan)),

            "T_N": extras["T_N"],
            "Ftx_b": extras["Ftx_b"], "Fty_b": extras["Fty_b"], "Ftz_b": extras["Ftz_b"],
            "Lthr_b": extras["Lthr_b"], "Mthr_b": extras["Mthr_b"], "Nthr_b": extras["Nthr_b"],
            "Ftotx_b": extras["Ftotx_b"], "Ftoty_b": extras["Ftoty_b"], "Ftotz_b": extras["Ftotz_b"],
            "Ltot_b": extras["Ltot_b"], "Mtot_b": extras["Mtot_b"], "Ntot_b": extras["Ntot_b"],

            # ============================================================
            # MPC diagnostics
            # These are written by controllers/mlp_fixed_thrust_mpc.py.
            # Unlike dynamicsNew.py, this file does not log cascaded/gain-scheduling
            # internals because the MPC controller does not use those loops.
            # ============================================================

            # Commands used by the MPC
            "mpc_yaw_cmd_deg": float(getattr(cfg, "heading_cmd_deg", getattr(cfg, "yaw0_deg", np.nan))),
            "mpc_pitch_cmd_deg": float(getattr(cfg, "pitch_cmd_deg", getattr(cfg, "pitch0_deg", np.nan))),
            "mpc_roll_cmd_deg": float(getattr(cfg, "mpc_roll_cmd_deg", 0.0)),
            "mpc_alpha_cmd_deg": float(getattr(cfg, "mpc_alpha_cmd_deg", np.nan)),
            "mpc_beta_cmd_deg": float(getattr(cfg, "mpc_beta_cmd_deg", 0.0)),

            # MPC tracking errors and airdata from the controller's current-state view
            "mpc_heading_err_deg": float(getattr(cfg, "_ctrl_e_psi_deg", getattr(cfg, "_mpc_heading_err_deg", np.nan))),
            "mpc_pitch_err_deg": float(getattr(cfg, "_ctrl_e_theta_deg", getattr(cfg, "_mpc_pitch_err_deg", np.nan))),
            "mpc_roll_err_deg": float(getattr(cfg, "_ctrl_e_phi_deg", getattr(cfg, "_mpc_roll_err_deg", np.nan))),
            "mpc_V_ctrl": float(getattr(cfg, "_ctrl_V", getattr(cfg, "_mpc_V", np.nan))),
            "mpc_alpha_ctrl_deg": float(getattr(cfg, "_ctrl_alpha_deg", getattr(cfg, "_mpc_alpha_deg", np.nan))),
            "mpc_beta_ctrl_deg": float(getattr(cfg, "_ctrl_beta_deg", getattr(cfg, "_mpc_beta_deg", np.nan))),

            # MPC optimizer outputs
            "mpc_best_cost": float(getattr(cfg, "_mpc_best_cost", np.nan)),
            "mpc_reused": float(getattr(cfg, "_mpc_reused", np.nan)),
            "mpc_deltaS_cmd_deg": float(getattr(cfg, "_mpc_deltaS_cmd_deg", ctrl.deltaS_deg)),
            "mpc_deltaD_cmd_deg": float(getattr(cfg, "_mpc_deltaD_cmd_deg", ctrl.deltaD_deg)),
            "mpc_control_update": float(getattr(cfg, "_mpc_control_update", np.nan)),

            # MPC settings copied into the CSV for reproducibility
            "mpc_horizon": float(getattr(cfg, "mpc_horizon", np.nan)),
            "mpc_num_candidates": float(getattr(cfg, "mpc_num_candidates", np.nan)),
            "mpc_pred_dt": float(getattr(cfg, "mpc_pred_dt", np.nan)),
            "mpc_control_period_s": float(getattr(cfg, "mpc_control_period_s", np.nan)),
            "mpc_Q_heading": float(getattr(cfg, "mpc_Q_heading", np.nan)),
            "mpc_Q_pitch": float(getattr(cfg, "mpc_Q_pitch", np.nan)),
            "mpc_Q_roll": float(getattr(cfg, "mpc_Q_roll", np.nan)),
            "mpc_Q_alpha": float(getattr(cfg, "mpc_Q_alpha", np.nan)),
            "mpc_Q_beta": float(getattr(cfg, "mpc_Q_beta", np.nan)),
        })

        if show_dashboard:
            now_wall = time.perf_counter()
            is_final = i >= n_steps
            if is_final or (now_wall - dashboard_last_wall) >= dashboard_update_every_s:
                dashboard_last_wall = now_wall
                _print_progress_dashboard(
                    i=i,
                    n_steps=n_steps,
                    t_sim=t,
                    t_final=float(cfg.t_final),
                    start_wall=dashboard_start_wall,
                    bar_width=dashboard_bar_width,
                    best_cost=float(getattr(cfg, "_mpc_best_cost", np.nan)),
                    control_update=float(getattr(cfg, "_mpc_control_update", np.nan)),
                    force_newline=is_final,
                )

        # Advance
        if i < n_steps:
            state = next_state
            t += dt

            if not (np.isfinite(state.pos_world).all() and np.isfinite(state.vel_world).all()
                    and np.isfinite(state.quat_body_to_world).all() and np.isfinite(state.omega_body).all()):
                print(f"STOP: non-finite state at t={t:.3f}")
                break

    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_csv, index=False)
    print(f"Wrote {cfg.output_csv} with {len(out)} rows.")
    return out
