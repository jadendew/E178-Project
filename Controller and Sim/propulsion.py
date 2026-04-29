# propulsion.py
"""
Simple propulsion/thrust model(s) for the 6DOF sim.

Current model:
- Throttle in [0,1] maps linearly to thrust magnitude:
    T = throttle * THRUST_MAX_N
- Force direction and application point are defined in BODY coordinates.
- Produces body-frame force and moment about the vehicle COM.

BODY axes convention (consistent with your sim):
- +X forward
- +Y right
- +Z down
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


@dataclass
class Thruster:
    """A single thruster definition in BODY coordinates."""
    enabled: bool
    thrust_max_n: float          # N at throttle=1
    dir_body: np.ndarray         # direction in body frame (will be normalized)
    pos_body: np.ndarray         # application point in body coords (m)


@dataclass
class PropulsionConfig:
    """Propulsion system configuration."""
    enabled: bool
    thrusters: List[Thruster]


def build_simple_single_thruster(
    enabled: bool,
    thrust_max_n: float,
    thrust_dir_body: Tuple[float, float, float],
    thrust_pos_body: Tuple[float, float, float],
) -> PropulsionConfig:
    """Convenience builder for a single-thruster setup."""
    thr = Thruster(
        enabled=bool(enabled),
        thrust_max_n=float(thrust_max_n),
        dir_body=np.array(thrust_dir_body, dtype=float),
        pos_body=np.array(thrust_pos_body, dtype=float),
    )
    return PropulsionConfig(enabled=bool(enabled), thrusters=[thr])


def thrust_force_moment_body(
    prop: PropulsionConfig,
    throttle: float,
    com_body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute total thrust force and moment about COM in BODY frame.

    Returns:
      F_thr_b (3,), M_thr_b (3,), T_total (scalar N)
    """
    if (prop is None) or (not prop.enabled):
        return np.zeros(3), np.zeros(3), 0.0

    thr = float(np.clip(throttle, 0.0, 1.0))
    F_sum = np.zeros(3, dtype=float)
    M_sum = np.zeros(3, dtype=float)
    T_total = 0.0

    for t in prop.thrusters:
        if not t.enabled:
            continue

        d = _unit(np.array(t.dir_body, dtype=float))
        T = thr * float(t.thrust_max_n)
        F = T * d

        r = np.array(t.pos_body, dtype=float) - np.array(com_body, dtype=float)
        M = np.cross(r, F)

        F_sum += F
        M_sum += M
        T_total += T

    return F_sum, M_sum, float(T_total)
