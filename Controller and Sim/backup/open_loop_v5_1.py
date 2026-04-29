"""
Open-loop controller: returns constant (deltaS, deltaD) or scheduled commands.

Expected cfg keys (from run_sim.py CFG dict):
  - DELTA_S_DEG
  - DELTA_D_DEG
"""

from __future__ import annotations


def controller(t: float, state: dict, cfg: dict) -> tuple[float, float]:
    """
    Parameters
    ----------
    t : float
        Time [s]
    state : dict
        A small state view passed in by the runner (optional; unused here)
    cfg : dict
        Run configuration (contains DELTA_S_DEG, DELTA_D_DEG)

    Returns
    -------
    (deltaS_deg, deltaD_deg) : tuple[float, float]
        Commanded symmetric and differential elevon deflections [deg]
    """
    # Constant open-loop commands
    deltaS = float(cfg["DELTA_S_DEG"])
    deltaD = float(cfg["DELTA_D_DEG"])

    # Example time scheduling (disabled by default):
    # if t > 2.0:
    #     deltaS += 2.0

    return deltaS, deltaD
