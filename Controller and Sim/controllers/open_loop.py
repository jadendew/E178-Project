# controllers/open_loop.py
"""
Open-loop controller: returns constant elevon deflections + throttle.

Return format:
  (deltaS_deg, deltaD_deg, throttle)

Throttle in [0,1].
"""

def controller(t, state, cfg):
    # Use run_sim.py config defaults
    deltaS = float(cfg.delta_s_deg)
    deltaD = float(cfg.delta_d_deg)
    throttle = float(cfg.throttle0)

    return (deltaS, deltaD, throttle)
