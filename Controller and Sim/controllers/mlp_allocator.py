# controllers/mlp_allocator.py

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class MLPControlAllocator:
    """
    MLP-based control allocator.

    Given desired moment coefficients Cl_cmd, Cm_cmd, Cn_cmd,
    find deltaS and deltaD that make the aero surrogate predict those moments.

    This uses the existing aero_model.query(alpha, beta, deltaS, deltaD).
    """

    def __init__(
        self,
        aero_model,
        deltaS_max_deg: float = 20.0,
        deltaD_max_deg: float = 10.0,
        elevon_max_deg: float = 25.0,
        w_cl: float = 5.0,
        w_cm: float = 10.0,
        w_cn: float = 1.0,
        w_effort: float = 0.0005,
        w_smooth: float = 0.02,
        w_elevon_limit: float = 100.0,
        maxiter: int = 20,
    ):
        self.aero_model = aero_model

        self.deltaS_max_deg = float(deltaS_max_deg)
        self.deltaD_max_deg = float(deltaD_max_deg)
        self.elevon_max_deg = float(elevon_max_deg)

        self.w_cl = float(w_cl)
        self.w_cm = float(w_cm)
        self.w_cn = float(w_cn)
        self.w_effort = float(w_effort)
        self.w_smooth = float(w_smooth)
        self.w_elevon_limit = float(w_elevon_limit)

        self.maxiter = int(maxiter)

        self.prev_deltaS = 0.0
        self.prev_deltaD = 0.0

    def allocate(
        self,
        alpha_deg: float,
        beta_deg: float,
        Cl_cmd: float,
        Cm_cmd: float,
        Cn_cmd: float = 0.0,
        initial_guess=None,
    ):
        alpha_deg = float(alpha_deg)
        beta_deg = float(beta_deg)

        Cl_cmd = float(Cl_cmd)
        Cm_cmd = float(Cm_cmd)
        Cn_cmd = float(Cn_cmd)

        if initial_guess is None:
            x0 = np.array([self.prev_deltaS, self.prev_deltaD], dtype=float)
        else:
            x0 = np.asarray(initial_guess, dtype=float)

        x0[0] = np.clip(x0[0], -self.deltaS_max_deg, self.deltaS_max_deg)
        x0[1] = np.clip(x0[1], -self.deltaD_max_deg, self.deltaD_max_deg)

        def cost(u):
            deltaS, deltaD = float(u[0]), float(u[1])

            coeffs = self.aero_model.query(
                alpha_deg,
                beta_deg,
                deltaS,
                deltaD,
            )

            Cl = float(coeffs["Cl"])
            Cm = float(coeffs["Cm"])
            Cn = float(coeffs["Cn"])

            # Actual physical elevon angles if:
            # deltaL = deltaS + deltaD
            # deltaR = deltaS - deltaD
            deltaL = deltaS + deltaD
            deltaR = deltaS - deltaD

            J = 0.0

            # Track desired moment coefficients
            J += self.w_cl * (Cl - Cl_cmd) ** 2
            J += self.w_cm * (Cm - Cm_cmd) ** 2
            J += self.w_cn * (Cn - Cn_cmd) ** 2

            # Prefer smaller actuator usage
            J += self.w_effort * (deltaS ** 2 + deltaD ** 2)

            # Prefer smooth changes from previous command
            J += self.w_smooth * (
                (deltaS - self.prev_deltaS) ** 2
                + (deltaD - self.prev_deltaD) ** 2
            )

            # Soft physical elevon limit penalty
            left_violation = max(0.0, abs(deltaL) - self.elevon_max_deg)
            right_violation = max(0.0, abs(deltaR) - self.elevon_max_deg)

            J += self.w_elevon_limit * (
                left_violation ** 2
                + right_violation ** 2
            )

            return float(J)

        result = minimize(
            cost,
            x0,
            method="SLSQP",
            bounds=[
                (-self.deltaS_max_deg, self.deltaS_max_deg),
                (-self.deltaD_max_deg, self.deltaD_max_deg),
            ],
            options={
                "maxiter": self.maxiter,
                "ftol": 1e-7,
                "disp": False,
            },
        )

        if result.success:
            deltaS, deltaD = result.x
        else:
            # Fall back to previous/safe guess if optimizer fails
            deltaS, deltaD = x0

        deltaS = float(np.clip(deltaS, -self.deltaS_max_deg, self.deltaS_max_deg))
        deltaD = float(np.clip(deltaD, -self.deltaD_max_deg, self.deltaD_max_deg))

        self.prev_deltaS = deltaS
        self.prev_deltaD = deltaD

        return deltaS, deltaD