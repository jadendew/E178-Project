"""
aero.py

Aero table model:
- Interpolates coefficients over (alpha, beta) on a fixed (deltaS, deltaD) slice.
- Supports continuous deflections:
  - Superposition mode (recommended for your separated pitch/roll sweeps):
      C(dS,dD) ≈ C00 + (C(dS,0)-C00) + (C(0,dD)-C00)
    with linear interpolation along the available dS and dD families.
  - Non-superposition mode:
      Bilinear interpolation in (dS, dD) using four corner slices
      (requires a full 4D grid of slices in the CSV).
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class AeroSlice2D:
    """Interpolates coefficients on a fixed (deltaS, deltaD) slice over (alpha, beta)."""
    def __init__(self, df: pd.DataFrame, deltaS_deg: float, deltaD_deg: float, tol: float = 1e-9):
        sl = df[
            np.isclose(df["deltaS_deg"], deltaS_deg, atol=tol) &
            np.isclose(df["deltaD_deg"], deltaD_deg, atol=tol)
        ].copy()
        if sl.empty:
            raise ValueError(f"No rows for deltaS={deltaS_deg}, deltaD={deltaD_deg}")

        self.deltaS_deg = float(deltaS_deg)
        self.deltaD_deg = float(deltaD_deg)

        self.alphas = np.sort(sl["alpha_deg"].unique())
        self.betas  = np.sort(sl["beta_deg"].unique())

        def grid(col: str) -> np.ndarray:
            piv = sl.pivot_table(index="alpha_deg", columns="beta_deg", values=col, aggfunc="mean")
            piv = piv.reindex(index=self.alphas, columns=self.betas)
            if piv.isna().any().any():
                raise ValueError(f"Slice missing grid points for {col}.")
            return piv.to_numpy(dtype=float)

        self.interp = {
            k: RegularGridInterpolator((self.alphas, self.betas), grid(k),
                                       bounds_error=False, fill_value=None)
            for k in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
        }

        self.alpha_min, self.alpha_max = float(self.alphas.min()), float(self.alphas.max())
        self.beta_min,  self.beta_max  = float(self.betas.min()),  float(self.betas.max())

    def query(self, alpha_deg: float, beta_deg: float) -> Dict[str, float]:
        a = float(np.clip(alpha_deg, self.alpha_min, self.alpha_max))
        b = float(np.clip(beta_deg,  self.beta_min,  self.beta_max))
        pt = np.array([a, b], dtype=float)
        out = {}
        for k in self.interp:
            out[k] = float(np.asarray(self.interp[k](pt)).reshape(-1)[0])
        return out


class AeroModel:
    """
    Provides coefficients for arbitrary (deltaS, deltaD).

    If use_superposition=True:
      - Requires baseline (0,0), plus:
          pitch family: deltaD == 0 with multiple deltaS
          roll  family: deltaS == 0 with multiple deltaD
      - Uses linear interpolation along each family + incremental superposition.

    If use_superposition=False:
      - Performs bilinear interpolation in (deltaS, deltaD) using corner slices.
      - Requires the dataset to include full grid slices.
    """
    def __init__(self, df: pd.DataFrame, tol: float = 1e-9, use_superposition: bool = True):
        self.df = df.copy()
        self.tol = float(tol)
        self.use_superposition = bool(use_superposition)

        # Global alpha/beta bounds
        self.alpha_min = float(self.df["alpha_deg"].min())
        self.alpha_max = float(self.df["alpha_deg"].max())
        self.beta_min  = float(self.df["beta_deg"].min())
        self.beta_max  = float(self.df["beta_deg"].max())

        # Deflection sets available (global)
        self.deltaS_vals_all = np.sort(self.df["deltaS_deg"].unique().astype(float))
        self.deltaD_vals_all = np.sort(self.df["deltaD_deg"].unique().astype(float))

        # Families for superposition
        self.deltaS_vals_pitch = np.sort(
            self.df[np.isclose(self.df["deltaD_deg"], 0.0, atol=self.tol)]["deltaS_deg"].unique().astype(float)
        )
        self.deltaD_vals_roll  = np.sort(
            self.df[np.isclose(self.df["deltaS_deg"], 0.0, atol=self.tol)]["deltaD_deg"].unique().astype(float)
        )

        self._slice_cache: Dict[Tuple[float, float], AeroSlice2D] = {}

        if self.use_superposition:
            has00 = np.any(
                np.isclose(self.df["deltaS_deg"], 0.0, atol=self.tol) &
                np.isclose(self.df["deltaD_deg"], 0.0, atol=self.tol)
            )
            if not has00:
                raise ValueError("Superposition requires deltaS=0, deltaD=0 rows in CSV.")

            if len(self.deltaS_vals_pitch) == 0 or len(self.deltaD_vals_roll) == 0:
                raise ValueError("Superposition requires deltaD==0 family and deltaS==0 family.")

            if not np.any(np.isclose(self.deltaS_vals_pitch, 0.0, atol=self.tol)):
                raise ValueError("Superposition requires deltaS=0 present in deltaD==0 family.")
            if not np.any(np.isclose(self.deltaD_vals_roll, 0.0, atol=self.tol)):
                raise ValueError("Superposition requires deltaD=0 present in deltaS==0 family.")

    def _get_slice(self, dS: float, dD: float) -> AeroSlice2D:
        key = (float(dS), float(dD))
        if key not in self._slice_cache:
            self._slice_cache[key] = AeroSlice2D(self.df, dS, dD, tol=self.tol)
        return self._slice_cache[key]

    @staticmethod
    def _bracket(vals: np.ndarray, x: float) -> Tuple[float, float, float]:
        """
        Return (x0, x1, w) where:
          x is clamped to [min,max]
          lerp = (1-w)*f(x0) + w*f(x1)
        """
        vals = np.asarray(vals, dtype=float)
        x = float(x)

        if x <= float(vals.min()):
            return float(vals.min()), float(vals.min()), 0.0
        if x >= float(vals.max()):
            return float(vals.max()), float(vals.max()), 0.0

        idx = int(np.searchsorted(vals, x))
        x0 = float(vals[idx - 1])
        x1 = float(vals[idx])
        if abs(x1 - x0) < 1e-12:
            return x0, x1, 0.0
        w = float((x - x0) / (x1 - x0))
        return x0, x1, w

    @staticmethod
    def _lerp_coeffs(c0: Dict[str, float], c1: Dict[str, float], w: float) -> Dict[str, float]:
        out = {}
        for k in c0:
            out[k] = (1.0 - w) * float(c0[k]) + w * float(c1[k])
        return out

    def clamp_deltas(self, deltaS_deg: float, deltaD_deg: float) -> Tuple[float, float]:
        """Clamp requested deltas to what is available in the dataset."""
        dS = float(deltaS_deg)
        dD = float(deltaD_deg)
        if self.use_superposition:
            dS = float(np.clip(dS, float(self.deltaS_vals_pitch.min()), float(self.deltaS_vals_pitch.max())))
            dD = float(np.clip(dD, float(self.deltaD_vals_roll.min()),  float(self.deltaD_vals_roll.max())))
        else:
            dS = float(np.clip(dS, float(self.deltaS_vals_all.min()), float(self.deltaS_vals_all.max())))
            dD = float(np.clip(dD, float(self.deltaD_vals_all.min()), float(self.deltaD_vals_all.max())))
        return dS, dD

    def query(self, alpha_deg: float, beta_deg: float, deltaS_deg: float, deltaD_deg: float) -> Dict[str, float]:
        # clamp alpha/beta to global grid
        a = float(np.clip(alpha_deg, self.alpha_min, self.alpha_max))
        b = float(np.clip(beta_deg,  self.beta_min,  self.beta_max))

        dS_req = float(deltaS_deg)
        dD_req = float(deltaD_deg)
        dS, dD = self.clamp_deltas(dS_req, dD_req)

        if self.use_superposition:
            base00 = self._get_slice(0.0, 0.0).query(a, b)

            # C(dS,0) interpolated along deltaS family
            s0, s1, ws = self._bracket(self.deltaS_vals_pitch, dS)
            c_s0 = self._get_slice(s0, 0.0).query(a, b)
            c_s1 = self._get_slice(s1, 0.0).query(a, b)
            c_s  = self._lerp_coeffs(c_s0, c_s1, ws)

            # C(0,dD) interpolated along deltaD family
            d0, d1, wd = self._bracket(self.deltaD_vals_roll, dD)
            c_d0 = self._get_slice(0.0, d0).query(a, b)
            c_d1 = self._get_slice(0.0, d1).query(a, b)
            c_d  = self._lerp_coeffs(c_d0, c_d1, wd)

            out = {}
            for k in base00:
                out[k] = float(base00[k]) + (float(c_s[k]) - float(base00[k])) + (float(c_d[k]) - float(base00[k]))
            return out

        # bilinear in (dS,dD) - needs full grid
        s0, s1, ws = self._bracket(self.deltaS_vals_all, dS)
        d0, d1, wd = self._bracket(self.deltaD_vals_all, dD)

        c00 = self._get_slice(s0, d0).query(a, b)
        c10 = self._get_slice(s1, d0).query(a, b)
        c01 = self._get_slice(s0, d1).query(a, b)
        c11 = self._get_slice(s1, d1).query(a, b)

        c0 = self._lerp_coeffs(c00, c10, ws)
        c1 = self._lerp_coeffs(c01, c11, ws)
        return self._lerp_coeffs(c0, c1, wd)
