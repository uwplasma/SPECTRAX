"""Third-order strong-stability-preserving Runge--Kutta method."""

from typing import ClassVar

import numpy as np
from diffrax import (
    AbstractERK,
    ButcherTableau,
    ThirdOrderHermitePolynomialInterpolation,
)

__all__ = ["SSPRK3"]


class SSPRK3(AbstractERK):
    """Three-stage, third-order SSP Runge--Kutta method (SSPRK(3,3)).

    Strong stability requires forward Euler to be strongly stable in the
    chosen convex functional and the time step to obey its stability bound; it
    is not an unconditional guarantee for Fourier spectral discretizations.
    """

    tableau: ClassVar[ButcherTableau] = ButcherTableau(
        a_lower=(np.array([1.0]), np.array([0.25, 0.25])),
        b_sol=np.array([1 / 6, 1 / 6, 2 / 3]),
        b_error=np.array([-1 / 3, -1 / 3, 2 / 3]),
        c=np.array([1.0, 0.5]),
    )
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        del terms
        return 3
