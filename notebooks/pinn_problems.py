"""The three physics problems solved by PINNs in 04_pinn_physics.ipynb.

Each problem exposes:
    .residual(model, x)  -> PDE/ODE residual tensor(s)
    .exact(x)            -> reference solution (analytic, or high-accuracy ODE)
    .loss_fn(model)      -> closure for pinn_core.train_pinn

All three have an independently-checkable ground truth, so the PINN is never
graded against itself.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.special import airy

from pinn_core import d, to_np


# ---------------------------------------------------------------------------
# 1. DIRAC EQUATION (1+1 D, stationary, two-component spinor)
# ---------------------------------------------------------------------------
class DiracProblem:
    r"""Stationary Dirac equation in 1+1 dimensions.

    In the standard two-component reduction the stationary Dirac system is the
    coupled first-order pair

        du/dx = +(E + m) v
        dv/dx = -(E - m) u

    Differentiating once decouples it:  u'' = -(E^2 - m^2) u = -k^2 u,
    which is exactly the relativistic dispersion relation  E^2 = k^2 + m^2.

    With u(0) = 1, v(0) = 0 the closed-form solution is

        u(x) = cos(kx),      v(x) = -(k/(E+m)) sin(kx),     k = sqrt(E^2 - m^2)

    so the PINN can be graded against an exact analytic benchmark.
    """

    name = "Dirac (1+1D)"

    def __init__(self, E: float = 2.0, m: float = 1.0, L: float = 6.0,
                 n_col: int = 400):
        assert E > m, "need E > m for a propagating (real-k) solution"
        self.E, self.m, self.L = E, m, L
        self.k = float(np.sqrt(E ** 2 - m ** 2))
        self.x = torch.linspace(0.0, L, n_col).reshape(-1, 1).requires_grad_(True)
        self.x0 = torch.zeros(1, 1)

    def exact(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.cos(self.k * x)
        v = -(self.k / (self.E + self.m)) * np.sin(self.k * x)
        return u, v

    def residual(self, model, x):
        out = model(x)
        u, v = out[:, 0:1], out[:, 1:2]
        r1 = d(u, x) - (self.E + self.m) * v
        r2 = d(v, x) + (self.E - self.m) * u
        return r1, r2

    def loss_fn(self, model):
        def closure():
            r1, r2 = self.residual(model, self.x)
            phys = (r1 ** 2).mean() + (r2 ** 2).mean()
            o0 = model(self.x0)
            bc = (o0[:, 0] - 1.0) ** 2 + (o0[:, 1] - 0.0) ** 2
            return phys + 10.0 * bc.mean()
        return closure


# ---------------------------------------------------------------------------
# 2. BRANS-DICKE COSMOLOGY (flat FLRW + dust)
# ---------------------------------------------------------------------------
class BransDickeProblem:
    r"""Brans-Dicke scalar-tensor cosmology, flat FLRW with dust.

    Field equations (units 8*pi = c = 1), with H = adot/a and rho = rho0 a^-3:

        (F)  3H^2 = rho/phi + (omega/2)(phidot/phi)^2 - 3H(phidot/phi)
        (S)  phiddot + 3H phidot = rho/(2 omega + 3)

    The Nariai power-law solution is exact:

        a(t) = t^q,   q = (2 + 2 omega)/(4 + 3 omega)
        phi(t) = t^s, s = 2/(4 + 3 omega)

    consistent iff rho0 = phi0 * s * (2 omega + 3).  As omega -> infinity,
    q -> 2/3, recovering the GR dust limit -- the standard sanity check that
    Brans-Dicke reduces to general relativity.
    """

    name = "Brans-Dicke (FLRW + dust)"

    def __init__(self, omega: float = 10.0, t0: float = 1.0, t1: float = 3.0,
                 n_col: int = 400):
        self.omega = omega
        D = 4.0 + 3.0 * omega
        self.q = (2.0 + 2.0 * omega) / D
        self.s = 2.0 / D
        self.rho0 = 1.0 * self.s * (2.0 * omega + 3.0)   # phi0 = 1
        self.t = torch.linspace(t0, t1, n_col).reshape(-1, 1).requires_grad_(True)
        self.tic = torch.tensor([[t0]])
        self.t0 = t0

    def exact(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return t ** self.q, t ** self.s

    def residual(self, model, t):
        out = model(t)
        # positivity of a and phi enforced structurally via exp
        a = torch.exp(out[:, 0:1])
        phi = torch.exp(out[:, 1:2])
        adot, phidot = d(a, t), d(phi, t)
        phiddot = d(phidot, t)
        H = adot / a
        rho = self.rho0 * a ** (-3)
        rF = (3 * H ** 2 - rho / phi
              - 0.5 * self.omega * (phidot / phi) ** 2
              + 3 * H * (phidot / phi))
        rS = phiddot + 3 * H * phidot - rho / (2 * self.omega + 3)
        return rF, rS

    def loss_fn(self, model):
        def closure():
            rF, rS = self.residual(model, self.t)
            phys = (rF ** 2).mean() + (rS ** 2).mean()
            o = model(self.tic)          # a(t0)=1, phi(t0)=1 -> log = 0
            bc = (o[:, 0] ** 2 + o[:, 1] ** 2).mean()
            return phys + 100.0 * bc
        return closure


# ---------------------------------------------------------------------------
# 3. WHEELER-DEWITT EQUATION (minisuperspace, closed FLRW + Lambda)
# ---------------------------------------------------------------------------
class WheelerDeWittProblem:
    r"""Wheeler-DeWitt equation in closed-FLRW minisuperspace.

        Psi''(a) + (p/a) Psi'(a) - U(a) Psi(a) = 0,
        U(a) = a^2 - (Lambda/3) a^4

    p is the Hartle-Hawking factor-ordering parameter (the operator-ordering
    ambiguity of quantising a curved minisuperspace).  The classical turning
    point sits at a_t = sqrt(3/Lambda): U < 0 (oscillatory, classically
    allowed) for a > a_t, U > 0 (exponential, under the barrier) for a < a_t
    -- the tunnelling-from-nothing configuration.

    Two independent ground truths:
      * `exact` integrates the full ODE with scipy's adaptive Radau solver at
        tight tolerance (rtol=1e-12);
      * `airy_asymptotic` gives the analytic Airy form valid near a_t, where
        linearising U(a) ~ U'(a_t)(a - a_t) turns the equation into Psi_zz = z Psi.
    """

    name = "Wheeler-DeWitt (minisuperspace)"

    def __init__(self, lam: float = 1.0, p: float = 0.0, a_lo: float = 0.3,
                 a_hi: float = 2.6, n_col: int = 500):
        self.lam, self.p = lam, p
        self.a_lo, self.a_hi = a_lo, a_hi
        self.a_turn = float(np.sqrt(3.0 / lam))
        self.a = torch.linspace(a_lo, a_hi, n_col).reshape(-1, 1).requires_grad_(True)
        self.aic = torch.tensor([[a_lo]])
        self._ref = None
        # initial data at a_lo, shared by the PINN BC and the ODE reference
        self.psi0, self.dpsi0 = 1.0, 0.0

    def U(self, a):
        return a ** 2 - (self.lam / 3.0) * a ** 4

    def exact(self, a_eval: np.ndarray) -> np.ndarray:
        """High-accuracy reference via adaptive implicit ODE integration."""
        def rhs(a, y):
            psi, dpsi = y
            return [dpsi, -(self.p / a) * dpsi + (a ** 2 - (self.lam / 3) * a ** 4) * psi]
        sol = solve_ivp(rhs, (self.a_lo, self.a_hi), [self.psi0, self.dpsi0],
                        t_eval=np.sort(a_eval), method="Radau",
                        rtol=1e-12, atol=1e-14)
        assert sol.success, sol.message
        return sol.y[0]

    def airy_asymptotic(self, a_eval: np.ndarray) -> np.ndarray:
        """Analytic Airy solution of the linearised equation near a_turn."""
        at = self.a_turn
        Uprime = 2 * at - (4 * self.lam / 3) * at ** 3      # = -2*sqrt(3) for lam=1
        c = abs(Uprime) ** (1.0 / 3.0)
        z = -c * (a_eval - at)
        return airy(z)[0]                                    # Ai(z)

    def residual(self, model, a):
        psi = model(a)
        dpsi = d(psi, a)
        d2psi = d(dpsi, a)
        return d2psi + (self.p / a) * dpsi - self.U(a) * psi

    def loss_fn(self, model):
        def closure():
            r = self.residual(model, self.a)
            phys = (r ** 2).mean()
            aic = self.aic.clone().requires_grad_(True)
            psi_ic = model(aic)
            dpsi_ic = d(psi_ic, aic)
            bc = ((psi_ic - self.psi0) ** 2).mean() + ((dpsi_ic - self.dpsi0) ** 2).mean()
            return phys + 100.0 * bc
        return closure
