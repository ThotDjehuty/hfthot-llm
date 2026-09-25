"""Physics-Informed Neural Network core for thotbook-AmentI.

CPU-only PINN machinery used by 04_pinn_physics.ipynb to solve the Dirac,
Brans-Dicke and Wheeler-DeWitt equations.

NOTE on this machine: PyTorch ships no macOS x86_64 wheel past 2.2.2, which
was built against NumPy 1.x while this env runs NumPy 2.x. Pure-torch autograd
(including the higher-order derivatives PINNs need) works fine; only the
`.numpy()` bridge is broken. Every tensor->array conversion here therefore
goes through `.tolist()`, never `.numpy()`.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)   # PINNs need the precision


def to_np(t: torch.Tensor) -> np.ndarray:
    """Tensor -> ndarray without the broken .numpy() bridge."""
    return np.asarray(t.detach().reshape(-1).tolist())


def d(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """d^order y / dx^order via repeated autograd (graph kept for PINN loss)."""
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=True)[0]
    return y


class MLP(nn.Module):
    """Fully-connected tanh network — the standard PINN trial function.

    tanh is used because PINN residuals need smooth higher-order derivatives;
    ReLU has a zero second derivative almost everywhere and cannot represent
    the curvature terms in any of the three PDEs here.
    """

    def __init__(self, n_in: int = 1, n_out: int = 1, width: int = 64,
                 depth: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        layers: list[nn.Module] = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:                      # Xavier init — PINN standard
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_pinn(model: nn.Module, loss_fn, n_adam: int = 3000,
               n_lbfgs: int = 500, lr: float = 1e-3, log_every: int = 500,
               verbose: bool = True) -> dict:
    """Two-stage Adam -> L-BFGS training, the canonical PINN recipe.

    Adam escapes the poor initial basin cheaply; L-BFGS (a quasi-Newton
    method using the curvature of the loss surface) then converges to far
    tighter residuals than any first-order method reaches alone. This is the
    optimisation pairing referenced in the notebook's optimiser chapter —
    explicitly *not* differential evolution, which cannot exploit the
    analytic gradients autograd already gives us.
    """
    hist: list[float] = []
    t0 = time.time()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(n_adam):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if verbose and (it % log_every == 0 or it == n_adam - 1):
            print(f"  [Adam ] {it:5d}  loss = {loss.item():.4e}")
    t_adam = time.time() - t0

    t1 = time.time()
    if n_lbfgs > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=n_lbfgs,
                                  history_size=50, tolerance_grad=1e-12,
                                  tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            hist.append(loss.item())
            return loss

        lbfgs.step(closure)
        if verbose:
            print(f"  [LBFGS] final loss = {hist[-1]:.4e}")
    t_lbfgs = time.time() - t1

    return {"history": np.asarray(hist), "final_loss": hist[-1],
            "t_adam": t_adam, "t_lbfgs": t_lbfgs, "t_total": t_adam + t_lbfgs,
            "n_params": sum(p.numel() for p in model.parameters())}


def l2_relative(pred: np.ndarray, exact: np.ndarray) -> float:
    """Relative L2 error ||pred-exact||_2 / ||exact||_2 — the standard PINN metric."""
    return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-300))
