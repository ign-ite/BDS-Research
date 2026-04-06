"""
Extension 1 — Automated Dirichlet Weight Optimisation (AWO)
============================================================
Maintains a 5-simplex weight vector w and updates it every K episodes
via projected gradient ascent on the scalarised return.

Objectives (indices):
    0: α  — normalised VM cost      (lower cost → higher α-reward)
    1: ε  — execution time          (lower time → higher ε-reward)
    2: τ  — resource utilisation    (higher util → higher τ-reward)
    3: δ  — throughput              (higher → higher δ-reward)
    4: μ  — deadline adherence      (higher → higher μ-reward)
"""

import numpy as np
from typing import Dict, List, Optional

# ---- w_mode presets ---------------------------------------------------------
PRESET_COST_FOCUS    = np.array([0.50, 0.10, 0.10, 0.15, 0.15], dtype=np.float64)
PRESET_BALANCED      = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float64)
DIRICHLET_INIT_ALPHA = np.ones(5, dtype=np.float64)          # uniform Dirichlet sample


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


class DirichletWeightOptimiser:
    """
    Maintains w ∈ Δ^5 and applies projected gradient ascent every K episodes.

    Parameters
    ----------
    w_mode : str
        One of "fixed_cost", "fixed_balanced", "auto_dirichlet".
    K : int
        Number of episodes between weight updates.
    lr_w : float
        Learning rate for weight update.
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        w_mode: str = "auto_dirichlet",
        K: int = 200,
        lr_w: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.w_mode = w_mode
        self.K = K
        self.lr_w = lr_w
        self._rng = np.random.default_rng(seed)

        # Initialise weight vector
        if w_mode == "fixed_cost":
            self._weights = PRESET_COST_FOCUS.copy()
        elif w_mode == "fixed_balanced":
            self._weights = PRESET_BALANCED.copy()
        else:  # "auto_dirichlet"
            self._weights = self._rng.dirichlet(DIRICHLET_INIT_ALPHA)

        # Episode logs — cleared every K episodes after gradient step
        self._episode_logs: List[Dict] = []   # [{objectives, total_return}, ...]

        # Full history for plotting
        self.weight_history: List[np.ndarray] = [self._weights.copy()]
        self.episode_count: int = 0

    # ------------------------------------------------------------------
    def get_weights(self) -> np.ndarray:
        """Return current weight vector (copy)."""
        return self._weights.copy()

    # ------------------------------------------------------------------
    def update(self, objectives: np.ndarray, total_return: float) -> None:
        """
        Record one episode's objective vector and scalar return.
        Triggers a weight update every K episodes (auto_dirichlet only).

        Parameters
        ----------
        objectives : np.ndarray, shape (5,)
            Per-objective normalised reward contributions [α, ε, τ, δ, μ].
        total_return : float
            Scalar episode return.
        """
        self.episode_count += 1
        self._episode_logs.append({
            "objectives": np.array(objectives, dtype=np.float64),
            "total_return": float(total_return),
        })
        # Always keep full weight history
        self.weight_history.append(self._weights.copy())

        if self.w_mode == "auto_dirichlet" and len(self._episode_logs) >= self.K:
            self._gradient_step()
            self._episode_logs.clear()

    # ------------------------------------------------------------------
    def _gradient_step(self) -> None:
        """
        Projected gradient ascent on E[w·objectives] w.r.t. w.

        Gradient estimate:  ∇_w ≈ (1/K) Σ_k (objectives_k - w·objectives_k · ones)
        We actually use the weighted-return gradient:
            ∇_w J(w) ≈ mean over K episodes of: R_k * objectives_k / ||objectives_k||
        This encourages w to align with high-return objective vectors.
        """
        logs = self._episode_logs
        K = len(logs)
        if K == 0:
            return

        objectives_mat = np.stack([l["objectives"] for l in logs])   # (K, 5)
        returns = np.array([l["total_return"] for l in logs])          # (K,)

        # Normalise objectives per episode to [0,1] range
        obj_norms = np.linalg.norm(objectives_mat, axis=1, keepdims=True) + 1e-8
        objectives_norm = objectives_mat / obj_norms                   # (K, 5)

        # Gradient: weighted average of objective vectors by return
        ret_weights = returns - returns.mean()                         # centre
        ret_std = returns.std() + 1e-8
        ret_weights /= ret_std
        grad = (objectives_norm * ret_weights[:, None]).mean(axis=0)   # (5,)

        # Ascent step
        w_new = self._weights + self.lr_w * grad
        self._weights = _project_simplex(w_new)

    # ------------------------------------------------------------------
    def get_weight_trajectory(self) -> np.ndarray:
        """
        Return full weight history as array of shape (T, 5).
        Each row corresponds to one episode tick.
        """
        return np.array(self.weight_history)   # (T, 5)

    # ------------------------------------------------------------------
    @staticmethod
    def make(w_mode: str, **kwargs) -> "DirichletWeightOptimiser":
        """Factory: create optimiser for the given w_mode."""
        return DirichletWeightOptimiser(w_mode=w_mode, **kwargs)
