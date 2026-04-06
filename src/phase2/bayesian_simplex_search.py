"""
Extension 1 — Bayesian Simplex Search
======================================
Replaces projected gradient ascent with Latin Hypercube Sampling on
the 5-simplex.  Generates N candidate weight vectors, evaluates each
for a short training run, and selects the best w*.

Objectives (indices):
    0: α  — normalised VM cost
    1: ε  — execution time
    2: τ  — resource utilisation
    3: δ  — throughput
    4: μ  — deadline adherence
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube
from typing import List, Optional

# ---- w_mode presets (unchanged from original) ----------------------------
PRESET_COST_FOCUS = np.array([0.50, 0.10, 0.10, 0.15, 0.15], dtype=np.float64)
PRESET_BALANCED   = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float64)


def generate_simplex_candidates(n: int = 20, d: int = 5,
                                 seed: int = 42) -> np.ndarray:
    """
    Generate *n* weight vectors on the *d*-simplex using Latin
    Hypercube Sampling followed by row-normalisation.

    Returns
    -------
    candidates : np.ndarray, shape (n, d)
        Each row sums to 1 and lies on the probability simplex.
    """
    sampler = LatinHypercube(d=d, seed=seed)
    raw = sampler.random(n=n)          # (n, d) in [0, 1]^d
    # Normalise each row to sum=1 → project onto simplex
    row_sums = raw.sum(axis=1, keepdims=True)
    candidates = raw / row_sums
    return candidates


class BayesianSimplexSearch:
    """
    Manages the LHS candidate pool, tracks evaluation results,
    and selects w*.

    Note: despite the class name, this implementation uses Latin
    Hypercube Sampling (LHS) for candidate generation, not a
    Gaussian-process surrogate model. The term 'Bayesian' refers to
    the budget-efficient, prior-informed sampling philosophy rather
    than strict Bayesian optimisation.

    Parameters
    ----------
    n_candidates : int
        Number of weight-vector candidates to generate.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, n_candidates: int = 20, seed: int = 42):
        self.n_candidates = n_candidates
        self.candidates = generate_simplex_candidates(n_candidates, seed=seed)
        self.scores: List[float] = []       # mean return per candidate
        self.costs: List[float] = []        # mean cost per candidate
        self.best_idx: Optional[int] = None
        self.best_w: Optional[np.ndarray] = None

    def record_score(self, idx: int, mean_return: float, mean_cost: float):
        """Record evaluation result for candidate *idx*."""
        if len(self.scores) <= idx:
            self.scores.extend([0.0] * (idx + 1 - len(self.scores)))
            self.costs.extend([0.0] * (idx + 1 - len(self.costs)))
        self.scores[idx] = mean_return
        self.costs[idx] = mean_cost

    def select_best(self) -> np.ndarray:
        """Select w* = argmax mean_return across all candidates."""
        self.best_idx = int(np.argmax(self.scores))
        self.best_w = self.candidates[self.best_idx].copy()
        return self.best_w

    def get_candidate(self, idx: int) -> np.ndarray:
        """Return candidate weight vector at index *idx*."""
        return self.candidates[idx].copy()

    def summary(self) -> dict:
        """Return a summary dict for reporting."""
        return {
            "candidates": self.candidates.tolist(),
            "scores": self.scores,
            "costs": self.costs,
            "best_idx": self.best_idx,
            "best_w": self.best_w.tolist() if self.best_w is not None else None,
        }
