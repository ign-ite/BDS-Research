"""
Extension 2 — Stochastic VM Pricing / Spot Instances
=====================================================
PricingModel wraps VM pricing with two modes:
  - "static"  : fixed base prices (Phase-1 behaviour)
  - "spot"    : Ornstein-Uhlenbeck process per VM type

OU parameters:  θ=0.15, σ=0.20, μ=base_price
Price clamped to [0.1×base, 3.0×base].
"""

import numpy as np
from typing import List, Optional


class PricingModel:
    """
    Controls VM pricing across episodes.

    Parameters
    ----------
    mode : str
        "static" or "spot".
    vms : list
        Initial VM objects (used to extract base prices).
    theta : float
        Mean-reversion speed for OU process.
    sigma : float
        Volatility for OU process.
    dt : float
        Time step size per episode tick.
    seed : Optional[int]
        RNG seed.
    """

    def __init__(
        self,
        mode: str = "static",
        vms: Optional[List] = None,
        theta: float = 0.15,
        sigma: float = 0.20,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.mode = mode
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self._rng = np.random.default_rng(seed)

        # Extract base prices from VMs (or default to 1.0)
        if vms:
            self._base_prices = np.array([vm.base_price for vm in vms], dtype=np.float64)
            self._current_prices = self._base_prices.copy()
        else:
            self._base_prices = np.array([], dtype=np.float64)
            self._current_prices = np.array([], dtype=np.float64)

        # Price history: list of snapshots, one per episode step
        # Shape grows to (n_episodes, n_vms)
        self._price_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def init_vms(self, vms: List) -> None:
        """Initialise (or re-initialise) base prices from a VM list."""
        self._base_prices = np.array([vm.base_price for vm in vms], dtype=np.float64)
        self._current_prices = self._base_prices.copy()

    # ------------------------------------------------------------------
    def step(self, vms: List) -> None:
        """
        Update vm.current_price for each VM using the pricing model.
        Called once per episode (or per step if desired).

        Parameters
        ----------
        vms : list of VM objects with attributes base_price, current_price.
        """
        if len(self._base_prices) == 0:
            self.init_vms(vms)

        if self.mode == "static":
            for vm in vms:
                vm.current_price = vm.base_price
            self._price_history.append(self._base_prices.copy())
            return

        # Spot: Euler-Maruyama discretisation of OU SDE
        #   dP = θ(μ - P)dt + σ√dt · dW
        n = len(vms)
        mu = np.array([vm.base_price for vm in vms], dtype=np.float64)
        dW = self._rng.standard_normal(n)

        self._current_prices = (
            self._current_prices
            + self.theta * (mu - self._current_prices) * self.dt
            + self.sigma * np.sqrt(self.dt) * dW
        )

        # Clamp to [0.1×base, 3.0×base]
        self._current_prices = np.clip(
            self._current_prices, 0.1 * mu, 3.0 * mu
        )

        for i, vm in enumerate(vms):
            vm.current_price = float(self._current_prices[i])

        self._price_history.append(self._current_prices.copy())

    # ------------------------------------------------------------------
    def get_price_history(self) -> np.ndarray:
        """
        Return price history as np.ndarray of shape (n_episodes, n_vms).
        """
        if not self._price_history:
            return np.empty((0, len(self._base_prices)))
        return np.array(self._price_history)

    # ------------------------------------------------------------------
    def reset_prices(self, vms: List) -> None:
        """Reset current prices to base values (call at episode start)."""
        if len(self._base_prices) == 0:
            self.init_vms(vms)
        self._current_prices = np.array([vm.base_price for vm in vms], dtype=np.float64)
        for vm in vms:
            vm.current_price = vm.base_price
