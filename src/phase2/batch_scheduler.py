"""
Extension 3 — Job Batching to Reduce Scheduling Overhead
=========================================================
BatchScheduler wraps ClusterEnv and groups up to B pending executor
requests into a single compound scheduling decision.

Metrics tracked per episode:
    scheduling_steps  — number of agent .step() calls
    overhead_ratio    — scheduling_steps / total_executors_scheduled
"""

import copy
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class BatchScheduler:
    """
    Wrapper around a ClusterEnv that issues up to B executor placements
    per agent decision step.

    Each call to step(actions) consumes a list/array of up to B VM
    indices (0 = wait, 1..V = place on VM i), one per pending executor.
    The underlying env is stepped for each action in sequence; the
    aggregate reward and final observation are returned.

    Parameters
    ----------
    env : ClusterEnv (py_environment)
        The underlying scheduling environment.
    B : int
        Batch size — maximum executors to schedule per agent call.
    """

    def __init__(self, env, B: int = 1):
        self.env = env
        self.B = B

        # Episode-level metrics
        self._scheduling_steps: int = 0
        self._total_executors: int = 0
        self._episode_exec_times: List[float] = []

        # History over runs
        self.scheduling_steps_history: List[int] = []
        self.overhead_ratio_history: List[float] = []
        self.exec_time_history: List[float] = []
        self.throughput_history: List[float] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment and episode counters."""
        self._scheduling_steps = 0
        self._total_executors = 0
        ts = self.env._reset()
        return np.array(ts.observation, dtype=np.int32)

    # ------------------------------------------------------------------
    def step(self, actions) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply up to B actions to the environment sequentially.

        Parameters
        ----------
        actions : int or array-like of ints
            Single action or list of up to B actions.

        Returns
        -------
        obs, cumulative_reward, done, info
        """
        if np.isscalar(actions):
            actions = [actions]
        actions = list(actions)[: self.B]

        self._scheduling_steps += 1
        cumulative_reward = 0.0
        done = False
        obs = None

        for action in actions:
            self._total_executors += 1
            ts = self.env._step(int(action))
            cumulative_reward += float(ts.reward)
            obs = np.array(ts.observation, dtype=np.int32)
            if ts.is_last():
                done = True
                break

        if obs is None:
            obs = np.zeros_like(self.env._state, dtype=np.int32)

        info: Dict[str, Any] = {}

        if done:
            self._finalise_episode()
            info["scheduling_steps"] = self._scheduling_steps
            info["overhead_ratio"] = (
                self._scheduling_steps / max(1, self._total_executors)
            )

        return obs, cumulative_reward, done, info

    # ------------------------------------------------------------------
    def _finalise_episode(self) -> None:
        """Compute and store episode-level metrics."""
        overhead = self._scheduling_steps / max(1, self._total_executors)
        exec_time = self.env.calculate_episode_makespan()
        throughput = self.env.calculate_throughput()
        # Accumulate total reward approximation (not tracked here; callers do it)

        self.scheduling_steps_history.append(self._scheduling_steps)
        self.overhead_ratio_history.append(overhead)
        self.exec_time_history.append(exec_time)
        self.throughput_history.append(throughput)

    # ------------------------------------------------------------------
    def get_batch_metrics(self) -> Dict[str, Any]:
        """Return summary statistics of batch metrics across episodes."""
        oh = np.array(self.overhead_ratio_history)
        et = np.array(self.exec_time_history)
        th = np.array(self.throughput_history)

        return {
            "B": self.B,
            "overhead_mean": float(np.mean(oh)) if len(oh) else 0.0,
            "overhead_std": float(np.std(oh)) if len(oh) else 0.0,
            "exec_time_mean": float(np.mean(et)) if len(et) else 0.0,
            "exec_time_std": float(np.std(et)) if len(et) else 0.0,
            "throughput_mean": float(np.mean(th)) if len(th) else 0.0,
            "throughput_std": float(np.std(th)) if len(th) else 0.0,
            "scheduling_steps_total": int(np.sum(self.scheduling_steps_history)),
            "n_episodes": len(self.overhead_ratio_history),
        }
