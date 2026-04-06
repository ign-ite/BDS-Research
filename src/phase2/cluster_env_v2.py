"""
ClusterEnv v2 — Scaled cluster for transfer-learning experiments
=================================================================
- 40 % more VMs (9 → 13):  4×vm1, 4×vm2, 5×vm3
- Job CPU/mem demands scaled by 1.25×
- Action space expanded: 0 = wait, 1-13 = place on VM 0-12
- Observation vector extended for additional VMs.
"""

import copy
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cluster
import constants
import workload
import definitions as defs
from rm_environment import ClusterEnv

from tf_agents.specs import array_spec


class ClusterEnv_v2(ClusterEnv):
    """
    Larger cluster environment for transfer-learning evaluation.
    Inherits from ClusterEnv and overrides cluster configuration.
    """

    # ── Scaled cluster parameters ──────────────────────────────────
    VM1_TOTAL, VM1_CPU, VM1_MEM, VM1_PRICE = 4, 4, 12, 1
    VM2_TOTAL, VM2_CPU, VM2_MEM, VM2_PRICE = 4, 8, 24, 2
    VM3_TOTAL, VM3_CPU, VM3_MEM, VM3_PRICE = 5, 12, 36, 3
    NUM_VMS = VM1_TOTAL + VM2_TOTAL + VM3_TOTAL  # 13

    JOB_SCALE = 1.25  # scale factor for CPU/mem demands

    def __init__(self, weight_vector=None, pricing_model=None):
        # Build scaled VMs locally (do NOT mutate global cluster module)
        self._v2_vms = []
        for i in range(self.VM1_TOTAL):
            self._v2_vms.append(defs.VM(len(self._v2_vms), self.VM1_CPU, self.VM1_MEM, self.VM1_PRICE))
        for i in range(self.VM2_TOTAL):
            self._v2_vms.append(defs.VM(len(self._v2_vms), self.VM2_CPU, self.VM2_MEM, self.VM2_PRICE))
        for i in range(self.VM3_TOTAL):
            self._v2_vms.append(defs.VM(len(self._v2_vms), self.VM3_CPU, self.VM3_MEM, self.VM3_PRICE))

        # Build scaled jobs
        self._v2_jobs = self._scale_jobs()

        # Compute observation/action dimensions
        batch_size = max(1, int(constants.job_batch_size))
        self._v2_obs_dim = self.NUM_VMS * 2 + 4 * batch_size
        self._v2_num_actions = self.NUM_VMS + 1  # 0=wait, 1..13=place

        # Call PyEnvironment.__init__ directly (skip ClusterEnv.__init__)
        super(ClusterEnv, self).__init__()

        # ── specs (override before any reset) ──────────────────────
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=self._v2_num_actions - 1,
            name='action',
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._v2_obs_dim,), dtype=np.int32,
            minimum=0, name='observation',
        )

        # ── internal state (mirrors ClusterEnv.__init__) ───────────
        self.weight_vector = weight_vector
        self.pricing_model = pricing_model
        self.episode_cost_accum = 0.0
        self._last_episode_cost = 0.0
        self._last_adherence = 0.0

        self.vms = copy.deepcopy(self._v2_vms)
        self.jobs = copy.deepcopy(self._v2_jobs)
        self.job_idx = 0
        self.clock = self.jobs[0].arrival_time if self.jobs else 0
        self.start_time = self.clock

        self._state = self._gen_state(0)
        self._episode_ended = False
        self.episode_success = False
        self.reward = 0
        self.completed_jobs = 0
        self.total_time = 0
        self.good_placement = 0

        self.total_cpu = sum(v.cpu for v in self.vms)
        self.total_mem = sum(v.mem for v in self.vms)
        self.total_cpu_used = 0
        self.total_mem_used = 0

        self.cpu_utilization_history = []
        self.mem_utilization_history = []
        self.last_reward_weights = (
            weight_vector if weight_vector is not None
            else np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        )

        # Episode reward CSV — create a no-op writer
        import io, csv
        self._csv_buf = io.StringIO()
        self.episode_reward_writer = csv.writer(self._csv_buf)

        # Job queue (PriorityQueue)
        from queue import PriorityQueue
        self.job_queue = PriorityQueue()

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _scale_jobs():
        """Deep-copy global workload jobs and scale CPU/mem by 1.25×."""
        scaled = copy.deepcopy(workload.JOBS_WORKLOAD)
        for j in scaled:
            j.cpu = max(1, int(j.cpu * ClusterEnv_v2.JOB_SCALE))
            j.mem = max(1, int(j.mem * ClusterEnv_v2.JOB_SCALE))
        return scaled

    def _gen_state(self, job_idx):
        """Generate observation vector for the v2 cluster."""
        state = []
        for v in self.vms:
            state.append(v.cpu_now)
            state.append(v.mem_now)
        batch_size = max(1, int(constants.job_batch_size))
        for offset in range(batch_size):
            idx = job_idx + offset
            if idx < len(self.jobs):
                state.append(self.jobs[idx].type)
                state.append(self.jobs[idx].cpu)
                state.append(self.jobs[idx].mem)
                state.append(self.jobs[idx].ex - self.jobs[idx].ex_placed)
            else:
                state.extend([0, 0, 0, 0])
        return state

    # ── override specs ─────────────────────────────────────────────
    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    # ── override _reset ────────────────────────────────────────────
    def _reset(self):
        self._last_episode_cost = (
            self.episode_cost_accum if self.episode_cost_accum > 0
            else self._last_episode_cost
        )
        self._last_adherence = self.calculate_deadline_adherence()
        self.episode_cost_accum = 0.0

        self.vms = copy.deepcopy(self._v2_vms)
        self.jobs = copy.deepcopy(self._v2_jobs)
        self.job_idx = 0
        self.clock = self.jobs[0].arrival_time if self.jobs else 0
        self.start_time = self.clock
        self._episode_ended = False
        self.episode_success = False
        self.reward = 0
        self.completed_jobs = 0
        self.total_time = 0
        self.good_placement = 0
        self.total_cpu_used = 0
        self.total_mem_used = 0
        self.cpu_utilization_history = []
        self.mem_utilization_history = []
        self.last_reward_weights = (
            self.weight_vector if self.weight_vector is not None
            else np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        )

        if self.pricing_model is not None:
            self.pricing_model.reset_prices(self.vms)

        from queue import PriorityQueue
        self.job_queue = PriorityQueue()

        self._state = self._gen_state(0)
        from tf_agents.trajectories import time_step as ts
        return ts.restart(np.array(self._state, dtype=np.int32))

    # ── override _step to handle expanded action space ─────────────
    def _step(self, action):
        import logging
        from tf_agents.trajectories import time_step as ts

        if self._episode_ended:
            return self.reset()

        action = int(action)

        # Update VM prices
        if self.pricing_model is not None:
            self.pricing_model.step(self.vms)
        else:
            cluster.update_vm_prices(self.vms)

        if action < 0 or action >= self._v2_num_actions:
            raise ValueError(
                f'action {action} out of range [0, {self._v2_num_actions - 1}]'
            )

        if action == 0:
            # same logic as base ClusterEnv
            current_job = self.jobs[self.job_idx]
            if current_job.ex_placed > 0:
                self.reward = -50
                self._episode_ended = True
            elif self.job_queue.empty():
                self.reward = -200
                self._episode_ended = True
            else:
                self.reward = -1
                _, y = self.job_queue.get()
                self.clock = y.finish_time
                self._finish_one_job(y)
        else:
            vm_idx = action - 1
            if vm_idx >= len(self.vms):
                self.reward = -200
                self._episode_ended = True
            elif self._execute_placement(vm_idx):
                if self._check_enough():
                    self.reward = 1
                else:
                    self.reward = -200
                    self._episode_ended = True
            else:
                self.reward = -200
                self._episode_ended = True

        if self._episode_ended:
            self._compute_terminal_reward()
            return ts.termination(
                np.array(self._state, dtype=np.int32), self.reward
            )
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=self.reward, discount=0.9,
            )

    # ── placement / finish helpers (simplified from base) ──────────
    def _execute_placement(self, vm_idx):
        current_job = self.jobs[self.job_idx]
        vm = self.vms[vm_idx]
        if current_job.cpu > vm.cpu_now or current_job.mem > vm.mem_now:
            return False

        # cost accumulator
        ex_cost = vm.current_price * (current_job.duration / max(current_job.ex, 1))
        self.episode_cost_accum += ex_cost

        vm.cpu_now -= current_job.cpu
        vm.mem_now -= current_job.mem
        self.total_cpu_used += current_job.cpu
        self.total_mem_used += current_job.mem

        current_job.ex_placed += 1
        current_job.ex_placement_list.append(vm_idx)

        if current_job.ex_placed == current_job.ex:
            # Apply placement penalty
            if constants.pp_apply == 'true':
                uniq = len(set(current_job.ex_placement_list))
                if current_job.type == 3:
                    if uniq != 1:
                        current_job.duration += current_job.duration * float(constants.placement_penalty) / 100
                    else:
                        self.good_placement += 1
                else:
                    if uniq < current_job.ex_placed:
                        current_job.duration += current_job.duration * float(constants.placement_penalty) / 100
                    else:
                        self.good_placement += 1

            current_job.running = True
            current_job.start_time = self.clock
            current_job.finish_time = self.clock + current_job.duration

            for i in range(len(current_job.ex_placement_list)):
                vid = current_job.ex_placement_list[i]
                if current_job.start_time > self.vms[vid].stop_use_clock:
                    self.vms[vid].used_time += current_job.duration
                    self.vms[vid].stop_use_clock = current_job.finish_time
                else:
                    if current_job.finish_time > self.vms[vid].stop_use_clock:
                        self.vms[vid].used_time += (
                            current_job.finish_time - self.vms[vid].stop_use_clock
                        )
                        self.vms[vid].stop_use_clock = current_job.finish_time

            self.job_queue.put((current_job.finish_time, current_job))
            if self.job_idx + 1 == len(self.jobs):
                self._episode_ended = True
                self.episode_success = True
            else:
                self.job_idx += 1

        self._state = self._gen_state(self.job_idx)
        self.update_resource_utilization()
        return True

    def _finish_one_job(self, finished_job):
        finished_job.finished = True
        finished_job.running = False
        for vid in finished_job.ex_placement_list:
            vm = self.vms[vid]
            vm.cpu_now += finished_job.cpu
            vm.mem_now += finished_job.mem
            self.completed_jobs += 1
            self.total_cpu_used += finished_job.cpu * finished_job.duration
            self.total_mem_used += finished_job.mem * finished_job.duration
        self._state = self._gen_state(self.job_idx)
        self.update_resource_utilization()

    def _check_enough(self):
        """Check there are enough cluster resources for remaining executors."""
        total_cpu_avail = sum(v.cpu_now for v in self.vms)
        total_mem_avail = sum(v.mem_now for v in self.vms)
        if self.job_idx < len(self.jobs):
            j = self.jobs[self.job_idx]
            needed_ex = j.ex - j.ex_placed
            if needed_ex * j.cpu > total_cpu_avail or needed_ex * j.mem > total_mem_avail:
                return False
        return True

    def _compute_terminal_reward(self):
        """Compute the multi-objective terminal reward (same logic as base)."""
        if not self.episode_success:
            return  # keep the negative reward already set

        cost = self.calculate_vm_cost()
        max_cost = sum(v.price for v in self._v2_vms) * sum(j.duration for j in self._v2_jobs)
        cost_norm = 1 - (cost / max_cost) if max_cost > 0 else 0

        avg_time = self.calculate_avg_time()
        min_avg = sum(j.duration for j in self._v2_jobs) / len(self._v2_jobs)
        max_avg = min_avg + min_avg * float(constants.placement_penalty) / 100
        span = max_avg - min_avg
        time_norm = 1 - ((avg_time - min_avg) / span if span > 0 else 0)

        util = self.calculate_average_utilization()
        thru = self.calculate_throughput()
        last_at = self.jobs[-1].arrival_time if self.jobs else 0
        max_thru = len(self.jobs) / last_at if (self.jobs and last_at > 0) else 1
        thru_norm = thru / max_thru
        adh = self.calculate_deadline_adherence()

        if self.weight_vector is not None:
            w = self.weight_vector
        else:
            w = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.last_reward_weights = w

        self.reward = float(constants.fixed_episodic_reward) * (
            w[0] * cost_norm +
            w[1] * time_norm +
            w[2] * util +
            w[3] * thru_norm +
            w[4] * adh
        )
