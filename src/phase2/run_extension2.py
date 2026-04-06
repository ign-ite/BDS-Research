"""
Extension 2 — Stochastic VM Pricing / Spot Instances Runner
============================================================
Trains Rainbow DQN under static and spot pricing.
Evaluates FIFO scheduler under spot pricing.
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import copy
from tqdm import tqdm

from tf_agents.environments import tf_py_environment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cluster
import constants
from rm_environment import ClusterEnv
from R_DQN_tfagent import RainbowQNetwork, PrioritizedReplayBuffer, collect_data, collect_step
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from baseline_schedulers import FIFOScheduler

from phase2.pricing_model import PricingModel

tf.compat.v1.enable_v2_behavior()
sns.set_theme(style="whitegrid")


def train_rainbow_dqn_ext2(
    pricing_mode: str,
    num_iterations: int = 10000,
) -> Dict:
    print(f"\n--- Training Rainbow DQN [pricing_mode: {pricing_mode}] ---")
    
    pm = PricingModel(mode=pricing_mode, vms=copy.deepcopy(cluster.VMS))
    
    def env_factory():
        return ClusterEnv(pricing_model=pm)
        
    train_py_env = env_factory()
    eval_py_env = env_factory()
    
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    
    q_net = RainbowQNetwork(
        train_env.observation_spec(), 
        train_env.action_spec(), 
        num_atoms=51, 
        fc_layer_params=(200,)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=9e-4)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99,
        train_step_counter=train_step_counter,
        n_step_update=2,
        target_update_period=200,
        epsilon_greedy=lambda: tf.maximum(0.1, 1 - train_step_counter.numpy() / num_iterations)
    )
    agent.initialize()
    agent.train = tf.function(agent.train, jit_compile=False)

    replay_buffer = PrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=50000,
        alpha=0.6,
        beta=0.4,
        anneal_step=num_iterations
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=3, single_deterministic_pass=False).prefetch(3)
    iterator = iter(dataset)

    cpu_util, mem_util, adherence = [], [], []
    
    print("Collecting initial data...")
    collect_data(train_env, agent.collect_policy, replay_buffer, 500, cpu_util, mem_util, adherence)

    costs = []
    
    pbar = tqdm(total=num_iterations, desc=f"Ext2/Rainbow [{pricing_mode}]")
    for step in range(1, num_iterations + 1):
        time_step = train_env.reset()
        
        while not time_step.is_last():
            action_step = agent.collect_policy.action(time_step)
            next_time_step = train_env.step(action_step.action)
            collect_step(train_env, agent.collect_policy, replay_buffer, cpu_util, mem_util, adherence)
            time_step = next_time_step

        experience, sample_info = next(iterator)
        weights = tf.reshape(tf.cast(sample_info.probabilities, tf.float32), [-1, 1])
        agent.train(experience, weights=weights)

        # Log episode metric — from the REAL env inside the TF wrapper
        cost = train_env.pyenv.envs[0].get_vm_cost()
        costs.append(cost)

        if step % 200 == 0:
            print(f"Step {step}/{num_iterations} | Cost: {cost:.1f}")
        
        pbar.update(1)

    pbar.close()
    return {
        "costs": costs,
        "pricing_history": pm.get_price_history()
    }


def run_fifo_spot(num_iterations: int = 10000) -> Dict:
    print("\n--- Running FIFO Baseline [pricing_mode: spot] ---")
    pm = PricingModel(mode="spot", vms=copy.deepcopy(cluster.VMS))
    
    # Needs to match the env workflow roughly, but baseline_schedulers 
    # re-schedules over a single job list deterministically.
    # To match 'episodes', we just repeatedly schedule the same initial jobs,
    # but varying the prices per episode.
    
    fifo = FIFOScheduler()
    costs = []
    
    pbar = tqdm(total=num_iterations, desc="Ext2/FIFO [spot]")
    for step in range(1, num_iterations + 1):
        vms = copy.deepcopy(cluster.VMS)
        jobs = copy.deepcopy(cluster.JOBS)
        
        # Step pricing model and apply to VMs
        pm.step(vms)
        
        fifo.schedule(jobs, vms)
        costs.append(fifo.episode_costs[-1])
        
        if step % 200 == 0:
            print(f"FIFO Step {step}/{num_iterations} | Cost: {costs[-1]:.1f}")
        
        pbar.update(1)
            
    pbar.close()
    return {
        "costs": costs,
        "pricing_history": pm.get_price_history()
    }


def run_extension2(num_iterations: int = 10000):
    res_static = train_rainbow_dqn_ext2("static", num_iterations)
    res_spot   = train_rainbow_dqn_ext2("spot", num_iterations)
    res_fifo   = run_fifo_spot(num_iterations)
    
    out_dir = os.path.join(constants.root, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    # --- Figure A: Spot Price Time Series ---
    # use history from spot run
    W = res_spot["pricing_history"]  # (T, 3)
    
    plt.figure(figsize=(10, 6))
    labels = ["VM1", "VM2", "VM3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i in range(3):
        p_trace = W[:, i]
        plt.plot(p_trace, label=labels[i], color=colors[i], alpha=0.8)
        std = np.std(p_trace)
        # simplistic +/- 1 std rolling bound for visualization
        rolling_mean = pd.Series(p_trace).rolling(50, min_periods=1).mean()
        plt.fill_between(range(len(p_trace)), rolling_mean - std, rolling_mean + std, color=colors[i], alpha=0.2)
        
    plt.title("Spot Price Time Series (OU Process)")
    plt.xlabel("Episode")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension2_spot_prices.png"), dpi=300)
    plt.close()

    # --- Figure B: Cumulative VM Cost ---
    # normalize using max cost from cluster to compare fairly
    max_c = cluster.max_episode_cost
    if max_c == 0:
        max_c = 1.0
        
    cum_static = np.cumsum(np.array(res_static["costs"]) / max_c)
    cum_spot   = np.cumsum(np.array(res_spot["costs"]) / max_c)
    cum_fifo   = np.cumsum(np.array(res_fifo["costs"]) / max_c)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cum_static, label="Rainbow-Static", color="#1f77b4", lw=2)
    plt.plot(cum_spot, label="Rainbow-Spot", color="#ff7f0e", lw=2)
    plt.plot(cum_fifo, label="FIFO-Spot", color="grey", lw=2, linestyle='--')
    
    plt.title("Cumulative Normalised VM Cost")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Normalised Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension2_cumulative_cost.png"), dpi=300)
    plt.close()

    # --- Figure C: Cost Distribution Box-plot ---
    tail = int(num_iterations * 0.2)
    data = []
    
    for c in res_static["costs"][-tail:]:
        data.append({"Config": "Rainbow-Static", "Cost": c / max_c})
    for c in res_spot["costs"][-tail:]:
        data.append({"Config": "Rainbow-Spot", "Cost": c / max_c})
    for c in res_fifo["costs"][-tail:]:
        data.append({"Config": "FIFO-Spot", "Cost": c / max_c})
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Config", y="Cost", palette=["#1f77b4", "#ff7f0e", "grey"])
    plt.title("Cost Volatility Distribution (Last 20% Episodes)")
    plt.ylabel("Normalised Episode Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension2_cost_boxplot.png"), dpi=300)
    plt.close()
    
    return {"static": res_static, "spot": res_spot, "fifo": res_fifo}

if __name__ == "__main__":
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()
    
    iters = 10000
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        iters = 50
        
    run_extension2(iters)
