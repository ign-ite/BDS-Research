"""
Extension 3 — Job Batching to Reduce Scheduling Overhead Runner
================================================================
Trains Rainbow DQN for 5000 iterations under various batch sizes
B ∈ {1, 2, 4, 8} and plots the results.
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from tf_agents.environments import tf_py_environment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cluster
import constants
from rm_environment import ClusterEnv
from R_DQN_tfagent import RainbowQNetwork, PrioritizedReplayBuffer, collect_data, collect_step
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from phase2.batch_scheduler import BatchScheduler

tf.compat.v1.enable_v2_behavior()
sns.set_theme(style="whitegrid")


def train_rainbow_dqn_ext3(
    batch_size: int,
    num_iterations: int = 5000,
) -> Dict:
    print(f"\n--- Training Rainbow DQN [batch_size B={batch_size}] ---")
    
    # We maintain a separate environment factory to not bleed state
    def env_factory():
        return ClusterEnv()
    
    train_py_env = env_factory()
    eval_py_env = env_factory()
    
    # Wrap train env with BatchScheduler for tracking overhead
    batch_scheduler = BatchScheduler(train_py_env, B=batch_size)
    
    tf_train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    
    q_net = RainbowQNetwork(
        tf_train_env.observation_spec(), 
        tf_train_env.action_spec(), 
        num_atoms=51, 
        fc_layer_params=(200,)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=9e-4)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        tf_train_env.time_step_spec(),
        tf_train_env.action_spec(),
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
        batch_size=tf_train_env.batch_size,
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
    collect_data(tf_train_env, agent.collect_policy, replay_buffer, 500, cpu_util, mem_util, adherence)

    pbar = tqdm(total=num_iterations, desc=f"Ext3 [B={batch_size}]")
    for step in range(1, num_iterations + 1):
        # Reset via TFPyEnvironment to properly initialise _current_time_step
        time_step = tf_train_env.reset()
        obs = batch_scheduler.reset()
        done = False
        
        while not done:
            # Batch decisions
            actions = []
            for _ in range(batch_size):
                action_step = agent.collect_policy.action(time_step)
                actions.append(action_step.action.numpy()[0])
                
            obs, reward, done, info = batch_scheduler.step(actions)
            
            # Refresh TF time_step from underlying env
            time_step = tf_train_env.current_time_step()

        # Pull a few steps for the buffer
        collect_step(tf_train_env, agent.collect_policy, replay_buffer, cpu_util, mem_util, adherence)

        experience, sample_info = next(iterator)
        weights = tf.reshape(tf.cast(sample_info.probabilities, tf.float32), [-1, 1])
        agent.train(experience, weights=weights)

        if step % 200 == 0:
            bm = batch_scheduler.get_batch_metrics()
            print(f"Step {step}/{num_iterations} | Batch: {batch_size} | Overhead Ratio: {bm['overhead_mean']:.2f}")

        pbar.update(1)

    pbar.close()
    return {"metrics": batch_scheduler.get_batch_metrics(), "scheduler": batch_scheduler}


def run_extension3(num_iterations: int = 5000):
    batch_sizes = [1, 2, 4, 8]
    results = {}

    for B in batch_sizes:
        results[B] = train_rainbow_dqn_ext3(batch_size=B, num_iterations=num_iterations)

    out_dir = os.path.join(constants.root, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    # Compile data
    bar_data = []
    line_data = []
    cdf_data = {}
    
    for B in batch_sizes:
        metrics = results[B]["metrics"]
        sched = results[B]["scheduler"]
        
        bar_data.append({
            "Batch Size": str(B),
            "Overhead Ratio": metrics["overhead_mean"],
            "Std": metrics["overhead_std"]
        })
        
        for t in sched.throughput_history:
            line_data.append({"Batch Size": B, "Throughput": t})
            
        cdf_data[B] = np.sort(sched.exec_time_history)

    # --- Figure A: Overhead Reduction Bar Chart ---
    df_bar = pd.DataFrame(bar_data)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_bar, x="Batch Size", y="Overhead Ratio", color="mediumpurple")
    # Add error bars
    plt.errorbar(x=range(len(batch_sizes)), y=df_bar["Overhead Ratio"], yerr=df_bar["Std"], 
                 fmt="none", c="black", capsize=5)
    plt.title("Scheduling Overhead Ratio vs Batch Size")
    plt.xlabel("Batch Size B")
    plt.ylabel("Overhead Ratio (Calls / Placements)")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension3_overhead_bar.png"), dpi=300)
    plt.close()

    # --- Figure B: Throughput vs Batch Size ---
    df_line = pd.DataFrame(line_data)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_line, x="Batch Size", y="Throughput", marker="o", ci=95, color="mediumpurple")
    plt.title("Job Throughput vs Batch Size")
    plt.xlabel("Batch Size B")
    plt.ylabel("Throughput (tasks/s)")
    plt.xticks(batch_sizes)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension3_throughput_vs_batch.png"), dpi=300)
    plt.close()

    # --- Figure C: Episode Makespan CDF ---
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    line_widths = [2.5, 2.0, 2.0, 2.0]
    line_styles = ["-", "--", "-.", ":"]

    for i, B in enumerate(batch_sizes):
        data_sorted = cdf_data[B]
        if len(data_sorted) < 2:
            continue
        p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        plt.plot(data_sorted, p, lw=line_widths[i], ls=line_styles[i],
                 label=f"B={B}", color=colors[i], alpha=0.85)

    plt.title("Episode Makespan CDF across Batch Sizes")
    plt.xlabel("Episode Makespan (simulated time units)")
    plt.ylabel("Cumulative Fraction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extension3_exec_time_cdf.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "extension3_exec_time_cdf.pdf"), dpi=300)
    plt.close()

    return results

if __name__ == "__main__":
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()
    
    iters = 5000
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        iters = 50
        
    run_extension3(iters)
