"""
Extension 1 — Bayesian Simplex Search (replaces AWO gradient ascent)
=====================================================================
1. Run fixed_cost and fixed_balanced baselines (full iterations).
2. Generate 20 LHS candidates on the 5-simplex.
3. Evaluate each candidate for 500 episodes → select w* = argmax return.
4. Run w* for 2000 evaluation episodes.
5. Generate figures:
   A. extension1_simplex_heatmap.png  (PCA scatter of 20 candidates)
   B. extension1_return_comparison.png (bar chart: 3 configs)
   C. extension1_pareto_scatter.png   (cost vs return, Pareto)
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict
from tqdm import tqdm
from sklearn.decomposition import PCA

from tf_agents.environments import tf_py_environment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cluster
import constants
from rm_environment import ClusterEnv
from R_DQN_tfagent import RainbowQNetwork, PrioritizedReplayBuffer, collect_data, collect_step
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from phase2.bayesian_simplex_search import (
    BayesianSimplexSearch, PRESET_COST_FOCUS, PRESET_BALANCED
)

tf.compat.v1.enable_v2_behavior()
sns.set_theme(style="whitegrid")

# ── Checkpoint helper ───────────────────────────────────────────────
CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints', 'phase1_rainbow'
)


def save_checkpoint(agent, q_net, train_step, path=CKPT_DIR):
    """Save agent weights via tf.train.Checkpoint."""
    os.makedirs(path, exist_ok=True)
    ckpt = tf.train.Checkpoint(
        q_network=q_net,
        train_step=train_step,
    )
    ckpt.write(os.path.join(path, 'ckpt'))


def _build_agent(train_env, num_iterations, lr=9e-4):
    """Build a fresh Rainbow DQN agent + replay buffer."""
    q_net = RainbowQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=51,
        fc_layer_params=(200,),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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
        epsilon_greedy=lambda: tf.maximum(
            0.1, 1 - train_step_counter.numpy() / max(num_iterations, 1)
        ),
    )
    agent.initialize()
    agent.train = tf.function(agent.train, jit_compile=False)

    replay_buffer = PrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=50000,
        alpha=0.6,
        beta=0.4,
        anneal_step=num_iterations,
    )
    return agent, q_net, replay_buffer, train_step_counter


# ── Training loop (shared by baselines and candidates) ──────────────
def _train_loop(
    weight_vector: np.ndarray,
    num_iterations: int,
    label: str,
    save_ckpt: bool = False,
):
    """
    Train Rainbow DQN with a *fixed* weight vector.
    Returns dict with lists: returns, costs, adherences.
    """
    train_py_env = ClusterEnv(weight_vector=weight_vector)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    agent, q_net, replay_buffer, step_ctr = _build_agent(train_env, num_iterations)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64,
        num_steps=3, single_deterministic_pass=False,
    ).prefetch(3)
    iterator = iter(dataset)

    cpu_u, mem_u, adh_buf = [], [], []
    collect_data(train_env, agent.collect_policy, replay_buffer, 500,
                 cpu_u, mem_u, adh_buf)

    returns, costs, adherences = [], [], []
    pbar = tqdm(total=num_iterations, desc=label)

    for step in range(1, num_iterations + 1):
        time_step = train_env.reset()
        ep_ret = 0.0

        # Inject weights into real underlying env
        real_env = train_env.pyenv.envs[0]
        real_env.weight_vector = weight_vector.copy()

        while not time_step.is_last():
            action_step = agent.collect_policy.action(time_step)
            next_ts = train_env.step(action_step.action)
            collect_step(train_env, agent.collect_policy, replay_buffer,
                         cpu_u, mem_u, adh_buf)
            ep_ret += time_step.reward.numpy()[0]
            time_step = next_ts

        experience, sample_info = next(iterator)
        weights = tf.reshape(
            tf.cast(sample_info.probabilities, tf.float32), [-1, 1]
        )
        agent.train(experience, weights=weights)

        real_env = train_env.pyenv.envs[0]
        cost = real_env.get_vm_cost()
        adh  = real_env.get_deadline_adherence()

        returns.append(ep_ret)
        costs.append(cost)
        adherences.append(adh)
        pbar.update(1)

    pbar.close()

    if save_ckpt:
        save_checkpoint(agent, q_net, step_ctr)
        print(f"  → Checkpoint saved to {CKPT_DIR}")

    return {"returns": returns, "costs": costs, "adherences": adherences}


# ── Public entry point ──────────────────────────────────────────────
def run_extension1(num_iterations: int = 10000):
    """
    Phase A: fixed_cost & fixed_balanced   (num_iterations each)
    Phase B: 20 LHS candidates             (500 episodes each)
    Phase C: w* evaluation                 (2000 episodes)
    """
    # Candidate / eval budgets scaled for smoke tests
    cand_iters = max(50, num_iterations // 20)   # 500 for full, 5 for smoke
    eval_iters = max(100, num_iterations // 5)   # 2000 for full, 20 for smoke

    results: Dict = {}

    # ── Phase A: Fixed baselines ────────────────────────────────────
    print("\n--- Training Rainbow DQN [w_mode: fixed_cost] ---")
    results["fixed_cost"] = _train_loop(
        PRESET_COST_FOCUS, num_iterations, "Ext1 [fixed_cost]", save_ckpt=True
    )

    print("\n--- Training Rainbow DQN [w_mode: fixed_balanced] ---")
    results["fixed_balanced"] = _train_loop(
        PRESET_BALANCED, num_iterations, "Ext1 [fixed_balanced]"
    )

    # ── Phase B: Bayesian Simplex Search ────────────────────────────
    print("\n--- Bayesian Simplex Search (20 LHS candidates) ---")
    searcher = BayesianSimplexSearch(n_candidates=20, seed=42)

    for i in range(searcher.n_candidates):
        w = searcher.get_candidate(i)
        w_str = ", ".join(f"{v:.3f}" for v in w)
        print(f"\n  Candidate {i+1}/20: w=[{w_str}]")
        res = _train_loop(w, cand_iters, f"Cand {i+1}/20")
        tail = max(1, len(res["returns"]) // 2)  # last 50%
        mean_ret  = float(np.mean(res["returns"][-tail:]))
        mean_cost = float(np.mean(res["costs"][-tail:]))
        searcher.record_score(i, mean_ret, mean_cost)
        print(f"    → Mean Return: {mean_ret:.2f}  Mean Cost: {mean_cost:.2f}")

    best_w = searcher.select_best()
    best_idx = searcher.best_idx
    print(f"\n  ★ Best candidate: #{best_idx+1}  w={best_w}")
    print(f"    Score: {searcher.scores[best_idx]:.2f}")

    # ── Phase C: Full evaluation of w* ──────────────────────────────
    print(f"\n--- Evaluating w* for {eval_iters} episodes ---")
    results["bayesian_w*"] = _train_loop(
        best_w, eval_iters, "Ext1 [bayesian_w*]"
    )

    # ── Plotting ────────────────────────────────────────────────────
    out_dir = os.path.join(constants.root, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    _plot_simplex_heatmap(searcher, out_dir)
    _plot_return_comparison(results, num_iterations, eval_iters, out_dir)
    _plot_pareto_scatter(results, num_iterations, eval_iters, out_dir)

    # Store searcher summary so run_phase2 can extract metrics
    results["_searcher"] = searcher.summary()
    return results


# ── Figure A: PCA Simplex Heatmap ──────────────────────────────────
def _plot_simplex_heatmap(searcher: BayesianSimplexSearch, out_dir: str):
    W = searcher.candidates                          # (20, 5)
    scores = np.array(searcher.scores)
    pca = PCA(n_components=2)
    W2 = pca.fit_transform(W)                        # (20, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        W2[:, 0], W2[:, 1],
        c=scores, cmap="viridis", s=120, edgecolors="k", linewidths=0.5,
    )
    # Mark w* with red star
    if searcher.best_idx is not None:
        ax.scatter(
            W2[searcher.best_idx, 0], W2[searcher.best_idx, 1],
            marker="*", s=350, c="red", edgecolors="k", zorder=5,
            label=f"w* (#{searcher.best_idx+1})",
        )
    plt.colorbar(sc, label="Mean Return")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Bayesian Simplex Search — LHS Candidates (PCA Projection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension1_simplex_heatmap.png"), dpi=300)
    plt.close(fig)


# ── Figure B: Return Comparison Bar Chart ──────────────────────────
def _plot_return_comparison(results, base_iters, eval_iters, out_dir):
    configs = ["fixed_cost", "fixed_balanced", "bayesian_w*"]
    iters_map = {
        "fixed_cost": base_iters,
        "fixed_balanced": base_iters,
        "bayesian_w*": eval_iters,
    }
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e"]
    bar_data = []
    for cfg in configs:
        n = iters_map[cfg]
        tail = max(1, int(n * 0.2))
        rets = results[cfg]["returns"][-tail:]
        bar_data.append({
            "Config": cfg,
            "Mean Return": np.mean(rets),
            "Std": np.std(rets),
        })

    df = pd.DataFrame(bar_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df["Config"], df["Mean Return"], color=colors,
                  edgecolor="k", linewidth=0.5)
    ax.errorbar(range(3), df["Mean Return"], yerr=df["Std"],
                fmt="none", c="black", capsize=5)
    ax.set_title("Extension 1 — Return Comparison (Last 20%)")
    ax.set_ylabel("Mean Episode Return")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension1_return_comparison.png"), dpi=300)
    plt.close(fig)


# ── Figure C: Pareto Scatter ───────────────────────────────────────
def _plot_pareto_scatter(results, base_iters, eval_iters, out_dir):
    configs = ["fixed_cost", "fixed_balanced", "bayesian_w*"]
    iters_map = {
        "fixed_cost": base_iters,
        "fixed_balanced": base_iters,
        "bayesian_w*": eval_iters,
    }
    colors = {"fixed_cost": "#1f77b4", "fixed_balanced": "#2ca02c",
              "bayesian_w*": "#ff7f0e"}
    markers = {"fixed_cost": "o", "fixed_balanced": "s", "bayesian_w*": "*"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg in configs:
        n = iters_map[cfg]
        tail = max(1, int(n * 0.2))
        mean_cost = np.mean(results[cfg]["costs"][-tail:])
        mean_ret  = np.mean(results[cfg]["returns"][-tail:])
        norm_cost = mean_cost / cluster.max_episode_cost if cluster.max_episode_cost > 0 else mean_cost
        sz = 250 if cfg == "bayesian_w*" else 150
        ax.scatter([norm_cost], [mean_ret], s=sz, marker=markers[cfg],
                   c=colors[cfg], edgecolors="k", label=cfg, zorder=5)
        ax.annotate(cfg, (norm_cost, mean_ret), xytext=(8, 5),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Normalised VM Cost (Lower → Better)")
    ax.set_ylabel("Mean Return (Higher → Better)")
    ax.set_title("Extension 1 — Pareto Scatter (Cost vs Return)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension1_pareto_scatter.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    iters = 10000
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        iters = 100
    run_extension1(iters)
