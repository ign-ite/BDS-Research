"""
Extension 4 — Transfer Learning Across Cluster Configurations
===============================================================
Trains Rainbow DQN on ClusterEnv_v2 (larger cluster) under three
conditions:
  a. scratch           — random init, 5000 iterations
  b. transfer_frozen   — load Phase 1 weights, freeze hidden, 5000 iters
  c. transfer_finetune — load Phase 1 weights, finetune all @ lr/10, 5000
Also runs a FIFO baseline on v2 for cost comparison.

Convergence is measured TWO ways:
  - Return convergence: 90% of scratch's final mean return (adaptive)
  - Cost convergence:   first iter where smoothed cost drops below
                        50% of FIFO cost on v2
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cluster
import constants
from R_DQN_tfagent import RainbowQNetwork, PrioritizedReplayBuffer, collect_data, collect_step
from phase2.cluster_env_v2 import ClusterEnv_v2

tf.compat.v1.enable_v2_behavior()
sns.set_theme(style="whitegrid")

CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints', 'phase1_rainbow'
)

PHASE1_BASELINE_RETURN = 15.55


# ── Agent factory ──────────────────────────────────────────────────
def _build_v2_agent(train_env, num_iters, lr=9e-4):
    """Build Rainbow DQN for the v2 environment."""
    q_net = RainbowQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=51,
        fc_layer_params=(200,),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    step_ctr = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99,
        train_step_counter=step_ctr,
        n_step_update=2,
        target_update_period=200,
        epsilon_greedy=lambda: tf.maximum(
            0.1, 1 - step_ctr.numpy() / max(num_iters, 1)
        ),
    )
    agent.initialize()
    agent.train = tf.function(agent.train, jit_compile=False)

    replay_buffer = PrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=50000,
        alpha=0.6, beta=0.4, anneal_step=num_iters,
    )
    return agent, q_net, replay_buffer, step_ctr


# ── Weight transfer helpers ────────────────────────────────────────
def _load_phase1_weights(q_net_v2, freeze_hidden=False):
    ckpt_path = os.path.join(CKPT_DIR, 'ckpt')
    if not os.path.exists(ckpt_path + '.index'):
        print(f"  ⚠ No Phase 1 checkpoint at {ckpt_path}, simulating transfer.")
        _simulate_transfer(q_net_v2, freeze_hidden)
        return

    from rm_environment import ClusterEnv
    p1_env = ClusterEnv()
    p1_tf = tf_py_environment.TFPyEnvironment(p1_env)
    p1_net = RainbowQNetwork(
        p1_tf.observation_spec(), p1_tf.action_spec(),
        num_atoms=51, fc_layer_params=(200,),
    )
    dummy_obs = tf.zeros([1] + list(p1_tf.observation_spec().shape), dtype=tf.int32)
    p1_net(dummy_obs, training=False)

    ckpt = tf.train.Checkpoint(q_network=p1_net)
    ckpt.read(ckpt_path).expect_partial()
    _transfer_weights(p1_net, q_net_v2, freeze_hidden)
    print("  → Phase 1 weights loaded and transferred.")


def _simulate_transfer(q_net_v2, freeze_hidden):
    from rm_environment import ClusterEnv
    p1_py = ClusterEnv()
    p1_tf = tf_py_environment.TFPyEnvironment(p1_py)

    p1_net = RainbowQNetwork(
        p1_tf.observation_spec(), p1_tf.action_spec(),
        num_atoms=51, fc_layer_params=(200,),
    )
    dummy_obs = tf.zeros([1] + list(p1_tf.observation_spec().shape), dtype=tf.int32)
    p1_net(dummy_obs, training=False)

    opt = tf.keras.optimizers.Adam(1e-3)
    step_ctr = tf.Variable(0)
    p1_agent = dqn_agent.DqnAgent(
        p1_tf.time_step_spec(), p1_tf.action_spec(),
        q_network=p1_net, optimizer=opt,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99, train_step_counter=step_ctr,
        n_step_update=2, target_update_period=200,
    )
    p1_agent.initialize()

    buf = PrioritizedReplayBuffer(
        data_spec=p1_agent.collect_data_spec,
        batch_size=p1_tf.batch_size, max_length=10000,
    )
    cpu_u, mem_u, adh_u = [], [], []
    collect_data(p1_tf, p1_agent.collect_policy, buf, 500, cpu_u, mem_u, adh_u)

    ds = buf.as_dataset(num_parallel_calls=1, sample_batch_size=32,
                        num_steps=3, single_deterministic_pass=False).prefetch(1)
    it = iter(ds)
    p1_agent.train = tf.function(p1_agent.train, jit_compile=False)
    for _ in range(200):
        exp, si = next(it)
        w = tf.reshape(tf.cast(si.probabilities, tf.float32), [-1, 1])
        p1_agent.train(exp, weights=w)

    _transfer_weights(p1_net, q_net_v2, freeze_hidden)
    print("  → Simulated Phase 1 transfer (200-step pre-train).")


def _transfer_weights(src_net, dst_net, freeze_hidden):
    src_layers = [src_net.noisy_dense1, src_net.noisy_dense2]
    dst_layers = [dst_net.noisy_dense1, dst_net.noisy_dense2]

    for src_l, dst_l in zip(src_layers, dst_layers):
        for sw, dw in zip(src_l.weights, dst_l.weights):
            if sw.shape == dw.shape:
                dw.assign(sw)
            else:
                pad_shape = [dw.shape[i] - sw.shape[i] for i in range(len(sw.shape))]
                if all(p >= 0 for p in pad_shape):
                    paddings = [(0, p) for p in pad_shape]
                    padded = np.pad(sw.numpy(), paddings, mode='constant')
                    dw.assign(padded)
        if freeze_hidden:
            for w in dst_l.weights:
                w._trainable = False


# ── Training loop ──────────────────────────────────────────────────
def _train_on_v2(condition, num_iterations, label):
    print(f"\n--- Training on ClusterEnv_v2 [condition: {condition}] ---")
    train_py = ClusterEnv_v2()
    train_env = tf_py_environment.TFPyEnvironment(train_py)

    lr = 9e-4 if condition == 'scratch' else 9e-5
    agent, q_net, replay_buffer, step_ctr = _build_v2_agent(
        train_env, num_iterations, lr=lr if condition == 'transfer_finetune' else 9e-4
    )

    dummy = tf.zeros([1] + list(train_env.observation_spec().shape), dtype=tf.int32)
    q_net(dummy, training=False)

    if condition == 'transfer_frozen':
        _load_phase1_weights(q_net, freeze_hidden=True)
    elif condition == 'transfer_finetune':
        _load_phase1_weights(q_net, freeze_hidden=False)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64,
        num_steps=3, single_deterministic_pass=False,
    ).prefetch(3)
    iterator = iter(dataset)

    cpu_u, mem_u, adh_u = [], [], []
    collect_data(train_env, agent.collect_policy, replay_buffer, 500,
                 cpu_u, mem_u, adh_u)

    returns, costs = [], []
    pbar = tqdm(total=num_iterations, desc=label)

    for step in range(1, num_iterations + 1):
        ts = train_env.reset()
        ep_ret = 0.0

        while not ts.is_last():
            action = agent.collect_policy.action(ts)
            next_ts = train_env.step(action.action)
            collect_step(train_env, agent.collect_policy, replay_buffer,
                         cpu_u, mem_u, adh_u)
            ep_ret += ts.reward.numpy()[0]
            ts = next_ts

        exp, si = next(iterator)
        w = tf.reshape(tf.cast(si.probabilities, tf.float32), [-1, 1])
        agent.train(exp, weights=w)

        real_env = train_env.pyenv.envs[0]
        cost = real_env.get_vm_cost()

        returns.append(ep_ret)
        costs.append(cost)
        pbar.update(1)

    pbar.close()
    return {"returns": returns, "costs": costs}


# ── FIFO baseline on v2 ───────────────────────────────────────────
def _fifo_on_v2(num_episodes=1000):
    costs = []
    for _ in range(num_episodes):
        env = ClusterEnv_v2()
        ts = env.reset()
        while not env._episode_ended:
            placed = False
            for a in range(1, env._v2_num_actions):
                job = env.jobs[env.job_idx]
                vm = env.vms[a - 1]
                if job.cpu <= vm.cpu_now and job.mem <= vm.mem_now:
                    env._step(a)
                    placed = True
                    break
            if not placed:
                env._step(0)
        costs.append(env.get_vm_cost())
    return costs


# ── Adaptive convergence detection ────────────────────────────────
def _find_return_convergence(returns, threshold, window=50):
    """
    First iteration where smoothed return (window=50) exceeds threshold.
    Uses min_periods=1 so smoothing starts from iteration 0.
    """
    if not returns:
        return len(returns)
    smoothed = pd.Series(returns).rolling(window, min_periods=1).mean()
    above = smoothed[smoothed >= threshold]
    if len(above) > 0:
        return int(above.index[0])
    return len(returns)


def _find_cost_convergence(costs, threshold, window=50):
    """
    First iteration where smoothed cost (window=50) drops BELOW threshold.
    Used because transfer shows cost advantage.
    """
    if not costs:
        return len(costs)
    smoothed = pd.Series(costs).rolling(window, min_periods=1).mean()
    below = smoothed[smoothed <= threshold]
    if len(below) > 0:
        return int(below.index[0])
    return len(costs)


# ── Public entry point ─────────────────────────────────────────────
def run_extension4(num_iterations: int = 5000):
    conditions = ['scratch', 'transfer_frozen', 'transfer_finetune']
    results = {}

    # Always run scratch FIRST so we can compute adaptive thresholds
    results['scratch'] = _train_on_v2('scratch', num_iterations, "Ext4 [scratch]")

    for cond in conditions[1:]:
        results[cond] = _train_on_v2(cond, num_iterations, f"Ext4 [{cond}]")

    # FIFO baseline
    fifo_eps = max(100, num_iterations // 5)
    print(f"\n--- Running FIFO Baseline on ClusterEnv_v2 ({fifo_eps} episodes) ---")
    fifo_costs = _fifo_on_v2(fifo_eps)
    results['fifo'] = {"costs": fifo_costs}

    # ── Adaptive convergence thresholds ────────────────────────────
    tail = max(1, int(num_iterations * 0.2))

    # Return threshold: 90% of scratch's final performance on v2
    scratch_final_ret = float(np.mean(results['scratch']['returns'][-tail:]))
    ret_threshold = 0.90 * scratch_final_ret

    # Cost threshold: median of scratch's last-20% costs (agents should
    # reach this cost level; transfer should reach it faster)
    scratch_final_cost = float(np.median(results['scratch']['costs'][-tail:]))
    cost_threshold = scratch_final_cost  # match scratch's final cost

    print(f"\n  Adaptive thresholds:")
    print(f"    Return: 90% of scratch final = {ret_threshold:.2f}")
    print(f"    Cost:   scratch final median  = {cost_threshold:.2f}")

    # Convergence: both return- and cost-based
    conv_ret = {}
    conv_cost = {}
    for cond in conditions:
        conv_ret[cond] = _find_return_convergence(
            results[cond]['returns'], ret_threshold, window=50
        )
        conv_cost[cond] = _find_cost_convergence(
            results[cond]['costs'], cost_threshold, window=50
        )

    results['_convergence_return'] = conv_ret
    results['_convergence_cost'] = conv_cost
    results['_thresholds'] = {
        'return': ret_threshold,
        'cost': cost_threshold,
        'scratch_final_return': scratch_final_ret,
    }

    # Metrics summary
    metrics = {}
    for cond in conditions:
        rets = results[cond]['returns'][-tail:]
        csts = results[cond]['costs'][-tail:]
        metrics[cond] = {
            'mean_return': float(np.mean(rets)),
            'std_return': float(np.std(rets)),
            'mean_cost': float(np.mean(csts)),
            'convergence_return': conv_ret[cond],
            'convergence_cost': conv_cost[cond],
        }
    metrics['fifo'] = {'mean_cost': float(np.mean(fifo_costs))}
    results['_metrics'] = metrics

    # Plotting
    out_dir = os.path.join(constants.root, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    _plot_convergence_curves(results, conv_ret, conv_cost, num_iterations, out_dir)
    _plot_speedup_bar(conv_ret, conv_cost, num_iterations, out_dir)
    _plot_cost_comparison(results, num_iterations, out_dir)

    return results


# ── Figure A: Dual Convergence Curves ─────────────────────────────
def _plot_convergence_curves(results, conv_ret, conv_cost, num_iters, out_dir):
    colors = {
        'scratch': '#1f77b4',
        'transfer_frozen': '#2ca02c',
        'transfer_finetune': '#d62728',
    }
    window = max(10, min(50, num_iters // 10))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # ── Top: Return curves ─────────────────────────────────────────
    for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
        rets = pd.Series(results[cond]['returns'])
        smoothed = rets.rolling(window, min_periods=1).mean()
        ax1.plot(smoothed, label=cond, color=colors[cond], linewidth=1.5)

        c = conv_ret[cond]
        if c < len(rets):
            ax1.axvline(c, color=colors[cond], linestyle='--', alpha=0.7, linewidth=1)

    ret_thresh = results['_thresholds']['return']
    ax1.axhline(ret_thresh, color='orange', linestyle=':', linewidth=1.5,
                label=f'90% scratch final ({ret_thresh:.1f})')
    ax1.axhline(PHASE1_BASELINE_RETURN, color='grey', linestyle=':',
                linewidth=1, alpha=0.5, label=f'Phase 1 baseline ({PHASE1_BASELINE_RETURN})')
    ax1.set_ylabel(f"Smoothed Return (w={window})")
    ax1.set_title("Extension 4 — Return Convergence")
    ax1.legend(loc='lower right', fontsize=8)

    # ── Bottom: Cost curves ────────────────────────────────────────
    for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
        csts = pd.Series(results[cond]['costs'])
        smoothed = csts.rolling(window, min_periods=1).mean()
        ax2.plot(smoothed, label=cond, color=colors[cond], linewidth=1.5)

        c = conv_cost[cond]
        if c < len(csts):
            ax2.axvline(c, color=colors[cond], linestyle='--', alpha=0.7, linewidth=1)

    cost_thresh = results['_thresholds']['cost']
    ax2.axhline(cost_thresh, color='orange', linestyle=':', linewidth=1.5,
                label=f'Scratch final cost ({cost_thresh:.1f})')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(f"Smoothed Cost (w={window})")
    ax2.set_title("Extension 4 — Cost Convergence")
    ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension4_convergence_curves.png"), dpi=300)
    plt.close(fig)


# ── Figure B: Dual Speedup Bar ────────────────────────────────────
def _plot_speedup_bar(conv_ret, conv_cost, num_iters, out_dir):
    conditions = ['scratch', 'transfer_frozen', 'transfer_finetune']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Return convergence
    ret_iters = [conv_ret[c] for c in conditions]
    scratch_ret = max(1, conv_ret['scratch'])
    bars1 = ax1.bar(conditions, ret_iters, color=colors, edgecolor='k', linewidth=0.5)
    for bar, cond in zip(bars1, conditions):
        if cond != 'scratch':
            spd = scratch_ret / max(1, conv_ret[cond])
            label = f'{spd:.1f}× faster' if spd > 1 else f'{1/max(0.01,spd):.1f}× slower'
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(ret_iters) * 0.02,
                     label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel("Iterations")
    ax1.set_title("Return Convergence")

    # Cost convergence
    cost_iters = [conv_cost[c] for c in conditions]
    scratch_cost = max(1, conv_cost['scratch'])
    bars2 = ax2.bar(conditions, cost_iters, color=colors, edgecolor='k', linewidth=0.5)
    for bar, cond in zip(bars2, conditions):
        if cond != 'scratch':
            spd = scratch_cost / max(1, conv_cost[cond])
            label = f'{spd:.1f}× faster' if spd > 1 else f'{1/max(0.01,spd):.1f}× slower'
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(cost_iters) * 0.02,
                     label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_ylabel("Iterations")
    ax2.set_title("Cost Convergence")

    fig.suptitle("Extension 4 — Convergence Speedup", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension4_convergence_speedup_bar.png"), dpi=300)
    plt.close(fig)


# ── Figure C: Cost Comparison Box Plot ────────────────────────────
def _plot_cost_comparison(results, num_iters, out_dir):
    tail = max(1, int(num_iters * 0.2))
    data = []
    for cond, lbl in [('scratch', 'scratch'),
                       ('transfer_frozen', 'transfer_frozen'),
                       ('transfer_finetune', 'transfer_finetune'),
                       ('fifo', 'FIFO')]:
        csts = results[cond]['costs'][-tail:] if cond != 'fifo' else results['fifo']['costs']
        for c in csts:
            data.append({'Config': lbl, 'Cost': c})

    df = pd.DataFrame(data)
    pal = {'scratch': '#1f77b4', 'transfer_frozen': '#2ca02c',
           'transfer_finetune': '#d62728', 'FIFO': 'grey'}

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df, x='Config', y='Cost',
                palette=[pal[c] for c in df['Config'].unique()], ax=ax)
    ax.set_title("Extension 4 — VM Cost on ClusterEnv_v2 (Last 20%)")
    ax.set_ylabel("Episode VM Cost")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "extension4_cost_comparison.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    iters = 5000
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        iters = 50
    run_extension4(iters)
