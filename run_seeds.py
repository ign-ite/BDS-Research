"""
Multi-Seed Experiment Runner
==============================
Runs Phase 2 extensions across multiple random seeds for statistical
significance.  Results are saved incrementally — completed seeds are
preserved if the process is interrupted.

Usage:
  python run_seeds.py --experiment 1 --seeds 0,1,2,3,4 --iterations 10000
  python run_seeds.py --experiment all --seeds 0,1,2 --iterations 5000
  python run_seeds.py --experiment 1 --seeds 0,1,2 --smoke-test
"""

import argparse
import json
import os
import sys
import random
import csv
import time

# ── GPU Setup: ensure conda CUDA libs are found ───────────────────
_conda_env = os.path.dirname(os.path.dirname(sys.executable))
_cuda_lib = os.path.join(_conda_env, 'lib')
if os.path.isdir(_cuda_lib):
    os.environ['LD_LIBRARY_PATH'] = _cuda_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# GPU JIT fix: point XLA to libdevice and configure GPU correctly
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # deterministic numerics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress INFO/WARNING (TensorRT etc.)

import numpy as np
import tensorflow as tf

# GPU setup — cap memory to prevent system crashes
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        # Limit to 4 GB of 16 GB — keeps OS/display very responsive
        tf.config.set_logical_device_configuration(
            _gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError:
        pass  # already configured
    tf.config.optimizer.set_jit(False)
    print(f"[GPU] {_gpus[0].name} — 8 GB limit, XLA JIT off")
else:
    print("[GPU] No GPU detected — running on CPU")

# Paths
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('src/phase2'))

RESULTS_ROOT = os.path.join('results', 'phase2', 'seeds')


def set_all_seeds(seed: int):
    """Set ALL random seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  [seed] All random seeds set to {seed}")


def _seed_dir(ext: int, seed: int) -> str:
    d = os.path.join(RESULTS_ROOT, f'ext{ext}', f'seed{seed}')
    os.makedirs(d, exist_ok=True)
    return d


def _seed_done(ext: int, seed: int) -> bool:
    """Check if a seed run already has saved results (resume support)."""
    d = _seed_dir(ext, seed)
    return os.path.exists(os.path.join(d, 'metrics.json'))


# ── Extractors: convert runner results → metrics + learning curve ──

def _extract_ext1(results: dict, num_iters: int) -> dict:
    """Extract metrics from run_extension1 results."""
    tail = max(1, int(num_iters * 0.2))
    eval_iters = max(100, num_iters // 5)
    tail_bw = max(1, int(eval_iters * 0.2))

    metrics = {}
    curves = {}
    for mode in ['fixed_cost', 'fixed_balanced']:
        r = results[mode]
        rets, costs, adhs = r['returns'][-tail:], r['costs'][-tail:], r['adherences'][-tail:]
        metrics[mode] = {
            'mean_return': float(np.mean(rets)),
            'std_return': float(np.std(rets)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'adherence_pct': float(np.mean(adhs) * 100),
        }
        curves[mode] = list(zip(range(len(r['returns'])), r['returns'], r['costs']))

    bw = results['bayesian_w*']
    bw_rets = bw['returns'][-tail_bw:]
    bw_costs = bw['costs'][-tail_bw:]
    bw_adhs  = bw['adherences'][-tail_bw:]
    metrics['bayesian_w*'] = {
        'mean_return': float(np.mean(bw_rets)),
        'std_return': float(np.std(bw_rets)),
        'mean_cost': float(np.mean(bw_costs)),
        'std_cost': float(np.std(bw_costs)),
        'adherence_pct': float(np.mean(bw_adhs) * 100),
    }
    curves['bayesian_w*'] = list(zip(
        range(len(bw['returns'])), bw['returns'], bw['costs']
    ))

    # Store searcher summary if available
    if '_searcher' in results:
        metrics['_searcher'] = results['_searcher']

    return metrics, curves


def _extract_ext2(results: dict, num_iters: int) -> dict:
    tail = max(1, int(num_iters * 0.2))
    metrics = {}
    curves = {}
    for mode in ['static', 'spot', 'fifo']:
        r = results[mode]
        costs = r['costs'][-tail:]
        metrics[mode] = {
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
        }
        curves[mode] = list(zip(range(len(r['costs'])), r['costs']))
    return metrics, curves


def _extract_ext3(results: dict, num_iters: int) -> dict:
    metrics = {}
    curves = {}
    for B in [1, 2, 4, 8]:
        m = results[B]['metrics']
        metrics[f'B={B}'] = {
            'overhead_mean': float(m['overhead_mean']),
            'overhead_std': float(m['overhead_std']),
        }
        curves[f'B={B}'] = []  # ext3 doesn't have per-iteration curves
    return metrics, curves


def _extract_ext4(results: dict, num_iters: int) -> dict:
    tail = max(1, int(num_iters * 0.2))
    metrics = {}
    curves = {}
    for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
        r = results[cond]
        rets, costs = r['returns'][-tail:], r['costs'][-tail:]
        metrics[cond] = {
            'mean_return': float(np.mean(rets)),
            'std_return': float(np.std(rets)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
        }
        curves[cond] = list(zip(range(len(r['returns'])), r['returns'], r['costs']))

    if '_convergence_return' in results:
        metrics['_convergence_return'] = {
            k: int(v) for k, v in results['_convergence_return'].items()
        }
    if '_convergence_cost' in results:
        metrics['_convergence_cost'] = {
            k: int(v) for k, v in results['_convergence_cost'].items()
        }
    if '_thresholds' in results:
        metrics['_thresholds'] = {
            k: float(v) for k, v in results['_thresholds'].items()
        }

    fifo_costs = results.get('fifo', {}).get('costs', [])
    if fifo_costs:
        metrics['fifo'] = {
            'mean_cost': float(np.mean(fifo_costs)),
            'std_cost': float(np.std(fifo_costs)),
        }

    return metrics, curves


def _save_results(ext: int, seed: int, metrics: dict, curves: dict):
    """Save metrics.json and learning_curve.csv for one seed."""
    d = _seed_dir(ext, seed)

    with open(os.path.join(d, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save learning curves (flatten all variants into one CSV)
    csv_path = os.path.join(d, 'learning_curve.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'iteration', 'return', 'cost'])
        for variant, rows in curves.items():
            if variant.startswith('_'):
                continue
            for row in rows:
                if len(row) == 3:
                    writer.writerow([variant, row[0], row[1], row[2]])
                elif len(row) == 2:
                    writer.writerow([variant, row[0], '', row[1]])

    print(f"  → Saved: {d}/metrics.json + learning_curve.csv")


# ── Per-extension runners ─────────────────────────────────────────

def run_ext1_seed(seed: int, num_iters: int):
    if _seed_done(1, seed):
        print(f"  Ext1 seed{seed}: already done, skipping.")
        return
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    set_all_seeds(seed)
    from phase2.run_extension1 import run_extension1
    results = run_extension1(num_iterations=num_iters)
    metrics, curves = _extract_ext1(results, num_iters)
    _save_results(1, seed, metrics, curves)


def run_ext2_seed(seed: int, num_iters: int):
    if _seed_done(2, seed):
        print(f"  Ext2 seed{seed}: already done, skipping.")
        return
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    set_all_seeds(seed)
    from phase2.run_extension2 import run_extension2
    results = run_extension2(num_iterations=num_iters)
    metrics, curves = _extract_ext2(results, num_iters)
    _save_results(2, seed, metrics, curves)


def run_ext3_seed(seed: int, num_iters: int):
    if _seed_done(3, seed):
        print(f"  Ext3 seed{seed}: already done, skipping.")
        return
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    set_all_seeds(seed)
    from phase2.run_extension3 import run_extension3
    results = run_extension3(num_iterations=num_iters)
    metrics, curves = _extract_ext3(results, num_iters)
    _save_results(3, seed, metrics, curves)


def run_ext4_seed(seed: int, num_iters: int):
    if _seed_done(4, seed):
        print(f"  Ext4 seed{seed}: already done, skipping.")
        return
    import utilities, workload, cluster
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    set_all_seeds(seed)
    from phase2.run_extension4 import run_extension4
    results = run_extension4(num_iterations=num_iters)
    metrics, curves = _extract_ext4(results, num_iters)
    _save_results(4, seed, metrics, curves)


# ── Main ──────────────────────────────────────────────────────────

EXT_RUNNERS = {1: run_ext1_seed, 2: run_ext2_seed,
               3: run_ext3_seed, 4: run_ext4_seed}

EXT_DEFAULT_ITERS = {1: 10000, 2: 10000, 3: 5000, 4: 5000}

def main():
    parser = argparse.ArgumentParser(description='Multi-Seed Phase 2 Runner')
    parser.add_argument('--experiment', type=str, default='all',
                        help='Extension number (1,2,3,4) or "all"')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4',
                        help='Comma-separated seed list')
    parser.add_argument('--iterations', type=int, default=0,
                        help='Iterations per run (0 = use defaults)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick test with minimal iterations')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    if args.experiment == 'all':
        exts = [1, 2, 3, 4]
    else:
        exts = [int(e.strip()) for e in args.experiment.split(',')]

    print("="*60)
    print(f"Multi-Seed Runner: exts={exts}, seeds={seeds}")
    print("="*60)

    for ext in exts:
        default_iters = EXT_DEFAULT_ITERS[ext]
        if args.smoke_test:
            iters = 100 if ext in [1, 2] else 50
        elif args.iterations > 0:
            iters = args.iterations
        else:
            iters = default_iters

        runner = EXT_RUNNERS[ext]
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Extension {ext}, Seed {seed}, Iterations {iters}")
            print(f"{'='*50}")
            t0 = time.time()
            runner(seed, iters)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed/60:.1f} minutes")

    print("\n" + "="*60)
    print("All seed runs complete.")
    print(f"Results saved to: {RESULTS_ROOT}")
    print("Run: python aggregate_seeds.py  to compute statistics.")
    print("="*60)


if __name__ == '__main__':
    main()
