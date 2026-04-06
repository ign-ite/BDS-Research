"""
Aggregate Multi-Seed Results + Statistical Significance Testing
================================================================
Loads all seed results per experiment, computes mean ± std,
runs significance tests with Bonferroni correction, and outputs
LaTeX-formatted results tables.

Usage:
  python aggregate_seeds.py
  python aggregate_seeds.py --ext 1   # single extension
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from scipy import stats as sp_stats

RESULTS_ROOT = os.path.join('results', 'phase2', 'seeds')

# Bonferroni correction for 3 primary tests
BONFERRONI_ALPHA = 0.05 / 3  # 0.0167


def _sig_marker(p: float) -> str:
    """Return significance marker with Bonferroni correction."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < BONFERRONI_ALPHA:
        return '*'
    else:
        return 'ns'


def _cohens_d(x, y):
    """Compute Cohen's d effect size (pooled std)."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def load_seed_metrics(ext: int) -> list:
    """Load all seed metrics for an extension. Returns list of dicts."""
    pattern = os.path.join(RESULTS_ROOT, f'ext{ext}', 'seed*', 'metrics.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ⚠ No seed results found for ext{ext} in {pattern}")
        return []
    metrics = []
    for f in files:
        with open(f) as fh:
            metrics.append(json.load(fh))
    return metrics


# ── Per-extension aggregation ─────────────────────────────────────

def aggregate_ext1(seed_metrics: list) -> dict:
    """Aggregate Extension 1 across seeds + t-test."""
    n = len(seed_metrics)
    print(f"\n{'='*50}")
    print(f"Extension 1 — Bayesian Simplex Search ({n} seeds)")
    print(f"{'='*50}")

    variants = ['fixed_cost', 'fixed_balanced', 'bayesian_w*']
    agg = {}

    for v in variants:
        returns = [m[v]['mean_return'] for m in seed_metrics if v in m]
        costs = [m[v]['mean_cost'] for m in seed_metrics if v in m]
        adh = [m[v].get('adherence_pct', 0) for m in seed_metrics if v in m]
        agg[v] = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns, ddof=1)) if len(returns) > 1 else 0,
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs, ddof=1)) if len(costs) > 1 else 0,
            'adherence_pct': float(np.mean(adh)),
            'per_seed_returns': returns,
            'per_seed_costs': costs,
            'n_seeds': len(returns),
        }

    # Statistical test: bayesian_w* vs fixed_balanced (returns)
    bw_rets = agg['bayesian_w*']['per_seed_returns']
    fb_rets = agg['fixed_balanced']['per_seed_returns']

    if len(bw_rets) >= 2 and len(fb_rets) >= 2:
        t_stat, p_val = sp_stats.ttest_ind(bw_rets, fb_rets, equal_var=False)
        d = _cohens_d(bw_rets, fb_rets)
        agg['_test_bss_vs_balanced'] = {
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'sig': _sig_marker(p_val),
        }
        print(f"  t-test (BSS vs balanced): t={t_stat:.3f}, p={p_val:.4f}, "
              f"d={d:.3f}, {_sig_marker(p_val)}")
    else:
        agg['_test_bss_vs_balanced'] = {
            'p_value': 1.0, 'sig': 'ns', 'cohens_d': 0,
            'note': f'insufficient seeds (bw={len(bw_rets)}, fb={len(fb_rets)})',
        }
        print(f"  ⚠ Not enough seeds for t-test")

    _print_table_ext1(agg)
    return agg


def aggregate_ext2(seed_metrics: list) -> dict:
    """Aggregate Extension 2 across seeds + Welch's t-test."""
    n = len(seed_metrics)
    print(f"\n{'='*50}")
    print(f"Extension 2 — Spot Pricing ({n} seeds)")
    print(f"{'='*50}")

    agg = {}
    for mode in ['static', 'spot', 'fifo']:
        costs = [m[mode]['mean_cost'] for m in seed_metrics if mode in m]
        agg[mode] = {
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs, ddof=1)) if len(costs) > 1 else 0,
            'per_seed_costs': costs,
            'n_seeds': len(costs),
        }

    # Welch's t-test: spot vs static costs
    spot_c = agg['spot']['per_seed_costs']
    stat_c = agg['static']['per_seed_costs']

    if len(spot_c) >= 2 and len(stat_c) >= 2:
        t_stat, p_val = sp_stats.ttest_ind(spot_c, stat_c, equal_var=False)
        d = _cohens_d(stat_c, spot_c)  # higher static = positive d
        agg['_test_spot_vs_static'] = {
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'sig': _sig_marker(p_val),
        }
        print(f"  Welch t-test (spot vs static cost): t={t_stat:.3f}, "
              f"p={p_val:.4f}, d={d:.3f}, {_sig_marker(p_val)}")
    else:
        agg['_test_spot_vs_static'] = {'p_value': 1.0, 'sig': 'ns'}

    _print_table_ext2(agg)
    return agg


def aggregate_ext3(seed_metrics: list) -> dict:
    """Aggregate Extension 3 across seeds."""
    n = len(seed_metrics)
    print(f"\n{'='*50}")
    print(f"Extension 3 — Job Batching ({n} seeds)")
    print(f"{'='*50}")

    agg = {}
    for B in [1, 2, 4, 8]:
        key = f'B={B}'
        overheads = [m[key]['overhead_mean'] for m in seed_metrics if key in m]
        agg[key] = {
            'overhead_mean': float(np.mean(overheads)),
            'overhead_std': float(np.std(overheads, ddof=1)) if len(overheads) > 1 else 0,
            'n_seeds': len(overheads),
        }

    _print_table_ext3(agg)
    return agg


def aggregate_ext4(seed_metrics: list) -> dict:
    """Aggregate Extension 4 across seeds + paired t-test."""
    n = len(seed_metrics)
    print(f"\n{'='*50}")
    print(f"Extension 4 — Transfer Learning ({n} seeds)")
    print(f"{'='*50}")

    agg = {}
    for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
        returns = [m[cond]['mean_return'] for m in seed_metrics if cond in m]
        costs = [m[cond]['mean_cost'] for m in seed_metrics if cond in m]
        agg[cond] = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns, ddof=1)) if len(returns) > 1 else 0,
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs, ddof=1)) if len(costs) > 1 else 0,
            'per_seed_returns': returns,
            'per_seed_costs': costs,
            'n_seeds': len(returns),
        }

    # Convergence iterations (paired by seed)
    conv_scratch = []
    conv_frozen = []
    for m in seed_metrics:
        cc = m.get('_convergence_cost', {})
        if 'scratch' in cc and 'transfer_frozen' in cc:
            conv_scratch.append(cc['scratch'])
            conv_frozen.append(cc['transfer_frozen'])

    if len(conv_scratch) >= 2:
        t_stat, p_val = sp_stats.ttest_rel(conv_scratch, conv_frozen)
        agg['_test_convergence'] = {
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'sig': _sig_marker(p_val),
            'scratch_iters': conv_scratch,
            'frozen_iters': conv_frozen,
        }
        print(f"  Paired t-test (cost convergence: scratch vs frozen): "
              f"t={t_stat:.3f}, p={p_val:.4f}, {_sig_marker(p_val)}")
    else:
        agg['_test_convergence'] = {'p_value': 1.0, 'sig': 'ns'}

    # FIFO
    fifo_costs = [m['fifo']['mean_cost'] for m in seed_metrics
                  if 'fifo' in m and 'mean_cost' in m.get('fifo', {})]
    if fifo_costs:
        agg['fifo'] = {
            'mean_cost': float(np.mean(fifo_costs)),
            'std_cost': float(np.std(fifo_costs, ddof=1)) if len(fifo_costs) > 1 else 0,
        }

    _print_table_ext4(agg)
    return agg


# ── LaTeX table formatters ────────────────────────────────────────

def _fmt(mean, std, decimals=2):
    """Format mean ± std."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _print_table_ext1(agg):
    test = agg.get('_test_bss_vs_balanced', {})
    sig = test.get('sig', 'ns')
    p = test.get('p_value', 1.0)
    d = test.get('cohens_d', 0)

    print(f"\n% LaTeX Table — Extension 1")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Extension 1: BSS vs Fixed Weights (mean $\pm$ std, " + str(agg['bayesian_w*']['n_seeds']) + " seeds)}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Variant & Mean Return & Mean Cost & Adh. \% \\")
    print(r"\midrule")
    for v in ['fixed\_cost', 'fixed\_balanced', 'bayesian\_w*']:
        v_key = v.replace('\\', '')
        a = agg[v_key]
        adh = f"{a['adherence_pct']:.1f}"
        print(f"  {v} & {_fmt(a['mean_return'], a['std_return'])} & "
              f"{_fmt(a['mean_cost'], a['std_cost'])} & {adh} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"% BSS vs balanced: p={p:.4f} ({sig}), Cohen's d={d:.3f}")
    print(r"\end{table}")


def _print_table_ext2(agg):
    test = agg.get('_test_spot_vs_static', {})
    sig = test.get('sig', 'ns')
    p = test.get('p_value', 1.0)

    print(f"\n% LaTeX Table — Extension 2")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Extension 2: Spot Pricing (mean $\pm$ std, " + str(agg['spot']['n_seeds']) + " seeds)}")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"Variant & Mean Cost \\")
    print(r"\midrule")
    for mode, label in [('static', 'Rainbow-static'), ('spot', 'Rainbow-spot'), ('fifo', 'FIFO-spot')]:
        a = agg[mode]
        print(f"  {label} & {_fmt(a['mean_cost'], a['std_cost'])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"% Spot vs static: p={p:.4f} ({sig})")
    print(r"\end{table}")


def _print_table_ext3(agg):
    print(f"\n% LaTeX Table — Extension 3")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Extension 3: Job Batching Overhead (" + str(agg['B=1']['n_seeds']) + " seeds)}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Batch Size & Overhead Ratio & Reduction \\")
    print(r"\midrule")
    for B in [1, 2, 4, 8]:
        a = agg[f'B={B}']
        reduction = (1 - a['overhead_mean']) * 100
        print(f"  B={B} & {_fmt(a['overhead_mean'], a['overhead_std'])} & "
              f"{reduction:.0f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _print_table_ext4(agg):
    test = agg.get('_test_convergence', {})
    sig = test.get('sig', 'ns')
    p = test.get('p_value', 1.0)

    print(f"\n% LaTeX Table — Extension 4")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Extension 4: Transfer Learning (" + str(agg['scratch']['n_seeds']) + " seeds)}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Condition & Mean Return & Mean Cost & Cost $\Delta$ \\")
    print(r"\midrule")
    scratch_cost = agg['scratch']['mean_cost']
    for cond, label in [('scratch', 'Scratch'), ('transfer\\_frozen', 'Frozen'),
                        ('transfer\\_finetune', 'Finetune')]:
        c_key = cond.replace('\\', '')
        a = agg[c_key]
        delta = ((a['mean_cost'] - scratch_cost) / scratch_cost * 100)
        delta_str = f"{delta:+.1f}\\%"
        print(f"  {label} & {_fmt(a['mean_return'], a['std_return'])} & "
              f"{_fmt(a['mean_cost'], a['std_cost'])} & {delta_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"% Cost convergence (scratch vs frozen): p={p:.4f} ({sig})")
    print(r"\end{table}")


# ── Save aggregated results ───────────────────────────────────────

def _save_aggregated(ext: int, agg: dict):
    """Save aggregated.json for an extension."""
    out_dir = os.path.join(RESULTS_ROOT, f'ext{ext}')
    os.makedirs(out_dir, exist_ok=True)

    # Remove per_seed lists for cleaner JSON (keep only stats)
    clean = {}
    for k, v in agg.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items()
                        if not kk.startswith('per_seed')}
        else:
            clean[k] = v

    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"  → Saved {out_dir}/aggregated.json")


# ── Main ──────────────────────────────────────────────────────────

AGGREGATORS = {
    1: aggregate_ext1,
    2: aggregate_ext2,
    3: aggregate_ext3,
    4: aggregate_ext4,
}


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed results')
    parser.add_argument('--ext', type=str, default='all',
                        help='Extension number or "all"')
    args = parser.parse_args()

    exts = [1, 2, 3, 4] if args.ext == 'all' else [int(args.ext)]

    print("="*60)
    print("Aggregating Multi-Seed Results")
    print(f"Bonferroni-corrected α = {BONFERRONI_ALPHA:.4f}")
    print("="*60)

    all_agg = {}
    for ext in exts:
        seed_metrics = load_seed_metrics(ext)
        if not seed_metrics:
            continue
        agg = AGGREGATORS[ext](seed_metrics)
        _save_aggregated(ext, agg)
        all_agg[ext] = agg

    # Summary of significance
    if all_agg:
        print(f"\n{'='*60}")
        print("Statistical Significance Summary")
        print(f"(Bonferroni α = {BONFERRONI_ALPHA:.4f})")
        print(f"{'='*60}")

        tests = [
            (1, '_test_bss_vs_balanced', 'BSS w* vs balanced (return)'),
            (2, '_test_spot_vs_static', 'Spot vs static (cost)'),
            (4, '_test_convergence', 'Transfer convergence (cost)'),
        ]
        for ext, key, desc in tests:
            if ext in all_agg and key in all_agg[ext]:
                t = all_agg[ext][key]
                print(f"  {desc}: p={t.get('p_value',1):.4f} {t.get('sig','ns')}")
            else:
                print(f"  {desc}: not available")


if __name__ == '__main__':
    main()
