"""
Phase 2 Top-level Orchestrator
===============================
Runs all four Phase 2 extensions + ablation study and outputs summary.
Pass '--smoke-test' for a rapid verification run.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('src/phase2'))

import utilities, workload, cluster
from phase2.run_extension1 import run_extension1
from phase2.run_extension2 import run_extension2
from phase2.run_extension3 import run_extension3
from phase2.run_extension4 import run_extension4
from phase2.run_ablation import run_ablation

def main():
    smoke_test = len(sys.argv) > 1 and sys.argv[1] == "--smoke-test"
    
    iters_ext12 = 100 if smoke_test else 10000
    iters_ext3 = 50 if smoke_test else 5000
    iters_ext4 = 50 if smoke_test else 5000
    
    print("="*60)
    print(f"Starting Phase 2 Batch Experiments (Smoke Test: {smoke_test})")
    print("="*60)
    
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    print("\n>>> Running Extension 1: Bayesian Simplex Search")
    res1 = run_extension1(num_iterations=iters_ext12)
    
    print("\n>>> Running Extension 2: Stochastic VM Pricing (Spot Instances)")
    res2 = run_extension2(num_iterations=iters_ext12)
    
    print("\n>>> Running Extension 3: Job Batching")
    res3 = run_extension3(num_iterations=iters_ext3)

    print("\n>>> Running Extension 4: Transfer Learning Across Clusters")
    res4 = run_extension4(num_iterations=iters_ext4)

    print("\n>>> Running Ablation Study")
    abl = run_ablation(res1, res2, res4,
                       num_iters_ext1=iters_ext12,
                       num_iters_ext4=iters_ext4)

    print("\n" + "="*60)
    print("Phase 2 Experiments Complete. Generating Summary...")
    print("="*60)

    # Build Summary DataFrame
    summary = []
    
    # ── Ext 1 ──────────────────────────────────────────────────────
    tail1 = max(1, int(iters_ext12 * 0.2))
    for mode in ["fixed_cost", "fixed_balanced"]:
        rets = res1[mode]["returns"][-tail1:]
        costs = res1[mode]["costs"][-tail1:]
        adhs = res1[mode]["adherences"][-tail1:]
        summary.append({
            "Experiment": "Ext 1 (BSS)",
            "Variant": mode,
            "Mean Return": np.mean(rets),
            "Mean Cost": np.mean(costs),
            "Adherence %": np.mean(adhs) * 100,
            "Overhead Ratio": 1.0
        })

    eval_iters = max(100, iters_ext12 // 5)
    tail_bw = max(1, int(eval_iters * 0.2))
    bw_rets = res1["bayesian_w*"]["returns"][-tail_bw:]
    bw_costs = res1["bayesian_w*"]["costs"][-tail_bw:]
    bw_adhs = res1["bayesian_w*"]["adherences"][-tail_bw:]
    summary.append({
        "Experiment": "Ext 1 (BSS)",
        "Variant": "bayesian_w*",
        "Mean Return": np.mean(bw_rets),
        "Mean Cost": np.mean(bw_costs),
        "Adherence %": np.mean(bw_adhs) * 100,
        "Overhead Ratio": 1.0
    })

    # ── Ext 2 ──────────────────────────────────────────────────────
    tail2 = max(1, int(iters_ext12 * 0.2))
    for mode in ["static", "spot", "fifo"]:
        costs = res2[mode]["costs"][-tail2:]
        summary.append({
            "Experiment": "Ext 2 (Pricing)",
            "Variant": f"Rainbow-{mode}" if mode != "fifo" else "FIFO-spot",
            "Mean Return": np.nan,
            "Mean Cost": np.mean(costs),
            "Adherence %": np.nan,
            "Overhead Ratio": 1.0
        })

    # ── Ext 3 ──────────────────────────────────────────────────────
    for b in [1, 2, 4, 8]:
        metrics = res3[b]["metrics"]
        summary.append({
            "Experiment": "Ext 3 (Batching)",
            "Variant": f"B={b}",
            "Mean Return": np.nan,
            "Mean Cost": np.nan,
            "Adherence %": np.nan,
            "Overhead Ratio": metrics["overhead_mean"]
        })

    # ── Ext 4 ──────────────────────────────────────────────────────
    ext4_metrics = res4.get('_metrics', {})
    for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
        m = ext4_metrics.get(cond, {})
        summary.append({
            "Experiment": "Ext 4 (Transfer)",
            "Variant": cond,
            "Mean Return": m.get('mean_return', np.nan),
            "Mean Cost": m.get('mean_cost', np.nan),
            "Adherence %": np.nan,
            "Overhead Ratio": np.nan,
        })
    fifo_m = ext4_metrics.get('fifo', {})
    summary.append({
        "Experiment": "Ext 4 (Transfer)",
        "Variant": "FIFO-v2",
        "Mean Return": np.nan,
        "Mean Cost": fifo_m.get('mean_cost', np.nan),
        "Adherence %": np.nan,
        "Overhead Ratio": np.nan,
    })

    df = pd.DataFrame(summary)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nPhase 2 Summary Metrics:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"))

    # Convergence info
    conv_ret = res4.get('_convergence_return', {})
    conv_cost = res4.get('_convergence_cost', {})
    thresholds = res4.get('_thresholds', {})
    if conv_ret:
        print(f"\nExt 4 Convergence (adaptive thresholds):")
        print(f"  Return threshold: 90% of scratch final = {thresholds.get('return', 0):.2f}")
        print(f"  Cost threshold:   scratch final median  = {thresholds.get('cost', 0):.2f}")
        for cond in ['scratch', 'transfer_frozen', 'transfer_finetune']:
            print(f"  {cond}: return@{conv_ret.get(cond,'?')}, cost@{conv_cost.get(cond,'?')}")
        s_ret = max(1, conv_ret.get('scratch', 1))
        s_cost = max(1, conv_cost.get('scratch', 1))
        for cond in ['transfer_frozen', 'transfer_finetune']:
            c_ret = max(1, conv_ret.get(cond, s_ret))
            c_cost = max(1, conv_cost.get(cond, s_cost))
            print(f"  Speedup ({cond}): return {s_ret/c_ret:.1f}×, cost {s_cost/c_cost:.1f}×")

    # Ablation summary
    print("\nAblation Study Summary:")
    a = abl.get('A', {})
    print(f"  A: BSS w* = {a.get('bss_return',0):.2f} vs Random = {a.get('random_mean_return',0):.2f} "
          f"(+{a.get('improvement_pct',0):.1f}%)")
    b = abl.get('B', {})
    print(f"  B: OU spot temporal advantage = {b.get('ou_advantage_pct',0):.1f}% over static")
    c = abl.get('C', {})
    print(f"  C: Transfer feature contribution = {c.get('transfer_contribution_pct',0):.1f}% "
          f"of cost improvement")

    print("\nPlots saved to: results/phase2/")
    
    out_dir = os.path.join("results", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "phase2_summary.csv"), index=False)
    
if __name__ == "__main__":
    main()
