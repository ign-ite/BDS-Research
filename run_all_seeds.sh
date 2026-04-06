#!/bin/bash
# Crash-proof multi-seed runner — each (ext, seed) is a SEPARATE process
# so memory is fully freed between runs. Monitors GPU memory.
#
# Usage: ./run_all_seeds.sh
# Resume: just re-run — completed seeds are auto-skipped

set -e

CONDA_ENV="/home/glitch/anaconda3/envs/rb_ddrl_p2"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
PYTHON="${CONDA_ENV}/bin/python"
SEEDS="0 1 2"
RESULTS="results/phase2/seeds"

echo "=============================================="
echo "  Crash-Proof Multi-Seed Runner (3 seeds)"
echo "  $(date)"
echo "=============================================="

TOTAL=0
DONE=0
FAILED=0

for EXT in 1 2 3 4; do
    for SEED in $SEEDS; do
        TOTAL=$((TOTAL + 1))
        METRIC_FILE="${RESULTS}/ext${EXT}/seed${SEED}/metrics.json"

        if [ -f "$METRIC_FILE" ]; then
            echo "[SKIP] Ext ${EXT} Seed ${SEED} — already done"
            DONE=$((DONE + 1))
            continue
        fi

        echo ""
        echo "======================================================"
        echo "[RUN] Extension ${EXT}, Seed ${SEED}  ($(date +%H:%M))"
        echo "======================================================"

        # Run as isolated process with lower CPU priority to prevent UI lag
        export OMP_NUM_THREADS=4  # Prevent TF from hogging all CPU cores
        if nice -n 19 $PYTHON run_seeds.py --experiment $EXT --seeds $SEED 2>&1; then
            DONE=$((DONE + 1))
            echo "[OK] Ext ${EXT} Seed ${SEED} complete (${DONE}/${TOTAL})"
        else
            FAILED=$((FAILED + 1))
            echo "[FAIL] Ext ${EXT} Seed ${SEED} failed — continuing"
        fi

        # Brief pause to let GPU memory fully release
        sleep 3
        echo "[MEM] GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo 'N/A')"
    done
done

echo ""
echo "=============================================="
echo "  COMPLETE: ${DONE} done, ${FAILED} failed, ${TOTAL} total"
echo "  $(date)"
echo "=============================================="
echo ""
echo "  Next: $PYTHON aggregate_seeds.py"
