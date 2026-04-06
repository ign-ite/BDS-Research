#!/bin/bash
# GPU wrapper — sets LD_LIBRARY_PATH before Python starts so TF finds CUDA 11.8
# Usage: ./gpu_run.sh run_seeds.py --experiment all --seeds 0,1,2,3,4

CONDA_ENV="/home/glitch/anaconda3/envs/rb_ddrl_p2"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

exec "${CONDA_ENV}/bin/python" "$@"
