#!/usr/bin/env bash
set -euo pipefail

cd /home/xcp/workspace/ICLR

PYTHON_BIN="${PYTHON_BIN:-/home/xcp/miniconda3/envs/d3/bin/python}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

exec "$PYTHON_BIN" scripts/train_carrybox.py \
  --mode baseline \
  --pickup_only \
  --pickup_rsi \
  --episode_length_s 5 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --save_interval "$SAVE_INTERVAL" \
  --run_suffix amp \
  --headless
