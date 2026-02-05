#!/usr/bin/env bash
# model="deepseek/deepseek-chat-v3.1"
# optional --vlm flag
model="anthropic/claude-sonnet-4"

python scripts/run_eval_astar_parallel.py \
  --data-root data/json_2.1.0/atomic \
  --pattern '' \
  --model "${model}" \
  --workers 10 \
  --debug
