#!/usr/bin/env bash
# Run a list of configs sequentially. Usage: bash scripts/run_queue.sh "cfgA.yaml --set seed=0" "cfgB.yaml" ...
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
for job in "$@"; do
  name=$(echo "$job" | tr ' /=.' '____' | cut -c1-90)
  echo "=== $(date +%H:%M:%S) START $job"
  .venv/Scripts/python.exe scripts/train.py $job > "results/logs/${name}.log" 2>&1
  echo "=== $(date +%H:%M:%S) END   $job (exit $?) -> $(tail -1 results/logs/${name}.log | cut -c1-200)"
done
