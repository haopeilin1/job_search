#!/bin/bash
cd "$(dirname "$0")/.."
PYTHON="./venv/Scripts/python.exe"

# Wait for gen to complete
while [ ! -f eval/v3_gen.json ]; do
    echo "[$(date)] Waiting for gen batch..."
    sleep 30
done
echo "[$(date)] gen batch completed"

# Run remaining batches sequentially
echo "[$(date)] Starting batch: wang"
$PYTHON -u eval/run_eval_v3.py --batch wang --output eval/v3_wang.json > eval/v3_wang.log 2>&1
echo "[$(date)] Batch wang done"

echo "[$(date)] Starting batch: li"
$PYTHON -u eval/run_eval_v3.py --batch li --output eval/v3_li.json > eval/v3_li.log 2>&1
echo "[$(date)] Batch li done"

echo "[$(date)] Starting batch: chen"
$PYTHON -u eval/run_eval_v3.py --batch chen --output eval/v3_chen.json > eval/v3_chen.log 2>&1
echo "[$(date)] Batch chen done"

echo "[$(date)] All batches completed"
