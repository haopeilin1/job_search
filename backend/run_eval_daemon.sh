#!/bin/bash
cd /home/hpl/job_search/backend
export PYTHONPATH=/home/hpl/job_search/backend
exec python3 -m eval.run_full_eval_v3 > eval/results/run.log 2>&1
