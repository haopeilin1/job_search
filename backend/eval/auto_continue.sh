#!/bin/bash
# 自动监控：第一轮完成后 -> Judge后处理 -> 第二轮 -> 第三轮

LOG="/home/hpl/job_search/backend/eval/results/auto_continue.log"
RUN1_DIR="/home/hpl/job_search/backend/eval/results/run1"

echo "$(date '+%Y-%m-%d %H:%M:%S') 监控任务启动，等待第一轮结束..." >> "$LOG"

# 等待第一轮进程结束
while kill -0 2904078 2>/dev/null; do
    sleep 30
    COUNT=$(ls "$RUN1_DIR"/*.json 2>/dev/null | grep -v "^_" | wc -l)
    echo "$(date '+%Y-%m-%d %H:%M:%S') 第一轮进度: $COUNT/55" >> "$LOG"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') 第一轮结束，开始 Judge 后处理..." >> "$LOG"
cd /home/hpl/job_search/backend
/opt/anaconda3/bin/python eval/judge_postprocess.py --run 1 >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') Judge 后处理完成，启动第二轮..." >> "$LOG"
/opt/anaconda3/bin/python -u eval/batch_eval_runner.py --runs 1 --start-run 2 >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') 第二轮完成，启动第三轮..." >> "$LOG"
/opt/anaconda3/bin/python -u eval/batch_eval_runner.py --runs 1 --start-run 3 >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') 全部完成！" >> "$LOG"
