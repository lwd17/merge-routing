#!/usr/bin/env bash
# Remaining pipeline at the compute-feasible scale: one decoding repetition
# on 200 confirmation questions (the paired interval is bootstrapped over
# questions, so question count buys more precision than repetition count).
set -u
cd "$(dirname "$0")/.."
export CQP_DATA_ROOT=/home/liangwd/agent26/cqp-pilot/data
export CQP_MODEL=Qwen3-8B
export CQP_API_BASES="http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8106/v1"
export CQP_NO_THINK=1
export CQP_WEXP_RESP=2.0
export CQP_WEXP_GS=1.0
LOG=runs/pipeline.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "=== STEP 5b: confirmation policy comparison (seeds=1, 200 questions)"
python scripts/run_experiment.py --manifest confirm_200.json --caches runs/caches \
    --out runs/confirm --workers 40 --seeds 1 \
    --arms capped_full,separable,structural,learned_phase,global_set,merge_response,structural_greedy \
    >> $LOG 2>&1
say "    confirm: $(wc -l < runs/confirm/records.jsonl) records"

say "=== STEP 6: response ablations"
python scripts/run_experiment.py --manifest confirm_200.json --caches runs/caches \
    --out runs/ablation --workers 40 --seeds 1 \
    --arms merge_response,resp_raw_encoding,resp_no_distance,resp_concave,resp_uncentered,resp_greedy \
    >> $LOG 2>&1
say "    ablation: $(wc -l < runs/ablation/records.jsonl) records"

say "=== STEP 7: frozen-model dataset transfer"
for DS in musique 2wiki; do
  CQP_DATASET=$DS python scripts/run_experiment.py --manifest transfer_150.json \
      --caches runs/caches_$DS --out runs/transfer_$DS --workers 40 --seeds 1 \
      --arms structural,merge_response >> $LOG 2>&1
  say "    $DS: $(wc -l < runs/transfer_$DS/records.jsonl 2>/dev/null || echo 0) records"
done

say "=== STEP 3b: resume N=6 / N=8 interventions"
python scripts/collect_interventions.py --records runs/dev_n6/records.jsonl \
    --out runs/interventions/n6.jsonl --n-agents 6 --states 14 --reps 3 \
    --workers 16 --splits runs/splits.json >> $LOG 2>&1
say "    n6: $(wc -l < runs/interventions/n6.jsonl) pairs"
python scripts/collect_interventions.py --records runs/dev_n8/records.jsonl \
    --out runs/interventions/n8.jsonl --n-agents 8 --states 14 --reps 3 \
    --workers 16 --splits runs/splits.json >> $LOG 2>&1
say "    n8: $(wc -l < runs/interventions/n8.jsonl) pairs"

say "=== STEP 8: analyses and report"
python scripts/analyze_policy.py --records runs/confirm/records.jsonl \
    --ref merge_response --out runs/report/policy.json >> $LOG 2>&1
python scripts/analyze_policy.py --records runs/ablation/records.jsonl \
    --ref merge_response --cost-base merge_response \
    --out runs/report/ablation.json >> $LOG 2>&1
python scripts/analyze_transfer.py \
    --cell "HotpotQA:Qwen3-8B:runs/confirm/records.jsonl" \
    --cell "MuSiQue:Qwen3-8B:runs/transfer_musique/records.jsonl" \
    --cell "2WikiMHQA:Qwen3-8B:runs/transfer_2wiki/records.jsonl" \
    --out runs/report/transfer.json >> $LOG 2>&1
python scripts/analyze_response.py \
    --interventions runs/interventions/n4.jsonl runs/interventions/n6.jsonl runs/interventions/n8.jsonl \
    --model src/merge/search/response_model.npz \
    --out runs/report/response.json >> $LOG 2>&1
python scripts/make_report.py --report-dir runs/report \
    --conditions runs/report/conditions.json --out RESULTS.md >> $LOG 2>&1
say "=== PIPELINE COMPLETE -> RESULTS.md"
