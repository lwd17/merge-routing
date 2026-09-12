#!/usr/bin/env bash
# Round 2: both implementation defects fixed.
#   (1) exposure weight re-selected on a grid that reaches the active regime
#       (gamma is in terminal-F1 units, so the structural O(1) grid was a dead
#        zone: the objective picked the separable package in 93% of states);
#   (2) local scorer retrained with the THIRD label condition of Appendix A.2
#       ("or improves the receiver's answer F1"), which lifted the positive rate
#       from 3.3% to 48.1% and un-saturated both heads (r_theta zero-floor
#       63.9%->18.2%, tanh(v_theta) at -1 92.3%->12.9%).
# Every arm that reads the local scorer is re-run; capped_full does not use it
# but is re-run too so the whole table comes from one scorer revision.
set -u
cd "$(dirname "$0")/.."
export CQP_DATA_ROOT=/home/liangwd/agent26/cqp-pilot/data
export CQP_MODEL=Qwen3-8B
export CQP_API_BASES="http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8106/v1,http://127.0.0.1:8107/v1"
export CQP_NO_THINK=1
LOG=runs/round2.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "=== S1: re-select the exposure weight on the fixed scorer (dev split)"
python scripts/select_weight.py --arm merge_response --env CQP_WEXP_RESP \
    --weights 3,5,10,20,30,50 --manifest selection_60.json \
    --out runs/select3 --workers 40 >> $LOG 2>&1
W=$(python -c "import json;print(json.load(open('runs/select3/selection_merge_response.json'))['selected']['weight'])")
export CQP_WEXP_RESP=$W
say "    CQP_WEXP_RESP=$W"
python scripts/select_weight.py --arm global_set --env CQP_WEXP_GS \
    --weights 1,3,10,30 --manifest selection_60.json \
    --out runs/select3 --workers 40 >> $LOG 2>&1
WG=$(python -c "import json;print(json.load(open('runs/select3/selection_global_set.json'))['selected']['weight'])")
export CQP_WEXP_GS=$WG
say "    CQP_WEXP_GS=$WG"

say "=== S2: re-fit learned-phase and global-set heads on the same interventions"
python scripts/train_response.py --interventions runs/interventions/n4.jsonl \
    --out src/merge/search/response_model.npz --fit-phase --fit-global-set >> $LOG 2>&1
say "    refitted"

say "=== S3: policy comparison, all arms, one scorer revision"
python scripts/run_experiment.py --manifest confirm_200.json --caches runs/caches \
    --out runs/round2_confirm --workers 40 --seeds 1 \
    --arms capped_full,separable,structural,learned_phase,global_set,merge_response,structural_greedy \
    >> $LOG 2>&1
say "    $(wc -l < runs/round2_confirm/records.jsonl) records"

say "=== S4: response ablations"
python scripts/run_experiment.py --manifest confirm_200.json --caches runs/caches \
    --out runs/round2_ablation --workers 40 --seeds 1 \
    --arms merge_response,resp_raw_encoding,resp_no_distance,resp_concave,resp_uncentered,resp_greedy \
    >> $LOG 2>&1
say "    $(wc -l < runs/round2_ablation/records.jsonl) records"

say "=== S5: frozen-model dataset transfer"
for DS in musique 2wiki; do
  CQP_DATASET=$DS python scripts/run_experiment.py --manifest transfer_150.json \
      --caches runs/caches_$DS --out runs/round2_transfer_$DS --workers 40 --seeds 1 \
      --arms structural,merge_response >> $LOG 2>&1
  say "    $DS: $(wc -l < runs/round2_transfer_$DS/records.jsonl) records"
done

say "=== S6: analyses and report"
python scripts/analyze_policy.py --records runs/round2_confirm/records.jsonl \
    --ref merge_response --out runs/report/policy.json >> $LOG 2>&1
python scripts/analyze_policy.py --records runs/round2_ablation/records.jsonl \
    --ref merge_response --cost-base merge_response --out runs/report/ablation.json >> $LOG 2>&1
python scripts/analyze_transfer.py \
    --cell "HotpotQA:Qwen3-8B:runs/round2_confirm/records.jsonl" \
    --cell "MuSiQue:Qwen3-8B:runs/round2_transfer_musique/records.jsonl" \
    --cell "2WikiMHQA:Qwen3-8B:runs/round2_transfer_2wiki/records.jsonl" \
    --out runs/report/transfer.json >> $LOG 2>&1
python scripts/analyze_solver.py --records runs/dev/records.jsonl --states 150 \
    --w-exp "$W" --out runs/report/solver.json >> $LOG 2>&1
python scripts/make_report.py --report-dir runs/report \
    --conditions runs/report/conditions.json --out RESULTS.md >> $LOG 2>&1
say "=== ROUND 2 COMPLETE -> RESULTS.md"
