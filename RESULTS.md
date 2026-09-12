# MERGE-Response: measured results

All numbers in this file were produced by the code in this repository on the runs recorded below. Cells that were not measured are marked NOT MEASURED; none is carried over from an earlier protocol.

## 0. Run conditions

| Setting | Value |
|---|---|
| backbone | Qwen3-8B (served with vLLM, 3 endpoints, CQP_NO_THINK=1) |
| backbone_deviation | The manuscript specifies Qwen2.5-14B / Qwen3-14B / Gemma-3-12B / Qwen3.6-27B. NONE of these weights are present on this machine (only Qwen3-4B and Qwen3-8B). Every number here therefore uses Qwen3-8B and is NOT comparable to the retained structural tables. |
| team_size | N=4 primary; N=6 and N=8 for the boundary replay |
| rounds | T=4 search rounds, communication before rounds 2-4 (T_comm=3) |
| retrieval | BM25 (Pyserini) top-5 unseen documents per query |
| envelope | 300 tokens per step team-wide, 120 tokens and 4 items per receiver |
| aggregation | strict majority, else plurality with source-count then lexical tie-breakers |
| clustering | frozen all-MiniLM-L6-v2, closest cluster above cosine 0.62 |
| local_scorer | frozen semantic_value.npz, trained earlier on main_200 development trajectories |
| splits | discovery/fitting/selection drawn from main_ext_200 (disjoint from the local-scorer training manifest main_200); confirmation drawn from locked4_400. All splits are question-disjoint and every state/branch of a question stays in its split. |
| confirmation_scale_deviation | The manuscript plans 400 confirmation questions x 3 matched repetitions. Measured GPU throughput here (~21 runs/min, 3xQwen3-8B on saturated A100s) allows 200 questions x 1 decoding repetition for the primary comparison; a partial set of questions also carries 2-3 repetitions from an earlier pass and those are averaged within question. Because the paired interval is bootstrapped over QUESTIONS after averaging repetitions within each question, question count buys more precision than repetition count at fixed compute. |
| rcr_router | External baseline from another paper; not reproduced (out of scope for this run). |
| transfer_deviation | Backbone transfer cannot be measured (single model available). Dataset transfer (MuSiQue, 2WikiMultiHopQA) is measured with the frozen N=4 HotpotQA response model. |
| selected_exposure_weight_response | 20.0 (selected on selection_60 after the scorer fix; the 3->20 segment is monotone, 20-50 is not, so the magnitude is identified but the point is not) |
| selected_exposure_weight_global_set | 3.0 (same split, same search budget) |
| solver_note | All optimized arms share the beam/compound solver. structural_greedy and resp_greedy measure both objectives under the retained positive-singleton greedy solver, so a change of objective is separated from a change of search. |
| round | Round 2 — after five implementation defects were found and fixed (see the defect log below). Round-1 numbers are superseded and are not carried into any table. |
| defect_1_exposure_weight | The response potential is in terminal-F1 units (gamma ~0.01-0.10) while the structural exposure functions are O(1) counts. The round-1 weight grid {0.25..2.0} was therefore a dead zone: the exposure term was 1-7% of the local value and merge_response selected the SAME package as separable in 93.3% of states. Re-selected on {3,5,10,20,30,50}; w_exp=20 chosen on the development split (86% of states change package). |
| defect_2_local_scorer | The local-scorer trainer implemented only two of the three label conditions in Appendix A.2, omitting 'or improves the receiver's answer F1'. The positive rate was 3.3%, so the 0.2 BCE calibration term drove both heads to their bounds: tanh(v_theta) saturated at -1 for 92.3% of transfers (a constant -0.25 penalty) and r_theta sat at zero for 63.9%. After adding the condition (requires a no_comm development arm) the positive rate is 48.1% and saturation falls to 12.9% / 18.2%. |
| defect_3_learned_phase | The learned-phase regression basis used the holder count h where the new-receiver count k was required, making the f_broad marginal 2h-1 instead of max(h-1,0)^2-max(h-2,0)^2. |
| defect_4_global_set | Global-Set counted transfers rather than distinct new receivers, so two items of one cluster to one receiver counted as two new holders. |
| defect_5_reporting | The ablation table's cost column was labelled against capped full while its base was MERGE-Response. |
| intervention_label_caveat | Terminal labels were collected under a continuation policy that used the PREVIOUS local-scorer revision. The cached cluster relevance q_c was recomputed with the current scorer (no LLM calls), but a fully clean run would re-collect the interventions under the new pi_ref. |

Response model as fitted:

- `inputs`: ['d_t', 'h_over_N', 'delta_over_N', 'delta', 'inv_N', 'q_c']
- `hidden`: 16
- `trainable_params`: 129
- `lambda`: 0.001
- `eta`: 0.0
- `selected_epoch`: 85
- `selection_error`: 0.022304806858301163
- `n_fit_pairs`: 252
- `n_fit_questions`: 30
- `search_budget`: 9
- `seed_primary`: 0
- `seed_stability`: {'0': {'selection_err': 0.022304806858301163, 'epoch': 85}, '1': {'selection_err': 0.02321111038327217, 'epoch': 27}, '2': {'selection_err': 0.022648708894848824, 'epoch': 75}}
- `data_sha256`: 461f631c8c2b3878
- `target`: paired terminal group F1 difference

## 1. Response-data collection summary

| Split | Questions | Logged states | Eligible states | Analyzed pairs |
|---|---:|---:|---:|---:|
| Discovery | 9 | 24 | 30 | 102 |
| Fitting | 39 | 109 | 113 | 372 |
| Selection | 21 | 59 | 73 | 222 |
| Confirmation | 0 | 0 | 0 | 0 |

Exclusions (states and pairs not analyzed), by recorded reason:

- `no_single_holder_cluster`: 192

## 2. Held-out terminal-response prediction (R1)

Both error columns are in F1 points. The constant predictor is fitted only on development data. N=6 and N=8 use the response model frozen after N=4 fitting.

| N | Tasks / pairs | Model MAE | Constant MAE | Calibration slope |
|---:|---:|---:|---:|---:|
| 4 | 25 / 204 | 10.73 | 10.09 | -0.001 |
| 6 | 6 / 54 | 10.81 | 8.21 | -0.538 |
| 8 | 8 / 66 | 9.15 | 6.26 | 0.148 |

## 3. Boundary alignment (R2)

Marginals are measured terminal-F1 effects in points at distance δ = m − h, on states matched across δ = 2, 1, 0. C₋ = μ(1) − μ(2) and C₊ = μ(1) − μ(0), each with a paired 95% bootstrap interval over questions. A positive late peak requires both contrasts positive.

| N | Stage | States | μ(2) | μ(1) | μ(0) | C₋ [95% CI] | C₊ [95% CI] |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | Early | 45 | +1.91 | +6.24 | +1.29 | +4.33 [+0.09, +8.66] | +4.94 [-2.21, +11.74] |
| 4 | Middle | 54 | +3.38 | +0.14 | +1.90 | -3.24 [-9.75, +2.61] | -1.77 [-7.78, +3.56] |
| 4 | Late | 53 | +4.34 | +5.26 | +2.29 | +0.92 [-6.62, +8.26] | +2.97 [-1.69, +8.22] |
| 6 | Early | 12 | +11.11 | +2.18 | +2.78 | -8.93 [-22.22, +0.79] | -0.60 [-8.93, +7.74] |
| 6 | Middle | 14 | +6.15 | +2.62 | +10.16 | -3.53 [-21.15, +8.38] | -7.54 [-19.05, +0.00] |
| 6 | Late | 14 | +0.00 | +0.00 | +0.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 8 | Early | 12 | +8.33 | +8.28 | +3.10 | -0.05 [-11.42, +14.09] | +5.19 [-0.74, +13.52] |
| 8 | Middle | 14 | +1.90 | -3.38 | -1.27 | -5.29 [-12.38, -0.05] | -2.11 [-7.14, +0.81] |
| 8 | Late | 14 | +2.38 | -4.76 | +0.00 | -7.14 [-19.05, +0.00] | -4.76 [-14.29, +0.00] |

## 4. Policy comparison (P1)

All optimized arms use the common beam/compound solver and the same frozen local scorer, on held-out confirmation questions. Δ is MERGE-Response minus the row's arm in F1 points. RCR-Router is an external baseline and was not reproduced.

| Arm | F1 | Δ [95% CI] | Comm. tokens | Routed items | Total Tokens | Avg Latency (s) | Cost vs. capped full | Questions × reps |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Capped full | 0.6301 | -2.21 [-5.82, +1.31] | 260.0 | 22.01 | 27490 | 58.5 | 1.00 | 200 × 1 |
| Separable | 0.6083 | -0.01 [-2.91, +2.90] | 70.4 | 5.75 | 26102 | 54.5 | 0.95 | 200 × 1 |
| Structural (original MERGE) | 0.5875 | +2.09 [-0.87, +5.19] | 75.2 | 6.17 | 26114 | 54.2 | 0.95 | 200 × 1 |
| Learned-Phase | 0.6038 | +0.45 [-1.89, +2.85] | 71.7 | 5.96 | 26069 | 55.0 | 0.95 | 200 × 1 |
| Global-Set | 0.6194 | -1.13 [-4.06, +1.66] | 102.7 | 7.68 | 26330 | 56.6 | 0.96 | 200 × 1 |
| **MERGE-Response** | 0.6082 | --- | 110.3 | 9.02 | 26405 | 56.6 | 0.96 | 198 × 1 |

## 5. Learned-response ablations (A1)

Differences are variant minus full MERGE-Response in F1 points, on the same questions and the same solver unless the variant is the solver ablation.

| Arm | F1 | Δ [95% CI] | Comm. tokens | Routed items | Total Tokens | Avg Latency (s) | Cost vs. merge_response | Questions × reps |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **MERGE-Response** | 0.6212 | --- | 108.4 | 8.95 | 26399 | 45.9 | 1.00 | 200 × 1 |
| Raw holder/team encoding | 0.6285 | -0.73 [-3.11, +1.58] | 159.7 | 13.21 | 26667 | 46.2 | 1.01 | 200 × 1 |
| No decision-distance input | 0.6047 | +1.65 [-0.90, +4.53] | 101.7 | 8.41 | 26274 | 45.3 | 1.00 | 200 × 1 |
| Concavity-constrained response | 0.6115 | +0.97 [-1.59, +3.65] | 123.9 | 10.15 | 26486 | 46.0 | 1.00 | 200 × 1 |
| Uncentered raw-response potential | 0.6153 | +0.59 [-2.38, +3.64] | 155.2 | 12.74 | 26651 | 46.1 | 1.01 | 200 × 1 |
| MERGE-Response under greedy-only construction | 0.6026 | +1.86 [-0.16, +4.24] | 87.2 | 7.11 | 26231 | 45.7 | 0.99 | 200 × 1 |

### 5b. Objective change vs. search change (A2)

Both objectives are measured under BOTH solvers on the confirmation questions, so a gain from the learned response can be separated from a gain from escaping positive-singleton construction.

| Objective | Beam / compound | Greedy-only | Solver effect |
|---|---:|---:|---:|
| Structural | 0.5875 | 0.6007 | -1.32 |
| MERGE-Response | 0.6082 | 0.6026 | +0.56 |

## 6. Small-state solver diagnostic (S1)

Both solvers use the same cached states and the same learned-response objective, with exact enumeration as reference. A match means the optimal objective value within 1e-8; ties count as matches rather than package-identity failures.

| Solver | States | Optimal-value match | Median gap | P90 gap | Latency (s) |
|---|---:|---:|---:|---:|---:|
| Positive-singleton greedy / single moves | 150 | 63.33% | 0.0000% | 76.0972% | 0.0013 |
| Beam / compound moves | 150 | 100.00% | 0.0000% | 0.0000% | 0.0161 |

## 7. Frozen learned-response transfer (T1)

The N=4 response model is frozen and applied without refitting. Δ is response minus structural F1 in points, paired over questions.

| Dataset | Backbone | Structural F1 | Response F1 | ΔF1 [95% CI] |
|---|---|---:|---:|---|
| HotpotQA | Qwen3-8B | 0.5875 | 0.6082 | +2.09 [-0.87, +5.19] |
| MuSiQue | Qwen3-8B | 0.2186 | 0.1974 | -2.12 [-5.79, +1.30] |
| 2WikiMHQA | Qwen3-8B | 0.3959 | 0.4239 | +2.80 [-2.44, +8.00] |

## 8. What these measurements support

Read against the predeclared decision rules of the protocol:

**RQ1 (response prediction): NOT SUPPORTED.** The model beats a development-fitted constant predictor on 0 of 3 team sizes. Negative calibration slopes mean the ranking is inverted where they occur, so a low fitting error is not evidence of a usable held-out response estimate.

**RQ2 (policy value): NOT SUPPORTED.** MERGE-Response reaches F1 0.6082. Intervals excluding zero: none. Against Separable, Structural, Learned-Phase and Global-Set every paired interval spans zero, so the learned response provides no measurable routing gain over a hand-specified exposure term or over dropping cross-receiver exposure entirely.

**RQ3 (boundary tracking): NOT SUPPORTED.** Both neighbour contrasts are positive in 2 of 9 (team size x stage) cells, and in 0 cells do both intervals exclude zero. The measured curvature does not track distance to majority across team sizes.

**RQ4 (frozen transfer): NEUTRAL.** On 3 of 3 measured dataset cells the paired interval spans zero. The frozen model neither gains nor collapses relative to the structural reference; the backbone dimension could not be measured at all (single model available).

**Threshold complementarity binds in practice.** Positive-singleton greedy reaches the exact optimum in only 63.33% of 150 enumerable states (P90 normalized gap 76.1%), while beam/compound search reaches it in 100.00%. The objective is not being optimized adequately by singleton construction. Beam costs 9x the objective evaluations (16 ms vs 1 ms per allocation). Unit tests (tests/test_allocation.py) construct states where greedy stops at the empty package and beam finds the majority-crossing package, establishing possibility; the table above measures how often it actually matters. End to end, swapping the solver costs 1.86 F1 points ([-0.16, +4.24]).

### Data-quality flags

- N=6 Early: only 12 matched states; the interval is too wide to test the contrast.
- N=6 Middle: only 14 matched states; the interval is too wide to test the contrast.
- N=6 Late: every matched state has a zero effect at all three distances (14 states) - degenerate, not evidence of a null effect.
- N=8 Early: only 12 matched states; the interval is too wide to test the contrast.
- N=8 Middle: only 14 matched states; the interval is too wide to test the contrast.
- N=8 Late: only 14 matched states; the interval is too wide to test the contrast.
- The capped-full row is NOT dose-matched: it transmits 260 tokens and 22.0 items per task against 110 tokens and 9.0 items for the optimized arms. Its advantage measures the value of a ~5x larger evidence dose, not a better allocation rule. The dose-matched comparison is the one among the remaining arms.
- The raw holder/team encoding variant also transmits far more (160 tokens, 13.2 items), so its nominally higher F1 is confounded with dose and must not be read as a better representation.
- The exposure weight was selected on 60 development questions with one repetition over 6 candidates spanning 5.2 F1 points; the ordering is not monotone across the whole grid, so the magnitude is identified but the point value is not. It was NOT retuned after seeing confirmation results.
- Cost-term constants (w_tok=0.004, w_load=0.5, S_load=400) are inherited from the structural controller's table and were NOT re-selected for the terminal-F1 scale. MERGE-Response sends 110 tokens per task against a 300-token per-step cap; whether that operating point is right for this objective is untested.

