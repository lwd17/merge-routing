# MERGE: Evidence Routing under Group Externalities for Multi-Agent Search

Anonymous code release accompanying the paper *"MERGE: Evidence Routing under
Group Externalities for Multi-Agent Search"* (under double-blind review).

MERGE treats inter-agent communication in multi-agent LLM search as a **joint
allocation problem over source-grounded evidence-to-receiver transfers**. At
each communication step a training-free, non-generative controller maximizes
an exposure-welfare objective (paper Eq. 12): receiver-local value (designed,
Eq. 8-9, or learned, Eq. 10-11) inside a saturating per-requirement coverage
term, a phase-dependent cross-receiver exposure term (reach vs. broadcast,
Eq. 4-6), a convex receiver-load cost, a token cost, and a within-receiver
duplicate correction (Eq. 13), under hard per-step feasibility limits. The
online controller makes **zero routing LLM calls**.

**Scope.** This repository contains the method implementation and the full
experiment pipeline (data preparation, arm runners, paired analysis, and
scorer-training scripts). It intentionally contains **no experimental
results, run logs, or evaluation records**: results live in the paper.
The shipped weight file is a reference artifact required to execute the
learned-value arms; no performance claim is made in this repository.

---

## Repository layout

```
merge-release/
├── README.md
├── LICENSE                      MIT
├── pyproject.toml               installable package metadata
├── requirements.txt             dependencies
├── src/merge/
│   ├── config.py                endpoints, model id, data root (env-driven)
│   ├── extract.py, rq2_metrics.py   claim extraction + token metrics
│   └── search/
│       ├── engine.py            LLM-call layer, seeds, cached round 1
│       ├── gurc.py              experiment engine + every arm's policy
│       ├── welfare.py           exposure-welfare objective, solver, MERGE arms
│       ├── memory_pool.py       global claim pool, provenance, holders, clustering
│       ├── semantic_value.py    learned local value (numpy inference) + encoder
│       ├── semantic_value.npz   reference r/v heads with training-data hash
│       ├── prompts.py           the three prompts (propose, finding, answer)
│       ├── retriever.py         BM25 (Pyserini) + optional dense rerank
│       └── qa_eval.py           answer normalization, F1/EM
└── scripts/
    ├── hotpotqa_setup.py        download HotpotQA, build corpus + BM25 index
    ├── musique_setup.py         download MuSiQue, build corpus + BM25 index
    ├── make_2wiki.py            build 2WikiMultiHopQA corpus + index
    ├── make_hotpot_manifest.py  seeded, disjoint evaluation manifests
    ├── make_musique_manifest.py seeded evaluation manifest (MuSiQue)
    ├── run_gurc_pilot.py        MAIN EXPERIMENT RUNNER (all arms)
    ├── analyze_paired.py        paired bootstrap contrasts (paper CI procedure)
    ├── build_transfer_log.py    transfer-level log from run records
    ├── train_semantic_value.py  train the learned local value (Appendix A.3)
    ├── replay_fixed_state.py    fixed-state branching replay (Section 5)
    ├── scifact_setup.py         SciFact corpus/index/manifests (RQ4)
    └── analyze_scifact.py       verdict macro-F1 + evidence-sentence F1
```

---

## 1. Installation

Requirements: Python >= 3.10, Java 21 (for Pyserini/Lucene), and an
OpenAI-compatible endpoint able to serve the evaluated models (we used vLLM).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 2. Serving models

All experiments talk to an OpenAI-compatible endpoint, e.g.:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct --port 8100 --max-model-len 8192
```

Client configuration:

| Variable | Meaning | Default |
|---|---|---|
| `CQP_API_BASE` | single endpoint URL | `http://localhost:8100/v1` |
| `CQP_API_BASES` | comma-separated fleet; each task is stably hashed to one endpoint so vLLM prefix caching keeps working | `CQP_API_BASE` |
| `CQP_MODEL` | model id passed to the endpoint | `Qwen/Qwen2.5-14B-Instruct` |
| `CQP_API_KEY` | API key | `local` |
| `CQP_NO_THINK` | suppress explicit thinking traces (reasoning-mode backbones) | off |

All backbones use an 8192-token context.

---

## 3. Data preparation

Each dataset lives under `data/<name>/` with the same layout: a pooled
paragraph corpus (`corpus/`), a Lucene BM25 index (`index_bm25/`), a
title-to-docid map, and evaluation manifests (`manifests/*.json`). Override
the root with `CQP_DATA_ROOT`; select the active dataset with
`CQP_DATASET` (`hotpotqa` | `musique` | `2wiki` | `scifact`).

```bash
python scripts/hotpotqa_setup.py
python scripts/musique_setup.py
# 2WikiMultiHopQA: place the official dev.json at data/2wiki_raw/dev.json, then
python scripts/make_2wiki.py
```

Evaluation manifests are seeded samples that are **disjoint from every
development set**:

```bash
python scripts/make_hotpot_manifest.py --n 400 --seed <SEED> \
    --exclude data/hotpotqa/manifests/probe_30.json \
    --exclude <every previously used manifest>

python scripts/make_musique_manifest.py
```

Never evaluate on a manifest that overlaps a set used for method iteration.

---

## 4. Running experiments

The runner evaluates a list of arms on one manifest. All agents share a
fixed investigator persona; round-1 retrieval is generated once per task,
cached, and **replayed verbatim in every arm**, so arm comparisons are
paired by construction and communication can only influence rounds >= 2:

```bash
python scripts/run_gurc_pilot.py \
    --manifest <manifest>.json \
    --caches runs/caches \
    --out runs/my_experiment \
    --arms no_comm,capped_full,merge_d \
    --workers 64
```

Repeat the same command with a different `--out` for additional repetitions.

### Arms (code name <-> paper name)

Factorial MERGE arms (Section 5):

| Code arm | Paper arm |
|---|---|
| `designed_sep` | DESIGNED-SEP (designed local value, receiver-separable) |
| `merge_d` | **MERGE-D** (designed local value, joint allocation) |
| `learned_sep` | LEARNED-SEP (learned local value, receiver-separable) |
| `merge_l` | **MERGE-L** (learned local value, joint allocation) |

Controlled baselines (Table 7) and agentic baselines (Table 9):

| Code arm | Paper arm |
|---|---|
| `no_comm` | No communication |
| `capped_full` | Capped full communication (broadcast under the shared hard limits) |
| `random_ev` | Random evidence routing (fixed 335-token dose) |
| `hmem` | HMem receiver-local memory |
| `agentprune` | AgentPrune-inspired (pruned sender edges) |
| `dytopo` | DyTopo-inspired (dynamic topology) |
| `global_set` | GLOBAL-SET (learned joint set scorer, no explicit exposure terms) |
| `solo_tool` | Solo agent, matched budget |
| `central_orch` | Centralized orchestrator |
| `lead_roles` | Lead-role decomposition |

The full registry is `POLICIES` in `src/merge/search/gurc.py`. The
separable variants drop only the cross-receiver exposure term, so the
objective factorizes exactly (Eq. 3); every other term is shared.

### Experiment toggles (environment variables)

| Variable | Effect |
|---|---|
| `CQP_DATA_ROOT`, `CQP_DATASET` | data root / dataset directory |
| `CQP_BUDGET_MAIN=B` | budget sweep (Appendix B.4): overrides the per-step global cap B for MERGE and the envelope-matched baselines. The cap is a pure per-step **feasibility constraint** — never a target dose; the per-receiver cap (120 tokens) and item cap (4) are unchanged |
| `CQP_CLUSTER_TAU` | encoder-clustering threshold (Appendix A.1; default 0.62) |
| `CQP_CLUSTER=jaccard` | lightweight token-Jaccard clustering fallback for environments without the encoder (convenience only; the paper mechanism is the default encoder mode) |
| `CQP_ROLES=1` | role-specialized search instructions (Appendix A.4: Grounder / Bridge explorer / Alternative explorer / Verifier), off by default |
| `CQP_SEMVAL` | path to learned-value weights (default: shipped `semantic_value.npz`) |
| `CQP_RETRIEVER=rerank` | hybrid retrieval: BM25 top-50 + dense cosine rerank |
| `CQP_NO_THINK=1` | suppress explicit thinking traces (reasoning-mode backbones) |
| `CQP_RAW_LOG` | path for a raw LLM call log (debugging) |

Backbone-specific serving options are applied identically to **every**
comparison arm; the routing configuration itself is identical across
datasets and backbones.

### Output format

Each run writes `<out>/records.jsonl`, one JSON object per (task, arm):

```
task_id, arm, question, answer (gold), gold_docids, n_agents, rounds,
events        per-agent per-round: query, retrieved docids/titles, routed items,
finals        per-agent final answers with per-agent F1/EM,
group         team answer (strict majority, else plurality with fixed
              source-count and lexical tie-breakers), with F1/EM,
comm_tokens   total transmitted inter-agent tokens (three communication steps),
alloc_log      per-step solver log: greedy value, final value, accepted moves,
              package size, wall-clock seconds (Appendix A.7),
memory_pool, pool_stats, team_gold_coverage, wall_seconds
```

---

## 5. Reproducing the paper's comparisons

1. Serve a backbone (Section 2), prepare a dataset (Section 3), draw a fresh
   seeded manifest (never reuse development ids).
2. Run all arms of interest with shared caches (Section 4); repeat the exact
   command per repetition with a new `--out`.
3. Compute paired contrasts with the paper's interval procedure (per-task
   repetition means -> paired differences -> 10,000-resample percentile
   bootstrap over tasks):

```bash
python scripts/analyze_paired.py \
    --arm-a merge_d --arm-b capped_full \
    --records runs/exp_r1/records.jsonl runs/exp_r2/records.jsonl
```

Budget sweep (Appendix B.4): rerun step 2 with `CQP_BUDGET_MAIN` set to each
cap (both MERGE and the envelope-matched baselines read it). Team-size
scaling: pass `--n-agents N` and use one frozen manifest for every N.
Retriever transfer: `CQP_RETRIEVER=rerank`.

---

## 6. Communication accounting

`comm_tokens` counts transmitted inter-agent content only; retrieved document
text is never counted unless transmitted. Every fixed-team arm communicates
at three steps per task. The shared per-step envelope is a hard feasibility
constraint: 300 tokens global, 120 tokens and 4 items per receiver
(Table 4); the random arm instead uses a fixed 335-token dose (Table 7).
Routed items are durable holdings: they enter the receiver's next propose
call and compete for the answer window exactly like the receiver's own
retrieved documents.

---

## 7. Learned local value, replay, and SciFact

`src/merge/search/welfare.py` implements the exposure-welfare objective and
the local-value factorial:

- the **designed local value** (Eq. 8-9): per-requirement mass
  `q_src [0.60 q_sem + 0.25 q_bridge + 0.15 q_scarce]` inside the
  saturating coverage term, plus `+0.20 q_question - 0.20 q_held`
  transfer-local corrections (frozen coefficients, Appendix A.2);
- the **learned local value** (Eq. 10-11, Appendix A.3): nonnegative
  per-requirement support `r_theta(x, g, s_i)` (softplus head) inside the
  same saturating coverage plus a bounded transfer residual `0.25 tanh(.)`,
  from frozen-encoder embeddings of the item, requirement, question, and
  last query, with scalar state features (load, holder count, round,
  source support, new-entity flag); hidden width 256;
- both are combined with the same phase-dependent exposure term, convex
  load cost, token cost, and within-receiver duplicate correction, and the
  same greedy + add/drop/swap solver (Appendix A.7).

Retrain the learned scorer on your own **development** logs (never on
locked evaluation records — that is evaluation contamination):

```bash
python scripts/train_semantic_value.py --run runs/<dev_run> --epochs 8 \
    --out src/merge/search/semantic_value.npz
```

Supervision (Eq. 10-11): within-receiver-state pairwise ranking with a
0.2-weight binary-productivity calibration term; a transfer is preferred
when the following search round recovers a new supporting document or the
transfer closes an annotated requirement. Reference weights (trained on a
small development pilot log; no performance claims) ship in
`src/merge/search/semantic_value.npz` with a training-data hash. Override
at runtime with `CQP_SEMVAL=/path/to/weights.npz`.

The fixed-state branching replay (Section 5, Table 2) clones logged
receiver states and compares separable against joint allocation at matched
dose, with the residual indivisibility gap logged (Appendix A.6):

```bash
python scripts/replay_fixed_state.py --records runs/<run>/records.jsonl \
    --states 900 --out runs/replay.jsonl [--learned] [--focal --holders]
```

SciFact claim verification (RQ4): `python scripts/scifact_setup.py` builds
the corpus, BM25 index, and SUPPORT/REFUTE manifests; run any arm with
`CQP_DATASET=scifact` and score with `scripts/analyze_scifact.py`
(verdict macro-F1 and evidence-sentence F1).

---

## 8. License

MIT (see `LICENSE`). The datasets (HotpotQA, MuSiQue, 2WikiMultiHopQA,
SciFact) and models retain their own licenses; this repository downloads or
reads them but does not redistribute them.
