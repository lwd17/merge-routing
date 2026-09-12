"""Paired terminal-utility interventions: the training labels for g_phi.

Protocol (paper Appendix "Paired terminal-utility interventions"):

  * Log states from a fixed development collection mixture (separable,
    capped-full, structural). States are read back from those records.
  * At a logged communication step, pick a cluster with exactly ONE actual
    holder and a focal non-holder receiver e.
  * Build A_h by assigning the SAME source-backed evidence to h-1 other
    receivers, so that including concurrent assignments the cluster is
    exposed to h holders. Compare A_h against A_h + {e}, holding the focal
    receiver state, the item, its token length, the candidate pool, the local
    scores, the relevance q_c, and all other assignments fixed.
  * Both branches continue to the final team answer under the same frozen
    reference policy pi_ref = structural, with three matched decoding
    repetitions.
  * The primary label is the terminal group-F1 difference; next-step F1 and
    supporting-evidence recall are logged separately as diagnostics.

For T=4 the communication steps t=1,2,3 (before rounds 2,3,4) are the early,
middle and late stages with d = 1, 1/2, 0. At N=4,6,8 the boundary triplets
delta = 2,1,0 correspond to h = (1,2,3), (2,3,4), (3,4,5). Only states
feasible at ALL of delta=2,1,0 enter the primary boundary contrast; every
exclusion reason is counted and reported.

Zero, negative and incorrect-answer outcomes are retained. States are never
selected for a desired response shape.

Usage:
  python scripts/collect_interventions.py --records runs/dev/records.jsonl \
      --out runs/interventions/n4.jsonl --n-agents 4 --states 120 --reps 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge.search import prompts, qa_eval, retriever  # noqa: E402
from merge.search.engine import _chat, _parse_finding, _parse_proposal, _seed  # noqa: E402
from merge.search.local_value import (  # noqa: E402
    B_RECV, ITEM_CAP, b_global, candidates, decision_distance, local_maps,
)
from merge.search.memory_pool import MemoryPool  # noqa: E402
from merge.search.pipeline import _clean_query, build_state, plurality_vote  # noqa: E402
from merge.search.policies import policy_structural  # noqa: E402

SEED = 20260902
PERSONA = {"role": "Investigator",
           "description": "A thorough researcher who answers questions by "
                          "searching Wikipedia."}


# ------------------------------------------------------------- state replay
def rebuild(rec: dict, rnd: int):
    """Clone the pool and per-agent state as they were before round `rnd`."""
    n = rec["n_agents"]
    pool = MemoryPool(n)
    mid_map = {}
    routed_before = defaultdict(list)
    for ev in rec["events"]:
        if ev["round"] < rnd:
            for m in (ev.get("routed_mids") or []):
                routed_before[ev["agent"]].append(m)
    for it in sorted(rec["memory_pool"], key=lambda x: x["mid"]):
        if it.get("round_created", 1) >= rnd:
            continue
        new = pool.add_claims(
            it.get("producer", 0), it.get("round_created", 1),
            [{"claim": it["claim"], "entities": it.get("entities", []),
              "source_title": it.get("source_title", "")}],
            {it.get("source_title", ""): it.get("source_docid", "")},
            {it.get("source_docid", "")}, None)
        if new:
            mid_map[it["mid"]] = new[0]
    received_tokens = [0] * n
    by_mid = {x["mid"]: x for x in pool.items}
    for i in range(n):
        for old in routed_before[i]:
            nm = mid_map.get(old)
            if nm is not None and i not in pool.holders[nm]:
                pool.holders[nm][i] = 2
                received_tokens[i] += by_mid[nm]["tokens"]
    ev_at = {(e["agent"], e["round"]): e for e in rec["events"]}
    sketches, own, seen = [], [], [set() for _ in range(n)]
    for i in range(n):
        prev = ev_at.get((i, rnd - 1), {})
        sk = prev.get("sketch", {}) or {}
        sketches.append({"unresolved_slots": (sk.get("unresolved_slots") or [])[:6],
                         "target_entities": (sk.get("target_entities") or [])[:6]})
        rounds_i = []
        for r in range(1, rnd):
            e = ev_at.get((i, r))
            if e:
                rounds_i.append({"round": r, "q1": e.get("q1", ""),
                                 "docs": [{"docid": d, "title": t,
                                           "excerpt": ""}
                                          for t, d in zip(e.get("titles", []),
                                                          e.get("docids", []))]})
                seen[i].update(e.get("docids", []))
        if not rounds_i:
            rounds_i = [{"round": 1, "q1": "", "docs": []}]
        own.append(rounds_i)
    return pool, sketches, own, seen, received_tokens


# --------------------------------------------------------- continuation run
def continue_to_answer(task, pool, sketches, own, seen, received_tokens,
                       package, rnd, rounds, n, seed_tag):
    """Apply `package`, then run rounds rnd..T under pi_ref = structural."""
    import copy
    pool = copy.deepcopy(pool)
    sketches = copy.deepcopy(sketches)
    own = copy.deepcopy(own)
    seen = [set(s) for s in seen]
    received_tokens = list(received_tokens)
    received = [[] for _ in range(n)]
    task_id = str(task["id"])
    question = task["question"]
    gold = {str(d) for d in task.get("gold_docids", [])}
    title2doc, doc_texts = {}, {}
    sysp = [prompts.system_prompt(PERSONA, n, agent_idx=i) for i in range(n)]

    def apply(routed):
        for i, items in routed.items():
            pool.mark_received(i, [it["mid"] for it in items])
            received[i].extend(items)
            received_tokens[i] += sum(it["tokens"] for it in items)

    apply(package)
    first_new_gold, first_f1 = 0, []
    team_seen = set().union(*seen) if seen else set()
    alloc_log = []

    for r in range(rnd, rounds + 1):
        if r > rnd:                                   # pi_ref for later steps
            st = build_state(pool=pool, n_agents=n, rounds=rounds, rnd=r,
                             sketches=sketches, own=own,
                             received_tokens=received_tokens,
                             question=question, task_id=task_id,
                             alloc_log=alloc_log)
            try:
                apply({i: v for i, v in (policy_structural(st) or {}).items() if v})
            except Exception:
                pass
        # agents within a round are independent: run their two LLM calls in
        # parallel, then publish to the shared pool serially in agent order
        # so the pool state stays deterministic.
        def agent_step(i):
            held = received[i][-8:] if received[i] else None
            prop = _parse_proposal(_chat(
                sysp[i], prompts.propose_user(question, own[i], held),
                _seed(task_id, i, r, seed_tag, "propose"), max_tokens=300,
                task_id=task_id, tag={"stage": "intv_propose"}))
            q1 = _clean_query(prop["query"])
            docs = retriever.search(q1, top_k=5, exclude_docids=seen[i])
            finding = _parse_finding(_chat(
                sysp[i], prompts.finding_user(question, q1, docs),
                _seed(task_id, i, r, seed_tag, "finding"), max_tokens=380,
                task_id=task_id, tag={"stage": "intv_finding"}))
            return i, prop, q1, docs, finding

        with ThreadPoolExecutor(max_workers=n) as ex:
            step = dict((res[0], res) for res in ex.map(agent_step, range(n)))
        for i in range(n):
            _i, prop, q1, docs, finding = step[i]
            sketches[i] = prop
            docids = [d["docid"] for d in docs]
            for d in docs:
                title2doc.setdefault(d["title"], d["docid"])
                doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
            if r == rnd:
                first_new_gold += len(set(docids) & gold - team_seen)
            seen[i].update(docids)
            own[i].append({"round": r, "q1": q1, "docs": docs})
            pool.add_claims(i, r, finding.get("claims", []), title2doc,
                            set(docids), doc_texts=doc_texts)

    def answer_step(i):
        evidence = [{**d, "round": rr["round"]} for rr in own[i]
                    for d in rr["docs"] if d.get("docid")]
        for it in received[i]:
            evidence.append({"docid": f"mem:{it['mid']}",
                             "title": it["source_title"] or "team evidence",
                             "excerpt": it["claim"],
                             "round": it["round_created"]})
        raw = _chat(sysp[i], prompts.answer_user(question, evidence),
                    _seed(task_id, i, 99, seed_tag, "answer"),
                    max_tokens=120, task_id=task_id,
                    tag={"stage": "intv_answer"})
        ans = prompts.parse_json_field(raw, "answer") or raw.strip()[:80]
        em, f1 = qa_eval.score(ans, task.get("answer", ""),
                               task.get("answer_aliases", []))
        return {"agent": i, "answer": ans, "em": em, "f1": f1}

    with ThreadPoolExecutor(max_workers=n) as ex:
        finals = list(ex.map(answer_step, range(n)))
    for f in finals:
        if f["agent"] in package:
            first_f1.append(f["f1"])
    ga = plurality_vote(finals, pool)
    g_em, g_f1 = qa_eval.score(ga, task.get("answer", ""),
                               task.get("answer_aliases", []))
    return {"group_f1": g_f1, "group_em": g_em,
            "new_gold": first_new_gold,
            "focal_f1": sum(first_f1) / len(first_f1) if first_f1 else 0.0}


# ------------------------------------------------------------ intervention
def build_packages(pool, n, focal_mid, focal_recv, others, h, base_items):
    """A_h (evidence to h-1 other receivers) and A_h + focal."""
    item = {x["mid"]: x for x in pool.items}[focal_mid]
    tok = item["tokens"]
    base = {}
    for j in others[: h - 1]:
        base.setdefault(j, []).append(item)
    full = {k: list(v) for k, v in base.items()}
    full.setdefault(focal_recv, []).append(item)
    # hard envelope must hold for BOTH branches
    for pkg in (base, full):
        total = sum(len(v) * tok for v in pkg.values())
        if total > b_global():
            return None, None
        for v in pkg.values():
            if len(v) * tok > B_RECV or len(v) > ITEM_CAP:
                return None, None
    return base, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--splits", default="",
                    help="JSON file mapping question id -> split name")
    args = ap.parse_args()

    n = args.n_agents
    m = n // 2 + 1
    triplet = [m - 2, m - 1, m]          # h values for delta = 2, 1, 0
    rng = random.Random(args.seed)
    splits = json.loads(Path(args.splits).read_text()) if args.splits else {}

    recs = [json.loads(l) for l in Path(args.records).read_text().splitlines()]
    recs = [r for r in recs if r.get("n_agents") == n]
    by_task = {}
    for r in recs:
        by_task.setdefault(r["task_id"], r)       # one record per question
    tasks_meta = {r["task_id"]: r for r in by_task.values()}

    # enumerate eligible (question, stage) states
    cand_states = []
    excl = Counter()
    for tid, rec in tasks_meta.items():
        for rnd in range(2, args.rounds + 1):
            try:
                pool, sketches, own, seen, rtok = rebuild(rec, rnd)
            except Exception:
                excl["rebuild_failed"] += 1
                continue
            if not pool.items:
                excl["empty_pool"] += 1
                continue
            # clusters with exactly ONE actual holder
            cl_holders = defaultdict(set)
            cl_items = defaultdict(list)
            for it in pool.items:
                c = it.get("semantic_cluster_id") or it["mid"]
                cl_holders[c] |= set(pool.holders[it["mid"]])
                cl_items[c].append(it)
            elig = [(c, cl_items[c][0]) for c in cl_holders
                    if len(cl_holders[c]) == 1]
            if not elig:
                excl["no_single_holder_cluster"] += 1
                continue
            cand_states.append((tid, rnd, [(c, it["mid"], sorted(cl_holders[c])[0])
                                           for c, it in elig]))
    rng.shuffle(cand_states)
    print(f"eligible (question, stage) states: {len(cand_states)}; "
          f"exclusions: {dict(excl)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["task_id"], r["round"], r["cluster"], r["h"]))
            except json.JSONDecodeError:
                pass

    jobs = []
    for tid, rnd, elig in cand_states:
        if len({j[0] for j in jobs}) >= args.states and tid not in {j[0] for j in jobs}:
            continue
        c, mid, holder = elig[rng.randrange(len(elig))]
        jobs.append((tid, rnd, c, mid, holder))
        if len(jobs) >= args.states * args.rounds:
            break

    lock = threading.Lock()
    fh = out_path.open("a")
    t0 = time.time()
    counters = Counter()

    def work(job):
        tid, rnd, c, mid, holder = job
        rec = tasks_meta[tid]
        task = {"id": tid, "question": rec["question"],
                "answer": rec.get("answer", ""),
                "answer_aliases": rec.get("answer_aliases", []),
                "gold_docids": rec.get("gold_docids", [])}
        pool, sketches, own, seen, rtok = rebuild(rec, rnd)
        state = build_state(pool=pool, n_agents=n, rounds=args.rounds, rnd=rnd,
                            sketches=sketches, own=own, received_tokens=rtok,
                            question=task["question"], task_id=tid,
                            alloc_log=[])
        d_t = decision_distance(state)
        # q_c cached BEFORE intervention, identical across the triplet
        try:
            cands = candidates(pool, n)
            _w, _mass, _corr, q_cluster = local_maps(pool, state, cands)
            q_c = float(q_cluster.get(c, 0.0))
        except Exception:
            return [("q_c_failed", None)]
        item_tokens = {x["mid"]: x["tokens"] for x in pool.items}[mid]
        # mean local score of this item across non-holders (package-independent)
        loc_vals = [sum(_mass.get((mid, j), ())) + _corr.get((mid, j), 0.0)
                    for j in range(n) if (mid, j) in _mass]
        local_score = sum(loc_vals) / len(loc_vals) if loc_vals else 0.0
        non_holders = [i for i in range(n) if i != holder]
        rng2 = random.Random(hash((tid, rnd, c)) & 0xffffffff)
        rng2.shuffle(non_holders)                 # prespecified balanced rule
        focal_recv = non_holders[0]
        others = non_holders[1:]
        rows = []
        for h in triplet:
            if h < 1 or h - 1 > len(others):
                rows.append(("infeasible_h", None))
                continue
            base, full = build_packages(pool, n, mid, focal_recv, others, h, None)
            if base is None:
                rows.append(("infeasible_package", None))
                continue
            acc = {"base": [], "full": []}
            for rep in range(args.reps):
                tag = f"intv{rep}"
                try:
                    b = continue_to_answer(task, pool, sketches, own, seen, rtok,
                                           base, rnd, args.rounds, n, tag)
                    f = continue_to_answer(task, pool, sketches, own, seen, rtok,
                                           full, rnd, args.rounds, n, tag)
                except Exception:
                    rows.append(("decoding_failure", None))
                    acc = None
                    break
                acc["base"].append(b)
                acc["full"].append(f)
            if not acc:
                continue
            mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)  # noqa: E731
            row = {
                "task_id": tid, "split": splits.get(tid, "unassigned"),
                "round": rnd, "stage_t": rnd - 1, "d_t": d_t,
                "n_agents": n, "cluster": str(c), "mid": mid,
                "q_c": q_c, "h": h, "delta": m - h,
                "focal_receiver": focal_recv, "reps": args.reps,
                "item_tokens": item_tokens, "local_score": local_score,
                "y_terminal_f1": mean(acc["full"], "group_f1") - mean(acc["base"], "group_f1"),
                "base_group_f1": mean(acc["base"], "group_f1"),
                "full_group_f1": mean(acc["full"], "group_f1"),
                "d_next_f1": mean(acc["full"], "focal_f1") - mean(acc["base"], "focal_f1"),
                "d_new_gold": mean(acc["full"], "new_gold") - mean(acc["base"], "new_gold"),
            }
            rows.append(("ok", row))
        return rows

    todo = [j for j in jobs
            if not all((j[0], j[1], str(j[2]), h) in done for h in triplet)]
    print(f"{len(todo)} intervention triplets to run "
          f"(reps={args.reps}, N={n})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j for j in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rows = fut.result()
            except Exception:
                counters["worker_error"] += 1
                continue
            with lock:
                for status, row in rows:
                    counters[status] += 1
                    if row is not None:
                        fh.write(json.dumps(row) + "\n")
                fh.flush()
            if k % 10 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(todo)} triplets  {el/60:.1f} min  "
                      f"{dict(counters)}", flush=True)
    fh.close()
    summary = {"n_agents": n, "eligible_states": len(cand_states),
               "state_exclusions": dict(excl), "pair_status": dict(counters)}
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
