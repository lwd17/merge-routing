"""Fixed-state branching replay (paper Section 5 and Table 2).

The replay clones logged receiver states and compares receiver-separable
against joint allocation with identical candidates, local scores, token
dose, and decoding seed. From each cloned state it executes exactly one
continuation step per affected receiver -- exposure -> next-query proposal
-> standard retrieval -> one answer call -- and measures

  * delta supporting-evidence recall: newly retrieved gold documents that
    no teammate had seen at the cloned state, and
  * next-step answer F1 from the receiver's one-step answer.

Because evidence items are indivisible, both packages are filled to the
largest common dose below the target and the residual gap is logged
(Appendix A.6). Optional probes:
  --focal    add one focal transfer whose cluster is already exposed to
             zero / one / two other receivers in the base package;
  --holders  additionally log the pre-existing holder count of every focal
             cluster so the transition into the three-agent majority can
             be analyzed.

Usage:
  CQP_DATA_ROOT=... CQP_MODEL=... CQP_API_BASES=... \\
  python scripts/replay_fixed_state.py --records runs/<run>/records.jsonl \\
      --states 900 --out runs/replay_<dataset>.jsonl [--learned] [--focal]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.rq2_metrics import content_tokens  # noqa: E402
from merge.search import prompts, qa_eval, retriever  # noqa: E402
from merge.search.engine import _chat  # noqa: E402
from merge.search.memory_pool import MemoryPool  # noqa: E402
from merge.search import welfare as W  # noqa: E402

SEED = 20260902


def rebuild_state(rec: dict, rnd: int):
    """Clone the routing-time state at communication round `rnd`."""
    n = rec["n_agents"]
    pool = MemoryPool(n)
    mid_map = {}
    routed_until = defaultdict(list)
    for ev in rec["events"]:
        if ev["round"] < rnd:
            for m in (ev.get("routed_mids") or []):
                routed_until[ev["agent"]].append(m)
    for it in sorted(rec["memory_pool"], key=lambda x: x["mid"]):
        if it.get("round_created", 1) >= rnd:
            continue
        new_mids = pool.add_claims(
            it.get("producer", 0), it.get("round_created", 1),
            [{"claim": it["claim"], "entities": it.get("entities", []),
              "source_title": it.get("source_title", "")}],
            {it.get("source_title", ""): it.get("source_docid", "")},
            {it.get("source_docid", "")},
            None,
        )
        if new_mids:
            mid_map[it["mid"]] = new_mids[0]
    received_tokens = [0] * n
    items_by_new = {x["mid"]: x for x in pool.items}
    for i in range(n):
        for old_mid in routed_until[i]:
            new_mid = mid_map.get(old_mid)
            if new_mid is not None:
                pool.holders[new_mid][i] = 2
                received_tokens[i] += items_by_new[new_mid]["tokens"]
    ev_at = {(e["agent"], e["round"]): e for e in rec["events"]}
    sketches, last_q, own, seen = [], {}, [], [set() for _ in range(n)]
    for i in range(n):
        prev = ev_at.get((i, rnd - 1), {})
        sk = prev.get("sketch", {}) or {}
        sketches.append({"unresolved_slots":
                         (sk.get("unresolved_slots") or [])[:6]})
        last_q[i] = prev.get("q1", "")
        rounds_i = []
        for r in range(1, rnd):
            e = ev_at.get((i, r))
            if e:
                rounds_i.append({
                    "round": r, "q1": e.get("q1", ""),
                    "docs": [{"title": t, "docid": d}
                             for t, d in zip(e.get("titles", []),
                                             e.get("docids", []))],
                })
                seen[i].update(e.get("docids", []))
        own.append(rounds_i)
    state = {
        "pool": pool, "n_agents": n, "round": rnd,
        "remaining_rounds": rec.get("rounds", 4) - rnd + 1,
        "question": rec["question"],
        "sketches": sketches,
        "held_tokens": [set().union(*(content_tokens(x["claim"])
                                      for x in pool.items
                                      if i in pool.holders[x["mid"]]))
                        if any(i in pool.holders[x["mid"]]
                               for x in pool.items) else set()
                        for i in range(n)],
        "received_tokens": received_tokens,
        "last_queries": last_q,
        "alloc_log": [],
    }
    return state, own, seen


def trim_to_dose(package, items, target):
    """Drop smallest-contribution transfers until total tokens <= target."""
    pkg = sorted(package, key=lambda tr: items[tr[0]]["tokens"])
    total = sum(items[m]["tokens"] for m, _ in pkg)
    out = list(pkg)
    while out and total > target:
        m, i = out.pop()  # drop the largest first to close the gap fastest
        total -= items[m]["tokens"]
    return set(out), total


def continuation(rec, state, own, seen, package, seed):
    """One exposure -> proposal -> retrieval -> answer step per receiver."""
    pool = state["pool"]
    items = {x["mid"]: x for x in pool.items}
    routed_by = defaultdict(list)
    for m, i in package:
        routed_by[i].append(items[m])
    team_seen = set().union(*seen) if seen else set()
    gold = set(rec.get("gold_docids", []))
    new_gold, f1s = 0, []
    for i, routed in sorted(routed_by.items()):
        sysp = prompts.system_prompt(
            {"role": "Investigator", "description": "replay clone."},
            state["n_agents"])
        raw = _chat(sysp,
                    prompts.propose_user(rec["question"], own[i][-4:],
                                         routed),
                    seed, task_id=rec["task_id"],
                    tag={"stage": "replay_propose"})
        q1 = (prompts.parse_json_field(raw, "query")
              or rec["question"])[:200]
        docs = retriever.search(q1, top_k=5, exclude_docids=seen[i])
        new_gold += len({d["docid"] for d in docs} & gold - team_seen)
        evidence = [{"docid": d["docid"], "title": d["title"],
                     "excerpt": d.get("excerpt", ""),
                     "round": state["round"]} for d in docs]
        evidence += [{"docid": f"mem:{it['mid']}",
                      "title": it["source_title"] or "team evidence",
                      "excerpt": it["claim"],
                      "round": it["round_created"]} for it in routed]
        ans_raw = _chat(sysp,
                        prompts.answer_user(rec["question"], evidence),
                        seed, task_id=rec["task_id"],
                        tag={"stage": "replay_answer"})
        ans = (prompts.parse_json_field(ans_raw, "answer")
               or ans_raw.strip()[:80])
        _, f1 = qa_eval.score(ans, rec.get("answer", ""),
                              rec.get("answer_aliases", []))
        f1s.append(f1)
    return new_gold, (sum(f1s) / len(f1s) if f1s else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", type=int, default=900)
    ap.add_argument("--learned", action="store_true",
                    help="use the learned local value (default: designed)")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--holders", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = [json.loads(x)
            for x in Path(args.records).read_text().splitlines()]
    states = [(ri, rnd) for ri in range(len(recs)) for rnd in (2, 3, 4)]
    rng.shuffle(states)
    states = states[: args.states]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build = W._learned_maps if args.learned else W._designed_maps
    n_done = 0
    with out.open("w") as fh:
        for ri, rnd in states:
            rec = recs[ri]
            try:
                state, own, seen = rebuild_state(rec, rnd)
            except Exception:
                continue
            pool = state["pool"]
            cands = W._candidates(pool, state["n_agents"])
            if not cands:
                continue
            weights, mass, corr, qc = build(pool, state, cands)
            mk = lambda phi: W.WelfareScorer(  # noqa: E731
                pool, state["n_agents"], weights, mass, corr, qc,
                state["received_tokens"], W._tau(state), use_phi=phi)
            sep_pkg, _ = W.optimize_welfare(mk(False), cands)
            joint_pkg, _ = W.optimize_welfare(mk(True), cands)
            items = {x["mid"]: x for x in pool.items}
            tot = lambda p: sum(items[m]["tokens"] for m, _ in p)  # noqa
            target = min(tot(sep_pkg), tot(joint_pkg))
            sep_pkg, sep_tok = trim_to_dose(sep_pkg, items, target)
            joint_pkg, joint_tok = trim_to_dose(joint_pkg, items, target)
            row = {
                "task_id": rec["task_id"], "arm": rec["arm"], "round": rnd,
                "dose_target": target, "sep_tokens": sep_tok,
                "joint_tokens": joint_tok,
                "residual_gap": target - min(sep_tok, joint_tok),
            }
            if args.focal and joint_pkg:
                focal = rng.choice(sorted(joint_pkg))
                c = mk(True).cluster_of[focal[0]]
                others = sum(1 for m, i in joint_pkg
                             if mk(True).cluster_of[m] == c
                             and (m, i) != focal)
                row["focal_other_receivers"] = others
                if args.holders:
                    row["focal_prior_holders"] = len(
                        mk(True).base_holders.get(c, ()))
            g_sep, f_sep = continuation(rec, state, own, seen,
                                        sep_pkg, args.seed)
            g_joint, f_joint = continuation(rec, state, own, seen,
                                            joint_pkg, args.seed)
            row.update({
                "sep_new_gold": g_sep, "joint_new_gold": g_joint,
                "sep_next_f1": round(f_sep, 4),
                "joint_next_f1": round(f_joint, 4),
            })
            fh.write(json.dumps(row) + "\n")
            n_done += 1
    print(f"replayed {n_done} states -> {out}")


if __name__ == "__main__":
    main()
