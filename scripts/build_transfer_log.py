"""Transfer-level log built post-hoc from
pilot records. One row per actual evidence transfer (c -> i, round t):

  cluster/provenance metadata, gap relevance at send time, context overlap,
  receiver's next-round query change / new docs / new global clusters,
  final F1/EM, paired terminal uplift vs the none arm, rescue/harm flags.

This is the raw material for training the learned local value
(scripts/train_semantic_value.py).

Usage:
  python scripts/build_transfer_log.py --records runs/gurc_pilot/records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge.rq2_metrics import content_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    args = ap.parse_args()

    recs = defaultdict(dict)
    for l in Path(args.records).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        recs[r["arm"]][r["task_id"]] = r

    none_arm = recs.get("no_comm", {})
    out_path = Path(args.records).with_name("transfer_log.jsonl")
    rows = 0
    with out_path.open("w") as out:
        for arm, by_task in recs.items():
            if arm == "no_comm":
                continue
            for tid, r in by_task.items():
                pool = {m["mid"]: m for m in r.get("memory_pool", [])}
                gold = set(r["gold_docids"])
                base = none_arm.get(tid)
                # index events by (agent, round)
                ev = {(e["agent"], e["round"]): e for e in r["events"]}
                team_docs_before = defaultdict(set)
                seen = set()
                for rnd in range(1, r["rounds"] + 1):
                    for a in range(r["n_agents"]):
                        team_docs_before[rnd] |= set()
                    for a in range(r["n_agents"]):
                        e = ev.get((a, rnd))
                        if e:
                            seen |= set(e["docids"])
                    team_docs_before[rnd + 1] = set(seen)

                for (a, rnd), e in sorted(ev.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                    if not e["routed_mids"]:
                        continue
                    nxt = ev.get((a, rnd + 1))
                    sk_prev = ev.get((a, rnd - 1), {}).get("sketch", {}) or {}
                    gap_tokens = set()
                    for s in (sk_prev.get("unresolved_slots") or []):
                        gap_tokens |= content_tokens(s)
                    held_docs = {
                        d for rr in range(1, rnd) for d in ev.get((a, rr), {}).get("docids", [])
                    }
                    for mid in e["routed_mids"]:
                        m = pool.get(mid, {})
                        claim_toks = content_tokens(m.get("claim", ""))
                        row = {
                            "task_id": tid,
                            "arm": arm,
                            "round": rnd,
                            "receiver": a,
                            "mid": mid,
                            "producer": m.get("producer"),
                            "source_family": m.get("source_family_id"),
                            "semantic_cluster": m.get("semantic_cluster_id"),
                            "tokens": m.get("tokens"),
                            "gold_source": m.get("source_docid") in gold,
                            "gap_relevance": len(claim_toks & gap_tokens),
                            "receiver_already_held_source": m.get("source_docid") in held_docs,
                            "next_query_change": (
                                None if not nxt else
                                len(content_tokens(nxt["q1"]) - content_tokens(e["q1"]))
                            ),
                            "next_new_docs": (
                                None if not nxt else
                                len(set(nxt["docids"]) - team_docs_before[rnd + 1])
                            ),
                            "next_new_gold": (
                                None if not nxt else
                                len((set(nxt["docids"]) & gold) - team_docs_before[rnd + 1])
                            ),
                            "final_group_em": r["group"]["em"],
                            "final_group_f1": r["group"]["f1"],
                            "uplift_em": (
                                None if base is None else r["group"]["em"] - base["group"]["em"]
                            ),
                            "uplift_f1": (
                                None if base is None else
                                round(r["group"]["f1"] - base["group"]["f1"], 4)
                            ),
                            "rescue": (
                                None if base is None else
                                int(r["group"]["em"] == 1 and base["group"]["em"] == 0)
                            ),
                            "harm": (
                                None if base is None else
                                int(r["group"]["em"] == 0 and base["group"]["em"] == 1)
                            ),
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows += 1
    print(f"wrote {rows} transfer rows -> {out_path}")


if __name__ == "__main__":
    main()
