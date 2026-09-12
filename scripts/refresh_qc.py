"""Recompute the cached cluster relevance q_c of logged interventions.

The response model reads q_c as one of its six inputs. q_c is produced by the
local scorer, so a scorer revision shifts that input's distribution and the
fitted coefficient no longer matches deployment. This rebuilds each logged
state and recomputes q_c with the CURRENT scorer, without re-running any LLM
call. The terminal labels are NOT touched: they were measured under the
continuation policy of the scorer revision in force at collection time, which
is recorded in the output so the limitation stays visible.

Usage:
  python scripts/refresh_qc.py --interventions runs/interventions/n4.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from merge.search.local_value import candidates, local_maps  # noqa: E402
from merge.search.pipeline import build_state  # noqa: E402
from merge.search.semantic_value import model_fingerprint  # noqa: E402
from collect_interventions import rebuild  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interventions", nargs="+", required=True)
    ap.add_argument("--records", nargs="+",
                    default=["runs/dev/records.jsonl",
                             "runs/dev_n6/records.jsonl",
                             "runs/dev_n8/records.jsonl"])
    args = ap.parse_args()

    recs = {}
    for p in args.records:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            recs.setdefault((r["task_id"], r["n_agents"]), r)
    fp = model_fingerprint()
    print(f"current local scorer fingerprint: {fp}")

    for path in args.interventions:
        p = Path(path)
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        cache = {}
        old, new, miss = [], [], 0
        for r in rows:
            key = (r["task_id"], r["n_agents"], r["round"])
            if key not in cache:
                rec = recs.get((r["task_id"], r["n_agents"]))
                if rec is None:
                    cache[key] = None
                else:
                    try:
                        pool, sk, own, seen, rtok = rebuild(rec, r["round"])
                        st = build_state(pool=pool, n_agents=r["n_agents"],
                                         rounds=rec.get("rounds", 4),
                                         rnd=r["round"], sketches=sk, own=own,
                                         received_tokens=rtok,
                                         question=rec["question"],
                                         task_id=r["task_id"], alloc_log=[])
                        c = candidates(pool, r["n_agents"])
                        _w, _m, _c, qc = local_maps(pool, st, c)
                        cache[key] = qc
                    except Exception:
                        cache[key] = None
            qc = cache[key]
            if qc is None or r["cluster"] not in qc:
                miss += 1
                continue
            old.append(r["q_c"])
            r["q_c_collected"] = r["q_c"]
            r["q_c"] = float(qc[r["cluster"]])
            r["scorer_fingerprint"] = fp
            new.append(r["q_c"])
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        o, n = np.array(old), np.array(new)
        print(f"{p.name}: refreshed {len(new)}/{len(rows)} rows "
              f"({miss} unresolvable)")
        if len(n):
            print(f"   q_c mean {o.mean():.4f} -> {n.mean():.4f} | "
                  f"median {np.median(o):.4f} -> {np.median(n):.4f} | "
                  f"frac<0.01 {100*(o<0.01).mean():.1f}% -> {100*(n<0.01).mean():.1f}%")


if __name__ == "__main__":
    main()
