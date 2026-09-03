"""
make_hotpot_manifest.py — Draw a fresh stratified HotpotQA manifest.

manifests/probe_30.json (25 questions) must NOT be reused for the headline
result: the top_k=2 regime was chosen by looking at those questions, so
reporting on them is selection on the outcome. This draws a disjoint pool from
the 7,405 dev questions and keeps probe_30 as the dev set for prompt iteration
and gate tuning.

  python scripts/make_hotpot_manifest.py --n 200 --out main_200.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="main_200.json")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--level", default="hard")
    ap.add_argument("--frac-bridge", type=float, default=0.8)
    # Comma-separated: every previously-touched pool must be excluded, not just
    # the most recent one. A single --exclude let probe_30 back into the n=200
    # extension pool; it was disjoint only by chance.
    ap.add_argument("--exclude", default="probe_30.json",
                    help="comma-separated manifest filenames to exclude")
    args = ap.parse_args()

    base = REPO_ROOT / "data" / "hotpotqa"
    data = json.loads((base / "hotpotqa_dev.json").read_text())
    title_to_docid = json.loads((base / "title_to_docid.json").read_text())

    excluded, ex_names = set(), []
    for name in [x.strip() for x in args.exclude.split(",") if x.strip()]:
        ex_path = base / "manifests" / name
        if ex_path.exists():
            ids = {str(t.get("id") or t.get("task_id")) for t in json.loads(ex_path.read_text())}
            excluded |= ids
            ex_names.append(f"{name}({len(ids)})")
        else:
            print(f"  WARNING: exclude file not found, NOT excluded: {name}")
    print(f"excluding {len(excluded)} ids from {', '.join(ex_names) or '(none)'}")

    pool = []
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id"))
        if qid in excluded:
            continue
        if args.level and ex.get("level") != args.level:
            continue
        titles = [t for t, _ in ex.get("supporting_facts", [])] if \
            isinstance(ex.get("supporting_facts"), list) else []
        if not titles:
            sf = ex.get("supporting_facts") or {}
            titles = sf.get("title", []) if isinstance(sf, dict) else []
        gold = sorted({title_to_docid[t] for t in set(titles) if t in title_to_docid})
        if len(gold) < 2:          # need >=2 gold docs or composition cannot matter
            continue
        pool.append({
            "id": qid,
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_aliases": [],
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "gold_docids": gold,
        })

    bridge = [x for x in pool if x["type"] == "bridge"]
    comp = [x for x in pool if x["type"] != "bridge"]
    print(f"pool: {len(pool)} ({len(bridge)} bridge, {len(comp)} comparison)")

    rng = random.Random(args.seed)
    rng.shuffle(bridge)
    rng.shuffle(comp)
    n_bridge = min(len(bridge), int(args.n * args.frac_bridge))
    n_comp = min(len(comp), args.n - n_bridge)
    chosen = bridge[:n_bridge] + comp[:n_comp]
    rng.shuffle(chosen)

    out_path = base / "manifests" / args.out
    out_path.write_text(json.dumps(chosen, indent=1))

    qrel_path = base / f"qrel_{Path(args.out).stem}.txt"
    with open(qrel_path, "w") as f:
        for t in chosen:
            for d in t["gold_docids"]:
                f.write(f"{t['id']} 0 {d} 1\n")

    from collections import Counter
    print(f"wrote {len(chosen)} -> {out_path}")
    print(f"       qrels -> {qrel_path}")
    print("types:", Counter(t["type"] for t in chosen))
    print("gold docs/q:", Counter(len(t["gold_docids"]) for t in chosen))


if __name__ == "__main__":
    main()
