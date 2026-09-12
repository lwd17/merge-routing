"""Build MuSiQue manifest + title_to_docid for the MERGE apparatus.

- gold_docids: supporting paragraphs matched to corpus docids by
  (title, body-prefix) — the corpus was built from task paragraphs.
- stratified by hop count (len(question_decomposition)): 160/120/120 for 2/3/4.
- title_to_docid.json: link-channel external index (first docid per title).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data" / "musique"
SEED = 20260811
N_BY_HOP = {2: 160, 3: 120, 4: 120}


def main():
    key2doc = {}
    title2doc = {}
    with open(BASE / "corpus" / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            contents = d["contents"]
            title, _, body = contents.partition("\n\n")
            title = title.strip()
            key2doc[(title, body.strip()[:80])] = str(d["id"])
            title2doc.setdefault(title, str(d["id"]))
    print(f"corpus: {len(key2doc)} docs, {len(title2doc)} titles")
    (BASE / "title_to_docid.json").write_text(
        json.dumps(title2doc, ensure_ascii=False))

    val = json.loads((BASE / "musique_val.json").read_text())
    pool = {2: [], 3: [], 4: []}
    unresolved = 0
    for t in val:
        if not t.get("answerable", True):
            continue
        hops = len(t.get("question_decomposition") or [])
        if hops not in pool:
            continue
        gold = []
        ok = True
        for p in t["paragraphs"]:
            if not p.get("is_supporting"):
                continue
            did = key2doc.get(
                (p["title"].strip(), p["paragraph_text"].strip()[:80]))
            if did is None:
                ok = False
                break
            gold.append(did)
        if not ok or len(gold) != hops:
            unresolved += 1
            continue
        pool[hops].append({
            "id": str(t["id"]), "question": t["question"],
            "answer": t["answer"],
            "answer_aliases": t.get("answer_aliases", []),
            "type": f"{hops}hop", "level": "musique",
            "gold_docids": sorted(set(gold)),
        })
    print("pool sizes:", {k: len(v) for k, v in pool.items()},
          "unresolved:", unresolved)
    rng = random.Random(SEED)
    out = []
    for h, n in N_BY_HOP.items():
        rng.shuffle(pool[h])
        out.extend(pool[h][:n])
    rng.shuffle(out)
    md = BASE / "manifests"
    md.mkdir(exist_ok=True)
    (md / "musique_400.json").write_text(json.dumps(out, ensure_ascii=False))
    from collections import Counter
    print("manifest:", len(out), dict(Counter(t["type"] for t in out)))


if __name__ == "__main__":
    main()
