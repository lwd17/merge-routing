"""Build 2WikiMultihopQA assets for the MERGE apparatus:
corpus.jsonl (from dev contexts) + BM25 index + title_to_docid + manifest.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "2wiki_raw" / "dev.json"
BASE = Path(__file__).resolve().parents[1] / "data" / "2wiki"
SEED = 20260812
N = 400


def main():
    data = json.loads(RAW.read_text())
    print(f"dev: {len(data)} tasks")
    BASE.mkdir(exist_ok=True)
    (BASE / "corpus").mkdir(exist_ok=True)
    (BASE / "manifests").mkdir(exist_ok=True)

    title2doc = {}
    docs = []
    for t in data:
        for title, sents in t["context"]:
            title = title.strip()
            if title in title2doc:
                continue
            did = str(len(docs))
            title2doc[title] = did
            docs.append({"id": did,
                         "contents": f"{title}\n\n{' '.join(sents)}"})
    with open(BASE / "corpus" / "corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (BASE / "title_to_docid.json").write_text(
        json.dumps(title2doc, ensure_ascii=False))
    print(f"corpus: {len(docs)} docs")

    pool = []
    for t in data:
        gold_titles = {sf[0].strip() for sf in t.get("supporting_facts", [])}
        gold = [title2doc[x] for x in gold_titles if x in title2doc]
        if len(gold) < 2:
            continue
        pool.append({
            "id": str(t["_id"]), "question": t["question"],
            "answer": t["answer"], "answer_aliases": [],
            "type": t.get("type", "?"), "level": "2wiki",
            "gold_docids": sorted(gold),
        })
    rng = random.Random(SEED)
    # stratify by type proportionally
    by_type = {}
    for p in pool:
        by_type.setdefault(p["type"], []).append(p)
    out = []
    total = sum(len(v) for v in by_type.values())
    for ty, items in sorted(by_type.items()):
        k = max(1, round(N * len(items) / total))
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    out = out[:N]
    (BASE / "manifests" / "2wiki_400.json").write_text(
        json.dumps(out, ensure_ascii=False))
    print("manifest:", len(out), dict(Counter(t["type"] for t in out)))

    r = subprocess.run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(BASE / "corpus"),
        "--index", str(BASE / "index_bm25"),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8", "--storeRaw",
    ], capture_output=True, text=True)
    print("index rc:", r.returncode)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        sys.exit(1)
    print("2wiki assets complete")


if __name__ == "__main__":
    main()
