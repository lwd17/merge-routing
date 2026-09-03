"""
musique_setup.py — Download MuSiQue, build pooled paragraph corpus, build BM25 index.

MuSiQue has 2-4 hop questions with 20 distractor paragraphs per question.
We only use answerable questions.

1. Download MuSiQue validation set (2417 answerable questions)
2. Pool all unique paragraphs → corpus
3. Save question manifest (30 probe questions) with gold doc IDs
4. Build pyserini BM25 index

Usage:
  python scripts/musique_setup.py
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "musique"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index_bm25"
MANIFEST_DIR = DATA_DIR / "manifests"

N_PROBE = 30
SEED = 20260803


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download MuSiQue ──
    raw_path = DATA_DIR / "musique_val.json"
    if raw_path.exists():
        print(f"Using cached {raw_path}")
        questions = json.loads(raw_path.read_text())
    else:
        print("Downloading MuSiQue validation set...")
        from datasets import load_dataset
        ds = load_dataset("bdsaglam/musique", split="validation")
        questions = [dict(item) for item in ds]
        raw_path.write_text(json.dumps(questions, ensure_ascii=False))
        print(f"  Saved {len(questions)} questions to {raw_path}")

    # Filter to answerable only
    answerable = [q for q in questions if q.get("answerable", True)]
    print(f"Total questions: {len(questions)}, answerable: {len(answerable)}")

    # ── Step 2: Build pooled paragraph corpus ──
    # Each paragraph: title + paragraph_text, dedup by (title, paragraph_text)
    seen = set()
    corpus = []
    doc_id_counter = 0
    # Map (title, paragraph_text_hash) → docid for gold matching
    para_to_docid = {}

    for q in answerable:
        for p in q["paragraphs"]:
            title = p["title"]
            text = p["paragraph_text"]
            key = (title, text[:200])  # dedup key
            if key not in seen:
                seen.add(key)
                docid = str(doc_id_counter)
                doc_id_counter += 1
                para_to_docid[key] = docid
                corpus.append({
                    "id": docid,
                    "contents": f"{title}\n\n{text}",
                })

    print(f"Corpus: {len(corpus)} unique paragraphs")

    # Write corpus as JSONL for pyserini
    corpus_path = CORPUS_DIR / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Corpus written to {corpus_path}")

    # ── Step 3: Build question manifest with gold doc IDs ──
    all_questions = []
    for q in answerable:
        gold_docids = []
        for p in q["paragraphs"]:
            if p["is_supporting"]:
                key = (p["title"], p["paragraph_text"][:200])
                did = para_to_docid.get(key)
                if did:
                    gold_docids.append(did)

        n_hops = len(q.get("question_decomposition", []))
        all_questions.append({
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "answer_aliases": q.get("answer_aliases", []),
            "n_hops": n_hops,
            "gold_docids": gold_docids,
            "n_gold": len(gold_docids),
        })

    # Distribution
    hop_dist = {}
    for q in all_questions:
        hop_dist[q["n_hops"]] = hop_dist.get(q["n_hops"], 0) + 1
    print(f"Hop distribution: {sorted(hop_dist.items())}")

    # ── Sample probe set ──
    rng = random.Random(SEED)

    # Prefer 3-4 hop questions (harder, more interesting for CQP)
    hop4 = [q for q in all_questions if q["n_hops"] == 4]
    hop3 = [q for q in all_questions if q["n_hops"] == 3]
    hop2 = [q for q in all_questions if q["n_hops"] == 2]

    print(f"  4-hop: {len(hop4)}, 3-hop: {len(hop3)}, 2-hop: {len(hop2)}")

    rng.shuffle(hop4)
    rng.shuffle(hop3)
    rng.shuffle(hop2)

    # Take mostly 3-4 hop (harder)
    probe = hop4[:12] + hop3[:12] + hop2[:6]
    rng.shuffle(probe)
    probe = probe[:N_PROBE]

    # Write probe manifest
    probe_path = MANIFEST_DIR / "probe_30.json"
    probe_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2))
    print(f"  Probe manifest: {len(probe)} questions → {probe_path}")

    # Write qrel file
    qrel_path = DATA_DIR / "qrel_gold.txt"
    with open(qrel_path, "w") as f:
        for q in probe:
            for docid in q["gold_docids"]:
                f.write(f"{q['id']} 0 {docid} 1\n")
    print(f"  Qrel file: {qrel_path}")

    # ── Step 4: Build pyserini BM25 index ──
    if INDEX_DIR.exists() and any(INDEX_DIR.glob("segments*")):
        print(f"Index already exists at {INDEX_DIR}, skipping build")
    else:
        print("Building BM25 index with pyserini...")
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(CORPUS_DIR),
            "--index", str(INDEX_DIR),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[:500]}")
            print(f"  Index build failed with code {result.returncode}")
            return
        print("  Index built successfully")

    # ── Step 5: Quick recall check ──
    print("\nQuick recall check on probe questions...")
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(str(INDEX_DIR))
        searcher.set_bm25(k1=0.9, b=0.4)

        recalls_at5 = []
        recalls_at20 = []
        for q in probe[:10]:
            hits = searcher.search(q["question"], k=20)
            hit_ids = [h.docid for h in hits]
            gold = set(q["gold_docids"])
            r5 = len(gold & set(hit_ids[:5])) / len(gold) if gold else 0
            r20 = len(gold & set(hit_ids[:20])) / len(gold) if gold else 0
            recalls_at5.append(r5)
            recalls_at20.append(r20)

        import numpy as np
        print(f"  Recall@5:  {np.mean(recalls_at5):.1%} (mean over 10 questions)")
        print(f"  Recall@20: {np.mean(recalls_at20):.1%}")
        regime = "good" if np.mean(recalls_at5) > 0.3 else "weak"
        print(f"  → Expected regime: BM25 recall is {regime}")
    except Exception as e:
        print(f"  Recall check failed: {e}")

    print(f"\n=== Setup complete ===")
    print(f"  Corpus: {len(corpus)} docs → {CORPUS_DIR}")
    print(f"  Index: {INDEX_DIR}")
    print(f"  Probe: {len(probe)} questions → {probe_path}")
    print(f"  Qrels: {qrel_path}")


if __name__ == "__main__":
    main()
