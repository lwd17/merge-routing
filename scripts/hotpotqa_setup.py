"""
hotpotqa_setup.py — Download HotpotQA, build pooled paragraph corpus, build BM25 index.

1. Download HotpotQA distractor dev set (~7405 questions)
2. Pool all unique paragraphs by title → corpus
3. Save question manifest (30 probe questions) with gold doc IDs
4. Build pyserini BM25 index

Usage:
  python scripts/hotpotqa_setup.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "hotpotqa"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index_bm25"
MANIFEST_DIR = DATA_DIR / "manifests"

N_PROBE = 30
SEED = 20260803


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download HotpotQA ────────────────────────────────────────────
    raw_path = DATA_DIR / "hotpotqa_dev.json"
    if raw_path.exists():
        print(f"Using cached {raw_path}")
        questions = json.loads(raw_path.read_text())
    else:
        print("Downloading HotpotQA distractor dev set...")
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        questions = [dict(item) for item in ds]
        raw_path.write_text(json.dumps(questions, ensure_ascii=False))
        print(f"  Saved {len(questions)} questions to {raw_path}")

    print(f"Total questions: {len(questions)}")

    # ── Step 2: Build pooled paragraph corpus ────────────────────────────────
    # Each doc: title + concatenated sentences
    # Dedup by title
    title_to_doc = {}
    for q in questions:
        ctx = q["context"]
        titles = ctx["title"]
        sentences_list = ctx["sentences"]
        for title, sents in zip(titles, sentences_list):
            if title not in title_to_doc:
                text = " ".join(sents)
                title_to_doc[title] = {
                    "title": title,
                    "text": text,
                }

    # Assign stable doc IDs (hash of title)
    corpus = []
    title_to_docid = {}
    for i, (title, doc) in enumerate(sorted(title_to_doc.items())):
        docid = str(i)
        title_to_docid[title] = docid
        corpus.append({
            "id": docid,
            "contents": f"{doc['title']}\n\n{doc['text']}",
        })

    print(f"Corpus: {len(corpus)} unique paragraphs (deduped by title)")

    # Write corpus as JSONL for pyserini
    corpus_path = CORPUS_DIR / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Corpus written to {corpus_path}")

    # Save title-to-docid mapping
    (DATA_DIR / "title_to_docid.json").write_text(
        json.dumps(title_to_docid, ensure_ascii=False, indent=2)
    )

    # ── Step 3: Build question manifest with gold doc IDs ────────────────────
    # For each question, identify gold doc IDs from supporting_facts titles
    all_questions = []
    for q in questions:
        sf = q["supporting_facts"]
        gold_titles = list(set(sf["title"]))
        gold_docids = [title_to_docid[t] for t in gold_titles if t in title_to_docid]

        all_questions.append({
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q["type"],
            "level": q["level"],
            "gold_titles": gold_titles,
            "gold_docids": gold_docids,
            "n_gold": len(gold_docids),
        })

    # Filter: only keep questions with exactly 2 gold docs (standard HotpotQA)
    valid = [q for q in all_questions if q["n_gold"] == 2]
    print(f"Valid questions (2 gold docs): {len(valid)}/{len(all_questions)}")

    # Sample probe set — prefer "hard" level, "bridge" type
    rng = random.Random(SEED)

    # Stratify: hard-bridge first, then hard-comparison, then medium
    hard_bridge = [q for q in valid if q["level"] == "hard" and q["type"] == "bridge"]
    hard_comparison = [q for q in valid if q["level"] == "hard" and q["type"] == "comparison"]
    medium = [q for q in valid if q["level"] == "medium"]

    print(f"  hard-bridge: {len(hard_bridge)}, hard-comparison: {len(hard_comparison)}, medium: {len(medium)}")

    # Take mostly hard-bridge (most similar to multi-hop deep research)
    rng.shuffle(hard_bridge)
    rng.shuffle(hard_comparison)
    rng.shuffle(medium)

    probe = hard_bridge[:20] + hard_comparison[:5] + medium[:5]
    rng.shuffle(probe)
    probe = probe[:N_PROBE]

    # Write probe manifest
    probe_path = MANIFEST_DIR / "probe_30.json"
    probe_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2))
    print(f"  Probe manifest: {len(probe)} questions → {probe_path}")

    # Also write qrel file (TREC format)
    qrel_path = DATA_DIR / "qrel_gold.txt"
    with open(qrel_path, "w") as f:
        for q in probe:
            for docid in q["gold_docids"]:
                f.write(f"{q['id']} 0 {docid} 1\n")
    print(f"  Qrel file: {qrel_path}")

    # ── Step 4: Build pyserini BM25 index ────────────────────────────────────
    if INDEX_DIR.exists() and (INDEX_DIR / "segments_1").exists():
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

    # ── Step 5: Quick recall check ───────────────────────────────────────────
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
            r5 = len(gold & set(hit_ids[:5])) / len(gold)
            r20 = len(gold & set(hit_ids[:20])) / len(gold)
            recalls_at5.append(r5)
            recalls_at20.append(r20)

        import numpy as np
        print(f"  Recall@5:  {np.mean(recalls_at5):.1%} (mean over 10 questions)")
        print(f"  Recall@20: {np.mean(recalls_at20):.1%}")
        print(f"  → Expected regime: BM25 recall is {'good' if np.mean(recalls_at5) > 0.3 else 'weak'}")
    except Exception as e:
        print(f"  Recall check failed: {e}")

    print("\n=== Setup complete ===")
    print(f"  Corpus: {len(corpus)} docs → {CORPUS_DIR}")
    print(f"  Index: {INDEX_DIR}")
    print(f"  Probe: {len(probe)} questions → {probe_path}")
    print(f"  Qrels: {qrel_path}")


if __name__ == "__main__":
    main()
