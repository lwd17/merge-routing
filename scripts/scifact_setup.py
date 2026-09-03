"""SciFact claim-verification setup (paper RQ4).

Builds the cross-regime transfer environment: the SciFact abstract corpus,
a BM25 index, and SUPPORT/REFUTE claim manifests. The evaluation manifest
samples 200 held-out labeled claims from the validation split; the
development manifest (scorer training and iteration) comes from the train
split, so development, scorer-training, and evaluation examples stay
disjoint.

Each task is phrased as a verification question whose gold answer is the
verdict string, so the shared agent stack (search, evidence items, routing,
answer aggregation) runs unchanged; only the retrieval corpus and the
answer target differ. Gold evidence sentences are stored next to the
manifest for the evidence-sentence F1 metric (scripts/analyze_scifact.py).

Usage:
  python scripts/scifact_setup.py           # writes data/scifact/...
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "scifact"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index_bm25"
MANIFEST_DIR = DATA_DIR / "manifests"

N_EVAL = 200
SEED = 20260902
VERDICT = {"SUPPORT": "SUPPORT", "CONTRADICT": "REFUTE"}


def main():
    for d in (DATA_DIR, CORPUS_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    corpus = load_dataset("allenai/scifact", "corpus", split="train")
    claims = {
        "train": load_dataset("allenai/scifact", "claims", split="train"),
        "validation": load_dataset("allenai/scifact", "claims",
                                   split="validation"),
    }

    # ---- corpus: one document per abstract ------------------------------
    sentences = {}
    title_to_docid = {}
    with (CORPUS_DIR / "docs.jsonl").open("w") as fh:
        for row in corpus:
            docid = str(row["doc_id"])
            sents = [s.strip() for s in row["abstract"]]
            sentences[docid] = sents
            title_to_docid[row["title"]] = docid
            fh.write(json.dumps({
                "id": docid,
                "contents": f"{row['title']}\n\n" + " ".join(sents),
            }) + "\n")
    (DATA_DIR / "sentences.json").write_text(json.dumps(sentences))
    (DATA_DIR / "title_to_docid.json").write_text(
        json.dumps(title_to_docid))
    print(f"corpus: {len(sentences)} abstracts")

    # ---- manifests ------------------------------------------------------
    def to_task(row):
        docs = [str(d) for d in row["cited_doc_ids"]]
        ev_sents = {}
        ev = row.get("evidence", {}) or {}
        if isinstance(ev, dict):
            for docid, anns in ev.items():
                idxs = sorted({i for ann in anns
                               for i in ann.get("sentences", [])})
                if idxs:
                    ev_sents[str(docid)] = idxs
        labels = {ann.get("label") for anns in (ev.values()
                  if isinstance(ev, dict) else []) for ann in anns}
        labels.discard(None)
        if len(labels) != 1:
            return None
        verdict = VERDICT.get(next(iter(labels)))
        if verdict is None:
            return None
        return {
            "id": f"scifact_{row['id']}",
            "question": ("Does the scientific literature SUPPORT or REFUTE "
                         f"this claim: \"{row['claim']}\"? "
                         "Answer with exactly one word: SUPPORT or REFUTE."),
            "answer": verdict,
            "gold_docids": docs,
            "evidence_sentences": ev_sents,
            "type": "claim_verification",
        }

    rng = random.Random(SEED)
    for split, name, cap in (("train", "scifact_dev.json", None),
                             ("validation", f"scifact_eval_{N_EVAL}.json",
                              N_EVAL)):
        tasks = [t for t in (to_task(r) for r in claims[split]) if t]
        rng.shuffle(tasks)
        if cap:
            tasks = tasks[:cap]
        (MANIFEST_DIR / name).write_text(json.dumps(tasks))
        print(f"{name}: {len(tasks)} claims")

    # ---- BM25 index -----------------------------------------------------
    cmd = [sys.executable, "-m", "pyserini.index.lucene",
           "--collection", "JsonCollection",
           "--input", str(CORPUS_DIR),
           "--index", str(INDEX_DIR),
           "--generator", "DefaultLuceneDocumentGenerator",
           "--threads", "4", "--storeRaw"]
    print("building BM25 index ...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise SystemExit("pyserini indexing failed")
    print(f"index ready at {INDEX_DIR}")
    print("Run with: CQP_DATASET=scifact python scripts/run_gurc_pilot.py "
          f"--manifest scifact_eval_{N_EVAL}.json ...")


if __name__ == "__main__":
    main()
