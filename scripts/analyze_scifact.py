"""SciFact metrics: verdict macro-F1 and evidence-sentence F1 (paper RQ4).

Verdict macro-F1 treats SUPPORT and REFUTE as two classes over the group
answer (an answer counts for a class when it contains exactly one of the
verdict words). Evidence-sentence F1 compares the gold evidence sentences
of each claim with the sentences implicated by the team's evidence items:
an evidence item implicates the sentence of its source abstract that best
matches the item text (token-F1), so precision counts implicated sentences
that are gold and recall counts gold sentences that were implicated. This
sentence-attribution rule is part of the released metric implementation.

Usage:
  python scripts/analyze_scifact.py --records runs/<run>/records.jsonl \\
      --data data/scifact [--arm merge_l]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.rq2_metrics import content_tokens  # noqa: E402


def token_f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def verdict_of(answer: str) -> str:
    low = answer.lower()
    sup, ref = "support" in low, "refut" in low
    if sup and not ref:
        return "SUPPORT"
    if ref and not sup:
        return "REFUTE"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--data", default="data/scifact")
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sentences = json.loads((root / args.data / "sentences.json").read_text())
    manifests = {}
    for mf in (root / args.data / "manifests").glob("*.json"):
        for t in json.loads(mf.read_text()):
            manifests[t["id"]] = t

    per_arm = defaultdict(lambda: {"conf": Counter(), "ef1": [], "comm": []})
    for line in Path(args.records).read_text().splitlines():
        rec = json.loads(line)
        if args.arm and rec["arm"] != args.arm:
            continue
        task = manifests.get(rec["task_id"])
        if task is None:
            continue
        agg = per_arm[rec["arm"]]
        pred = verdict_of(rec["group"]["answer"])
        agg["conf"][(task["answer"], pred)] += 1
        agg["comm"].append(rec.get("comm_tokens", 0))

        gold_sents = {(d, i) for d, idxs in
                      (task.get("evidence_sentences") or {}).items()
                      for i in idxs}
        implicated = set()
        for it in rec.get("memory_pool", []):
            docid = str(it.get("source_docid", ""))
            sents = sentences.get(docid)
            if not sents:
                continue
            toks = content_tokens(it["claim"])
            best, best_f1 = None, 0.0
            for si, s in enumerate(sents):
                f1 = token_f1(toks, content_tokens(s))
                if f1 > best_f1:
                    best, best_f1 = si, f1
            if best is not None and best_f1 >= 0.3:
                implicated.add((docid, best))
        if gold_sents:
            inter = len(gold_sents & implicated)
            p = inter / len(implicated) if implicated else 0.0
            r = inter / len(gold_sents)
            agg["ef1"].append(2 * p * r / (p + r) if p + r else 0.0)

    for arm, agg in sorted(per_arm.items()):
        f1s = []
        for cls in ("SUPPORT", "REFUTE"):
            tp = agg["conf"][(cls, cls)]
            fp = sum(v for (g, p), v in agg["conf"].items()
                     if p == cls and g != cls)
            fn = sum(v for (g, p), v in agg["conf"].items()
                     if g == cls and p != cls)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_ = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec_ / (prec + rec_)
                       if prec + rec_ else 0.0)
        n = sum(agg["conf"].values())
        ev = (sum(agg["ef1"]) / len(agg["ef1"])) if agg["ef1"] else 0.0
        comm = sum(agg["comm"]) / max(1, len(agg["comm"]))
        print(f"{arm:14s} n={n:4d}  verdict macro-F1={sum(f1s)/2:.3f}  "
              f"evidence-sentence F1={ev:.3f}  comm={comm:.0f}")


if __name__ == "__main__":
    main()
