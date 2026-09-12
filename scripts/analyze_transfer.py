"""Frozen-response transfer table: structural vs response per dataset/backbone.

The N=4 HotpotQA response model is frozen and applied without refitting.
For each (dataset, backbone) cell, both arms are run on the same questions
with the same solver; the paired difference is averaged within question over
repetitions and bootstrapped over questions.

Usage:
  python scripts/analyze_transfer.py \\
      --cell HotpotQA:Qwen3-8B:runs/confirm/records.jsonl \\
      --cell MuSiQue:Qwen3-8B:runs/transfer_musique/records.jsonl \\
      --out runs/report/transfer.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 10000
BOOT_SEED = 20260912


def per_question_f1(path, arm):
    """Per-question mean F1 over repetitions, de-duplicated on (task, seed)."""
    acc = defaultdict(list)
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue              # interleaved append from a concurrent writer
        if r["arm"] != arm:
            continue
        key = (r["task_id"], r.get("seed", 0))
        if key in seen:
            continue
        seen.add(key)
        acc[r["task_id"]].append(r["group"]["f1"])
    return {t: float(np.mean(v)) for t, v in acc.items()}


def paired(a, b, seed=BOOT_SEED, iters=BOOT):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float("nan"), (float("nan"), float("nan")), 0
    d = np.array([a[t] - b[t] for t in common])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(iters, len(d)))
    m = d[idx].mean(1)
    return (float(d.mean()),
            (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))),
            len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="Dataset:Backbone:path/to/records.jsonl")
    ap.add_argument("--struct-arm", default="structural")
    ap.add_argument("--resp-arm", default="merge_response")
    ap.add_argument("--out", default="runs/report/transfer.json")
    args = ap.parse_args()

    rows = []
    for spec in args.cell:
        ds, bb, path = spec.split(":", 2)
        if not Path(path).exists():
            rows.append({"dataset": ds, "backbone": bb,
                         "structural_f1": None, "response_f1": None,
                         "delta_points": None, "ci_points": None,
                         "questions": 0})
            continue
        s = per_question_f1(path, args.struct_arm)
        r = per_question_f1(path, args.resp_arm)
        d, ci, n = paired(r, s)
        rows.append({
            "dataset": ds, "backbone": bb,
            "structural_f1": float(np.mean(list(s.values()))) if s else None,
            "response_f1": float(np.mean(list(r.values()))) if r else None,
            "delta_points": d * 100 if d == d else None,
            "ci_points": [c * 100 for c in ci] if ci[0] == ci[0] else None,
            "questions": n,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))
    for r in rows:
        sf = "n/a" if r["structural_f1"] is None else f"{r['structural_f1']:.4f}"
        rf = "n/a" if r["response_f1"] is None else f"{r['response_f1']:.4f}"
        dd = "n/a" if r["delta_points"] is None else f"{r['delta_points']:+.2f}"
        print(f"{r['dataset']:10s} {r['backbone']:12s} struct={sf} resp={rf} "
              f"delta={dd} n={r['questions']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
