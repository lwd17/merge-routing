"""Select the exposure weight on the development selection split.

The response potential is on a terminal-F1 scale, so the historical
w_exp = 0.4 of the structural controller is not assumed appropriate. This
script runs the candidate weights on the SELECTION split only, under a fixed
search budget shared by the learned alternatives, and reports the choice.
Evaluation questions are never used here.

Usage:
  python scripts/select_weight.py --arm merge_response --env CQP_WEXP_RESP \\
      --weights 0.25,0.5,1.0,2.0 --manifest selection_60.json --out runs/select
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_f1(path, arm):
    acc = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] == arm:
            acc[r["task_id"]].append(r["group"]["f1"])
    if not acc:
        return float("nan"), 0
    return float(np.mean([np.mean(v) for v in acc.values()])), len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="merge_response")
    ap.add_argument("--env", default="CQP_WEXP_RESP")
    ap.add_argument("--weights", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--manifest", default="selection_60.json")
    ap.add_argument("--out", default="runs/select")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--caches", default="runs/caches")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for w in [x.strip() for x in args.weights.split(",")]:
        d = out_root / f"{args.arm}_w{w}"
        env = dict(os.environ, **{args.env: w})
        cmd = [sys.executable, "scripts/run_experiment.py",
               "--manifest", args.manifest, "--caches", args.caches,
               "--out", str(d), "--arms", args.arm,
               "--workers", str(args.workers), "--seeds", "1"]
        print(f"--- {args.env}={w}", flush=True)
        subprocess.run(cmd, env=env, check=True)
        f1, n = mean_f1(d / "records.jsonl", args.arm)
        results.append({"weight": float(w), "f1": f1, "questions": n})
        print(f"    {args.env}={w}: F1 {f1:.4f} over {n} questions", flush=True)
    results.sort(key=lambda r: -r["f1"])
    best = results[0]
    (out_root / f"selection_{args.arm}.json").write_text(
        json.dumps({"arm": args.arm, "env": args.env, "split": args.manifest,
                    "candidates": results, "selected": best}, indent=1))
    print(f"\nselected {args.env}={best['weight']} (selection F1 {best['f1']:.4f})")


if __name__ == "__main__":
    main()
