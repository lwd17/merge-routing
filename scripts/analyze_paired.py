"""Paired contrast analysis over one or more experiment runs.

Implements the paper's interval procedure exactly: for each arm, every task's
score is first averaged over the repetitions supplied for that arm; the paired
per-task difference is formed on the common task set; and the 95% confidence
interval is a nonparametric percentile bootstrap over tasks (10,000 resamples,
fixed seed). Repetitions are nested within tasks and are never treated as
independent observations.

Usage:
  # Two repetitions per arm, one records.jsonl each:
  python scripts/analyze_paired.py \
      --arm-a merge_d --arm-b capped_full \
      --records runs/exp_r1/records.jsonl runs/exp_r2/records.jsonl

  # Different runs may hold different arms; every file is scanned for both.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BOOT_ITERS = 10_000
BOOT_SEED = 20260810


def load(records_paths, arm):
    """task_id -> list of per-repetition (f1, em, comm) tuples."""
    per_task = defaultdict(list)
    for path in records_paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("arm") != arm:
                continue
            g = rec["group"]
            per_task[str(rec["task_id"])].append(
                (float(g["f1"]), float(g.get("em", 0.0)),
                 float(rec.get("comm_tokens", 0)))
            )
    return per_task


def rep_mean(per_task, idx):
    return {t: sum(v[idx] for v in reps) / len(reps)
            for t, reps in per_task.items()}


def bootstrap_ci(diffs):
    rng = random.Random(BOOT_SEED)
    n = len(diffs)
    means = []
    for _ in range(BOOT_ITERS):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * BOOT_ITERS)], means[int(0.975 * BOOT_ITERS) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True, help="first arm (reported as A-B)")
    ap.add_argument("--arm-b", required=True, help="second arm")
    ap.add_argument("--records", nargs="+", required=True,
                    help="one or more records.jsonl files (repetitions)")
    args = ap.parse_args()

    a = load(args.records, args.arm_a)
    b = load(args.records, args.arm_b)
    if not a or not b:
        raise SystemExit(
            f"arm not found: {args.arm_a}={len(a)} tasks, "
            f"{args.arm_b}={len(b)} tasks"
        )

    common = sorted(set(a) & set(b))
    print(f"tasks: {args.arm_a}={len(a)}  {args.arm_b}={len(b)}  paired={len(common)}")

    for name, idx in (("F1", 0), ("EM", 1)):
        ma, mb = rep_mean(a, idx), rep_mean(b, idx)
        diffs = [ma[t] - mb[t] for t in common]
        mean = sum(diffs) / len(diffs)
        lo, hi = bootstrap_ci(diffs)
        star = "*" if lo > 0 or hi < 0 else ""
        print(
            f"{name}: {args.arm_a}={sum(ma[t] for t in common)/len(common):.4f}  "
            f"{args.arm_b}={sum(mb[t] for t in common)/len(common):.4f}  "
            f"paired diff={mean*100:+.2f} pts  "
            f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}]{star}"
        )

    ca, cb = rep_mean(a, 2), rep_mean(b, 2)
    print(
        f"comm tokens: {args.arm_a}={sum(ca[t] for t in common)/len(common):.0f}  "
        f"{args.arm_b}={sum(cb[t] for t in common)/len(common):.0f}"
    )


if __name__ == "__main__":
    main()
