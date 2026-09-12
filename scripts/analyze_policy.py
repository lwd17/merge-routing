"""Paired policy comparison with question-level bootstrap intervals.

Repetitions (decoding seeds) are averaged WITHIN each question first, so
they are never treated as independent observations; the paired per-question
difference is then formed on the common question set and the 95% interval is
a percentile bootstrap over questions (10,000 resamples, fixed seed).

Reported per arm: terminal group F1, paired difference against the reference
arm, Total Tokens (input+output across every LLM call), average end-to-end
latency, and cost relative to same-manifest capped full communication.

Usage:
  python scripts/analyze_policy.py --records runs/confirm/records.jsonl \\
      --ref merge_response --out runs/report/policy.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 10000
BOOT_SEED = 20260912


def load(paths):
    """arm -> task -> [records], de-duplicated on (task, arm, seed).

    The records file is an append-only log, so a resumed or concurrent run
    can append the same (task, arm, seed) twice. Keeping the first occurrence
    prevents a question from being double-weighted in the paired means.
    """
    per = defaultdict(lambda: defaultdict(list))
    seen = set()
    dupes = 0
    bad = 0
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # interleaved append from a concurrent writer
                continue
            key = (r["task_id"], r["arm"], r.get("seed", 0))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            per[r["arm"]][r["task_id"]].append(r)
    if dupes or bad:
        print(f"(dropped {dupes} duplicate and {bad} malformed records)")
    return per


def arm_means(per_arm):
    """Per-question means over repetitions."""
    return {
        "f1": {t: float(np.mean([x["group"]["f1"] for x in v]))
               for t, v in per_arm.items()},
        "em": {t: float(np.mean([x["group"]["em"] for x in v]))
               for t, v in per_arm.items()},
        "tokens": {t: float(np.mean([x.get("total_tokens", 0) for x in v]))
                   for t, v in per_arm.items()},
        "latency": {t: float(np.mean([x.get("wall_seconds", 0) for x in v]))
                    for t, v in per_arm.items()},
        "comm": {t: float(np.mean([x.get("comm_tokens", 0) for x in v]))
                 for t, v in per_arm.items()},
        "routed_items": {t: float(np.mean([
            sum(1 for e in x.get("events", []) for _ in (e.get("routed_mids") or []))
            for x in v])) for t, v in per_arm.items()},
        "reps": {t: len(v) for t, v in per_arm.items()},
    }


def paired_ci(a_map, b_map, seed=BOOT_SEED, iters=BOOT):
    """Paired percentile bootstrap over the common question set (a - b)."""
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 2:
        return float("nan"), (float("nan"), float("nan")), 0
    d = np.array([a_map[t] - b_map[t] for t in common], dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(iters, len(d)))
    means = d[idx].mean(1)
    return (float(d.mean()),
            (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
            len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--ref", default="merge_response")
    ap.add_argument("--cost-base", default="capped_full")
    ap.add_argument("--out", default="runs/report/policy.json")
    args = ap.parse_args()

    per = load(args.records)
    stats = {arm: arm_means(v) for arm, v in per.items()}
    if args.ref not in stats:
        raise SystemExit(f"reference arm {args.ref} absent; have {list(stats)}")
    base_tok = (float(np.mean(list(stats[args.cost_base]["tokens"].values())))
                if args.cost_base in stats else float("nan"))

    rows = []
    for arm in sorted(stats):
        s = stats[arm]
        f1 = float(np.mean(list(s["f1"].values())))
        em = float(np.mean(list(s["em"].values())))
        tok = float(np.mean(list(s["tokens"].values())))
        lat = float(np.mean(list(s["latency"].values())))
        comm = float(np.mean(list(s["comm"].values())))
        routed = float(np.mean(list(s["routed_items"].values())))
        if arm == args.ref:
            diff, ci, npair = float("nan"), (float("nan"), float("nan")), 0
        else:
            diff, ci, npair = paired_ci(stats[args.ref]["f1"], s["f1"])
        rows.append({
            "arm": arm, "questions": len(s["f1"]),
            "reps": int(np.mean(list(s["reps"].values()))),
            "f1": f1, "em": em,
            "delta_vs_ref_points": diff * 100 if diff == diff else None,
            "ci_points": [c * 100 for c in ci] if ci[0] == ci[0] else None,
            "paired_questions": npair,
            "total_tokens": tok, "avg_latency_s": lat, "comm_tokens": comm,
            "routed_items": routed,
            "cost_vs_capped_full": tok / base_tok if base_tok == base_tok else None,
        })
    rows.sort(key=lambda r: -r["f1"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = {"capped_full": "capped full"}.get(args.cost_base, args.cost_base)
    out.write_text(json.dumps({"reference": args.ref,
                               "cost_base": args.cost_base,
                               "cost_base_label": label,
                               "rows": rows}, indent=1, default=float))
    print(f"{'arm':18s} {'F1':>6} {'d vs ref':>18} {'tokens':>8} {'lat':>6} {'cost':>5} {'n':>4}")
    for r in rows:
        d = ("---" if r["delta_vs_ref_points"] is None
             else f"{r['delta_vs_ref_points']:+.2f} [{r['ci_points'][0]:+.2f},{r['ci_points'][1]:+.2f}]")
        print(f"{r['arm']:18s} {r['f1']:.4f} {d:>18} {r['total_tokens']:8.0f} "
              f"{r['avg_latency_s']:6.1f} "
              f"{(r['cost_vs_capped_full'] or float('nan')):5.2f} {r['questions']:4d}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
