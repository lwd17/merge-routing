"""Response-model evaluation: collection summary, held-out prediction, boundary.

Produces the three measurement tables of the learned-response protocol.

1. Collection summary  -- questions, logged states, eligible states, and
   analyzed branch pairs per split, plus exclusion counts.

2. Held-out prediction -- task-balanced absolute error of g_phi against the
   MEASURED terminal labels, compared with a constant predictor fitted only
   on development data, plus a calibration slope (OLS of label on
   prediction). N=6,8 use the model frozen after N=4 fitting.

3. Boundary alignment  -- mu_{N,t}(delta), the task-balanced terminal
   marginal response at distance delta = m - h, with the two neighbour
   contrasts C_- = mu(1) - mu(2) and C_+ = mu(1) - mu(0). Both are computed
   on MATCHED states (questions holding both distances) and carry paired
   95% bootstrap intervals over questions. A positive late peak requires
   both contrasts positive; both are reported even when they disagree.

All marginals are measured effects in F1 points; model outputs are reported
separately and never presented as measurements.

Usage:
  python scripts/analyze_response.py --interventions runs/interventions/*.jsonl \\
      --model src/merge/search/response_model.npz --out runs/report/response.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.search.response import g_value, load_model, response_features  # noqa: E402

BOOT = 10000
BOOT_SEED = 20260912
STAGE = {1: "Early", 2: "Middle", 3: "Late"}


def load(paths):
    rows = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        for line in pp.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_balanced(vals_by_task):
    """Mean over questions of the within-question mean."""
    per = [np.mean(v) for v in vals_by_task.values() if len(v)]
    return float(np.mean(per)) if per else float("nan")


def boot_ci(per_task_vals, seed=BOOT_SEED, iters=BOOT):
    """Percentile bootstrap over questions of the task-balanced mean."""
    arr = np.array([np.mean(v) for v in per_task_vals if len(v)], dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(iters, len(arr)))
    means = arr[idx].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def collection_summary(rows, summaries):
    out = {}
    for split in ("discovery", "fitting", "selection", "confirmation",
                  "unassigned"):
        sub = [r for r in rows if r.get("split") == split]
        if not sub and split == "unassigned":
            continue
        out[split] = {
            "questions": len({r["task_id"] for r in sub}),
            "logged_states": len({(r["task_id"], r["round"]) for r in sub}),
            "eligible_states": len({(r["task_id"], r["round"], r["cluster"])
                                    for r in sub}),
            "analyzed_pairs": len(sub),
        }
    excl = Counter()
    for s in summaries:
        for k, v in (s.get("state_exclusions") or {}).items():
            excl[k] += v
        for k, v in (s.get("pair_status") or {}).items():
            if k != "ok":
                excl[k] += v
    out["exclusions"] = dict(excl)
    return out


def prediction_table(rows, model):
    """Held-out prediction vs a development-fitted constant predictor."""
    fit = [r for r in rows if r.get("split") == "fitting"]
    const = float(np.mean([r["y_terminal_f1"] for r in fit])) if fit else 0.0
    out = []
    for n in sorted({r["n_agents"] for r in rows}):
        held = [r for r in rows
                if r["n_agents"] == n and r.get("split") in
                ("selection", "confirmation", "discovery")]
        if not held:
            continue
        by_m, by_c, by_pair = defaultdict(list), defaultdict(list), defaultdict(list)
        preds, ys = [], []
        for r in held:
            x = response_features(r["d_t"], r["h"], r["n_agents"], r["q_c"])
            p = g_value(model, x)
            y = r["y_terminal_f1"]
            preds.append(p); ys.append(y)
            by_m[r["task_id"]].append(abs(p - y) * 100)     # F1 points
            by_c[r["task_id"]].append(abs(const - y) * 100)
            by_pair[r["task_id"]].append(1)
        preds_a, ys_a = np.array(preds), np.array(ys)
        if preds_a.std() > 1e-9:
            slope = float(np.polyfit(preds_a, ys_a, 1)[0])
        else:
            slope = float("nan")
        out.append({
            "N": n, "tasks": len(by_m), "pairs": len(held),
            "model_mae": task_balanced(by_m),
            "constant_mae": task_balanced(by_c),
            "calibration_slope": slope,
            "constant_value_points": const * 100,
        })
    return out


def boundary_table(rows):
    """mu(delta) and the two neighbour contrasts on matched states."""
    out = []
    for n in sorted({r["n_agents"] for r in rows}):
        for t in (1, 2, 3):
            sub = [r for r in rows if r["n_agents"] == n and r["stage_t"] == t]
            if not sub:
                continue
            # matched states: those observed at all of delta = 2, 1, 0
            by_state = defaultdict(dict)
            for r in sub:
                by_state[(r["task_id"], r["round"], r["cluster"])][r["delta"]] = r
            matched = {k: v for k, v in by_state.items()
                       if all(d in v for d in (0, 1, 2))}
            if not matched:
                out.append({"N": n, "stage": STAGE[t], "states": 0})
                continue
            mu, per_task = {}, {d: defaultdict(list) for d in (0, 1, 2)}
            for (tid, _r, _c), v in matched.items():
                for d in (0, 1, 2):
                    per_task[d][tid].append(v[d]["y_terminal_f1"] * 100)
            for d in (0, 1, 2):
                mu[d] = task_balanced(per_task[d])
            cm, cp = defaultdict(list), defaultdict(list)
            for (tid, _r, _c), v in matched.items():
                cm[tid].append((v[1]["y_terminal_f1"] - v[2]["y_terminal_f1"]) * 100)
                cp[tid].append((v[1]["y_terminal_f1"] - v[0]["y_terminal_f1"]) * 100)
            out.append({
                "N": n, "stage": STAGE[t], "states": len(matched),
                "questions": len({k[0] for k in matched}),
                "mu2": mu[2], "mu1": mu[1], "mu0": mu[0],
                "C_minus": task_balanced(cm),
                "C_minus_ci": boot_ci(list(cm.values())),
                "C_plus": task_balanced(cp),
                "C_plus_ci": boot_ci(list(cp.values())),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interventions", nargs="+", required=True)
    ap.add_argument("--model", default="src/merge/search/response_model.npz")
    ap.add_argument("--out", default="runs/report/response.json")
    args = ap.parse_args()

    rows = load(args.interventions)
    summaries = []
    for p in args.interventions:
        sp = Path(str(p) + ".summary.json")
        if sp.exists():
            summaries.append(json.loads(sp.read_text()))
    print(f"loaded {len(rows)} intervention pairs")
    model = load_model(args.model)
    report = {
        "collection": collection_summary(rows, summaries),
        "prediction": prediction_table(rows, model) if model else [],
        "boundary": boundary_table(rows),
        "model_meta": model["meta"] if model else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["collection"], indent=1))
    for r in report["prediction"]:
        print(f"N={r['N']} pairs={r['pairs']} model MAE={r['model_mae']:.2f} "
              f"const MAE={r['constant_mae']:.2f} slope={r['calibration_slope']:.3f}")
    for r in report["boundary"]:
        if r.get("states"):
            print(f"N={r['N']} {r['stage']:6s} states={r['states']:3d} "
                  f"mu2={r['mu2']:+.2f} mu1={r['mu1']:+.2f} mu0={r['mu0']:+.2f} "
                  f"C-={r['C_minus']:+.2f}{np.round(r['C_minus_ci'],2)} "
                  f"C+={r['C_plus']:+.2f}{np.round(r['C_plus_ci'],2)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
