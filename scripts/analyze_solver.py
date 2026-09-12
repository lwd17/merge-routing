"""Small-state solver diagnostic: greedy vs beam/compound vs exact.

On cached allocation states with at most 18 eligible candidates, enumerate
every feasible package exactly and compare it with the two search procedures
under the SAME learned-response objective:

  * positive-singleton greedy construction with single add/drop/swap moves
    (the retained solver), and
  * beam construction with compound local moves (the common solver).

Reported per solver: number of states, optimal-value match rate, median and
P90 normalized gap, and latency. A match means the objective value is within
an absolute tolerance of 1e-8 of the exact optimum -- ties count as matches
rather than package-identity failures. The normalized gap is
100 (F_exact - F_solver) / max(|F_exact|, 1e-8). The empty feasible action
has value zero and is a candidate in all three searches.

Usage:
  python scripts/analyze_solver.py --records runs/dev/records.jsonl \\
      --states 150 --out runs/report/solver.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.search.local_value import (  # noqa: E402
    candidates, decision_distance, local_maps,
)
from merge.search.pipeline import build_state  # noqa: E402
from merge.search.response import ResponseObjective, ResponsePotential, load_model  # noqa: E402
from merge.search.solver import beam_solve, exact_solve, greedy_solve  # noqa: E402

TOL = 1e-8
MAX_CANDS = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--model", default="src/merge/search/response_model.npz")
    ap.add_argument("--states", type=int, default=150)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDS)
    ap.add_argument("--w-exp", type=float, default=1.0)
    ap.add_argument("--out", default="runs/report/solver.json")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from collect_interventions import rebuild

    model = load_model(args.model)
    if model is None:
        raise SystemExit("no response model; train it first")

    recs = []
    for p in args.records:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(recs)

    res = {"greedy_single": [], "beam_compound": []}
    n_states = 0
    for rec in recs:
        if n_states >= args.states:
            break
        for rnd in (2, 3, 4):
            if n_states >= args.states:
                break
            try:
                pool, sketches, own, seen, rtok = rebuild(rec, rnd)
            except Exception:
                continue
            if not pool.items:
                continue
            n = rec["n_agents"]
            state = build_state(pool=pool, n_agents=n, rounds=rec.get("rounds", 4),
                                rnd=rnd, sketches=sketches, own=own,
                                received_tokens=rtok, question=rec["question"],
                                task_id=rec["task_id"], alloc_log=[])
            cands = candidates(pool, n)
            if not cands or len(cands) > args.max_candidates:
                continue
            try:
                w, mass, corr, qc = local_maps(pool, state, cands)
            except Exception:
                continue
            pot = ResponsePotential(model, decision_distance(state), n, qc)
            obj = ResponseObjective(pool, n, w, mass, corr, qc, rtok,
                                    state=state, potential=pot,
                                    w_exp=args.w_exp)
            _a, le = exact_solve(obj, cands, max_candidates=args.max_candidates)
            fx = le["final_value"]
            for solver, key in ((greedy_solve, "greedy_single"),
                                (beam_solve, "beam_compound")):
                _b, lg = solver(obj, cands)
                gap = 100.0 * (fx - lg["final_value"]) / max(abs(fx), TOL)
                res[key].append({
                    "gap": gap,
                    "match": abs(fx - lg["final_value"]) <= TOL,
                    "latency": lg["wall_seconds"],
                    "evals": lg["objective_evals"],
                })
            n_states += 1
            if n_states % 25 == 0:
                print(f"  {n_states} states", flush=True)

    rows = []
    for key, vals in res.items():
        if not vals:
            continue
        gaps = np.array([v["gap"] for v in vals])
        rows.append({
            "solver": key, "states": len(vals),
            "optimal_match_pct": 100.0 * np.mean([v["match"] for v in vals]),
            "median_gap_pct": float(np.median(gaps)),
            "p90_gap_pct": float(np.percentile(gaps, 90)),
            "mean_latency_s": float(np.mean([v["latency"] for v in vals])),
            "mean_objective_evals": float(np.mean([v["evals"] for v in vals])),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tolerance": TOL,
                               "max_candidates": args.max_candidates,
                               "rows": rows}, indent=1, default=float))
    for r in rows:
        print(f"{r['solver']:16s} states={r['states']:4d} "
              f"match={r['optimal_match_pct']:6.2f}% "
              f"median gap={r['median_gap_pct']:.4f}% "
              f"p90={r['p90_gap_pct']:.4f}% "
              f"lat={r['mean_latency_s']*1000:.1f}ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
