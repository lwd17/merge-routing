"""Main experiment runner: evaluates routing arms on a task manifest.

Arms (see merge.search.policies.POLICIES):
  no_comm, capped_full                      references
  separable, structural, learned_phase      controls
  global_set                                generic joint set predictor
  merge_response                            the learned-response method
  resp_*                                    response ablation variants

All agents share a fixed investigator persona; role specialization
(CQP_ROLES=1) is optional. Round 1 is generated once per (task, team size),
cached in <caches>/round0_n<N>.json, and replayed verbatim in every arm, so
arms are paired by construction and communication only affects rounds >= 2.

Usage:
  python scripts/run_experiment.py --out runs/main --manifest locked_200.json \
      --arms no_comm,capped_full,separable,structural,merge_response \
      --n-agents 4 --seeds 1 --workers 48
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge.config import HOTPOT_DIR  # noqa: E402
from merge.search.pipeline import run_task  # noqa: E402
from merge.search.policies import POLICIES  # noqa: E402

INVESTIGATOR = {
    "role": "Investigator",
    "description": "A thorough researcher who answers questions by "
                   "searching Wikipedia.",
}


def load_round0(caches: Path, tasks, n_agents: int, workers: int):
    """Arm-independent round 1, cached per team size."""
    caches.mkdir(parents=True, exist_ok=True)
    path = caches / f"round0_n{n_agents}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [t for t in tasks if str(t["id"]) not in cache]
    if missing:
        from merge.search.engine import build_round0
        personas = [dict(INVESTIGATOR) for _ in range(n_agents)]
        lock = threading.Lock()

        def gen(task):
            r0 = build_round0(task, personas, n_agents=n_agents)
            with lock:
                cache[str(task["id"])] = r0
                path.write_text(json.dumps(cache, ensure_ascii=False))

        print(f"round0 (N={n_agents}): generating {len(missing)} ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(gen, t) for t in missing]):
                f.result()
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="locked_200.json")
    ap.add_argument("--caches", default="runs/caches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of decoding-seed repetitions")
    ap.add_argument("--arms", default="no_comm,capped_full,separable,"
                                      "structural,merge_response")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in POLICIES, f"unknown arm {a}"

    tasks = json.loads((HOTPOT_DIR / "manifests" / args.manifest).read_text())
    if args.limit:
        tasks = tasks[: args.limit]

    personas = [dict(INVESTIGATOR) for _ in range(args.n_agents)]
    round0 = load_round0(Path(args.caches), tasks, args.n_agents, args.workers)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["arm"], r.get("seed", 0)))
            except json.JSONDecodeError:
                pass

    jobs = [(t, arm, s) for t in tasks for arm in arms
            for s in range(args.seeds)
            if (str(t["id"]), arm, s) not in done]
    random.Random(7).shuffle(jobs)
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)

    lock = threading.Lock()
    t0 = time.time()
    n_done = n_err = 0

    def run_job(job):
        task, arm, s = job
        tid = str(task["id"])
        rec = run_task(task, arm, personas=personas, round0=round0[tid],
                       n_agents=args.n_agents, rounds=args.rounds,
                       seed_key=f"merge{s}" if s else "merge")
        rec["seed"] = s
        rec["manifest"] = args.manifest
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            task, arm, s = futs[fut]
            try:
                rec = fut.result()
                with lock:
                    with records_path.open("a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
            except Exception:
                n_err += 1
                if n_err <= 5:
                    print(f"ERROR {task['id']}/{arm}/s{s}\n"
                          f"{traceback.format_exc(limit=3)}", flush=True)
            if (n_done + n_err) % 50 == 0:
                el = time.time() - t0
                print(f"  {n_done+n_err}/{len(jobs)} ({n_err} err) "
                      f"{n_done/max(1e-9,el):.2f}/s", flush=True)

    print(f"finished: {n_done} ok, {n_err} errors -> {records_path}")


if __name__ == "__main__":
    main()
