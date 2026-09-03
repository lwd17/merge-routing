"""Main experiment runner: evaluates communication arms on a task manifest.

Arms (see merge.search.gurc.POLICIES; equal LLM calls per agent per round):
  no_comm, capped_full, random_ev, hmem, agentprune, dytopo, global_set,
  designed_sep, merge_d, learned_sep, merge_l  (routing-policy arms)
  solo_tool, central_orch, lead_roles          (agentic baselines)

All agents share a fixed investigator persona; the optional role
specialization of Appendix A.4 is enabled with CQP_ROLES=1. Round 1 is
generated once per task (cached in <caches>/round0.json) and replayed
verbatim in every arm, so communication can only influence rounds >= 2.

Usage:
  CQP_RAW_LOG=runs/gurc_pilot/raw_log.jsonl \
  python scripts/run_gurc_pilot.py --out runs/gurc_pilot --workers 160 \
      --arms no_comm,capped_full,merge_d
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
from merge.search.gurc import POLICIES, run_gurc_task  # noqa: E402

INVESTIGATOR = {
    "role": "Investigator",
    "description": "A thorough researcher who answers questions by "
                   "searching Wikipedia.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="main_200.json")
    ap.add_argument("--caches", default="runs/caches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--arms", default=",".join(POLICIES))
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        assert a in POLICIES, f"unknown arm {a}"

    tasks = json.loads((HOTPOT_DIR / "manifests" / args.manifest).read_text())
    if args.limit:
        tasks = tasks[: args.limit]

    personas = [dict(INVESTIGATOR) for _ in range(args.n_agents)]

    caches = Path(args.caches)
    caches.mkdir(parents=True, exist_ok=True)
    round0_path = caches / "round0.json"
    round0 = (json.loads(round0_path.read_text())
              if round0_path.exists() else {})

    # generate the shared common-start cache for tasks not covered yet
    missing_r = [t for t in tasks if str(t["id"]) not in round0]
    if missing_r:
        from merge.search.engine import build_round0

        glock = threading.Lock()

        def gen_r(task):
            r0 = build_round0(task, personas, n_agents=args.n_agents)
            with glock:
                round0[str(task["id"])] = r0
                round0_path.write_text(
                    json.dumps(round0, ensure_ascii=False))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            print(f"round0: {len(missing_r)} ...")
            for f in as_completed([ex.submit(gen_r, t) for t in missing_r]):
                f.result()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["arm"]))
            except json.JSONDecodeError:
                pass

    jobs = [
        (t, arm) for t in tasks for arm in arms
        if (str(t["id"]), arm) not in done
    ]
    random.Random(7).shuffle(jobs)
    print(f"{len(jobs)} pilot runs to do ({len(done)} done)")
    lock = threading.Lock()
    t0 = time.time()
    n_done = n_err = 0

    def run_job(job):
        task, arm = job
        tid = str(task["id"])
        return run_gurc_task(
            task, arm, personas=personas, round0=round0[tid],
            n_agents=args.n_agents,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            task, arm = futs[fut]
            try:
                rec = fut.result()
                with lock:
                    with records_path.open("a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
            except Exception:
                n_err += 1
                print(f"ERROR {task['id']}/{arm}\n{traceback.format_exc(limit=2)}")
            if (n_done + n_err) % 100 == 0:
                print(f"  {n_done+n_err}/{len(jobs)} ({n_err} err), "
                      f"{n_done/max(1e-9,time.time()-t0):.2f}/s")

    print(f"finished: {n_done} ok, {n_err} errors -> {records_path}")


if __name__ == "__main__":
    main()
