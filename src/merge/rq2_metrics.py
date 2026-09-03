"""RQ2 metrics (v2, audit-ready).

Primary comparisons are DELTA curves relative to the shared round 1 (common
start, Audit A), triangulated across four measures (Audit D):
  raw query similarity, residual query similarity (question tokens removed),
  document overlap, team union gold recall.
Toward/away directionality reports raw event counts and task-clustered
bootstrap CIs (Audit risk 5).
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_STOP = set(
    "the a an of in on for to and or is was are were what which who when where "
    "how did does do with by from at as that this it its be been his her their "
    "s".split()
)


def content_tokens(q: str) -> set:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", q.lower())
        if t not in _STOP and len(t) > 1
    }


def jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _pairwise(values: List[set]) -> Optional[float]:
    sims = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            s = jaccard(values[i], values[j])
            if s is not None:
                sims.append(s)
    return sum(sims) / len(sims) if sims else None


# ------------------------------------------------------------ per-task ----

def per_task_round(records: List[dict]) -> List[dict]:
    """rows: {task_id, arm, round, query_sim, query_sim_residual,
    doc_overlap, gold_union_recall}"""
    rows = []
    for rec in records:
        q_tokens = content_tokens(rec["question"])
        gold = set(rec.get("gold_docids", []))
        by_round = defaultdict(list)
        for e in rec["events"]:
            by_round[e["round"]].append(e)
        union: set = set()
        for rnd, evs in sorted(by_round.items()):
            evs = sorted(evs, key=lambda e: e["agent"])
            union |= {d for e in evs for d in e["docids"]}
            rows.append(
                {
                    "task_id": rec["task_id"],
                    "arm": rec["arm"],
                    "round": rnd,
                    "query_sim": _pairwise(
                        [content_tokens(e["q1"]) for e in evs]
                    ),
                    "query_sim_residual": _pairwise(
                        [content_tokens(e["q1"]) - q_tokens for e in evs]
                    ),
                    "doc_overlap": _pairwise([set(e["docids"]) for e in evs]),
                    "gold_union_recall": (
                        len(union & gold) / len(gold) if gold else None
                    ),
                }
            )
    return rows


METRIC_KEYS = ["query_sim", "query_sim_residual", "doc_overlap", "gold_union_recall"]


def curves(rows: List[dict]) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["arm"], r["round"])].append(r)
    for (arm, rnd), rs in grouped.items():
        entry = {"n": len(rs)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in rs if r[key] is not None]
            entry[key] = sum(vals) / len(vals) if vals else None
        out[arm][rnd] = entry
    return out


def delta_curves(cs: Dict[str, Dict[int, dict]]) -> Dict[str, Dict[int, dict]]:
    """S_t - S_1 per arm/metric (meaningful because round 1 is shared)."""
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for arm, per_round in cs.items():
        base = per_round.get(1, {})
        for rnd, entry in per_round.items():
            out[arm][rnd] = {
                key: (
                    entry[key] - base[key]
                    if entry.get(key) is not None and base.get(key) is not None
                    else None
                )
                for key in METRIC_KEYS
            }
    return out


# ------------------------------------------------------- common start ----

def round1_consistency(records: List[dict]) -> dict:
    """Audit A check: per task, round-1 (q1, docids) must be identical in
    every arm."""
    sig: Dict[str, set] = defaultdict(set)
    for rec in records:
        r1 = sorted(
            ((e["agent"], e["q1"], tuple(e["docids"])) for e in rec["events"] if e["round"] == 1)
        )
        sig[rec["task_id"]].add(str(r1))
    mismatched = [t for t, s in sig.items() if len(s) > 1]
    return {
        "n_tasks": len(sig),
        "n_mismatched": len(mismatched),
        "mismatched_tasks": mismatched[:10],
        "pass": not mismatched,
    }


# ---------------------------------------------------------- permutation ----

def permutation_null(
    records: List[dict], arm: str, n_perm: int = 200, seed: int = 17
) -> Dict[int, dict]:
    rng = random.Random(seed)
    by_round_agent: Dict[int, Dict[int, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        if rec["arm"] != arm:
            continue
        for e in rec["events"]:
            by_round_agent[e["round"]][e["agent"]].append(e)
    out: Dict[int, dict] = {}
    for rnd, agents in sorted(by_round_agent.items()):
        idxs = sorted(agents)
        if min(len(agents[i]) for i in idxs) < 2:
            continue
        q_sims, d_sims = [], []
        for _ in range(n_perm):
            group = [rng.choice(agents[i]) for i in idxs]
            qs = _pairwise([content_tokens(e["q1"]) for e in group])
            ds = _pairwise([set(e["docids"]) for e in group])
            if qs is not None:
                q_sims.append(qs)
            if ds is not None:
                d_sims.append(ds)
        out[rnd] = {
            "query_sim_null": sum(q_sims) / len(q_sims) if q_sims else None,
            "doc_overlap_null": sum(d_sims) / len(d_sims) if d_sims else None,
        }
    return out


# --------------------------------------------------------- directionality ----

def divergence_events(records: List[dict]) -> List[dict]:
    rows = []
    for rec in records:
        for e in rec["events"]:
            if e.get("package_none") or not e["package"]:
                continue
            pkg_tokens = set()
            for p in e["package"]:
                pkg_tokens |= content_tokens(p["text"])
            t0 = content_tokens(e["q0"])
            t1 = content_tokens(e["q1"])
            s0 = jaccard(t0, pkg_tokens)
            s1 = jaccard(t1, pkg_tokens)
            if s0 is None or s1 is None:
                continue
            rows.append(
                {
                    "task_id": rec["task_id"],
                    "arm": rec["arm"],
                    "round": e["round"],
                    "agent": e["agent"],
                    "position": e.get("position", e["agent"]),
                    "revised": e["revised"],
                    "toward_peers": s1 > s0,
                    "away_from_peers": s1 < s0,
                    "delta_sim_to_pkg": s1 - s0,
                    "gold_new_team": bool(e["gold_new_team"]),
                    "new_team_docs": len(e["new_team"]),
                }
            )
    return rows


def direction_summary(
    div: List[dict], arm: str, iters: int = 2000, seed: int = 23
) -> dict:
    """Raw counts + task-clustered bootstrap CI for
    P(gold_new_team | toward) - P(gold_new_team | away)."""
    ar = [d for d in div if d["arm"] == arm]
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for d in ar:
        by_task[d["task_id"]].append(d)
    tasks = list(by_task)

    def stats(pool: List[dict]) -> Tuple[Optional[float], int, int]:
        hits = sum(d["gold_new_team"] for d in pool)
        n = len(pool)
        return (hits / n if n else None), hits, n

    away = [d for d in ar if d["away_from_peers"]]
    toward = [d for d in ar if d["toward_peers"]]
    same = [d for d in ar if not d["away_from_peers"] and not d["toward_peers"]]
    p_away, h_away, n_away = stats(away)
    p_toward, h_toward, n_toward = stats(toward)
    p_same, h_same, n_same = stats(same)

    deltas = []
    if tasks:
        rng = random.Random(seed)
        for _ in range(iters):
            sample = [by_task[rng.choice(tasks)] for _ in tasks]
            pool = [d for chunk in sample for d in chunk]
            pa = stats([d for d in pool if d["toward_peers"]])[0]
            pb = stats([d for d in pool if d["away_from_peers"]])[0]
            if pa is not None and pb is not None:
                deltas.append(pa - pb)
    deltas.sort()
    ci = (
        [deltas[int(0.025 * len(deltas))], deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]]
        if deltas
        else None
    )
    return {
        "toward": {"p": p_toward, "hits": h_toward, "n": n_toward},
        "away": {"p": p_away, "hits": h_away, "n": n_away},
        "unchanged": {"p": p_same, "hits": h_same, "n": n_same},
        "delta_toward_minus_away": (
            p_toward - p_away if p_toward is not None and p_away is not None else None
        ),
        "ci95_cluster_bootstrap": ci,
    }


# ------------------------------------------------------ position gradient ----

def position_gradient(div: List[dict], arm: str) -> Dict[int, dict]:
    """Audit E: adoption behaviour by within-round processing position."""
    out: Dict[int, dict] = {}
    ar = [d for d in div if d["arm"] == arm]
    by_pos: Dict[int, List[dict]] = defaultdict(list)
    for d in ar:
        by_pos[d["position"]].append(d)
    for pos, ds in sorted(by_pos.items()):
        out[pos] = {
            "n": len(ds),
            "revise_rate": sum(d["revised"] for d in ds) / len(ds),
            "toward_rate": sum(d["toward_peers"] for d in ds) / len(ds),
            "mean_delta_sim": sum(d["delta_sim_to_pkg"] for d in ds) / len(ds),
        }
    return out
