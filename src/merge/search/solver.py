"""Common allocation solver: beam construction + compound local moves.

Specification (paper Appendix: Search under complementarity). Every
comparison arm uses this solver so that differences between arms come from
their objectives, not from their search:

  * beam width 16, starting from the empty package;
  * at each depth expand all feasible one-transfer additions and retain the
    16 best distinct packages ranked by the COMPLETE objective value --
    an intermediate package is allowed to have negative value, which is what
    lets threshold complementarities survive construction;
  * the incumbent is the best feasible package over all depths, including
    the empty package (value 0);
  * depth is bounded by the largest number of transfers that can fit in the
    communication budget given the positive item lengths;
  * the incumbent is then refined with improving add, drop, one-for-one
    swap, same-cluster two-receiver addition, and one-drop/two-add moves,
    capped at 64 accepted moves;
  * ties break deterministically on the sorted transfer key.

Each candidate package is evaluated in full (never by incremental marginal
gains), so a non-submodular objective is handled correctly. The solver log
records objective-evaluation counts and wall-clock time for every arm.

`greedy_solve` is the retained positive-singleton construction with single
moves, kept only for the solver-only diagnostic (Table: solver test).
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Sequence, Set, Tuple

Transfer = Tuple[int, int]  # (mid, receiver)

BEAM_WIDTH = 16
MAX_MOVES = 64


class _Counter:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


def _key(action: Set[Transfer]) -> tuple:
    return tuple(sorted(action))


def _depth_bound(objective, cands: Sequence[Transfer]) -> int:
    """Largest transfer count that can fit the team budget."""
    lens = sorted(max(1, objective.tokens_of(tr)) for tr in cands)
    budget = objective.team_budget()
    total, k = 0, 0
    for L in lens:
        if total + L > budget:
            break
        total += L
        k += 1
    return max(0, min(k, len(cands)))


def beam_solve(objective, cands: Sequence[Transfer],
               beam_width: int = BEAM_WIDTH,
               max_moves: int = MAX_MOVES) -> Tuple[Set[Transfer], dict]:
    """Beam construction followed by compound local improvement."""
    t0 = time.time()
    ev = _Counter()

    def score(a: Set[Transfer]) -> float:
        ev.n += 1
        return objective.score(a)

    cands = list(cands)
    best_action: Set[Transfer] = set()
    best_value = 0.0                      # the empty package is always a candidate
    beam: List[Tuple[float, tuple, Set[Transfer]]] = [(0.0, (), set())]
    depth_cap = _depth_bound(objective, cands)
    construct_depth = 0

    for _depth in range(depth_cap):
        seen: Dict[tuple, bool] = {}
        nxt: List[Tuple[float, tuple, Set[Transfer]]] = []
        for _v, _k, action in beam:
            for tr in cands:
                if tr in action:
                    continue
                cand = action | {tr}
                k = _key(cand)
                if k in seen:
                    continue
                if not objective.feasible(cand):
                    continue
                seen[k] = True
                nxt.append((score(cand), k, cand))
        if not nxt:
            break
        # deterministic: value desc, then sorted transfer key asc
        nxt.sort(key=lambda x: (-x[0], x[1]))
        beam = nxt[:beam_width]
        construct_depth = _depth + 1
        if beam[0][0] > best_value:
            best_value, best_action = beam[0][0], set(beam[0][2])

    construct_value = best_value

    # ---- compound local improvement on the incumbent --------------------
    moves = 0
    while moves < max_moves:
        cur = best_action
        best_gain, best_next = 0.0, None

        def offer(cand: Set[Transfer]):
            nonlocal best_gain, best_next
            if not objective.feasible(cand):
                return
            g = score(cand) - best_value
            if g > best_gain + 1e-12:
                best_gain, best_next = g, cand

        outside = [tr for tr in cands if tr not in cur]
        for tr in outside:                                   # add
            offer(cur | {tr})
        for tr in list(cur):                                 # drop
            offer(cur - {tr})
        for old in list(cur):                                # one-for-one swap
            for new in outside:
                offer((cur - {old}) | {new})
        # same-cluster two-receiver addition (threshold synergies)
        by_cluster: Dict[object, List[Transfer]] = {}
        for tr in outside:
            by_cluster.setdefault(objective.cluster_of_transfer(tr), []).append(tr)
        for _c, group in by_cluster.items():
            for a, b in itertools.combinations(group, 2):
                if a[1] == b[1]:
                    continue          # distinct receivers only
                offer(cur | {a, b})
        # one-drop / two-add
        for old in list(cur):
            base = cur - {old}
            for a, b in itertools.combinations(outside, 2):
                offer(base | {a, b})
        if best_next is None:
            break
        best_action, best_value = best_next, best_value + best_gain
        best_value = score(best_action)
        moves += 1

    log = {
        "construct_value": round(construct_value, 6),
        "final_value": round(best_value, 6),
        "construct_depth": construct_depth,
        "moves": moves,
        "package_size": len(best_action),
        "objective_evals": ev.n,
        "beam_width": beam_width,
        "wall_seconds": round(time.time() - t0, 5),
        "solver": "beam_compound",
    }
    return best_action, log


def greedy_solve(objective, cands: Sequence[Transfer],
                 max_moves: int = MAX_MOVES) -> Tuple[Set[Transfer], dict]:
    """Retained positive-singleton greedy + single add/drop/swap moves.

    Used only for the solver-only diagnostic: construction stops as soon as
    no single feasible addition has a positive marginal gain, so threshold
    complementarities that need two simultaneous additions are invisible.
    """
    t0 = time.time()
    ev = _Counter()

    def score(a: Set[Transfer]) -> float:
        ev.n += 1
        return objective.score(a)

    cands = list(cands)
    action: Set[Transfer] = set()
    current = 0.0
    while True:                                    # positive-singleton construction
        best_gain, best_cand = 0.0, None
        for tr in cands:
            if tr in action:
                continue
            cand = action | {tr}
            if not objective.feasible(cand):
                continue
            g = score(cand) - current
            if g > best_gain + 1e-12:
                best_gain, best_cand = g, cand
        if best_cand is None:
            break
        action, current = best_cand, current + best_gain
    construct_value = current

    moves = 0
    while moves < max_moves:                       # single-transfer moves only
        best_gain, best_next = 0.0, None

        def offer(cand: Set[Transfer]):
            nonlocal best_gain, best_next
            if not objective.feasible(cand):
                return
            g = score(cand) - current
            if g > best_gain + 1e-12:
                best_gain, best_next = g, cand

        outside = [tr for tr in cands if tr not in action]
        for tr in outside:
            offer(action | {tr})
        for tr in list(action):
            offer(action - {tr})
        for old in list(action):
            for new in outside:
                offer((action - {old}) | {new})
        if best_next is None:
            break
        action = best_next
        current = score(action)
        moves += 1

    log = {
        "construct_value": round(construct_value, 6),
        "final_value": round(current, 6),
        "moves": moves,
        "package_size": len(action),
        "objective_evals": ev.n,
        "wall_seconds": round(time.time() - t0, 5),
        "solver": "greedy_single",
    }
    return action, log


def exact_solve(objective, cands: Sequence[Transfer],
                max_candidates: int = 18) -> Tuple[Set[Transfer], dict]:
    """Exhaustive enumeration over all feasible subsets (diagnostic only)."""
    t0 = time.time()
    cands = list(cands)
    if len(cands) > max_candidates:
        raise ValueError(f"exact enumeration needs <= {max_candidates} candidates")
    ev = 0
    best_action: Set[Transfer] = set()
    best_value = 0.0                                  # empty package
    n = len(cands)
    for mask in range(1, 1 << n):
        action = {cands[i] for i in range(n) if mask >> i & 1}
        if not objective.feasible(action):
            continue
        ev += 1
        v = objective.score(action)
        if v > best_value + 1e-12:
            best_value, best_action = v, action
    return best_action, {
        "final_value": round(best_value, 6),
        "package_size": len(best_action),
        "objective_evals": ev,
        "wall_seconds": round(time.time() - t0, 5),
        "solver": "exact",
    }
