"""Allocation policies: one objective per arm, one shared solver for all.

Every optimized arm builds the same candidate set and the same learned local
value, then differs ONLY in its exposure term:

  separable        no exposure term (receiver-separable)
  structural       tau_t schedule with f_broad / f_reach   (original MERGE)
  learned_phase    fitted tau_eta(d) with the same structural terms
  global_set       generic joint set predictor over the package
  merge_response    centered learned exposure potential Psi_phi  (this paper)

plus two non-optimized references:

  no_comm          never communicate
  capped_full      broadcast under the same hard caps

All optimized arms run the common beam/compound solver, so solver strength is
held constant across the comparison; `merge_response_greedy` is the solver-only
control that swaps in positive-singleton greedy construction.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

from .local_value import (
    B_RECV, ITEM_CAP, SeparableObjective, b_global,
    candidates, decision_distance, local_maps,
)
from .memory_pool import MemoryPool
from .response import (
    GlobalSetObjective, ResponseObjective, ResponsePotential,
    load_global_set, load_model,
)
from .solver import beam_solve, greedy_solve
from .structural import LearnedPhaseObjective, StructuralObjective

DEFAULT_W_EXP_RESPONSE = 1.0     # selected on development data


def w_exp_response() -> float:
    return float(os.environ.get("CQP_WEXP_RESP", DEFAULT_W_EXP_RESPONSE))


def w_exp_globalset() -> float:
    return float(os.environ.get("CQP_WEXP_GS", DEFAULT_W_EXP_RESPONSE))


def _prepare(state: dict):
    pool: MemoryPool = state["pool"]
    n = state["n_agents"]
    cands = candidates(pool, n)
    if not cands:
        return None
    weights, mass, corr, q_cluster = local_maps(pool, state, cands)
    return pool, n, cands, weights, mass, corr, q_cluster


def _finish(state: dict, objective, cands, solver=beam_solve) -> Dict[int, list]:
    action, log = solver(objective, cands)
    slog = state.get("alloc_log")
    if isinstance(slog, list):
        slog.append({"round": state["round"], "arm": objective.name,
                     "solver": log})
    pool: MemoryPool = state["pool"]
    items = {it["mid"]: it for it in pool.items}
    routed: Dict[int, list] = defaultdict(list)
    for mid, i in action:
        routed[i].append(items[mid])
    return dict(routed)


def _base_kwargs(prep):
    pool, n, cands, weights, mass, corr, q_cluster = prep
    return dict(pool=pool, n_agents=n, req_weights=weights, mass=mass,
                corr=corr, q_cluster=q_cluster)


def _build(prep, state, cls, **extra):
    pool, n, cands, weights, mass, corr, q_cluster = prep
    return cls(pool, n, weights, mass, corr, q_cluster,
               state["received_tokens"], state=state, **extra)


# ---------------------------------------------------------------- arms ----
def policy_no_comm(state: dict) -> Dict[int, list]:
    return {}


def policy_separable(state: dict) -> Dict[int, list]:
    prep = _prepare(state)
    if prep is None:
        return {}
    obj = _build(prep, state, SeparableObjective)
    return _finish(state, obj, prep[2])


def policy_structural(state: dict, solver=beam_solve) -> Dict[int, list]:
    prep = _prepare(state)
    if prep is None:
        return {}
    tau = 1.0 - decision_distance(state)
    obj = _build(prep, state, StructuralObjective, tau=tau)
    return _finish(state, obj, prep[2], solver=solver)


def policy_structural_greedy(state: dict) -> Dict[int, list]:
    """Structural objective under the RETAINED greedy solver.

    Paired with `resp_greedy`, this separates a change of objective from a
    change of search: both objectives are measured under both solvers.
    """
    return policy_structural(state, solver=greedy_solve)


def policy_learned_phase(state: dict) -> Dict[int, list]:
    prep = _prepare(state)
    if prep is None:
        return {}
    obj = _build(prep, state, LearnedPhaseObjective,
                 d_t=decision_distance(state))
    return _finish(state, obj, prep[2])


def policy_global_set(state: dict) -> Dict[int, list]:
    prep = _prepare(state)
    if prep is None:
        return {}
    gs = load_global_set()
    if gs is None:
        raise RuntimeError(
            "global_set requires global_set.npz (scripts/train_response.py "
            "--fit-global-set)")
    obj = _build(prep, state, GlobalSetObjective, model=gs,
                 d_t=decision_distance(state), w_exp=w_exp_globalset())
    return _finish(state, obj, prep[2])


def _response_policy(state: dict, variant: str = "full", centered: bool = True,
                     concave: bool = False, solver=beam_solve) -> Dict[int, list]:
    prep = _prepare(state)
    if prep is None:
        return {}
    model = load_model(os.environ.get("CQP_RESPONSE_" + variant.upper())
                       or None)
    if model is None:
        raise RuntimeError(
            "merge_response requires response_model.npz "
            "(scripts/train_response.py)")
    pot = ResponsePotential(model, decision_distance(state), state["n_agents"],
                            prep[6], variant=variant, centered=centered,
                            concave=concave)
    obj = _build(prep, state, ResponseObjective, potential=pot,
                 w_exp=w_exp_response())
    return _finish(state, obj, prep[2], solver=solver)


def policy_merge_response(state: dict) -> Dict[int, list]:
    return _response_policy(state)


# ---- ablation variants (Table: learned-response ablations) ---------------
def policy_response_raw_encoding(state: dict) -> Dict[int, list]:
    return _response_policy(state, variant="raw_encoding")


def policy_response_no_distance(state: dict) -> Dict[int, list]:
    return _response_policy(state, variant="no_distance")


def policy_response_concave(state: dict) -> Dict[int, list]:
    return _response_policy(state, concave=True)


def policy_response_uncentered(state: dict) -> Dict[int, list]:
    return _response_policy(state, centered=False)


def policy_response_greedy(state: dict) -> Dict[int, list]:
    return _response_policy(state, solver=greedy_solve)


def policy_capped_full(state: dict) -> Dict[int, list]:
    """Ordinary broadcast under the same hard team and receiver limits.

    Evidence items and their receiver bundles are indivisible: iterate the
    pool in creation order and broadcast each unseen item to every receiver
    that lacks it, skipping a bundle that does not fit. A run can therefore
    leave residual capacity.
    """
    pool: MemoryPool = state["pool"]
    n = state["n_agents"]
    per_tok: Dict[int, int] = defaultdict(int)
    per_cnt: Dict[int, int] = defaultdict(int)
    total = 0
    routed: Dict[int, list] = defaultdict(list)
    for it in sorted(pool.items, key=lambda x: (x["round_created"], x["mid"])):
        mid, t = it["mid"], it["tokens"]
        targets = [i for i in range(n) if i not in pool.holders[mid]]
        if not targets:
            continue
        if total + t * len(targets) > b_global():
            continue
        if any(per_tok[i] + t > B_RECV or per_cnt[i] + 1 > ITEM_CAP
               for i in targets):
            continue
        for i in targets:
            routed[i].append(it)
            per_tok[i] += t
            per_cnt[i] += 1
        total += t * len(targets)
    return dict(routed)


POLICIES = {
    "no_comm": policy_no_comm,
    "capped_full": policy_capped_full,
    "separable": policy_separable,
    "structural": policy_structural,
    "learned_phase": policy_learned_phase,
    "global_set": policy_global_set,
    "merge_response": policy_merge_response,
    # ablations
    "resp_raw_encoding": policy_response_raw_encoding,
    "resp_no_distance": policy_response_no_distance,
    "resp_concave": policy_response_concave,
    "resp_uncentered": policy_response_uncentered,
    "resp_greedy": policy_response_greedy,
    "structural_greedy": policy_structural_greedy,
}
