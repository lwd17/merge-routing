"""Shared learned local value U^theta and the shared allocation costs.

Paper Appendix "Shared learned local value":

  U^theta_{i,t}(A_i) = sum_g w_ig [1 - exp(-sum_{(x->i) in A_i} r_theta(x,g,s_i))]
                     + sum_{(x->i) in A_i} v_theta(x,s_i)
                     + sum_{(x->i) in A_i} [0.20 q_question(x) - 0.20 q_held(x,i)]

r_theta >= 0 (softplus) is predicted semantic support of item x for the
receiver's unresolved requirement g; v_theta is a bounded transfer residual
(0.25 tanh). Both come from the frozen two-layer MLP of semantic_value.py
over frozen-encoder embeddings of the item, requirement, question, and the
receiver's last query, plus five scalar state features.

This module also owns the pieces every arm shares: the candidate set, the
per-step communication envelope (a hard feasibility constraint), the convex
load cost, the token cost, and the within-receiver duplicate correction.
Only the cross-receiver exposure term differs between arms.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..rq2_metrics import content_tokens
from .memory_pool import MemoryPool

Transfer = Tuple[int, int]  # (mid, receiver)

# ---- shared constants (paper: constants and communication envelope) -------
W_LOAD = 0.5
W_TOK = 0.004
W_DUP = 0.1
S_LOAD = 400.0
B_GLOBAL = 300          # per-step team token cap
B_RECV = 120            # per-step receiver token cap
ITEM_CAP = 4            # items per receiver per step
C_QUESTION, C_HELD = 0.20, 0.20


def b_global() -> int:
    """Per-step global token cap B (pure feasibility constraint).

    CQP_BUDGET_MAIN overrides B for the budget sweep; it is never a target
    dose. The receiver cap and item cap are unchanged by the sweep.
    """
    return int(os.environ.get("CQP_BUDGET_MAIN", B_GLOBAL))


# ---------------------------------------------------------------- features
def q_sem(item_toks: Set[str], g_toks: Set[str]) -> float:
    """Lexical requirement match in [0,1] (used for training labels only)."""
    if not g_toks:
        return 0.0
    return min(1.0, len(item_toks & g_toks) / len(g_toks))


def q_question(item_toks: Set[str], q_toks: Set[str]) -> float:
    return min(len(item_toks & q_toks), 4) / 4.0


def q_held(item_toks: Set[str], held_toks: Set[str]) -> float:
    if not item_toks:
        return 0.0
    return len(item_toks & held_toks) / len(item_toks)


# ------------------------------------------------------------ state prep
def requirements(state: dict, i: int) -> List[str]:
    reqs = (state["sketches"][i].get("unresolved_slots") or [])[:6]
    return [s for s in reqs if content_tokens(s)]


def held_state(pool: MemoryPool, state: dict, i: int):
    held_ents: Set[str] = set()
    for it in pool.items:
        if i in pool.holders[it["mid"]]:
            held_ents |= {e.lower() for e in it["entities"]}
    return held_ents, state["held_tokens"][i]


def candidates(pool: MemoryPool, n: int) -> List[Transfer]:
    """Eligible transfers: any pool item to any agent that does not hold it."""
    out = []
    for i in range(n):
        for it in pool.unseen_for(i):
            out.append((it["mid"], i))
    return out


def decision_distance(state: dict) -> float:
    """d_t = (T_comm - t) / (T_comm - 1); 1 at the first step, 0 at the last."""
    r, remaining = state["round"], state["remaining_rounds"]
    horizon = r + remaining - 1          # last search round
    t = r - 1                            # communication step index, 1-based
    t_comm = horizon - 1                 # communication precedes rounds 2..T
    if t_comm <= 1:
        return 0.0
    return (t_comm - t) / (t_comm - 1)


def local_maps(pool: MemoryPool, state: dict, cands: Sequence[Transfer]):
    """Precompute U^theta ingredients for every candidate transfer.

    Returns (req_weights, mass, corr, q_cluster) where `mass[(mid,i)]` is the
    per-requirement r_theta vector, `corr[(mid,i)]` the transfer-local
    correction (bounded residual + 0.20/-0.20 line), and `q_cluster[c]` the
    cluster relevance fixed before allocation (max normalized predicted
    support mass in the cluster, clipped to [0,1]).
    """
    from . import semantic_value as sv
    model = sv.load_model()
    if model is None:
        raise RuntimeError(
            "the learned local value requires semantic_value.npz "
            "(set CQP_SEMVAL or run scripts/train_semantic_value.py)")
    _emb = sv.embed
    import numpy as np

    n = state["n_agents"]
    reqs = {i: requirements(state, i) for i in range(n)}
    weights = [[1.0 / len(reqs[i])] * len(reqs[i]) if reqs[i] else []
               for i in range(n)]
    items = {it["mid"]: it for it in pool.items}
    held = {i: held_state(pool, state, i) for i in range(n)}
    q_toks = content_tokens(state.get("question", ""))
    tok_cache = {mid: content_tokens(it["claim"] + " " + " ".join(it["entities"]))
                 for mid, it in items.items()}
    last_q = state.get("last_queries") or {}
    texts = ([items[mid]["claim"] for mid, _ in cands]
             + [g for i in reqs for g in reqs[i]]
             + [state.get("question", "")]
             + [last_q.get(i, "") for i in range(n) if last_q.get(i)])
    uniq = list(dict.fromkeys(t for t in texts if t))
    lookup = dict(zip(uniq, _emb(uniq)))
    horizon = state["round"] + state["remaining_rounds"] - 1
    mass: Dict[Transfer, List[float]] = {}
    corr: Dict[Transfer, float] = {}
    support_mass: Dict[int, float] = {}
    res_scale = float(model["meta"].get("res_scale", 0.25))
    for mid, i in cands:
        it = items[mid]
        hx = np.asarray(lookup[it["claim"]], dtype=np.float32)
        ents = {e.lower() for e in it["entities"]}
        held_ents, held_tok = held[i]
        scal = [
            min(state["received_tokens"][i], 800) / 800.0,
            min(len(pool.holders[mid]), n) / max(1, n),
            state["round"] / max(1, horizon),
            float(it.get("q_support", 1.0)),
            1.0 if (ents - held_ents) else 0.0,
        ]
        row = []
        for g in reqs[i]:
            hg = np.asarray(lookup[g], dtype=np.float32)
            row.append(sv.r_value(model, hx, hg, scal))
        mass[(mid, i)] = row
        anchor = last_q.get(i) or state.get("question", "")
        hq = np.asarray(lookup.get(anchor, hx), dtype=np.float32)
        toks = tok_cache[mid]
        corr[(mid, i)] = (
            res_scale * sv.v_value(model, hx, hq, scal)
            + C_QUESTION * q_question(toks, q_toks)
            - C_HELD * q_held(toks, held_tok))
        support_mass[mid] = max(support_mass.get(mid, 0.0),
                                sum(row) if row else 0.0)
    top = max(support_mass.values(), default=1.0) or 1.0
    q_cluster: Dict[str, float] = {}
    for mid, m in support_mass.items():
        c = items[mid].get("semantic_cluster_id") or mid
        q_cluster[c] = max(q_cluster.get(c, 0.0), min(1.0, m / top))
    return weights, mass, corr, q_cluster


# ---------------------------------------------------------------- objective
class BaseObjective:
    """Shared terms of every arm's objective.

    score(A) = sum_i U^theta_{i,t}(A_i) + exposure(A)
               - w_load C_load(A) - w_tok T(A) - w_dup C_dup(A)

    Subclasses implement `exposure`. `feasible` is the hard per-step
    envelope and is identical for every arm.
    """

    name = "base"

    def __init__(self, pool: MemoryPool, n_agents: int,
                 req_weights: List[List[float]],
                 mass: Dict[Transfer, List[float]],
                 corr: Dict[Transfer, float],
                 q_cluster: Dict[str, float],
                 received_tokens: Sequence[int],
                 state: Optional[dict] = None):
        self.items = {it["mid"]: it for it in pool.items}
        self.holders = pool.holders
        self.n_agents = n_agents
        self.req_weights = req_weights
        self.mass = mass
        self.corr = corr
        self.q_cluster = q_cluster
        self.received = list(received_tokens)
        self.state = state or {}
        self.cluster_of = {
            it["mid"]: (it.get("semantic_cluster_id") or it["mid"])
            for it in pool.items}
        # B_{c,t}: actual holders of each cluster BEFORE communication
        self.base_holders: Dict[object, Set[int]] = defaultdict(set)
        for it in pool.items:
            for a in pool.holders[it["mid"]]:
                self.base_holders[self.cluster_of[it["mid"]]].add(a)
        self.maj = self.n_agents // 2 + 1

    # ---- solver interface -------------------------------------------
    def tokens_of(self, tr: Transfer) -> int:
        return self.items[tr[0]]["tokens"]

    def team_budget(self) -> int:
        return b_global()

    def cluster_of_transfer(self, tr: Transfer):
        return self.cluster_of[tr[0]]

    def feasible(self, action: Set[Transfer]) -> bool:
        per_tok: Dict[int, int] = defaultdict(int)
        per_cnt: Dict[int, int] = defaultdict(int)
        total = 0
        for mid, i in action:
            t = self.items[mid]["tokens"]
            per_tok[i] += t
            per_cnt[i] += 1
            total += t
        return (total <= b_global()
                and all(v <= B_RECV for v in per_tok.values())
                and all(v <= ITEM_CAP for v in per_cnt.values()))

    # ---- objective terms ---------------------------------------------
    def new_receivers(self, action: Set[Transfer]) -> Dict[object, Set[int]]:
        """k_c(A): distinct NEW receivers per cluster (holders excluded)."""
        out: Dict[object, Set[int]] = defaultdict(set)
        for mid, i in action:
            c = self.cluster_of[mid]
            if i not in self.base_holders.get(c, ()):
                out[c].add(i)
        return out

    def local_value(self, action: Set[Transfer]) -> Tuple[float, int, Dict[int, int]]:
        per_tok: Dict[int, int] = defaultdict(int)
        total_tok = 0
        sat: Dict[Tuple[int, int], float] = defaultdict(float)
        corr_sum = 0.0
        for mid, i in action:
            t = self.items[mid]["tokens"]
            per_tok[i] += t
            total_tok += t
            for g_idx, m in enumerate(self.mass.get((mid, i), ())):
                sat[(i, g_idx)] += m
            corr_sum += self.corr.get((mid, i), 0.0)
        u = corr_sum
        for (i, g_idx), z in sat.items():
            u += self.req_weights[i][g_idx] * (1.0 - math.exp(-z))
        return u, total_tok, per_tok

    def exposure(self, action: Set[Transfer]) -> float:
        return 0.0

    def costs(self, action: Set[Transfer], total_tok: int,
              per_tok: Dict[int, int]) -> float:
        c_load = sum(((self.received[i] + t) ** 2 - self.received[i] ** 2)
                     / S_LOAD ** 2 for i, t in per_tok.items())
        n_ic: Dict[Tuple[int, object], int] = defaultdict(int)
        for mid, i in action:
            n_ic[(i, self.cluster_of[mid])] += 1
        c_dup = sum(min(max(n - 1, 0), 2) for n in n_ic.values())
        return W_LOAD * c_load + W_TOK * total_tok + W_DUP * c_dup

    def score(self, action: Set[Transfer]) -> float:
        if not action:
            return 0.0
        u, total_tok, per_tok = self.local_value(action)
        return u + self.exposure(action) - self.costs(action, total_tok, per_tok)


class SeparableObjective(BaseObjective):
    """Same local scorer, no cross-receiver exposure term (w_exp = 0).

    With fixed local states and scores the objective factorizes exactly as
    Q(A) = sum_i Q_i(A_i); only the shared budgets couple receivers.
    """
    name = "separable"


# Role-specialized search instructions (retained shared component)
ROLE_INSTRUCTIONS = [
    "Grounder. Identify the central entities, time, place, and a reliable "
    "first-hop fact.",
    "Bridge explorer. Follow intermediate entities and relations to find "
    "the next missing hop.",
    "Alternative explorer. Test a plausible path that differs from the "
    "current main path, especially when an entity or relation is ambiguous.",
    "Verifier. Seek an independent source and check or refute the current "
    "relation chain and candidate answer.",
]
ROLE_SUFFIX = ("This is your primary responsibility, but you may retrieve "
               "any evidence needed to answer the question.")


def role_instruction(agent_idx: int) -> Optional[str]:
    """Role sentence for CQP_ROLES=1 runs; None when roles are disabled."""
    if not os.environ.get("CQP_ROLES"):
        return None
    text = ROLE_INSTRUCTIONS[agent_idx % len(ROLE_INSTRUCTIONS)]
    return f"{text} {ROLE_SUFFIX}"
