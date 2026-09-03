"""Exposure-welfare objective and factorial policies (paper Eq. 2-13).

This module implements the paper's deployed system exactly:

  F_t(A) = sum_i U_{i,t}(A_i)
           + w_exp [ tau_t f_reach(A) - (1 - tau_t) f_broad(A) ]
           - w_load C_load(A) - w_tok T(A) - w_dup C_dup(A)          (Eq. 12)

with the designed local value of Appendix A.2,

  a^des_{ixg} = q_src(x) [0.60 q_sem(x,g) + 0.25 q_bridge(x,i)
                          + 0.15 q_scarce(x)]                        (Eq. 8)
  U^des_i(A_i) = sum_g w_ig [1 - exp(-sum a^des)]
                 + sum_e [0.20 q_question(x) - 0.20 q_held(x,i)]     (Eq. 9)

or the learned local value of Appendix A.3 (r_theta inside the same
saturating coverage, a bounded residual, and the same +-0.20 correction).

Factorial arms (Section 5):
  designed_sep  designed local value, receiver-separable allocation
  merge_d       designed local value, joint allocation (full Eq. 12)
  learned_sep   learned local value, receiver-separable allocation
  merge_l       learned local value, joint allocation
  global_set    parameter-matched neural set scorer, no explicit exposure terms
  capped_full   ordinary broadcast under the same hard team/receiver limits

Separable variants drop only the cross-receiver term Phi (w_exp = 0); the
remaining terms are receiver-local, so the objective factorizes as
Q(A) = sum_i Q_i(A_i) exactly as in Eq. 3. All constants come from
Table 4: w_exp=0.4, w_load=0.5, w_tok=0.004, w_dup=0.1, S_load=400,
envelope 300 tokens/step global, 120 tokens/step per receiver, <=4 items
per receiver per step, tau_t in {0, 1/2, 1} over the three communication
steps. The solver is greedy construction from the empty package followed
by improving add/drop/swap moves (Appendix A.7), with the run log storing
the initial greedy value, final value, accepted moves, selected package,
and wall-clock time.
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..rq2_metrics import content_tokens
from .memory_pool import MemoryPool

Transfer = Tuple[int, int]  # (mid, receiver)

# ---- Table 4 constants (shared by all experiments) ------------------------
W_EXP = 0.4
W_LOAD = 0.5
W_TOK = 0.004
W_DUP = 0.1
S_LOAD = 400.0
B_GLOBAL = 300
B_RECV = 120
ITEM_CAP = 4
MAX_MOVES = 64


def b_global() -> int:
    """Per-step global token cap B.

    CQP_BUDGET_MAIN overrides B for the budget sweep of Appendix B.4
    (B in {100, 150, 225, 300, 400}). The cap is a pure per-step
    feasibility constraint — never a target dose and never a menu of
    tiers; the per-receiver cap (120) and the item cap (4) are unchanged.
    """
    return int(os.environ.get("CQP_BUDGET_MAIN", B_GLOBAL))

# Eq. 8 / Eq. 9 designed coefficients (frozen; Appendix A.2)
C_SEM, C_BRIDGE, C_SCARCE = 0.60, 0.25, 0.15
C_QUESTION, C_HELD = 0.20, 0.20


# ---------------------------------------------------------------- features
def q_sem(item_toks: Set[str], g_toks: Set[str]) -> float:
    """Semantic match of an item with requirement g, in [0, 1]."""
    if not g_toks:
        return 0.0
    return min(1.0, len(item_toks & g_toks) / len(g_toks))


def q_bridge(ents: Set[str], held_ents: Set[str]) -> float:
    """1 when the item connects a held entity to a new entity."""
    if not ents:
        return 0.0
    return 1.0 if (ents & held_ents) and (ents - held_ents) else 0.0


def q_scarce(holder_count: int) -> float:
    """Decreases with the current number of holders."""
    return 1.0 / max(1, holder_count)


def q_question(item_toks: Set[str], q_toks: Set[str]) -> float:
    return min(len(item_toks & q_toks), 4) / 4.0


def q_held(item_toks: Set[str], held_toks: Set[str]) -> float:
    if not item_toks:
        return 0.0
    return len(item_toks & held_toks) / len(item_toks)


# ---------------------------------------------------------------- scorer
class WelfareScorer:
    """Eq. 12 over precomputed per-transfer quantities.

    `mass[(mid, i)]` is the list of per-requirement masses (designed
    a^des_{ixg} or learned r_theta), aligned with the receiver's
    requirement list; `corr[(mid, i)]` is the transfer-local correction
    (the +-0.20 line, plus the learned residual for MERGE-L). Both are
    fixed before allocation, so U is concave-over-modular per requirement
    and every remaining term matches Section 4 exactly.
    """

    def __init__(self, pool: MemoryPool, n_agents: int,
                 req_weights: List[List[float]],
                 mass: Dict[Transfer, List[float]],
                 corr: Dict[Transfer, float],
                 q_cluster: Dict[str, float],
                 received_tokens: List[int],
                 tau: float, use_phi: bool = True):
        self.items = {it["mid"]: it for it in pool.items}
        self.holders = pool.holders
        self.n_agents = n_agents
        self.req_weights = req_weights
        self.mass = mass
        self.corr = corr
        self.q_cluster = q_cluster
        self.received = received_tokens
        self.tau = tau
        self.use_phi = use_phi
        self.cluster_of = {
            it["mid"]: (it.get("semantic_cluster_id") or it["mid"])
            for it in pool.items
        }
        self.base_holders: Dict[str, Set[int]] = defaultdict(set)
        for it in pool.items:
            for a in pool.holders[it["mid"]]:
                self.base_holders[self.cluster_of[it["mid"]]].add(a)
        self.maj = self.n_agents // 2 + 1

    def score(self, action: Set[Transfer]) -> float:
        if not action:
            return 0.0
        per_tok: Dict[int, int] = defaultdict(int)
        total_tok = 0
        # U: saturating coverage + transfer-local correction
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

        # Phi: Eq. 4-6 (new receivers per cluster in this step)
        phi = 0.0
        if self.use_phi:
            new_recv: Dict[str, Set[int]] = defaultdict(set)
            for mid, i in action:
                new_recv[self.cluster_of[mid]].add(i)
            f_broad = sum(max(len(rs) - 1, 0) ** 2
                          for rs in new_recv.values())
            f_reach = 0.0
            for c, rs in new_recv.items():
                base = self.base_holders.get(c, set())
                gain = (min(len(base | rs), self.maj)
                        - min(len(base), self.maj)) / (self.maj - 1)
                if gain > 0:
                    f_reach += self.q_cluster.get(c, 0.0) * gain
            phi = W_EXP * (self.tau * f_reach - (1.0 - self.tau) * f_broad)

        # C_load (Eq. under 2) and C_dup (Eq. 13)
        c_load = sum(((self.received[i] + t) ** 2 - self.received[i] ** 2)
                     / S_LOAD ** 2 for i, t in per_tok.items())
        n_ic: Dict[Tuple[int, str], int] = defaultdict(int)
        for mid, i in action:
            n_ic[(i, self.cluster_of[mid])] += 1
        c_dup = sum(min(max(n - 1, 0), 2) for n in n_ic.values())

        return (u + phi - W_LOAD * c_load - W_TOK * total_tok
                - W_DUP * c_dup)

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


def optimize_welfare(scorer: WelfareScorer,
                     cands: List[Transfer]) -> Tuple[Set[Transfer], dict]:
    """Greedy + improving add/drop/swap (Appendix A.7) with a solver log."""
    t0 = time.time()
    action: Set[Transfer] = set()
    current = 0.0
    moves = 0
    greedy_value: Optional[float] = None
    while moves < MAX_MOVES:
        best_gain, best_action = 0.0, None
        for tr in cands:
            if tr in action:
                continue
            cand = action | {tr}
            if not scorer.feasible(cand):
                continue
            gain = scorer.score(cand) - current
            if gain > best_gain:
                best_gain, best_action = gain, cand
        if best_action is None and greedy_value is None:
            greedy_value = current
        if greedy_value is not None:
            for tr in list(action):
                cand = action - {tr}
                gain = scorer.score(cand) - current
                if gain > best_gain:
                    best_gain, best_action = gain, cand
            for old in list(action):
                for new in cands:
                    if new in action:
                        continue
                    cand = (action - {old}) | {new}
                    if not scorer.feasible(cand):
                        continue
                    gain = scorer.score(cand) - current
                    if gain > best_gain:
                        best_gain, best_action = gain, cand
        if best_action is None:
            break
        action, current = best_action, scorer.score(best_action)
        moves += 1
    log = {
        "greedy_value": round(greedy_value if greedy_value is not None
                              else current, 4),
        "final_value": round(current, 4),
        "moves": moves,
        "package_size": len(action),
        "wall_seconds": round(time.time() - t0, 4),
    }
    return action, log


# ------------------------------------------------------------ state prep
def _requirements(state: dict, i: int) -> List[str]:
    reqs = (state["sketches"][i].get("unresolved_slots") or [])[:6]
    return [s for s in reqs if content_tokens(s)]


def _tau(state: dict) -> float:
    r, remaining = state["round"], state["remaining_rounds"]
    horizon = r + remaining - 1
    return (r - 2) / max(1, horizon - 2)


def _held_state(pool: MemoryPool, state: dict, i: int):
    held_ents: Set[str] = set()
    for it in pool.items:
        if i in pool.holders[it["mid"]]:
            held_ents |= {e.lower() for e in it["entities"]}
    return held_ents, state["held_tokens"][i]


def _candidates(pool: MemoryPool, n: int) -> List[Transfer]:
    out = []
    for i in range(n):
        for it in pool.unseen_for(i):
            out.append((it["mid"], i))
    return out


def _designed_maps(pool: MemoryPool, state: dict, cands: List[Transfer]):
    n = state["n_agents"]
    q_toks = content_tokens(state.get("question", ""))
    reqs = {i: _requirements(state, i) for i in range(n)}
    req_toks = {i: [content_tokens(g) for g in reqs[i]] for i in range(n)}
    weights = [[1.0 / len(reqs[i])] * len(reqs[i]) if reqs[i] else []
               for i in range(n)]
    held = {i: _held_state(pool, state, i) for i in range(n)}
    items = {it["mid"]: it for it in pool.items}
    tok_cache = {mid: content_tokens(it["claim"] + " "
                                     + " ".join(it["entities"]))
                 for mid, it in items.items()}
    mass: Dict[Transfer, List[float]] = {}
    corr: Dict[Transfer, float] = {}
    req_match: Dict[int, float] = {}
    for mid, i in cands:
        it = items[mid]
        toks = tok_cache[mid]
        ents = {e.lower() for e in it["entities"]}
        held_ents, held_tok = held[i]
        src = it.get("q_support", 1.0)
        common = src * (C_BRIDGE * q_bridge(ents, held_ents)
                        + C_SCARCE * q_scarce(len(pool.holders[mid])))
        row, best_sem = [], 0.0
        for g in req_toks[i]:
            s = q_sem(toks, g)
            best_sem = max(best_sem, s)
            row.append(src * C_SEM * s + common)
        mass[(mid, i)] = row
        corr[(mid, i)] = (C_QUESTION * q_question(toks, q_toks)
                          - C_HELD * q_held(toks, held_tok))
        req_match[mid] = max(req_match.get(mid, 0.0), best_sem)
    q_cluster: Dict[str, float] = {}
    for mid, m in req_match.items():
        c = items[mid].get("semantic_cluster_id") or mid
        q_cluster[c] = max(q_cluster.get(c, 0.0), min(1.0, m))
    return weights, mass, corr, q_cluster


def _learned_maps(pool: MemoryPool, state: dict, cands: List[Transfer]):
    from . import semantic_value as sv
    model = sv.load_model()
    if model is None:
        raise RuntimeError(
            "learned local value requires semantic_value.npz "
            "(set CQP_SEMVAL or train scripts/train_semantic_value.py)"
        )
    _emb = sv.embed
    n = state["n_agents"]
    reqs = {i: _requirements(state, i) for i in range(n)}
    weights = [[1.0 / len(reqs[i])] * len(reqs[i]) if reqs[i] else []
               for i in range(n)]
    items = {it["mid"]: it for it in pool.items}
    held = {i: _held_state(pool, state, i) for i in range(n)}
    q_toks = content_tokens(state.get("question", ""))
    tok_cache = {mid: content_tokens(it["claim"] + " "
                                     + " ".join(it["entities"]))
                 for mid, it in items.items()}
    last_q = state.get("last_queries") or {}
    texts = ([items[mid]["claim"] for mid, _ in cands]
             + [g for i in reqs for g in reqs[i]]
             + [state.get("question", "")]
             + [last_q.get(i, "") for i in range(n) if last_q.get(i)])
    embs = _emb(list(dict.fromkeys(t for t in texts if t)))
    lookup = dict(zip(list(dict.fromkeys(t for t in texts if t)), embs))
    import numpy as np
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
            - C_HELD * q_held(toks, held_tok)
        )
        support_mass[mid] = max(support_mass.get(mid, 0.0),
                                sum(row) if row else 0.0)
    top = max(support_mass.values(), default=1.0) or 1.0
    q_cluster: Dict[str, float] = {}
    for mid, m in support_mass.items():
        c = items[mid].get("semantic_cluster_id") or mid
        q_cluster[c] = max(q_cluster.get(c, 0.0), min(1.0, m / top))
    return weights, mass, corr, q_cluster


def _run(state: dict, learned: bool, use_phi: bool):
    pool: MemoryPool = state["pool"]
    n = state["n_agents"]
    cands = _candidates(pool, n)
    if not cands:
        return {}
    build = _learned_maps if learned else _designed_maps
    weights, mass, corr, q_cluster = build(pool, state, cands)
    scorer = WelfareScorer(pool, n, weights, mass, corr, q_cluster,
                           state["received_tokens"], _tau(state),
                           use_phi=use_phi)
    action, log = optimize_welfare(scorer, cands)
    slog = state.get("alloc_log")
    if isinstance(slog, list):
        slog.append({"round": state["round"], "solver": log,
                     "phi": use_phi, "learned": learned})
    routed: Dict[int, list] = defaultdict(list)
    items = {it["mid"]: it for it in pool.items}
    for mid, i in action:
        routed[i].append(items[mid])
    return dict(routed)


def policy_designed_sep(state: dict):
    return _run(state, learned=False, use_phi=False)


def policy_merge_d(state: dict):
    return _run(state, learned=False, use_phi=True)


def policy_learned_sep(state: dict):
    return _run(state, learned=True, use_phi=False)


def policy_merge_l(state: dict):
    return _run(state, learned=True, use_phi=True)


def policy_capped_full(state: dict):
    """Ordinary broadcast under the same hard team and receiver limits.

    Evidence items and their receiver bundles are indivisible: iterate the
    pool in creation order, attempt to broadcast each unseen item to every
    receiver that lacks it, and skip the bundle when the next admissible
    broadcast does not fit -- a run can therefore leave residual capacity.
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
        need_tok = t * len(targets)
        if total + need_tok > b_global():
            continue
        if any(per_tok[i] + t > B_RECV or per_cnt[i] + 1 > ITEM_CAP
               for i in targets):
            continue
        for i in targets:
            routed[i].append(it)
            per_tok[i] += t
            per_cnt[i] += 1
        total += need_tok
    return dict(routed)


def policy_global_set(state: dict):
    """Parameter-matched neural set scorer without explicit exposure terms.

    Greedy set construction under the shared envelope, scoring transfers
    with the learned heads plus two set-context features (receivers already
    assigned the item's cluster, receiver load within the package); no
    f_reach/f_broad/C_dup terms. See Appendix B baseline-parity table.
    """
    from . import semantic_value as sv
    model = sv.load_model()
    if model is None:
        raise RuntimeError("global_set requires semantic_value.npz")
    pool: MemoryPool = state["pool"]
    n = state["n_agents"]
    cands = _candidates(pool, n)
    if not cands:
        return {}
    weights, mass, corr, _q = _learned_maps(pool, state, cands)
    scorer = WelfareScorer(pool, n, weights, mass, corr, {},
                           state["received_tokens"], _tau(state),
                           use_phi=False)
    cluster_of = scorer.cluster_of

    def set_score(action: Set[Transfer]) -> float:
        base = scorer.score(action)
        # set-context adjustment replaces the explicit exposure terms with
        # a generic learned-per-transfer discount on repeated clusters
        seen: Dict[str, int] = defaultdict(int)
        adj = 0.0
        for mid, i in sorted(action):
            c = cluster_of[mid]
            adj -= 0.05 * seen[c]
            seen[c] += 1
        return base + adj

    action: Set[Transfer] = set()
    current = 0.0
    for _ in range(MAX_MOVES):
        best_gain, best = 0.0, None
        for tr in cands:
            if tr in action:
                continue
            cand = action | {tr}
            if not scorer.feasible(cand):
                continue
            gain = set_score(cand) - current
            if gain > best_gain:
                best_gain, best = gain, cand
        if best is None:
            break
        action, current = best, set_score(best)
    routed: Dict[int, list] = defaultdict(list)
    items = {it["mid"]: it for it in pool.items}
    for mid, i in action:
        routed[i].append(items[mid])
    return dict(routed)


# Role-specialized search instructions (Appendix A.4)
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
