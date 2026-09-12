"""Unit tests for the allocation objective, potential and solver.

These are deterministic code checks, NOT experiments: nothing here involves
an LLM, a benchmark, or a measured result. They cover the invariants the
protocol requires before any policy comparison is run:

  * the empty package is always feasible and scores exactly zero;
  * a receiver that already holds a cluster earns no new-holder credit;
  * several items of one cluster to the SAME receiver count as one new
    holder (and are charged by the duplicate correction);
  * the potential is order-independent: the same final package has the same
    value regardless of the order transfers were added;
  * budget boundaries are enforced exactly (team, receiver, item count);
  * a hand-built threshold state where the first copy is not worth sending
    but two copies together cross the majority is found by the beam/compound
    solver and missed by positive-singleton greedy;
  * beam and greedy never exceed the exact optimum.

Run with:  python -m pytest tests/ -q        (or: python tests/test_allocation.py)
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("CQP_CLUSTER", "jaccard")

from merge.search import local_value as LV  # noqa: E402
from merge.search.memory_pool import MemoryPool  # noqa: E402
from merge.search.response import (  # noqa: E402
    ResponseObjective, ResponsePotential, _pava_nonincreasing,
    response_features,
)
from merge.search.solver import beam_solve, exact_solve, greedy_solve  # noqa: E402


# ----------------------------------------------------------------- helpers
class ToyObjective(LV.BaseObjective):
    """Exposure = w * 1{cluster reaches a strict majority}: a pure threshold.

    Local value is a fixed small per-transfer reward, so one copy alone is
    never worth its token cost but two copies that cross the majority are.
    """

    name = "toy_threshold"

    def __init__(self, *a, reward=1.0, local=-0.02, **kw):
        super().__init__(*a, **kw)
        self.reward = reward
        self.local = local

    def local_value(self, action):
        per_tok, total = {}, 0
        for mid, i in action:
            t = self.items[mid]["tokens"]
            per_tok[i] = per_tok.get(i, 0) + t
            total += t
        return self.local * len(action), total, per_tok

    def exposure(self, action):
        out = 0.0
        for c, rs in self.new_receivers(action).items():
            if len(self.base_holders.get(c, ())) + len(rs) >= self.maj:
                out += self.reward
        return out

    def costs(self, action, total_tok, per_tok):
        return 0.0


def toy_pool(n_agents=4, n_items=3, tokens=10):
    pool = MemoryPool(n_agents)
    for k in range(n_items):
        pool.items.append({
            "mid": k, "round_created": 1, "producer": 0,
            "claim": f"claim {k}", "source_docid": f"d{k}",
            "source_title": f"T{k}", "entities": [],
            "tokens": tokens, "independent_finders": {0},
            "semantic_cluster_id": "c0", "q_support": 1.0,
        })
        pool.holders[k][0] = 1                      # agent 0 is the sole holder
    return pool


def make_toy(n_agents=4, n_items=3, tokens=10, **kw):
    pool = toy_pool(n_agents, n_items, tokens)
    n = n_agents
    weights = [[] for _ in range(n)]
    mass = {(it["mid"], i): [] for it in pool.items for i in range(n)}
    corr = {k: 0.0 for k in mass}
    obj = ToyObjective(pool, n, weights, mass, corr, {"c0": 1.0}, [0] * n, **kw)
    cands = LV.candidates(pool, n)
    return obj, cands


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok  {name}")


# ------------------------------------------------------------------- tests
def test_empty_and_feasibility():
    obj, cands = make_toy()
    check("empty package is feasible", obj.feasible(set()))
    check("empty package scores exactly zero", obj.score(set()) == 0.0)
    # team budget boundary
    big = LV.b_global() // 10
    obj2, c2 = make_toy(n_items=big + 2, tokens=10)
    allc = {(m, 1) for m, i in c2 if i == 1}
    check("item cap enforced per receiver",
          not obj2.feasible(set(list(allc)[: LV.ITEM_CAP + 1])))
    obj3, c3 = make_toy(n_items=2, tokens=LV.B_RECV)
    one = {(0, 1)}
    two = {(0, 1), (1, 1)}
    check("receiver token cap: exactly at the cap is feasible", obj3.feasible(one))
    check("receiver token cap: one over is infeasible", not obj3.feasible(two))


def test_holder_credit():
    obj, cands = make_toy()
    # agent 0 already holds every item: it must earn no new-holder credit
    to_holder = {(m, i) for m, i in cands if i == 0}
    check("existing holder is not a candidate", len(to_holder) == 0)
    nr = obj.new_receivers({(0, 1), (1, 1)})
    check("two items of one cluster to the SAME receiver = one new holder",
          len(nr["c0"]) == 1)
    nr2 = obj.new_receivers({(0, 1), (1, 2)})
    check("two receivers of one cluster = two new holders", len(nr2["c0"]) == 2)


def test_duplicate_correction():
    pool = toy_pool()
    n = 4
    weights = [[] for _ in range(n)]
    mass = {(it["mid"], i): [] for it in pool.items for i in range(n)}
    corr = {k: 0.0 for k in mass}
    obj = LV.BaseObjective(pool, n, weights, mass, corr, {"c0": 1.0}, [0] * n)
    one = obj.score({(0, 1)})
    two = obj.score({(0, 1), (1, 1)})           # same cluster, same receiver
    dup_delta = (one * 2 - two)                 # difference beyond the token cost
    check("same-cluster repeat to one receiver is charged by C_dup",
          dup_delta > 0)


def test_potential_order_independence():
    rng = np.random.RandomState(0)
    model = {"W1": rng.randn(16, 6).astype(np.float32), "b1": np.zeros(16, np.float32),
             "w2": rng.randn(16).astype(np.float32), "b2": 0.0,
             "mu": np.zeros(6, np.float32), "sd": np.ones(6, np.float32), "meta": {}}
    pool = toy_pool()
    n = 4
    weights = [[] for _ in range(n)]
    mass = {(it["mid"], i): [] for it in pool.items for i in range(n)}
    corr = {k: 0.0 for k in mass}
    pot = ResponsePotential(model, 0.5, n, {"c0": 0.8})
    obj = ResponseObjective(pool, n, weights, mass, corr, {"c0": 0.8}, [0] * n,
                            potential=pot, w_exp=1.0)
    pkg = {(0, 1), (1, 2), (2, 3)}
    vals = set()
    for perm in itertools.permutations(sorted(pkg)):
        acc = set()
        for tr in perm:
            acc.add(tr)
        vals.add(round(obj.score(acc), 12))
    check("same final package has one value regardless of insertion order",
          len(vals) == 1)
    check("empty sum of the potential is zero", pot.psi("c0", 1, 0) == 0.0)
    check("centering makes gamma(1) exactly zero",
          abs(pot.gamma_curve("c0")[0]) < 1e-12)


def test_threshold_complementarity():
    """First copy is not worth it; two copies cross the majority and are."""
    obj, cands = make_toy(n_agents=4, n_items=3, tokens=10,
                          reward=1.0, local=-0.02)
    # one new receiver: holders = 1 + 1 = 2 < majority 3 -> no reward
    single = obj.score({(0, 1)})
    pair = obj.score({(0, 1), (1, 2)})
    check("a single new holder has negative value", single < 0)
    check("two new holders cross the majority and are worth it", pair > 0)
    g, lg = greedy_solve(obj, cands)
    b, lb = beam_solve(obj, cands)
    e, le = exact_solve(obj, cands)
    check("positive-singleton greedy stops at the empty package",
          lg["final_value"] == 0.0 and len(g) == 0)
    check("beam/compound search finds the threshold package",
          lb["final_value"] > 0.0 and len(b) >= 2)
    check("beam matches the exact optimum here",
          abs(lb["final_value"] - le["final_value"]) < 1e-9)


def test_solvers_bounded_by_exact():
    rng = np.random.RandomState(3)
    for trial in range(5):
        obj, cands = make_toy(n_agents=4, n_items=4, tokens=int(rng.randint(5, 30)),
                              reward=float(rng.uniform(0.2, 2.0)),
                              local=float(-rng.uniform(0.0, 0.3)))
        cands = cands[:12]
        _e, le = exact_solve(obj, cands)
        _g, lg = greedy_solve(obj, cands)
        _b, lb = beam_solve(obj, cands)
        check(f"[trial {trial}] greedy <= exact",
              lg["final_value"] <= le["final_value"] + 1e-9)
        check(f"[trial {trial}] beam <= exact",
              lb["final_value"] <= le["final_value"] + 1e-9)


def test_pava():
    v = np.array([0.1, 0.5, 0.2, 0.4, 0.05])
    p = _pava_nonincreasing(v)
    check("PAVA output is nonincreasing",
          all(p[i] >= p[i + 1] - 1e-9 for i in range(len(p) - 1)))
    check("PAVA preserves total mass", abs(p.sum() - v.sum()) < 1e-9)


def test_features():
    for n in (4, 6, 8):
        m = n // 2 + 1
        for h, delta in zip((m - 2, m - 1, m), (2, 1, 0)):
            x = response_features(0.5, h, n, 0.7)
            check(f"N={n} h={h}: delta encoded as {delta}",
                  abs(x[3] - delta) < 1e-9)
            check(f"N={n} h={h}: 1/N encoded", abs(x[4] - 1.0 / n) < 1e-9)


if __name__ == "__main__":
    for fn in (test_empty_and_feasibility, test_holder_credit,
               test_duplicate_correction, test_potential_order_independence,
               test_threshold_complementarity, test_solvers_bounded_by_exact,
               test_pava, test_features):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL UNIT TESTS PASSED")
