"""Learned exposure response: g_phi, the centered potential, and F^resp.

This is the method of the paper. Instead of prescribing how the value of a
cluster changes as more agents receive it, we MEASURE it with paired
interventions and fit a small response model.

Response model (Section "Learning without Prescribing the Curvature"):
a six-input, one-hidden-layer MLP with 16 tanh units and a bounded (tanh)
scalar output. For a package-conditioned holder count h, with
m = floor(N/2)+1 and delta = m - h, the input is

    x_{t,c,h} = (d_t, h/N, delta/N, delta, 1/N, q_c),

where d_t = (T_comm - t)/(T_comm - 1) is normalized remaining decision
distance and q_c is cluster relevance fixed before allocation. Its training
target is the paired terminal effect

    g_phi(x_{t,c,h}) ~ E[ Y_T(S_t, A_h + {e}) - Y_T(S_t, A_h) | x_{t,c,h} ],

with Y_T terminal group F1 on the 0-1 scale. No monotonicity constraint,
fixed early/late sign, or majority-peak template is imposed, so the model may
recover diminishing returns, a threshold peak, or neither.

Centering and integration (Eq. centered-response / response-potential):

    gamma_phi(t,c,h) = g_phi(x_{t,c,h}) - g_phi(x_{t,c,1})
    Psi_phi(c,A)     = sum_{r=b_ct}^{b_ct + k_c(A) - 1} gamma_phi(t,c,r)

The empty sum is zero, so a first copy contributes no potential: local
utility supplies first-copy usefulness and the potential only adjusts its
value under surrounding exposure. This is an identification convention for
the additive surrogate, not an exact decomposition of terminal F1.

Objective (Eq. exposure-welfare):

    F^resp(A) = sum_i U^theta_{i,t}(A_i) + w_exp sum_c Psi_phi(c,A)
                - w_load C_load(A) - w_tok T(A) - w_dup C_dup(A)

Neither tau_t, f_broad, nor f_reach appears here; they are retained only in
the STRUCTURAL and LEARNED-PHASE controls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .local_value import BaseObjective, Transfer

N_INPUTS = 6
HIDDEN = 16
FEATURE_LAYOUT = ["d_t", "h_over_N", "delta_over_N", "delta", "inv_N", "q_c"]
# delta and 1/N are kept in natural scale (constant variance at fixed N)
STANDARDIZE = [True, True, True, False, False, True]


# ----------------------------------------------------------------- features
def response_features(d_t: float, h: int, n_agents: int, q_c: float,
                      variant: str = "full") -> np.ndarray:
    """x_{t,c,h}; `variant` selects the ablation encodings."""
    m = n_agents // 2 + 1
    delta = m - h
    if variant == "raw_encoding":
        # replaces holder/boundary inputs by raw (h, N); majority distance is
        # still recoverable from (h, N), so this tests representation only
        return np.array([d_t, float(h), float(n_agents), q_c, 0.0, 0.0],
                        dtype=np.float32)
    if variant == "no_distance":
        return np.array([0.0, h / n_agents, delta / n_agents, float(delta),
                         1.0 / n_agents, q_c], dtype=np.float32)
    return np.array([d_t, h / n_agents, delta / n_agents, float(delta),
                     1.0 / n_agents, q_c], dtype=np.float32)


def _default_path() -> str:
    env = os.environ.get("CQP_RESPONSE")
    if env:
        return env
    return str(Path(__file__).resolve().parent / "response_model.npz")


_cache: Dict[str, Optional[dict]] = {}


def load_model(path: Optional[str] = None) -> Optional[dict]:
    path = path or _default_path()
    if path in _cache:
        return _cache[path]
    p = Path(path)
    if not p.exists():
        _cache[path] = None
        return None
    z = np.load(p, allow_pickle=False)
    model = {
        "W1": z["W1"], "b1": z["b1"], "w2": z["w2"], "b2": float(z["b2"]),
        "mu": z["mu"], "sd": z["sd"],
        "meta": json.loads(str(z["meta"])),
    }
    _cache[path] = model
    return model


def g_value(model: dict, x: np.ndarray) -> float:
    """Raw bounded response prediction g_phi(x) in [-1, 1]."""
    z = (x - model["mu"]) / model["sd"]
    h = np.tanh(model["W1"] @ z + model["b1"])
    return float(np.tanh(model["w2"] @ h + model["b2"]))


def _pava_nonincreasing(v: np.ndarray) -> np.ndarray:
    """Least-squares projection onto nonincreasing sequences (PAVA)."""
    v = np.asarray(v, dtype=np.float64).copy()
    n = len(v)
    if n <= 1:
        return v
    vals: List[float] = []
    wts: List[float] = []
    for x in v:
        vals.append(float(x))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] < vals[-1]:   # enforce nonincreasing
            w = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals.pop(); wts.pop()
            vals[-1] = m; wts[-1] = w
        # (loop merges violating blocks)
    out = []
    for m, w in zip(vals, wts):
        out.extend([m] * int(round(w)))
    return np.asarray(out[:n], dtype=np.float64)


class ResponsePotential:
    """Cached gamma_phi increments per (cluster, holder count) at one state."""

    def __init__(self, model: dict, d_t: float, n_agents: int,
                 q_cluster: Dict[object, float], variant: str = "full",
                 centered: bool = True, concave: bool = False):
        self.model = model
        self.d_t = d_t
        self.n = n_agents
        self.q_cluster = q_cluster
        self.variant = variant
        self.centered = centered
        self.concave = concave
        self._gamma: Dict[object, np.ndarray] = {}

    def gamma_curve(self, c: object) -> np.ndarray:
        """gamma_phi(t,c,h) for h = 1 .. N (index h-1)."""
        cur = self._gamma.get(c)
        if cur is not None:
            return cur
        q = float(self.q_cluster.get(c, 0.0))
        raw = np.array([
            g_value(self.model, response_features(self.d_t, h, self.n, q,
                                                  self.variant))
            for h in range(1, self.n + 1)
        ], dtype=np.float64)
        gam = raw - raw[0] if self.centered else raw
        if self.concave:
            gam = _pava_nonincreasing(gam)
        self._gamma[c] = gam
        return gam

    def psi(self, c: object, b: int, k: int) -> float:
        """Psi_phi(c,A) = sum_{r=b}^{b+k-1} gamma_phi(t,c,r); empty sum = 0."""
        if k <= 0:
            return 0.0
        gam = self.gamma_curve(c)
        total = 0.0
        for r in range(b, b + k):
            idx = min(max(r, 1), self.n) - 1
            total += float(gam[idx])
        return total


class ResponseObjective(BaseObjective):
    """F^resp: learned local value + centered exposure potential - costs."""

    name = "merge_response"

    def __init__(self, *a, potential: ResponsePotential, w_exp: float, **kw):
        super().__init__(*a, **kw)
        self.potential = potential
        self.w_exp = w_exp

    def exposure(self, action: Set[Transfer]) -> float:
        total = 0.0
        for c, rs in self.new_receivers(action).items():
            b = len(self.base_holders.get(c, ()))
            total += self.potential.psi(c, max(b, 1), len(rs))
        return self.w_exp * total


class GlobalSetObjective(BaseObjective):
    """Generic joint set predictor over the whole package (control arm).

    A DeepSets-style network: each transfer is embedded from the SAME
    primitive decision, relevance, holder and local-score information the
    response model sees, the embeddings are summed, and a head maps the pooled
    vector to a package value fitted on terminal package outcomes. Its
    distinction from the response model is that it is an unstructured set
    predictor, not an information disadvantage: it is given holder counts and
    decision distance as inputs.
    """

    name = "global_set"

    def __init__(self, *a, model: dict, d_t: float, w_exp: float, **kw):
        super().__init__(*a, **kw)
        self.gs = model
        self.d_t = d_t
        self.w_exp = w_exp

    def _transfer_feat(self, mid: int, i: int, action: Set[Transfer]) -> np.ndarray:
        c = self.cluster_of[mid]
        base = self.base_holders.get(c, ())
        b = len(base)
        # distinct NEW receivers of this cluster, matching k_c(A) in the main
        # objective: two items of one cluster to the same receiver are one new
        # holder, not two.
        k = len({i2 for m2, i2 in action
                 if self.cluster_of[m2] == c and i2 not in base})
        h = b + k
        n = self.n_agents
        m = n // 2 + 1
        loc = sum(self.mass.get((mid, i), ())) + self.corr.get((mid, i), 0.0)
        return np.array([
            self.d_t, h / n, (m - h) / n, float(m - h), 1.0 / n,
            float(self.q_cluster.get(c, 0.0)),
            float(loc), self.items[mid]["tokens"] / 120.0,
        ], dtype=np.float32)

    def exposure(self, action: Set[Transfer]) -> float:
        if not action:
            return 0.0
        W1, b1 = self.gs["W1"], self.gs["b1"]
        w2, b2 = self.gs["w2"], self.gs["b2"]
        mu, sd = self.gs["mu"], self.gs["sd"]
        pooled = np.zeros(W1.shape[0], dtype=np.float64)
        for mid, i in sorted(action):
            x = (self._transfer_feat(mid, i, action) - mu) / sd
            pooled += np.tanh(W1 @ x + b1)
        return self.w_exp * float(np.tanh(w2 @ pooled + b2))


def load_global_set(path: Optional[str] = None) -> Optional[dict]:
    path = path or os.environ.get(
        "CQP_GLOBALSET",
        str(Path(__file__).resolve().parent / "global_set.npz"))
    if path in _cache:
        return _cache[path]
    p = Path(path)
    if not p.exists():
        _cache[path] = None
        return None
    z = np.load(p, allow_pickle=False)
    model = {"W1": z["W1"], "b1": z["b1"], "w2": z["w2"], "b2": float(z["b2"]),
             "mu": z["mu"], "sd": z["sd"],
             "meta": json.loads(str(z["meta"]))}
    _cache[path] = model
    return model
