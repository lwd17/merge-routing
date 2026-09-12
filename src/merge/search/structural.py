"""Structural exposure reference and the learned-phase control.

The retained STRUCTURAL controller (the original MERGE) prices exposure with
two hand-specified functions over the distinct new receivers k_c(A):

    f_broad(A) = sum_c [max(k_c(A) - 1, 0)]^2
    f_reach(A) = sum_c q_c [min(b_ct + k_c(A), m) - min(b_ct, m)] / (m - 1)

combined by a deterministic phase schedule

    tau_t = (t - 1) / (T_comm - 1) = 1 - d_t,

so early steps penalize broad duplication and late steps reward reach up to
the majority cap m = floor(N/2) + 1:

    F^struct(A) = sum_i U_i(A_i) + w_exp [tau_t f_reach - (1 - tau_t) f_broad]
                  - w_load C_load - w_tok T - w_dup C_dup.

LEARNED-PHASE keeps both structural functions but replaces the fixed
schedule with a fitted scalar tau_eta(d) = sigmoid(a0 + a1 d), trained on the
same development interventions as the response model. It isolates "fit the
phase" from "fit the whole state-conditioned response".
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Set

from .local_value import BaseObjective, Transfer

W_EXP_STRUCT = 0.4          # retained structural constant


class StructuralObjective(BaseObjective):
    """Original MERGE exposure term with the deterministic phase schedule."""

    name = "structural"

    def __init__(self, *a, tau: float = 0.0, w_exp: float = W_EXP_STRUCT, **kw):
        super().__init__(*a, **kw)
        self.tau = tau
        self.w_exp = w_exp

    def f_broad(self, new_recv: Dict[object, Set[int]]) -> float:
        return sum(max(len(rs) - 1, 0) ** 2 for rs in new_recv.values())

    def f_reach(self, new_recv: Dict[object, Set[int]]) -> float:
        if self.maj <= 1:
            return 0.0
        out = 0.0
        for c, rs in new_recv.items():
            b = len(self.base_holders.get(c, ()))
            gain = (min(b + len(rs), self.maj) - min(b, self.maj)) / (self.maj - 1)
            if gain > 0:
                out += self.q_cluster.get(c, 0.0) * gain
        return out

    def exposure(self, action: Set[Transfer]) -> float:
        nr = self.new_receivers(action)
        return self.w_exp * (self.tau * self.f_reach(nr)
                             - (1.0 - self.tau) * self.f_broad(nr))


def _phase_path() -> str:
    env = os.environ.get("CQP_PHASE")
    if env:
        return env
    return str(Path(__file__).resolve().parent / "learned_phase.json")


_phase_cache: Dict[str, Optional[dict]] = {}


def load_phase(path: Optional[str] = None) -> Optional[dict]:
    """Load the fitted {a0, a1, w_exp} of the learned-phase control."""
    path = path or _phase_path()
    if path in _phase_cache:
        return _phase_cache[path]
    p = Path(path)
    obj = json.loads(p.read_text()) if p.exists() else None
    _phase_cache[path] = obj
    return obj


class LearnedPhaseObjective(StructuralObjective):
    """Structural terms with a fitted phase scalar tau_eta(d) = sigmoid(a0 + a1 d)."""

    name = "learned_phase"

    def __init__(self, *a, d_t: float = 0.0, params: Optional[dict] = None, **kw):
        params = params or load_phase()
        if params is None:
            raise RuntimeError(
                "learned_phase requires learned_phase.json "
                "(run scripts/train_response.py --fit-phase)")
        tau = 1.0 / (1.0 + math.exp(-(params["a0"] + params["a1"] * d_t)))
        kw.pop("tau", None)
        super().__init__(*a, tau=tau, w_exp=float(params.get("w_exp", W_EXP_STRUCT)), **kw)
        self.d_t = d_t
        self.params = params
