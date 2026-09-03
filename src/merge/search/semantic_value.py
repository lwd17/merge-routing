"""Structure-preserving learned semantic utility (MERGE extension).

Two small frozen MLP heads over MiniLM embeddings replace the *lexical*
ingredients of receiver-local value while leaving the welfare geometry
untouched:

  r_theta(x, g, s_i) >= 0   semantic resolution mass of claim x for the
                            receiver's unresolved slot g. It substitutes the
                            token-overlap count inside the concave coverage
                            term  1 - exp(-sum_e r_theta), which is
                            concave-over-modular, so the submodular-backbone
                            argument (Prop. 2) is unchanged: r is computed
                            from the pre-decision state and is a non-negative
                            constant per transfer while the solver runs.

  v_theta(x, s_i)           a modular residual for transfer-local usefulness
                            that slot coverage cannot express.

Both heads are trained OFFLINE on development logs only (see
scripts/train_semantic_value.py), exported to a compressed .npz with a
content hash, and executed here with numpy — the online controller still
makes zero routing LLM calls.

This module also hosts the shared frozen sentence encoder (`embed`, with an
in-process cache) used by the learned local value, the encoder-based
semantic clustering (Appendix A.1), and the need/offer matching of the
HMem / DyTopo / AgentPrune baselines.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ENCODER_ID = "all-MiniLM-L6-v2"
_embed_model = None
_emb_cache: Dict[str, list] = {}


def embed(texts: List[str]) -> List[list]:
    """Frozen-encoder sentence embeddings (normalized), cached per process."""
    global _embed_model
    todo = [t for t in texts if t not in _emb_cache]
    if todo:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(ENCODER_ID)
        for t, v in zip(todo,
                        _embed_model.encode(todo, normalize_embeddings=True)):
            _emb_cache[t] = v.tolist()
    return [_emb_cache[t] for t in texts]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _default_path() -> str:
    env = os.environ.get("CQP_SEMVAL")
    if env:
        return env
    here = Path(__file__).resolve().parent
    npz = here / "semantic_value.npz"
    return str(npz if npz.exists() else here / "semantic_value.json")


DEFAULT_PATH = _default_path()

_model_cache: Dict[str, Optional[dict]] = {}


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def load_model(path: Optional[str] = None) -> Optional[dict]:
    path = path or _default_path()
    if path in _model_cache:
        return _model_cache[path]
    model = None
    if os.path.exists(path):
        if path.endswith(".npz"):
            raw = np.load(path, allow_pickle=False)
            model = {
                head: {k: np.asarray(raw[f"{head}_{k}"], dtype=np.float32)
                       for k in ("W1", "b1", "w2", "b2")}
                for head in ("r", "v")
            }
            model["meta"] = json.loads(str(raw["meta"]))
        else:
            raw = json.loads(Path(path).read_text())
            model = {
                head: {k: np.asarray(v, dtype=np.float32)
                       for k, v in raw[head].items()}
                for head in ("r", "v")
            }
            model["meta"] = raw.get("meta", {})
    _model_cache[path] = model
    return model


def pair_features(hx: np.ndarray, hg: np.ndarray,
                  scalars: Sequence[float]) -> np.ndarray:
    return np.concatenate(
        [hx, hg, hx * hg, np.abs(hx - hg),
         np.asarray(scalars, dtype=np.float32)]
    )


def _head_forward(head: dict, phi: np.ndarray) -> float:
    h = np.tanh(head["W1"] @ phi + head["b1"])
    return float(head["w2"] @ h + head["b2"])


def r_value(model: dict, hx: np.ndarray, hg: np.ndarray,
            scalars: Sequence[float]) -> float:
    """Non-negative semantic resolution mass (softplus head)."""
    return float(_softplus(np.asarray(
        _head_forward(model["r"], pair_features(hx, hg, scalars))
    )))


def v_value(model: dict, hx: np.ndarray, hneed: np.ndarray,
            scalars: Sequence[float]) -> float:
    """Bounded modular residual in [-1, 1] (tanh head)."""
    return float(np.tanh(
        _head_forward(model["v"], pair_features(hx, hneed, scalars))
    ))


def transfer_scalars(round_no: int, horizon: int, claim_tokens: int,
                     holder_count: int, n_agents: int,
                     receiver_load: int) -> List[float]:
    """State scalars, all normalized and fixed before the solver runs."""
    return [
        round_no / max(1, horizon),
        min(claim_tokens, 80) / 80.0,
        min(holder_count, n_agents) / max(1, n_agents),
        min(receiver_load, 800) / 800.0,
    ]


def semantic_maps(
    model: dict,
    cands: Sequence[Tuple[int, int]],
    items: Dict[int, dict],
    holders: Dict[int, set],
    slot_texts: Dict[int, List[str]],
    need_texts: Dict[int, str],
    received_tokens: Sequence[int],
    round_no: int,
    horizon: int,
    n_agents: int,
    emb_fn,
) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[Tuple[int, int], float]]:
    """Precompute r (per candidate x slot) and v (per candidate) maps.

    Everything here depends only on the pre-decision state, never on the
    candidate set A being optimized, so each returned number is a constant
    during solving (the modularity requirement of Prop. 2).
    """
    texts: List[str] = []
    for mid, _i in cands:
        texts.append(items[mid]["claim"])
    for i, slots in slot_texts.items():
        texts.extend(slots)
    texts.extend(need_texts.values())
    embs = emb_fn(list(dict.fromkeys(texts)))
    lookup = dict(zip(list(dict.fromkeys(texts)), embs))

    sem_r: Dict[Tuple[int, int], List[float]] = {}
    sem_v: Dict[Tuple[int, int], float] = {}
    for mid, i in cands:
        it = items[mid]
        hx = np.asarray(lookup[it["claim"]], dtype=np.float32)
        scal = transfer_scalars(
            round_no, horizon, it.get("tokens", 20),
            len(holders.get(mid, ())), n_agents,
            received_tokens[i] if i < len(received_tokens) else 0,
        )
        rs: List[float] = []
        for g_text in slot_texts.get(i, []):
            hg = np.asarray(lookup[g_text], dtype=np.float32)
            rs.append(r_value(model, hx, hg, scal))
        sem_r[(mid, i)] = rs
        hneed = np.asarray(
            lookup[need_texts.get(i, it["claim"])], dtype=np.float32
        )
        sem_v[(mid, i)] = v_value(model, hx, hneed, scal)
    return sem_r, sem_v


def model_fingerprint(path: Optional[str] = None) -> str:
    path = path or _default_path()
    if not os.path.exists(path):
        return "absent"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
