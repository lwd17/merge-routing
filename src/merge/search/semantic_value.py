"""Shared learned local scorer: the frozen r_theta and v_theta heads.

Two small frozen MLP heads over MiniLM embeddings supply the receiver-local
ingredients of U^theta (see local_value.py):

  r_theta(x, g, s_i) >= 0   predicted semantic support of item x for the
                            receiver's unresolved requirement g. It sits
                            inside the saturating coverage term
                            1 - exp(-sum r_theta), so each r is computed from
                            the pre-decision state and is a non-negative
                            constant per transfer while the solver runs.

  v_theta(x, s_i)           a bounded modular residual for transfer-local
                            usefulness that requirement coverage cannot
                            express.

Both heads are trained OFFLINE on development logs only (see
scripts/train_semantic_value.py), exported to a compressed .npz with a
content hash, and executed here with numpy: the online allocator makes zero
routing LLM calls.

This module also hosts the shared frozen sentence encoder (`embed`, cached
in process), used by the local scorer and by online semantic clustering.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def model_fingerprint(path: Optional[str] = None) -> str:
    path = path or _default_path()
    if not os.path.exists(path):
        return "absent"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
