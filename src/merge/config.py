"""Shared endpoints and paths for the MERGE package."""
from __future__ import annotations

import os
from pathlib import Path

ENDPOINT = os.environ.get("CQP_API_BASE", "http://localhost:8100/v1")
MODEL = os.environ.get("CQP_MODEL", "Qwen/Qwen2.5-14B-Instruct")
API_KEY = os.environ.get("CQP_API_KEY", "local")

# Fleet of identical servers (same model, same max_model_len). One task's
# calls always hit the same endpoint (stable hash) so vLLM prefix caching
# keeps working; tasks spread across the fleet.
ENDPOINTS = [
    e.strip()
    for e in os.environ.get("CQP_API_BASES", ENDPOINT).split(",")
    if e.strip()
]


def endpoint_for(key: str) -> str:
    import hashlib

    h = int(hashlib.blake2b(str(key).encode(), digest_size=4).hexdigest(), 16)
    return ENDPOINTS[h % len(ENDPOINTS)]

REPO_ROOT = Path(__file__).resolve().parents[2]          # repository root
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

# Dataset root: data/<name> (hotpotqa | musique | 2wiki), overridable via env.
# The constant keeps its historical name — every consumer (retriever,
# title index, runner) resolves through it.
DATA_ROOT = Path(os.environ.get("CQP_DATA_ROOT", REPO_ROOT / "data"))
HOTPOT_DIR = DATA_ROOT / os.environ.get("CQP_DATASET", "hotpotqa")
