"""Shared LLM-call layer for the search experiments.

Provides the OpenAI-compatible chat wrapper with deterministic per-call
seeds, tolerant JSON parsing for proposals and findings, the optional raw
call log (CQP_RAW_LOG), and the cached arm-independent round 1
(build_round0): round 1 is generated once per task and replayed verbatim
in every arm, so communication can only influence rounds >= 2.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, List, Optional

from openai import OpenAI

from ..config import API_KEY, MODEL, endpoint_for
from . import prompts, retriever

# Raw audit log (GARC_PREREG layer 1): every LLM call appended as JSONL when
# CQP_RAW_LOG points at a file.
_RAW_PATH = os.environ.get("CQP_RAW_LOG")
_raw_lock = threading.Lock()


def _raw_log(entry: dict) -> None:
    if not _RAW_PATH:
        return
    with _raw_lock:
        with open(_RAW_PATH, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

_clients: Dict[str, OpenAI] = {}


def _get_client(task_id: str) -> OpenAI:
    base = endpoint_for(task_id)
    if base not in _clients:
        _clients[base] = OpenAI(base_url=base, api_key=API_KEY, timeout=180.0)
    return _clients[base]


def _seed(task_id: str, agent: int, rnd: int, arm: str, stage: str) -> int:
    key = f"{task_id}|{agent}|{rnd}|{arm}|{stage}".encode()
    return int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16) % (2**31)


def _chat(
    system: str,
    user: str,
    seed: int,
    max_tokens: int = 220,
    task_id: str = "",
    tag: Optional[dict] = None,
) -> str:
    if os.environ.get("CQP_NO_THINK"):
        # Qwen3 soft switch: disable thinking so the token budget goes to
        # the answer (reasoning otherwise consumes the entire max_tokens)
        user = user + "\n/no_think"
    resp = _get_client(task_id).chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
        seed=seed,
    )
    text = resp.choices[0].message.content or ""
    if _RAW_PATH:
        _raw_log(
            {**(tag or {}), "seed": seed, "system": system, "user": user,
             "output": text}
        )
    return text


def _parse_finding(raw: str) -> dict:
    obj = prompts.parse_json_obj(raw) or {}
    report = obj.get("report")
    if not isinstance(report, str) or not report.strip():
        report = raw[:300]
    claims = obj.get("claims")
    if not isinstance(claims, list):
        claims = []
    claims = [
        {"claim": str(c.get("claim", ""))[:300],
         "source_title": str(c.get("source_title", ""))[:120]}
        for c in claims
        if isinstance(c, dict)
    ][:8]
    hyp = obj.get("hypothesis")
    conf = obj.get("confidence")
    return {
        "report": report.strip(),
        "claims": claims,
        "hypothesis": hyp.strip() if isinstance(hyp, str) else "",
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
    }


def _parse_proposal(raw: str) -> dict:
    obj = prompts.parse_json_obj(raw) or {}
    q = obj.get("query")
    if not isinstance(q, str) or not q.strip():
        q = raw.strip()[:120]
    ents = obj.get("target_entities")
    slots = obj.get("unresolved_slots")
    return {
        "query": q.strip(),
        "target_entities": [str(e)[:80] for e in ents][:8]
        if isinstance(ents, list)
        else [],
        "unresolved_slots": [str(s)[:120] for s in slots][:8]
        if isinstance(slots, list)
        else [],
        "have_enough": bool(obj.get("have_enough"))
        if isinstance(obj.get("have_enough"), bool)
        else None,
    }


def build_round0(
    task: dict,
    personas: List[dict],
    n_agents: int = 4,
    top_k: int = 5,
) -> List[dict]:
    """Arm-independent round 1: propose -> retrieve -> finding, per agent.
    Cached by the runner and replayed identically in every arm (Audit A)."""
    task_id = str(task["id"])
    question = task["question"]
    out = []
    for i in range(n_agents):
        sysp = prompts.system_prompt(personas[i], n_agents, agent_idx=i)
        proposal = _parse_proposal(
            _chat(
                sysp,
                prompts.propose_user(question, []),
                _seed(task_id, i, 1, "common", "propose"),
                max_tokens=300,
                task_id=task_id,
                tag={"task_id": task_id, "arm": "common", "agent": i,
                     "round": 1, "stage": "propose"},
            )
        )
        q = proposal["query"]
        docs = retriever.search(q, top_k=top_k)
        finding = _parse_finding(
            _chat(
                sysp,
                prompts.finding_user(question, q, docs),
                _seed(task_id, i, 1, "common", "finding"),
                max_tokens=380,
                task_id=task_id,
                tag={"task_id": task_id, "arm": "common", "agent": i,
                     "round": 1, "stage": "finding"},
            )
        )
        out.append(
            {
                "agent": i,
                "q1": q,
                "docs": docs,
                "report": finding["report"],
                "hypothesis": finding["hypothesis"],
                "proposal": proposal,
                "finding": finding,
            }
        )
    return out
