"""Search-layer prompts (paper Appendix A.5).

Every agent call is a single JSON-emitting request: propose (declare the
private search state and the next query), finding (report claims with
source titles), and answer (final short phrase from the agent's own
evidence window). There is no revise call anywhere in the pipeline.
"""
from __future__ import annotations

import json
from typing import List, Optional


def _role_line(agent_idx: int):
    """Role-specialized search instruction (paper Appendix A.4).

    Active only when CQP_ROLES=1; agent index selects Grounder / Bridge
    explorer / Alternative explorer / Verifier. Roles guide search and do
    not change tools, access, or answer weights."""
    if agent_idx < 0:
        return None
    from .welfare import role_instruction
    line = role_instruction(agent_idx)
    return (line + " ") if line else None


def system_prompt(persona: dict, n_agents: int,
                  agent_idx: int = -1) -> str:
    return (
        f"You are {persona['role']}: {persona['description']} "
        + (_role_line(agent_idx) or "")
        + f"You are one of {n_agents} investigators answering a question by "
        "searching a Wikipedia snapshot with a keyword search engine (BM25). "
        "Short keyword queries work better than full sentences. "
        "Always answer with a single JSON object and nothing else."
    )


def own_log_block(own_rounds: List[dict]) -> str:
    if not own_rounds:
        return "(no searches yet)"
    lines = []
    for r in own_rounds:
        titles = ", ".join(d["title"] for d in r["docs"]) or "(nothing new)"
        lines.append(f"round {r['round']}: query='{r['q1']}' -> {titles}")
    return "\n".join(lines)


def propose_user(
    question: str,
    own_rounds: List[dict],
    routed_items: Optional[List[dict]] = None,
    suggest: Optional[List[str]] = None,
) -> str:
    """The proposal doubles as the private state declaration. When the
    allocator routed team evidence to this agent, it appears here — inside
    the agent's single normal call, no extra call of any kind.

    `suggest` carries the orchestrator/lead instruction line of the
    central_orch and lead_roles baselines; it is unused by the MERGE arms."""
    routed_block = ""
    if routed_items:
        lines = "\n".join(
            f"- {it['claim']} (source: {it['source_title'] or 'unknown'})"
            for it in routed_items
        )
        routed_block = (
            f"\nTeam evidence routed to you (from teammates' searches):\n"
            f"{lines}\n"
            "USE this evidence: if an item resolves one of your unresolved "
            "slots, treat that slot as answered and aim your next query at "
            "what is STILL missing — especially queries that connect an "
            "entity above to your remaining gaps.\n"
        )
    suggest_block = ""
    if suggest:
        suggest_block = "Your assigned focus: " + "; ".join(suggest) + "\n"
    return (
        f"Question: {question}\n\n"
        f"Your search log so far:\n{own_log_block(own_rounds)}\n"
        f"{routed_block}{suggest_block}\n"
        "Declare your private search state, then your next query.\n"
        "JSON: {\"query\": \"...\", "
        "\"target_entities\": [\"entities your query is after\"], "
        "\"unresolved_slots\": [\"facts still missing to answer\"], "
        "\"have_enough\": true/false}"
    )


def finding_user(question: str, q1: str, docs: List[dict]) -> str:
    doc_lines = "\n".join(
        f"- [{d['title']}] {d['excerpt'][:600]}" for d in docs
    ) or "(no new documents)"
    return (
        f"Question: {question}\n\n"
        f"You ran query: {q1}\nNew documents:\n{doc_lines}\n\n"
        "Write a report for your teammates (max 2 sentences) describing what "
        "the sources say that helps, or what direction is a dead end. The "
        "report must describe evidence only — do NOT state your final-answer "
        "guess in it. Separately list the claims you now hold with their "
        "source titles. Then your private best guess "
        "for the final answer, and your confidence in that guess (0-1).\n"
        'JSON: {"report": "...", '
        '"claims": [{"claim": "...", "source_title": "..."}], '
        '"hypothesis": "...", "confidence": 0.0}'
    )


def answer_user(question: str, evidence: List[dict]) -> str:
    """Answer stage: the agent's own evidence window (retrieved documents
    plus received claims), relevance-ranked before the budget cut.
    Ranking: lexical overlap with the question, recency tie-break; budget
    12000 chars (~3k tokens, safe in the 8k context)."""
    from ..rq2_metrics import content_tokens

    q_tokens = content_tokens(question)

    def score(d):
        text = f"{d['title']} {d['excerpt'][:400]}"
        return (
            len(q_tokens & content_tokens(text)),
            d.get("round", 0),
        )

    ranked = sorted(evidence, key=score, reverse=True)

    seen_ids = set()
    ev_lines = []
    budget = 12000
    for d in ranked:
        if d["docid"] in seen_ids:
            continue
        seen_ids.add(d["docid"])
        line = f"- [{d['title']}] {d['excerpt'][:500]}"
        if budget - len(line) < 0:
            break
        budget -= len(line)
        ev_lines.append(line)
    ev_text = "\n".join(ev_lines) or "(none)"
    return (
        f"Question: {question}\n\n"
        f"Your collected evidence:\n{ev_text}\n\n"
        "Give your final answer as a short phrase (an entity, date, number, "
        'or yes/no).\nJSON: {"answer": "..."}'
    )


def parse_json_obj(text: str) -> Optional[dict]:
    """Tolerant whole-object JSON parse (json_repair fallback).
    Leading garbage (think residue, role tokens, BOM, prose preambles) is
    truncated: parsing starts at the first '{'."""
    if text:
        i = text.find("{")
        if i > 0:
            text = text[i:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            obj = json.loads(repair_json(text))
        except Exception:
            return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    return obj if isinstance(obj, dict) else None


def parse_json_field(text: str, field: str) -> Optional[str]:
    obj = parse_json_obj(text)
    if isinstance(obj, dict):
        val = obj.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
