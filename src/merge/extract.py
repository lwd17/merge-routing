"""Answer extraction: per-turn per-agent solution string -> option letter.

Mirrors common multichoice conventions (reference[0] starts with the label
letter) but is applied to every trajectory point, not just the final draft.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_PATTERNS = [
    re.compile(r"^\s*\(?([A-J])\)?\s*[).:\]]"),          # "B) ...", "(B): ..."
    re.compile(r"^\s*\(?([A-J])\)?\s*$"),                 # bare letter
    re.compile(r"(?i)final solution\s*:\s*\(?([A-J])\b"),
    re.compile(r"(?i)\b(?:answer|option|choice)\s*(?:is)?\s*:?\s*\(?([A-J])\)?\s*[).:\]\s]"),
    re.compile(r"(?i)\b(?:answer|option|choice)\s*(?:is)?\s*:?\s*\(?([A-J])\)?\s*$"),
]


def parse_options(question: str) -> Dict[str, str]:
    """Extract {letter: normalized option text} from an 'A) ...' style block."""
    options: Dict[str, str] = {}
    for m in re.finditer(r"(?m)^\s*([A-J])\)\s*(.+?)\s*$", question):
        options[m.group(1)] = _norm(m.group(2))
    return options


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def extract_letter(solution: Optional[str], question: str) -> Optional[str]:
    if not solution:
        return None
    text = solution.strip()
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    options = parse_options(question)
    # Yes/No tasks phrased as options: accept the bare word.
    norm_sol = _norm(text)
    for letter, opt in options.items():
        if opt and (opt == norm_sol or (len(opt) > 3 and opt in norm_sol)):
            return letter
    # Last resort: a unique "X)" mention anywhere in the text.
    letters = set(re.findall(r"\b([A-J])\)", text))
    if len(letters) == 1:
        return letters.pop()
    return None


def trajectory(record: dict) -> Dict[str, Dict[int, Optional[str]]]:
    """record -> {agent_id: {turn: letter-or-None}} using each agent's latest
    solution-bearing message within a turn."""
    question = record["question"]
    latest: Dict[str, Dict[int, tuple]] = {}
    for m in record["memory"]:
        if m["contribution"] not in ("draft", "improve"):
            continue
        turn_map = latest.setdefault(m["agent_id"], {})
        prev = turn_map.get(m["turn"])
        if prev is None or m["message_id"] > prev[0]:
            turn_map[m["turn"]] = (m["message_id"], m["solution"])
    out: Dict[str, Dict[int, Optional[str]]] = {}
    for agent_id, turn_map in latest.items():
        out[agent_id] = {
            turn: extract_letter(sol, question)
            for turn, (_mid, sol) in sorted(turn_map.items())
        }
    return out


def reference_letter(record: dict) -> Optional[str]:
    refs: List[str] = record.get("references") or []
    if not refs or not refs[0]:
        return None
    return refs[0][0].upper()
