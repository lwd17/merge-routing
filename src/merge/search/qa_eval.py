"""HotpotQA-style answer scoring (EM + token F1, alias-aware)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Tuple


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: Iterable[str]) -> int:
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds if g))


def f1(pred: str, golds: Iterable[str]) -> float:
    best = 0.0
    p_toks = normalize(pred).split()
    for g in golds:
        if not g:
            continue
        g_toks = normalize(g).split()
        common = Counter(p_toks) & Counter(g_toks)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec = overlap / len(p_toks) if p_toks else 0.0
        rec = overlap / len(g_toks) if g_toks else 0.0
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score(pred: str, answer: str, aliases: Iterable[str] = ()) -> Tuple[int, float]:
    golds = [answer, *aliases]
    return em(pred, golds), f1(pred, golds)
