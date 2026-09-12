"""Global memory pool: claim-level evidence repository with provenance,
holder masks, and dedup-without-deleting-provenance (paper Section 3).

No LLM calls anywhere in this module.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

from ..rq2_metrics import content_tokens

_ENTITY_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+[A-Z][a-zA-Z0-9'\-]+)*|\d{4})\b"
)


def extract_entities(text: str) -> List[str]:
    """Mechanical entity proxy: capitalized spans + years."""
    ents = []
    for m in _ENTITY_RE.finditer(text or ""):
        span = m.group(1).strip()
        if len(span) > 2 or span.isdigit():
            ents.append(span)
    return list(dict.fromkeys(ents))[:10]


class MemoryPool:
    """Evidence repository. Cluster key = source docid (exact) plus
    near-duplicate claim merging inside a cluster; provenance always kept."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.items: List[dict] = []           # memory entries
        self.holders: Dict[int, Dict[int, int]] = defaultdict(dict)
        # holders[mid][agent] = 1 independent | 2 received

    # ------------------------------------------------------------ write ----
    def add_claims(
        self,
        producer: int,
        rnd: int,
        claims: List[dict],
        title_to_docid: Dict[str, str],
        producer_docids: set,
        doc_texts: Optional[Dict[str, str]] = None,
    ) -> List[int]:
        """Register a round's claims. source resolution: claim.source_title
        matched against the producer's retrieved titles this task."""
        new_ids = []
        for c in claims:
            claim_text = (c.get("claim") or "").strip()
            if len(claim_text) < 10:
                continue
            src_title = (c.get("source_title") or "").strip()
            src_docid = title_to_docid.get(src_title)
            dup = self._find_duplicate(claim_text, src_docid)
            if dup is not None:
                self.items[dup]["independent_finders"].add(producer)
                self.holders[dup][producer] = min(
                    self.holders[dup].get(producer, 1), 1
                )
                continue
            mid = len(self.items)
            self.items.append(
                {
                    "mid": mid,
                    "round_created": rnd,
                    "producer": producer,
                    "claim": claim_text[:300],
                    "source_docid": src_docid,
                    "source_title": src_title,
                    "source_family_id": f"fam_{src_docid or src_title or mid}",
                    "provenance_trace": f"trace://r{rnd}/a{producer}",
                    "semantic_cluster_id": self._semantic_cluster(claim_text),
                    "entities": extract_entities(claim_text),
                    "tokens": len(claim_text.split()),
                    "independent_finders": {producer},
                    # source support q_src: token support of the claim
                    # inside its source doc text; unverifiable -> 0.5
                    "q_support": (
                        (lambda ct, dt: len(ct & dt) / max(1, len(ct)))(
                            content_tokens(claim_text),
                            content_tokens((doc_texts or {}).get(src_docid or "", "")),
                        )
                        if doc_texts and src_docid and (doc_texts or {}).get(src_docid)
                        else 0.5
                    ),
                }
            )
            self.holders[mid][producer] = 1
            new_ids.append(mid)
        return new_ids

    CLUSTER_ENCODER = "all-MiniLM-L6-v2"

    def _semantic_cluster(self, claim_text: str) -> str:
        """Cross-source semantic grouping: near-duplicate claims from
        different sources share a semantic cluster id but remain separate
        items — provenance and source families are never merged away.

        Default mode (paper Appendix A.1): a frozen sentence encoder embeds
        the claim and the item joins the CLOSEST existing cluster when
        cosine similarity exceeds a development-selected threshold
        (CQP_CLUSTER_TAU, default 0.62); otherwise it starts a new cluster.
        No gold supporting facts or final answers are used.

        CQP_CLUSTER=jaccard switches to a lightweight token-Jaccard > 0.7
        first-match fallback for environments without the encoder; it is a
        convenience only, not the paper mechanism. The mode, encoder
        version, and threshold are recorded in stats().
        """
        import os as _os
        if _os.environ.get("CQP_CLUSTER", "encoder") == "encoder":
            tau = float(_os.environ.get("CQP_CLUSTER_TAU", "0.62"))
            self._cluster_mode = ("encoder", self.CLUSTER_ENCODER, tau)
            from .semantic_value import embed as _emb
            import numpy as np
            h = np.asarray(_emb([claim_text])[0], dtype=np.float32)
            best_sim, best_cid = -1.0, None
            reps = getattr(self, "_cluster_reps", None)
            if reps is None:
                reps = self._cluster_reps = {}
            for cid, rep in reps.items():
                sim = float(h @ rep)
                if sim > best_sim:
                    best_sim, best_cid = sim, cid
            if best_cid is not None and best_sim > tau:
                return best_cid
            cid = f"sem_{len(self.items)}"
            reps[cid] = h
            return cid
        self._cluster_mode = ("jaccard", None, 0.7)
        t_new = content_tokens(claim_text)
        for it in self.items:
            t_old = content_tokens(it["claim"])
            inter = len(t_new & t_old)
            union = len(t_new | t_old) or 1
            if inter / union > 0.7:
                return it["semantic_cluster_id"]
        return f"sem_{len(self.items)}"

    def _find_duplicate(self, claim_text: str, src_docid: Optional[str]):
        """Same-source near-duplicate detection (union-Jaccard > 0.6):
        a duplicate adds an independent finder instead of a new item."""
        t_new = content_tokens(claim_text)
        for it in self.items:
            if src_docid and it["source_docid"] == src_docid:
                t_old = content_tokens(it["claim"])
                inter = len(t_new & t_old)
                union = len(t_new | t_old) or 1
                if inter / union > 0.6:
                    return it["mid"]
        return None

    def mark_received(self, agent: int, mids: List[int]) -> None:
        for mid in mids:
            if agent not in self.holders[mid]:
                self.holders[mid][agent] = 2

    # ------------------------------------------------------------- read ----
    def unseen_for(self, agent: int) -> List[dict]:
        return [it for it in self.items if agent not in self.holders[it["mid"]]]

    # ------------------------------------------------------- diagnostics ----
    def possession_matrix(self) -> Dict[int, Dict[int, int]]:
        return {mid: dict(h) for mid, h in self.holders.items()}

    def stats(self) -> dict:
        n_ind = sum(
            1 for h in self.holders.values() for v in h.values() if v == 1
        )
        n_rec = sum(
            1 for h in self.holders.values() for v in h.values() if v == 2
        )
        mode = getattr(self, "_cluster_mode",
               ("encoder", self.CLUSTER_ENCODER, 0.62))
        return {
            "n_items": len(self.items),
            "cluster_mode": mode[0],
            "cluster_encoder": mode[1],
            "cluster_threshold": mode[2],
            "independent_holdings": n_ind,
            "received_holdings": n_rec,
        }

    def export(self) -> List[dict]:
        out = []
        for it in self.items:
            e = dict(it)
            e["independent_finders"] = sorted(it["independent_finders"])
            e["holders"] = {str(a): v for a, v in self.holders[it["mid"]].items()}
            out.append(e)
        return out
