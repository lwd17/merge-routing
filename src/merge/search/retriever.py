"""BM25 retrieval over the HotpotQA index built in the pilot repo.

Behavior is ported unchanged from cqp-pilot/src/cqp/retriever.py:
  * per-agent self-deduplication only (an agent never re-reads its own
    documents; cross-agent dedup would build the intervention into the
    apparatus and destroy the collision quantity being measured)
  * raw document JSON lives in the Lucene "raw" field

A process-wide lock serializes JVM access; BM25 latency is negligible next to
the LLM calls.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..config import HOTPOT_DIR

_lock = threading.Lock()
_searcher = None

# hybrid retrieval (CQP_RETRIEVER=rerank): BM25 top-50 candidates reranked
# by MiniLM cosine — extra_plan §17.4 cross-retriever generalization; the
# exposure-utility claims must not be BM25-only. Default path unchanged.
_EMB_MODEL = None
_EMB_CACHE: Dict[str, list] = {}
_emb_lock = threading.Lock()


def _embed(texts: List[str]) -> List[list]:
    global _EMB_MODEL
    with _emb_lock:
        todo = [t for t in texts if t not in _EMB_CACHE]
        if todo:
            if _EMB_MODEL is None:
                from sentence_transformers import SentenceTransformer
                _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = _EMB_MODEL.encode(todo, normalize_embeddings=True)
            for t, v in zip(todo, vecs):
                _EMB_CACHE[t] = v.tolist()
        return [_EMB_CACHE[t] for t in texts]


def _get_searcher(index_dir: Path):
    global _searcher
    with _lock:
        if _searcher is None:
            from pyserini.search.lucene import LuceneSearcher

            _searcher = LuceneSearcher(str(index_dir))
            _searcher.set_bm25(k1=0.9, b=0.4)
    return _searcher


def fetch_doc(
    docid: str, index_dir: Optional[Path] = None,
    excerpt_chars: int = 1200,
) -> Optional[Dict]:
    """Direct fetch of one document by docid (offline tooling)."""
    searcher = _get_searcher(index_dir or HOTPOT_DIR / "index_bm25")
    with _lock:
        d = searcher.doc(str(docid))
        if d is None:
            return None
        raw_field = d.lucene_document().getField("raw")
        raw = json.loads(raw_field.stringValue()) if raw_field else {}
    contents = raw.get("contents", raw.get("text", ""))
    title = raw.get("title", "")
    body = contents
    if not title and contents:
        head, _, rest = contents.partition("\n\n")
        if rest and len(head) < 120:
            title, body = head.strip(), rest
    return {"docid": str(docid), "title": title,
            "excerpt": body[:excerpt_chars], "score": 0.0}


def search(
    query: str,
    top_k: int = 5,
    exclude_docids=None,
    index_dir: Optional[Path] = None,
    excerpt_chars: int = 1200,
) -> List[Dict]:
    """Returns [{docid, title, excerpt, score}] with self-dedup applied."""
    searcher = _get_searcher(index_dir or HOTPOT_DIR / "index_bm25")
    exclude = set(exclude_docids or ())
    rerank = os.environ.get("CQP_RETRIEVER") == "rerank"
    pool_k = 50 if rerank else top_k
    fetch = pool_k if not exclude else min(pool_k + len(exclude) + 20, 300)
    with _lock:
        hits = searcher.search(query, k=fetch)
        results = []
        for hit in hits:
            if hit.docid in exclude:
                continue
            if len(results) >= pool_k:
                break
            raw_field = hit.lucene_document.getField("raw")
            raw = json.loads(raw_field.stringValue()) if raw_field else {}
            contents = raw.get("contents", raw.get("text", ""))
            title = raw.get("title", "")
            body = contents
            if not title and contents:
                # HotpotQA corpus stores contents as "Title\n\nBody".
                head, _, rest = contents.partition("\n\n")
                if rest and len(head) < 120:
                    title, body = head.strip(), rest
                else:
                    m = re.search(r"^title:\s*(.+)$", contents, re.MULTILINE)
                    if m:
                        title = m.group(1).strip().strip('"').strip("'")
            results.append(
                {
                    "docid": str(hit.docid),
                    "title": title,
                    "excerpt": body[:excerpt_chars],
                    "score": float(hit.score),
                }
            )
    if rerank and results:
        q_vec = _embed([query])[0]
        d_vecs = _embed([
            f"{d['title']} {d['excerpt'][:300]}" for d in results])
        scored = sorted(
            zip(results, d_vecs),
            key=lambda dv: -sum(a * b for a, b in zip(q_vec, dv[1])),
        )
        results = []
        for d, v in scored[:top_k]:
            d = dict(d)
            d["score"] = float(sum(a * b for a, b in zip(q_vec, v)))
            results.append(d)
    return results
