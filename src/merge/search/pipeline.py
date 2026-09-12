"""Multi-agent search pipeline with pluggable evidence-routing policies.

Round structure (equal LLM calls for every agent under every arm):
  private search -> publish claims to the shared pool -> build the global
  state -> allocate evidence transfers (zero LLM calls) -> routed items
  appear inside the receiver's next-round NORMAL propose call.

Communication precedes rounds 2..T, so with T=4 there are T_comm=3
communication steps. Routed items are durable holdings: they join the
receiver's evidence and compete for the answer window like retrieved
documents. A strict majority wins the group answer; otherwise plurality
with fixed source-count and lexical tie-breakers.

Round 1 is generated once per (task, team size) and replayed verbatim in
every arm, so arms are paired by construction and communication can only
influence rounds >= 2.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional

from . import prompts, qa_eval, retriever
from ..rq2_metrics import content_tokens
from .engine import _chat, _parse_finding, _parse_proposal, _seed, reset_usage, usage
from .memory_pool import MemoryPool
from .policies import POLICIES

RoutingPolicy = Callable[[dict], Dict[int, list]]


def _clean_query(q: str) -> str:
    """Sanitize search strings: drop JSON/markup characters and anything
    after a newline; keep the genuinely useful part."""
    q = (q or "").split("\n")[0]
    q = re.sub(r'[{}\[\]"`*<>|]', " ", q)
    return " ".join(q.split())[:200]


def plurality_vote(finals: List[dict], pool: MemoryPool) -> str:
    """Strict majority; otherwise plurality with fixed tie-breakers
    (distinct supporting sources, then lexicographic). Zero LLM calls."""
    norm_counts: Dict[str, int] = {}
    norm_first: Dict[str, str] = {}
    for f in finals:
        n = qa_eval.normalize(f["answer"])
        norm_counts[n] = norm_counts.get(n, 0) + 1
        norm_first.setdefault(n, f["answer"])

    def src_support(nrm: str) -> int:
        toks = content_tokens(nrm)
        if not toks:
            return 0
        return len({it.get("source_docid") for it in pool.items
                    if toks <= content_tokens(it["claim"])})

    best = max(norm_counts,
               key=lambda n: (norm_counts[n], src_support(n),
                              [-ord(ch) for ch in n]))
    return norm_first[best]


def build_state(*, pool, n_agents, rounds, rnd, sketches, own,
                received_tokens, question, task_id, alloc_log) -> dict:
    """Global allocation state at communication step before round `rnd`."""
    held_tokens = []
    for i in range(n_agents):
        toks = set()
        for it in pool.items:
            if i in pool.holders[it["mid"]]:
                toks |= content_tokens(it["claim"])
        held_tokens.append(toks)
    return {
        "task_id": task_id, "round": rnd, "n_agents": n_agents,
        "pool": pool, "sketches": sketches,
        "held_tokens": held_tokens, "received_tokens": list(received_tokens),
        "remaining_rounds": rounds - rnd + 1,
        "question": question,
        "alloc_log": alloc_log,
        "last_queries": {i: own[i][-1]["q1"] for i in range(n_agents)},
    }


def run_task(
    task: dict,
    arm: str,
    *,
    personas: List[dict],
    round0: List[dict],
    policy: Optional[RoutingPolicy] = None,
    n_agents: int = 4,
    rounds: int = 4,
    top_k: int = 5,
    seed_key: str = "merge",
) -> dict:
    if policy is None:
        if arm not in POLICIES:
            raise KeyError(arm)
        policy = POLICIES[arm]
    reset_usage()
    t0 = time.time()
    task_id = str(task["id"])
    question = task["question"]
    gold = {str(d) for d in task.get("gold_docids", [])}

    sys_prompts = [prompts.system_prompt(p, n_agents, agent_idx=i)
                   for i, p in enumerate(personas)]
    own: List[List[dict]] = [[] for _ in range(n_agents)]
    seen: List[set] = [set() for _ in range(n_agents)]
    received: List[List[dict]] = [[] for _ in range(n_agents)]
    title2doc: Dict[str, str] = {}
    doc_texts: Dict[str, str] = {}
    pool = MemoryPool(n_agents)
    events: List[dict] = []
    sketches: List[dict] = [{} for _ in range(n_agents)]
    comm_tokens = 0
    alloc_log: List[dict] = []

    def publish(agent: int, rnd: int, finding: dict, docids: List[str],
                docs: Optional[List[dict]] = None):
        if docs:
            for d in docs:
                doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
        pool.add_claims(agent, rnd, finding.get("claims", []), title2doc,
                        set(docids), doc_texts=doc_texts)

    # ---- round 1: common start (no reception anywhere) ----
    for i in range(n_agents):
        r0 = round0[i]
        docids = [d["docid"] for d in r0["docs"]]
        for d in r0["docs"]:
            title2doc.setdefault(d["title"], d["docid"])
        seen[i].update(docids)
        own[i].append({"round": 1, "q1": r0["q1"], "docs": r0["docs"]})
        sketches[i] = r0.get("proposal", {})
        publish(i, 1, r0.get("finding", {}), docids, docs=r0["docs"])
        events.append({
            "agent": i, "round": 1, "q1": r0["q1"], "docids": docids,
            "titles": [d["title"] for d in r0["docs"]],
            "routed_mids": [], "receiver": False,
            "gold_hits": sorted(set(docids) & gold),
            "sketch": sketches[i],
        })

    received_tokens = [0] * n_agents

    # ---- rounds >= 2: allocate, then search ----
    for rnd in range(2, rounds + 1):
        state = build_state(pool=pool, n_agents=n_agents, rounds=rounds,
                            rnd=rnd, sketches=sketches, own=own,
                            received_tokens=received_tokens,
                            question=question, task_id=task_id,
                            alloc_log=alloc_log)
        routed: Dict[int, List[dict]] = {
            i: items for i, items in (policy(state) or {}).items() if items}
        for i, items in routed.items():
            pool.mark_received(i, [it["mid"] for it in items])
            received[i].extend(items)
            tok = sum(it["tokens"] for it in items)
            comm_tokens += tok
            received_tokens[i] += tok

        for i in range(n_agents):
            held_recv = received[i][-8:] if received[i] else None
            proposal = _parse_proposal(_chat(
                sys_prompts[i],
                prompts.propose_user(question, own[i], held_recv),
                _seed(task_id, i, rnd, seed_key, "propose"),
                max_tokens=300, task_id=task_id,
                tag={"task_id": task_id, "arm": arm, "agent": i,
                     "round": rnd, "stage": "propose"}))
            sketches[i] = proposal
            q1 = _clean_query(proposal["query"])
            docs = retriever.search(q1, top_k=top_k, exclude_docids=seen[i])
            docids = [d["docid"] for d in docs]
            for d in docs:
                title2doc.setdefault(d["title"], d["docid"])
            seen[i].update(docids)
            finding = _parse_finding(_chat(
                sys_prompts[i], prompts.finding_user(question, q1, docs),
                _seed(task_id, i, rnd, seed_key, "finding"),
                max_tokens=380, task_id=task_id,
                tag={"task_id": task_id, "arm": arm, "agent": i,
                     "round": rnd, "stage": "finding"}))
            own[i].append({"round": rnd, "q1": q1, "docs": docs})
            publish(i, rnd, finding, docids, docs=docs)
            supported = [it["mid"] for it in pool.items
                         if pool.holders[it["mid"]].get(i) == 1]
            events.append({
                "agent": i, "round": rnd, "q1": q1, "docids": docids,
                "titles": [d["title"] for d in docs],
                "routed_mids": [it["mid"] for it in routed.get(i, [])],
                "receiver": i in routed,
                "gold_hits": sorted(set(docids) & gold),
                "sketch": {k: proposal.get(k) for k in
                           ("target_entities", "unresolved_slots",
                            "have_enough")},
                "mech": {
                    "cumulative_received_tokens": received_tokens[i],
                    "remaining_rounds": rounds - rnd,
                    "private_cluster_count": len(supported),
                    "received_cluster_count": sum(
                        1 for it in pool.items
                        if pool.holders[it["mid"]].get(i) == 2),
                },
            })

    # ---- decide: own docs + received claims in the same window ----
    finals = []
    for i in range(n_agents):
        evidence = [{**d, "round": r["round"]} for r in own[i] for d in r["docs"]]
        for it in received[i]:
            evidence.append({
                "docid": f"mem:{it['mid']}",
                "title": it["source_title"] or "team evidence",
                "excerpt": it["claim"], "round": it["round_created"]})
        ans_raw = _chat(
            sys_prompts[i], prompts.answer_user(question, evidence),
            _seed(task_id, i, 99, seed_key, "answer"),
            max_tokens=120, task_id=task_id,
            tag={"task_id": task_id, "arm": arm, "agent": i,
                 "round": rounds + 1, "stage": "answer"})
        ans = prompts.parse_json_field(ans_raw, "answer") or ans_raw.strip()[:80]
        em, f1 = qa_eval.score(ans, task.get("answer", ""),
                               task.get("answer_aliases", []))
        finals.append({"agent": i, "answer": ans, "em": em, "f1": round(f1, 4)})

    group_answer = plurality_vote(finals, pool)
    g_em, g_f1 = qa_eval.score(group_answer, task.get("answer", ""),
                               task.get("answer_aliases", []))
    team_seen = set().union(*seen)
    u = usage()
    return {
        "task_id": task_id, "arm": arm, "question": question,
        "answer": task.get("answer", ""), "gold_docids": sorted(gold),
        "n_agents": n_agents, "rounds": rounds,
        "events": events, "alloc_log": alloc_log,
        "received_sources": {
            str(i): sorted({it["source_docid"] for it in received[i]
                            if it.get("source_docid")})
            for i in range(n_agents)},
        "finals": finals,
        "group": {"answer": group_answer, "em": g_em, "f1": round(g_f1, 4),
                  "override": bool(g_em == 0 and any(f["em"] for f in finals))},
        "memory_pool": pool.export(), "pool_stats": pool.stats(),
        "comm_tokens": comm_tokens,
        "total_tokens": u["total_tokens"],
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "llm_calls": u["llm_calls"],
        "team_gold_coverage": len(team_seen & gold) / max(1, len(gold)),
        "wall_seconds": round(time.time() - t0, 2),
    }
