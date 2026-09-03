"""MERGE engine: multi-agent tool-using search with pluggable
evidence-routing policies (paper Sections 3-5).

Round structure (equal LLM calls for every agent under every arm):
  Private Search -> Publish claims to the shared memory pool -> Build
  global state -> Allocate evidence transfers (zero-LLM) -> routed items
  appear in the receiver's next-round NORMAL propose call. No extra call
  of any kind exists on the communication path.

Received items are durable holdings (possession=2): they are added to the
receiver's evidence and compete for the answer window like retrieved docs.

Routing policies are callables state -> {receiver: [pool items]} (an empty
dict means no communication this step), so the MERGE arms and every
baseline share one code path. All policies are zero-LLM.

Arms (Table 7 / Table 9):
  no_comm       no communication
  capped_full   ordinary broadcast under the shared hard limits
  random_ev     random transfers at a fixed 335-token dose
  hmem          receiver-local memory pull (HMem)
  agentprune    pruned sender edges (AgentPrune-inspired)
  dytopo        dynamic need/offer topology (DyTopo-inspired)
  global_set    learned joint set scorer, no explicit exposure terms
  designed_sep / merge_d / learned_sep / merge_l   factorial MERGE arms
  solo_tool / central_orch / lead_roles            agentic baselines
"""
from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Callable, Dict, List, Optional

from . import prompts, qa_eval, retriever
from ..rq2_metrics import content_tokens
from .engine import _chat, _parse_finding, _parse_proposal, _seed
from .memory_pool import MemoryPool
from .semantic_value import cosine as _cos, embed as _emb
from .welfare import (
    B_RECV, ITEM_CAP, b_global,
    policy_capped_full, policy_designed_sep, policy_global_set,
    policy_learned_sep, policy_merge_d, policy_merge_l,
)

RoutingPolicy = Callable[[dict], Dict[int, list]]

RANDOM_DOSE = 335  # Table 7: fixed per-step dose of the random arm


def policy_none(state: dict) -> Dict[int, list]:
    return {}


def policy_random_evidence(state: dict) -> Dict[int, list]:
    """Random evidence routing (Table 7): uniformly random (item, receiver)
    transfers at a FIXED 335-token per-step dose — adjudicates whether
    evidence-level selection needs an objective or merely a dose."""
    pool = state["pool"]
    n = state["n_agents"]
    seed = int(hashlib.md5(
        f"{state['task_id']}:{state['round']}:rev".encode()
    ).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pairs = [(it, i) for i in range(n) for it in pool.unseen_for(i)]
    rng.shuffle(pairs)
    routed: Dict[int, list] = {}
    g = 0
    for it, i in pairs:
        t = it["tokens"]
        if g + t > RANDOM_DOSE:
            continue
        routed.setdefault(i, []).append(it)
        g += t
    return routed


def _clean_query(q: str) -> str:
    """Sanitize search strings: drop JSON/markup special characters and
    anything after a newline; keep the genuinely useful part."""
    q = (q or "").split("\n")[0]
    q = re.sub(r'[{}\[\]"`*<>|]', " ", q)
    return " ".join(q.split())[:200]


def policy_hmem(state: dict) -> Dict[int, list]:
    """HMem: hierarchical shared-local memory (memory-architecture
    baseline). Global pool = M_G, per-agent holdings = M_L — identical
    structure to MERGE. Sync operator: each agent PULLS from M_G by its
    OWN relevance (lexical slot/question overlap + dense cosine), under
    the shared envelope. The single difference from MERGE is the missing
    team-coupled objective — this arm isolates 'globally optimized
    allocation vs per-agent greedy retrieval'."""
    pool = state["pool"]
    n = state["n_agents"]
    q_toks = content_tokens(state.get("question", ""))
    scored_pairs = []
    for i in range(n):
        sk = state["sketches"][i]
        slots = (sk.get("unresolved_slots") or [])[:4]
        slot_toks = [content_tokens(s) for s in slots]
        need_txt = " ; ".join(slots) if slots else state.get("question", "")
        need_emb = _emb([need_txt])[0]
        for it in pool.unseen_for(i):
            toks = content_tokens(it["claim"])
            lex = max((len(toks & st) for st in slot_toks), default=0) \
                + len(toks & q_toks)
            d = _cos(_emb([it["claim"]])[0], need_emb)
            scored_pairs.append((lex + d, it, i))
    scored_pairs.sort(key=lambda x: -x[0])
    cap = b_global()
    routed: Dict[int, list] = {}
    g = 0
    per_tok: Dict[int, int] = {}
    per_cnt: Dict[int, int] = {}
    for _s, it, i in scored_pairs:
        t = it["tokens"]
        if g + t > cap or per_tok.get(i, 0) + t > min(B_RECV, cap) \
                or per_cnt.get(i, 0) >= ITEM_CAP:
            continue
        routed.setdefault(i, []).append(it)
        g += t
        per_tok[i] = per_tok.get(i, 0) + t
        per_cnt[i] = per_cnt.get(i, 0) + 1
    return routed


def policy_dytopo(state: dict) -> Dict[int, list]:
    """DyTopo-inspired baseline (idea-faithful zero-LLM adaptation):
    round-wise DYNAMIC semantic need/offer topology. Per receiver, keep
    the 2 senders whose unseen claims best match the receiver's declared
    need (dynamic edges recomputed every round), then route the
    best-matching claims along surviving edges under the shared envelope
    for cost parity (the budget sweep of Appendix B.4 moves this cap and
    MERGE's together)."""
    pool = state["pool"]
    n = state["n_agents"]
    needs = {}
    for i in range(n):
        sk = state["sketches"][i]
        slots = (sk.get("unresolved_slots") or [])[:4]
        needs[i] = _emb([" ; ".join(slots) if slots
                         else state.get("question", "")])[0]
    scored = []
    for i in range(n):
        aff: Dict[int, float] = {}
        by_sender: Dict[int, list] = {}
        for it in pool.unseen_for(i):
            j = it["producer"]
            if j == i:
                continue
            c = _cos(_emb([it["claim"]])[0], needs[i])
            aff[j] = max(aff.get(j, -1.0), c)
            by_sender.setdefault(j, []).append((c, it))
        keep = sorted(aff, key=lambda j: -aff[j])[:2]
        for j in keep:
            for c, it in by_sender[j]:
                scored.append((c, it, i))
    scored.sort(key=lambda x: -x[0])
    cap = b_global()
    routed: Dict[int, list] = {}
    g = 0
    per_tok: Dict[int, int] = {}
    per_cnt: Dict[int, int] = {}
    for _c, it, i in scored:
        t = it["tokens"]
        if g + t > cap or per_tok.get(i, 0) + t > min(B_RECV, cap) \
                or per_cnt.get(i, 0) >= ITEM_CAP:
            continue
        routed.setdefault(i, []).append(it)
        g += t
        per_tok[i] = per_tok.get(i, 0) + t
        per_cnt[i] = per_cnt.get(i, 0) + 1
    return routed


def policy_agentprune(state: dict) -> Dict[int, list]:
    """AgentPrune-inspired baseline (idea-faithful zero-LLM adaptation):
    spatial-temporal pruning of the communication graph. Round 2 observes
    the full graph; from round 3 each directed edge carries an EMA utility
    = adoption of items routed along it (routed entities appearing in the
    receiver's next query); edges at/below the round median are PRUNED and
    stay judged every round (temporal decay via EMA). Routing along
    surviving edges mirrors dytopo's semantic matching under the same
    matched envelope."""
    pool = state["pool"]
    n = state["n_agents"]
    rnd = state["round"]
    ema = getattr(pool, "_ap_ema", None)
    if ema is None:
        ema = {}
        pool._ap_ema = ema
    last = getattr(pool, "_ap_last", {})
    # update edge EMA from last round's routed items vs current queries
    for (j, i), ents in last.items():
        q = state.get("prev_q_toks", {}).get(i, set())
        hit = 1.0 if (ents & q) else 0.0
        ema[(j, i)] = 0.5 * ema.get((j, i), 0.5) + 0.5 * hit
    alive = {(j, i) for j in range(n) for i in range(n) if j != i}
    if rnd >= 3 and ema:
        vals = sorted(ema.get(e, 0.5) for e in alive)
        med = vals[len(vals) // 2]
        pruned = {e for e in alive if ema.get(e, 0.5) > med}
        if pruned:
            alive = pruned
    needs = {}
    for i in range(n):
        sk = state["sketches"][i]
        slots = (sk.get("unresolved_slots") or [])[:4]
        needs[i] = _emb([" ; ".join(slots) if slots
                         else state.get("question", "")])[0]
    scored = []
    for i in range(n):
        for it in pool.unseen_for(i):
            j = it["producer"]
            if j == i or (j, i) not in alive:
                continue
            scored.append((_cos(_emb([it["claim"]])[0], needs[i]), it, i))
    scored.sort(key=lambda x: -x[0])
    cap = b_global()
    routed: Dict[int, list] = {}
    g = 0
    per_tok: Dict[int, int] = {}
    per_cnt: Dict[int, int] = {}
    new_last: Dict[tuple, set] = {}
    for _c, it, i in scored:
        t = it["tokens"]
        if g + t > cap or per_tok.get(i, 0) + t > min(B_RECV, cap) \
                or per_cnt.get(i, 0) >= ITEM_CAP:
            continue
        routed.setdefault(i, []).append(it)
        g += t
        per_tok[i] = per_tok.get(i, 0) + t
        per_cnt[i] = per_cnt.get(i, 0) + 1
        ents = set()
        for e in it.get("entities", []):
            ents |= content_tokens(e)
        key = (it["producer"], i)
        new_last.setdefault(key, set()).update(ents)
    pool._ap_last = new_last
    return routed


POLICIES: Dict[str, RoutingPolicy] = {
    "no_comm": policy_none,
    "capped_full": policy_capped_full,
    "random_ev": policy_random_evidence,
    "hmem": policy_hmem,
    "agentprune": policy_agentprune,
    "dytopo": policy_dytopo,
    "global_set": policy_global_set,
    "designed_sep": policy_designed_sep,
    "merge_d": policy_merge_d,
    "learned_sep": policy_learned_sep,
    "merge_l": policy_merge_l,
    # agentic baselines: dedicated runners, no routing policy
    "solo_tool": policy_none,
    "central_orch": policy_none,
    "lead_roles": policy_none,
}


def _plurality_vote(finals: List[dict], pool: MemoryPool):
    """Group answer (paper Section 5): a strict majority wins; otherwise
    plurality with fixed tie-breakers — more distinct supporting sources
    first, then lexicographic order. Zero LLM calls."""
    norm_counts: Dict[str, int] = {}
    norm_first: Dict[str, str] = {}
    for f in finals:
        n = qa_eval.normalize(f["answer"])
        norm_counts[n] = norm_counts.get(n, 0) + 1
        norm_first.setdefault(n, f["answer"])

    def _src_support(nrm: str) -> int:
        toks = content_tokens(nrm)
        if not toks:
            return 0
        return len({
            it.get("source_docid")
            for it in pool.items
            if toks <= content_tokens(it["claim"])
        })

    best_norm = max(
        norm_counts,
        key=lambda n: (norm_counts[n], _src_support(n),
                       [-ord(ch) for ch in n]),
    )
    return norm_first[best_norm]


def _run_solo(task: dict, round0: List[dict], seed_key: str = "gurc",
              top_k: int = 5, solo_rounds: int = 17) -> dict:
    """solo_tool baseline (single-agent tool loop, budget-matched):
    ONE agent, linear tool loop (propose -> BM25 -> finding), context
    compression via an accumulated claim log (recent raw log + all claims),
    single final answer. Online calls: 16x2+1 = 33 <= team's ~36."""
    t0 = time.time()
    task_id = str(task["id"])
    question = task["question"]
    gold = {str(d) for d in task.get("gold_docids", [])}
    sys_p = prompts.system_prompt(
        {"role": "Research Agent",
         "description": "A thorough solo researcher who answers questions "
                        "by iteratively searching Wikipedia."}, 1)
    own: List[dict] = []
    seen: set = set()
    pool = MemoryPool(1)
    title2doc: Dict[str, str] = {}
    doc_texts: Dict[str, str] = {}
    r0 = round0[0]
    own.append({"round": 1, "q1": r0["q1"], "docs": r0["docs"]})
    seen.update(d["docid"] for d in r0["docs"])
    for d in r0["docs"]:
        title2doc.setdefault(d["title"], d["docid"])
        doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
    pool.add_claims(0, 1, r0.get("claims", []), title2doc,
                    seen, doc_texts=doc_texts)
    for rnd in range(2, solo_rounds + 1):
        # context compression: recent 4 rounds of raw log; claims persist
        claim_log = [
            {"claim": it["claim"], "source_title": it["source_title"],
             "entities": it["entities"], "mid": it["mid"],
             "source_docid": it["source_docid"],
             "round_created": it["round_created"], "tokens": it["tokens"]}
            for it in pool.items
        ][-12:]
        proposal = _parse_proposal(_chat(
            sys_p, prompts.propose_user(question, own[-4:], claim_log or None),
            _seed(task_id, 0, rnd, seed_key, "propose"), max_tokens=300,
            task_id=task_id,
            tag={"task_id": task_id, "arm": "solo_tool", "agent": 0,
                 "round": rnd, "stage": "propose"}))
        q1 = _clean_query(proposal["query"])
        docs = retriever.search(q1, top_k=top_k, exclude_docids=seen)
        for d in docs:
            title2doc.setdefault(d["title"], d["docid"])
            doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
        seen.update(d["docid"] for d in docs)
        finding = _parse_finding(_chat(
            sys_p, prompts.finding_user(question, q1, docs),
            _seed(task_id, 0, rnd, seed_key, "finding"), max_tokens=380,
            task_id=task_id,
            tag={"task_id": task_id, "arm": "solo_tool", "agent": 0,
                 "round": rnd, "stage": "finding"}))
        own.append({"round": rnd, "q1": q1, "docs": docs})
        pool.add_claims(0, rnd, finding.get("claims", []), title2doc,
                        {d["docid"] for d in docs}, doc_texts=doc_texts)
    evidence = [{**d, "round": r["round"]} for r in own for d in r["docs"]]
    ans_raw = _chat(
        sys_p, prompts.answer_user(question, evidence),
        _seed(task_id, 0, 99, seed_key, "answer"), max_tokens=120,
        task_id=task_id,
        tag={"task_id": task_id, "arm": "solo_tool", "agent": 0,
             "round": solo_rounds + 1, "stage": "answer"})
    ans = prompts.parse_json_field(ans_raw, "answer") or ans_raw.strip()[:80]
    em, f1 = qa_eval.score(
        ans, task.get("answer", ""), task.get("answer_aliases", []))
    return {
        "task_id": task_id, "arm": "solo_tool", "question": question,
        "answer": task.get("answer", ""), "gold_docids": sorted(gold),
        "n_agents": 1, "rounds": solo_rounds, "events": [],
        "alloc_log": [], "received_sources": {},
        "finals": [{"agent": 0, "answer": ans, "em": em,
                    "f1": round(f1, 4)}],
        "group": {"answer": ans, "em": em, "f1": round(f1, 4),
                  "override": False},
        "memory_pool": pool.export(), "pool_stats": pool.stats(),
        "comm_tokens": 0,
        "team_gold_coverage": len(seen & gold) / max(1, len(gold)),
        "wall_seconds": round(time.time() - t0, 2),
    }


def _run_central(task: dict, *, personas: List[dict], round0: List[dict],
                 seed_key: str = "gurc", top_k: int = 5,
                 n_agents: int = 4, rounds: int = 4) -> dict:
    """central_orch baseline (budget-matched): a main orchestrator LLM
    dynamically instructs sub-agents each round — each assignment is an
    (instruction, context) tuple; sub-agents execute propose+search+finding
    under that instruction. Communication is the centralized instruction
    channel (its tokens are the comm cost). Online calls:
    rounds2-4 x (1 orch + 4 propose + 4 finding) + 4 answers
    = 31 (team arms: 28) — slight budget advantage TO the baseline."""
    t0 = time.time()
    task_id = str(task["id"])
    question = task["question"]
    gold = {str(d) for d in task.get("gold_docids", [])}
    sys_prompts = [prompts.system_prompt(p, n_agents) for p in personas]
    own: List[List[dict]] = [[] for _ in range(n_agents)]
    seen: List[set] = [set() for _ in range(n_agents)]
    title2doc: Dict[str, str] = {}
    doc_texts: Dict[str, str] = {}
    pool = MemoryPool(n_agents)
    comm_tokens = 0
    for i in range(n_agents):
        r0 = round0[i]
        docids = [d["docid"] for d in r0["docs"]]
        for d in r0["docs"]:
            title2doc.setdefault(d["title"], d["docid"])
            doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
        own[i].append({"round": 1, "q1": r0["q1"], "docs": r0["docs"]})
        seen[i].update(docids)
        pool.add_claims(i, 1, r0.get("claims", []), title2doc, set(docids),
                        doc_texts=doc_texts)
    for rnd in range(2, rounds + 1):
        claims_txt = "\n".join(
            f"- {it['claim'][:150]} (src: {it['source_title']})"
            for it in pool.items[-12:])
        state_txt = "\n".join(
            f"agent{i} last query: {own[i][-1]['q1']}"
            for i in range(n_agents))
        orch_raw = _chat(
            "You are the lead orchestrator of a 4-person search team. "
            "Output only JSON.",
            f"Question: {question}\n\nTeam findings so far:\n{claims_txt}\n\n"
            f"{state_txt}\n\n"
            "Assign each agent a focused sub-task for the next search round "
            "(cover different missing facts; name concrete entities).\n"
            'JSON: {"assignments": ["instruction for agent 0", '
            '"instruction for agent 1", "instruction for agent 2", '
            '"instruction for agent 3"]}',
            _seed(task_id, 9, rnd, seed_key, "orch"), max_tokens=280,
            task_id=task_id,
            tag={"task_id": task_id, "arm": "central_orch", "agent": -1,
                 "round": rnd, "stage": "orch"})
        obj = prompts.parse_json_obj(orch_raw) or {}
        assigns = obj.get("assignments") or []
        assigns = [str(a) for a in assigns][:n_agents]
        while len(assigns) < n_agents:
            assigns.append("Continue searching for missing facts.")
        comm_tokens += sum(len(a.split()) for a in assigns)
        for i in range(n_agents):
            proposal = _parse_proposal(_chat(
                sys_prompts[i],
                prompts.propose_user(question, own[i], None,
                                     suggest=[assigns[i]]),
                _seed(task_id, i, rnd, seed_key, "propose"),
                max_tokens=300, task_id=task_id,
                tag={"task_id": task_id, "arm": "central_orch", "agent": i,
                     "round": rnd, "stage": "propose"}))
            q1 = _clean_query(proposal["query"])
            docs = retriever.search(q1, top_k=top_k, exclude_docids=seen[i])
            for d in docs:
                title2doc.setdefault(d["title"], d["docid"])
                doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
            seen[i].update(d["docid"] for d in docs)
            finding = _parse_finding(_chat(
                sys_prompts[i], prompts.finding_user(question, q1, docs),
                _seed(task_id, i, rnd, seed_key, "finding"),
                max_tokens=380, task_id=task_id,
                tag={"task_id": task_id, "arm": "central_orch", "agent": i,
                     "round": rnd, "stage": "finding"}))
            own[i].append({"round": rnd, "q1": q1, "docs": docs})
            pool.add_claims(i, rnd, finding.get("claims", []), title2doc,
                            {d["docid"] for d in docs}, doc_texts=doc_texts)
    finals = []
    for i in range(n_agents):
        evidence = [{**d, "round": r["round"]}
                    for r in own[i] for d in r["docs"]]
        ans_raw = _chat(
            sys_prompts[i],
            prompts.answer_user(question, evidence),
            _seed(task_id, i, 99, seed_key, "answer"), max_tokens=120,
            task_id=task_id,
            tag={"task_id": task_id, "arm": "central_orch", "agent": i,
                 "round": rounds + 1, "stage": "answer"})
        ans = (prompts.parse_json_field(ans_raw, "answer")
               or ans_raw.strip()[:80])
        em, f1 = qa_eval.score(
            ans, task.get("answer", ""), task.get("answer_aliases", []))
        finals.append({"agent": i, "answer": ans, "em": em,
                       "f1": round(f1, 4)})
    group_answer = _plurality_vote(finals, pool)
    g_em, g_f1 = qa_eval.score(
        group_answer, task.get("answer", ""), task.get("answer_aliases", []))
    team_seen = set().union(*seen)
    grp = {"answer": group_answer, "em": g_em, "f1": round(g_f1, 4),
           "override": bool(g_em == 0 and any(f["em"] for f in finals))}
    return {
        "task_id": task_id, "arm": "central_orch", "question": question,
        "answer": task.get("answer", ""), "gold_docids": sorted(gold),
        "n_agents": n_agents, "rounds": rounds, "events": [],
        "alloc_log": [], "received_sources": {},
        "finals": finals, "group": grp,
        "memory_pool": pool.export(), "pool_stats": pool.stats(),
        "comm_tokens": comm_tokens,
        "team_gold_coverage": len(team_seen & gold) / max(1, len(gold)),
        "wall_seconds": round(time.time() - t0, 2),
    }


def _run_lead_roles(task: dict, *, personas: List[dict], round0: List[dict],
                    seed_key: str = "gurc", top_k: int = 5,
                    n_agents: int = 4, rounds: int = 4) -> dict:
    """lead_roles baseline (budget-matched): a lead LLM decomposes the
    question ONCE into per-role subtasks; the role subagents then work
    their fixed subtasks INDEPENDENTLY (no inter-agent communication, no
    per-round re-instruction); finally the lead SYNTHESIZES the answer
    from the team's findings and the subagents' answers. Communication =
    one-shot assignment tokens + the findings forwarded up to the lead at
    synthesis. Online calls: 1 + 3x(4+4) + 4 + 1 = 30 (team arms: 28)."""
    t0 = time.time()
    task_id = str(task["id"])
    question = task["question"]
    gold = {str(d) for d in task.get("gold_docids", [])}
    sys_prompts = [prompts.system_prompt(p, n_agents) for p in personas]
    own: List[List[dict]] = [[] for _ in range(n_agents)]
    seen: List[set] = [set() for _ in range(n_agents)]
    title2doc: Dict[str, str] = {}
    doc_texts: Dict[str, str] = {}
    pool = MemoryPool(n_agents)
    comm_tokens = 0
    for i in range(n_agents):
        r0 = round0[i]
        docids = [d["docid"] for d in r0["docs"]]
        for d in r0["docs"]:
            title2doc.setdefault(d["title"], d["docid"])
            doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
        own[i].append({"round": 1, "q1": r0["q1"], "docs": r0["docs"]})
        seen[i].update(docids)
        pool.add_claims(i, 1, r0.get("claims", []), title2doc, set(docids),
                        doc_texts=doc_texts)
    # ---- ONE-SHOT decomposition by the lead ----
    roles_txt = "\n".join(
        f"agent{i} ({personas[i].get('role','agent')}): "
        f"{personas[i].get('description','')[:120]}"
        for i in range(n_agents))
    lead_raw = _chat(
        "You are the lead of a research team. Output only JSON.",
        f"Question: {question}\n\nYour team:\n{roles_txt}\n\n"
        "Decompose the question into one focused, self-contained research "
        "subtask per agent (cover different facts; name concrete entities; "
        "the union of subtasks must suffice to answer).\n"
        'JSON: {"subtasks": ["subtask for agent 0", "subtask for agent 1", '
        '"subtask for agent 2", "subtask for agent 3"]}',
        _seed(task_id, 9, 1, seed_key, "lead"), max_tokens=300,
        task_id=task_id,
        tag={"task_id": task_id, "arm": "lead_roles", "agent": -1,
             "round": 1, "stage": "lead"})
    obj = prompts.parse_json_obj(lead_raw) or {}
    subs = [str(a) for a in (obj.get("subtasks") or [])][:n_agents]
    while len(subs) < n_agents:
        subs.append("Search for facts needed to answer the question.")
    comm_tokens += sum(len(a.split()) for a in subs)
    # ---- rounds 2..R: independent work on fixed subtasks ----
    for rnd in range(2, rounds + 1):
        for i in range(n_agents):
            proposal = _parse_proposal(_chat(
                sys_prompts[i],
                prompts.propose_user(question, own[i], None,
                                     suggest=[subs[i]]),
                _seed(task_id, i, rnd, seed_key, "propose"),
                max_tokens=300, task_id=task_id,
                tag={"task_id": task_id, "arm": "lead_roles", "agent": i,
                     "round": rnd, "stage": "propose"}))
            q1 = _clean_query(proposal["query"])
            docs = retriever.search(q1, top_k=top_k, exclude_docids=seen[i])
            for d in docs:
                title2doc.setdefault(d["title"], d["docid"])
                doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
            seen[i].update(d["docid"] for d in docs)
            finding = _parse_finding(_chat(
                sys_prompts[i], prompts.finding_user(question, q1, docs),
                _seed(task_id, i, rnd, seed_key, "finding"),
                max_tokens=380, task_id=task_id,
                tag={"task_id": task_id, "arm": "lead_roles", "agent": i,
                     "round": rnd, "stage": "finding"}))
            own[i].append({"round": rnd, "q1": q1, "docs": docs})
            pool.add_claims(i, rnd, finding.get("claims", []), title2doc,
                            {d["docid"] for d in docs}, doc_texts=doc_texts)
    # ---- per-agent answers (as in every arm), then LEAD synthesis ----
    finals = []
    for i in range(n_agents):
        evidence = [{**d, "round": r["round"]}
                    for r in own[i] for d in r["docs"]]
        ans_raw = _chat(
            sys_prompts[i],
            prompts.answer_user(question, evidence),
            _seed(task_id, i, 99, seed_key, "answer"), max_tokens=120,
            task_id=task_id,
            tag={"task_id": task_id, "arm": "lead_roles", "agent": i,
                 "round": rounds + 1, "stage": "answer"})
        ans = (prompts.parse_json_field(ans_raw, "answer")
               or ans_raw.strip()[:80])
        em, f1 = qa_eval.score(
            ans, task.get("answer", ""), task.get("answer_aliases", []))
        finals.append({"agent": i, "answer": ans, "em": em,
                       "f1": round(f1, 4)})
    claims_txt = "\n".join(
        f"- {it['claim'][:150]} (src: {it['source_title']})"
        for it in pool.items[:24])
    ans_txt = "\n".join(
        f"agent{f['agent']} answer: {f['answer']}" for f in finals)
    comm_tokens += len(claims_txt.split()) + len(ans_txt.split())
    syn_raw = _chat(
        "You are the lead of a research team. Output only JSON.",
        f"Question: {question}\n\nTeam findings:\n{claims_txt}\n\n"
        f"{ans_txt}\n\nSynthesize the single best final answer as a short "
        'phrase.\nJSON: {"answer": "..."}',
        _seed(task_id, 9, 99, seed_key, "synth"), max_tokens=120,
        task_id=task_id,
        tag={"task_id": task_id, "arm": "lead_roles", "agent": -1,
             "round": rounds + 1, "stage": "synth"})
    g_ans = (prompts.parse_json_field(syn_raw, "answer")
             or syn_raw.strip()[:80])
    g_em, g_f1 = qa_eval.score(
        g_ans, task.get("answer", ""), task.get("answer_aliases", []))
    team_seen = set().union(*seen)
    grp = {"answer": g_ans, "em": g_em, "f1": round(g_f1, 4),
           "override": bool(g_em == 0 and any(f["em"] for f in finals))}
    return {
        "task_id": task_id, "arm": "lead_roles", "question": question,
        "answer": task.get("answer", ""), "gold_docids": sorted(gold),
        "n_agents": n_agents, "rounds": rounds, "events": [],
        "alloc_log": [], "received_sources": {},
        "finals": finals, "group": grp,
        "memory_pool": pool.export(), "pool_stats": pool.stats(),
        "comm_tokens": comm_tokens,
        "team_gold_coverage": len(team_seen & gold) / max(1, len(gold)),
        "wall_seconds": round(time.time() - t0, 2),
    }


def run_gurc_task(
    task: dict,
    arm: str,
    *,
    personas: List[dict],
    round0: List[dict],
    policy: Optional[RoutingPolicy] = None,
    n_agents: int = 4,
    rounds: int = 4,
    top_k: int = 5,
    seed_key: str = "gurc",
) -> dict:
    if arm == "solo_tool":
        return _run_solo(task, round0, seed_key=seed_key, top_k=top_k)
    if arm == "central_orch":
        return _run_central(task, personas=personas, round0=round0,
                            seed_key=seed_key, top_k=top_k,
                            n_agents=n_agents, rounds=rounds)
    if arm == "lead_roles":
        return _run_lead_roles(task, personas=personas, round0=round0,
                               seed_key=seed_key, top_k=top_k,
                               n_agents=n_agents, rounds=rounds)
    if policy is None:
        if arm not in POLICIES:
            raise KeyError(arm)
        policy = POLICIES[arm]
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
    pool = MemoryPool(n_agents)
    events: List[dict] = []
    sketches: List[dict] = [{} for _ in range(n_agents)]
    comm_tokens = 0
    doc_texts: Dict[str, str] = {}

    def publish(agent: int, rnd: int, finding: dict, docids: List[str],
                docs: Optional[List[dict]] = None):
        if docs:
            for d in docs:
                doc_texts.setdefault(d["docid"], d.get("excerpt", ""))
        pool.add_claims(
            agent, rnd, finding.get("claims", []), title2doc, set(docids),
            doc_texts=doc_texts,
        )

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
        events.append(
            {
                "agent": i, "round": 1, "q1": r0["q1"], "docids": docids,
                "titles": [d["title"] for d in r0["docs"]],
                "routed_mids": [], "receiver": False,
                "gold_hits": sorted(set(docids) & gold),
                "sketch": sketches[i],
            }
        )

    received_tokens = [0] * n_agents
    alloc_log: List[dict] = []

    # ---- rounds >= 2 ----
    for rnd in range(2, rounds + 1):
        # Build global state from everything published up to round rnd-1,
        # allocate transfers (zero-LLM), route BEFORE this round's calls.
        held_tokens = []
        for i in range(n_agents):
            toks = set()
            for it in pool.items:
                if i in pool.holders[it["mid"]]:
                    toks |= content_tokens(it["claim"])
            held_tokens.append(toks)
        state = {
            "task_id": task_id, "round": rnd, "n_agents": n_agents,
            "pool": pool, "sketches": sketches,
            "held_tokens": held_tokens, "received_tokens": received_tokens,
            "remaining_rounds": rounds - rnd + 1,
            "question": question,
            "alloc_log": alloc_log,
            "last_queries": {i: own[i][-1]["q1"] for i in range(n_agents)},
            "prev_q_toks": {i: content_tokens(own[i][-1]["q1"])
                            for i in range(n_agents)},
        }
        routed: Dict[int, List[dict]] = {
            i: items for i, items in (policy(state) or {}).items() if items
        }
        for i, items in routed.items():
            pool.mark_received(i, [it["mid"] for it in items])
            received[i].extend(items)
            tok = sum(it["tokens"] for it in items)
            comm_tokens += tok
            received_tokens[i] += tok

        for i in range(n_agents):
            held_recv = received[i][-8:] if received[i] else None
            proposal = _parse_proposal(
                _chat(
                    sys_prompts[i],
                    prompts.propose_user(question, own[i], held_recv),
                    _seed(task_id, i, rnd, seed_key, "propose"),
                    max_tokens=300,
                    task_id=task_id,
                    tag={"task_id": task_id, "arm": arm, "agent": i,
                         "round": rnd, "stage": "propose"},
                )
            )
            sketches[i] = proposal
            q1 = _clean_query(proposal["query"])
            docs = retriever.search(q1, top_k=top_k, exclude_docids=seen[i])
            docids = [d["docid"] for d in docs]
            for d in docs:
                title2doc.setdefault(d["title"], d["docid"])
            seen[i].update(docids)
            finding = _parse_finding(
                _chat(
                    sys_prompts[i],
                    prompts.finding_user(question, q1, docs),
                    _seed(task_id, i, rnd, seed_key, "finding"),
                    max_tokens=380,
                    task_id=task_id,
                    tag={"task_id": task_id, "arm": arm, "agent": i,
                         "round": rnd, "stage": "finding"},
                )
            )
            own[i].append({"round": rnd, "q1": q1, "docs": docs})
            publish(i, rnd, finding, docids, docs=docs)
            supported = [
                it["mid"] for it in pool.items
                if pool.holders[it["mid"]].get(i) == 1
            ]
            events.append(
                {
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
                            if pool.holders[it["mid"]].get(i) == 2
                        ),
                        "supported_cluster_ids": supported[:20],
                    },
                }
            )

    # ---- decide: own docs + received claims compete in the fair window ----
    finals = []
    for i in range(n_agents):
        evidence = [
            {**d, "round": r["round"]} for r in own[i] for d in r["docs"]
        ]
        for it in received[i]:
            evidence.append(
                {
                    "docid": f"mem:{it['mid']}",
                    "title": it["source_title"] or "team evidence",
                    "excerpt": it["claim"],
                    "round": it["round_created"],
                }
            )
        ans_raw = _chat(
            sys_prompts[i],
            prompts.answer_user(question, evidence),
            _seed(task_id, i, 99, seed_key, "answer"),
            max_tokens=120,
            task_id=task_id,
            tag={"task_id": task_id, "arm": arm, "agent": i,
                 "round": rounds + 1, "stage": "answer"},
        )
        ans = prompts.parse_json_field(ans_raw, "answer") or ans_raw.strip()[:80]
        em, f1 = qa_eval.score(
            ans, task.get("answer", ""), task.get("answer_aliases", [])
        )
        finals.append({"agent": i, "answer": ans, "em": em, "f1": round(f1, 4)})

    group_answer = _plurality_vote(finals, pool)
    g_em, g_f1 = qa_eval.score(
        group_answer, task.get("answer", ""), task.get("answer_aliases", [])
    )

    team_seen = set().union(*seen)
    return {
        "task_id": task_id,
        "arm": arm,
        "question": question,
        "answer": task.get("answer", ""),
        "gold_docids": sorted(gold),
        "n_agents": n_agents,
        "rounds": rounds,
        "events": events,
        "alloc_log": alloc_log,
        "received_sources": {
            str(i): sorted(
                {it["source_docid"] for it in received[i]
                 if it.get("source_docid")}
            )
            for i in range(n_agents)
        },
        "finals": finals,
        "group": {
            "answer": group_answer, "em": g_em, "f1": round(g_f1, 4),
            "override": bool(g_em == 0 and any(f["em"] for f in finals)),
        },
        "memory_pool": pool.export(),
        "pool_stats": pool.stats(),
        "comm_tokens": comm_tokens,
        "team_gold_coverage": len(team_seen & gold) / max(1, len(gold)),
        "wall_seconds": round(time.time() - t0, 2),
    }
