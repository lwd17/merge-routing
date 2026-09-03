"""Train the learned local value of Appendix A.3 (r_theta, v_theta).

Data discipline: DEVELOPMENT logs only (records.jsonl + transfer_log.jsonl of
development runs). Never train on locked evaluation records; development,
scorer-training, and evaluation examples must stay disjoint.

Specification (paper Eq. 10-11):
  - a frozen text encoder represents the evidence item, the requirement, the
    question, and the receiver's last query;
  - scalar state features: current load, holder count, retrieval round,
    source support, and whether the item introduces a new entity;
  - a two-layer MLP with hidden width 256 predicts r_theta through a
    softplus output and the transfer residual through 0.25 tanh(.);
  - training pairs are formed within the same receiver state; a transfer is
    preferred when the following search round recovers a new benchmark
    supporting document or the transfer closes an annotated requirement;
  - loss  L = -log sigma(s+ - s-) + 0.2 BCE(sigma(s), y)  with the binary
    productivity label y (the calibration term averages the two members).

The exported .npz stores float32 weights plus a meta record with the
encoder id, feature layout, residual scale, and a sha256 of the training
rows so the frozen scorer is auditable. Validation metrics are printed but
never written into the artifact.

Usage:
  python scripts/train_semantic_value.py --run runs/<dev_run> --epochs 8 \\
      --out src/merge/search/semantic_value.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.rq2_metrics import content_tokens  # noqa: E402
from merge.search.semantic_value import pair_features  # noqa: E402
from merge.search.welfare import q_sem  # noqa: E402

HORIZON = 4
N_AGENTS = 4
SEED = 20260902
RES_SCALE = 0.25
N_SCALARS = 5


def load_examples(run_dir: Path):
    """Join transfer_log rows with texts/state reconstructed from records."""
    records = {}
    for line in (run_dir / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["task_id"], r["arm"])
        pool = {it["mid"]: it for it in r["memory_pool"]}
        slots, queries, routed = {}, {}, defaultdict(list)
        for ev in r["events"]:
            slots[(ev["agent"], ev["round"])] = [
                s for s in (ev["sketch"].get("unresolved_slots") or [])[:6]
                if content_tokens(s)
            ]
            queries[(ev["agent"], ev["round"])] = ev.get("q1", "")
            for mid in (ev.get("routed_mids") or []):
                routed[ev["agent"]].append((ev["round"], mid))
        records[key] = (pool, slots, queries, routed,
                        r.get("question", ""))

    def held_entities(pool, routed, receiver, upto_round):
        ents = set()
        for it in pool.values():
            if (it.get("producer") == receiver
                    and it.get("round_created", 9) <= upto_round):
                ents |= {e.lower() for e in it.get("entities", [])}
        for rr, mid in routed.get(receiver, []):
            if rr <= upto_round and mid in pool:
                ents |= {e.lower() for e in pool[mid].get("entities", [])}
        return ents

    examples = []
    for line in (run_dir / "transfer_log.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("next_new_docs") is None:
            continue  # final round: no next-hop outcome
        key = (row["task_id"], row["arm"])
        if key not in records:
            continue
        pool, slots, queries, routed, question = records[key]
        it = pool.get(row["mid"])
        if it is None:
            continue
        i, rnd = row["receiver"], row["round"]
        reqs = slots.get((i, rnd - 1), [])
        if not reqs:
            continue
        toks = content_tokens(it["claim"] + " "
                              + " ".join(it.get("entities", [])))
        closed = set(reqs) - set(slots.get((i, rnd), []))
        closes_req = any(
            q_sem(toks, content_tokens(g)) >= 0.5 for g in closed
        )
        y = float((row.get("next_new_gold") or 0) > 0 or closes_req)
        held_ents = held_entities(pool, routed, i, rnd - 1)
        ents = {e.lower() for e in it.get("entities", [])}
        scal = [
            0.0,  # current load unavailable in this log; recorded as zero
            min(len(it.get("holders", {})), N_AGENTS) / N_AGENTS,
            rnd / HORIZON,
            float(it.get("q_support", 1.0)),
            1.0 if (ents - held_ents) else 0.0,
        ]
        examples.append({
            "group": (row["task_id"], row["arm"], rnd, i),
            "task": row["task_id"],
            "claim": it["claim"],
            "reqs": reqs,
            "anchor": queries.get((i, rnd - 1)) or question,
            "scalars": scal,
            "y": y,
        })
    return examples


class Head(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.W1 = torch.nn.Linear(dim, hidden)
        self.w2 = torch.nn.Linear(hidden, 1)

    def forward(self, phi):
        return self.w2(torch.tanh(self.W1(phi))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/gurc_pilot")
    ap.add_argument("--out", default="src/merge/search/semantic_value.npz")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    root = Path(__file__).resolve().parents[1]
    examples = load_examples(root / args.run)
    print(f"joined examples: {len(examples)} "
          f"(pos rate {np.mean([e['y'] for e in examples]):.3f})")

    data_sha = hashlib.sha256(json.dumps(
        [(e["group"], e["claim"][:40], e["y"]) for e in examples],
        sort_keys=True, default=list,
    ).encode()).hexdigest()[:16]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    texts = sorted({e["claim"] for e in examples}
                   | {g for e in examples for g in e["reqs"]}
                   | {e["anchor"] for e in examples})
    embs = enc.encode(texts, normalize_embeddings=True,
                      batch_size=256, show_progress_bar=False)
    E = {t: torch.tensor(v, dtype=torch.float32) for t, v in zip(texts, embs)}
    dim_pair = 4 * embs.shape[1] + N_SCALARS

    def feats(e):
        hx = E[e["claim"]]
        req_phi = torch.stack([
            torch.tensor(pair_features(hx.numpy(), E[g].numpy(),
                                       e["scalars"]), dtype=torch.float32)
            for g in e["reqs"]
        ])
        anc_phi = torch.tensor(
            pair_features(hx.numpy(), E[e["anchor"]].numpy(), e["scalars"]),
            dtype=torch.float32)
        return req_phi, anc_phi

    cache = [feats(e) for e in examples]

    tasks = sorted({e["task"] for e in examples})
    random.shuffle(tasks)
    val_tasks = set(tasks[: max(1, len(tasks) // 5)])
    groups = defaultdict(list)
    for idx, e in enumerate(examples):
        groups[e["group"]].append(idx)
    train_pairs, val_pairs = [], []
    for _g, idxs in groups.items():
        pos = [i for i in idxs if examples[i]["y"] > 0.5]
        neg = [i for i in idxs if examples[i]["y"] < 0.5]
        for a in pos:
            for b in neg:
                (val_pairs if examples[a]["task"] in val_tasks
                 else train_pairs).append((a, b))
    print(f"within-state pairs: train={len(train_pairs)} val={len(val_pairs)}")

    r_head = Head(dim_pair, args.hidden)
    v_head = Head(dim_pair, args.hidden)
    opt = torch.optim.Adam(
        list(r_head.parameters()) + list(v_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    bce = torch.nn.functional.binary_cross_entropy

    def score(idx):
        req_phi, anc_phi = cache[idx]
        r = torch.nn.functional.softplus(r_head(req_phi)).sum()
        v = RES_SCALE * torch.tanh(v_head(anc_phi))
        return r + v

    def pair_acc(pairs):
        with torch.no_grad():
            return (sum(float(score(a) > score(b)) for a, b in pairs)
                    / max(1, len(pairs)))

    best = (0.0, None)
    for ep in range(args.epochs):
        random.shuffle(train_pairs)
        total = 0.0
        for a, b in train_pairs:
            sa, sb = score(a), score(b)
            loss = -torch.nn.functional.logsigmoid(sa - sb)
            loss = loss + 0.2 * 0.5 * (
                bce(torch.sigmoid(sa),
                    torch.tensor(examples[a]["y"]))
                + bce(torch.sigmoid(sb),
                      torch.tensor(examples[b]["y"]))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        va = pair_acc(val_pairs)
        print(f"epoch {ep}: loss {total/max(1,len(train_pairs)):.4f}  "
              f"val pair-acc {va:.3f}")
        if va >= best[0]:
            best = (va, {
                "r": {k: p.detach().clone()
                      for k, p in r_head.state_dict().items()},
                "v": {k: p.detach().clone()
                      for k, p in v_head.state_dict().items()},
            })
    if best[1] is not None:
        r_head.load_state_dict(best[1]["r"])
        v_head.load_state_dict(best[1]["v"])
    print(f"selected checkpoint val pair-acc {best[0]:.3f}")

    def export(head):
        sd = head.state_dict()
        return {
            "W1": sd["W1.weight"].numpy().astype(np.float32),
            "b1": sd["W1.bias"].numpy().astype(np.float32),
            "w2": sd["w2.weight"].numpy().reshape(-1).astype(np.float32),
            "b2": np.float32(sd["w2.bias"].numpy()[0]),
        }

    meta = {
        "encoder": "all-MiniLM-L6-v2",
        "dim_pair": int(dim_pair),
        "scalars": N_SCALARS,
        "scalar_layout": ["load_frac", "holder_frac", "round_frac",
                          "source_support", "new_entity"],
        "res_scale": RES_SCALE,
        "hidden": args.hidden,
        "trained_on": str(args.run),
        "data_sha256": data_sha,
        "n_examples": len(examples),
        "n_train_pairs": len(train_pairs),
        "seed": SEED,
    }
    out_path = root / args.out
    r_w, v_w = export(r_head), export(v_head)
    np.savez_compressed(
        out_path,
        meta=json.dumps(meta),
        **{f"r_{k}": v for k, v in r_w.items()},
        **{f"v_{k}": v for k, v in v_w.items()},
    )
    print(f"wrote {out_path}  "
          f"sha256 {hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
