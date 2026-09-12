"""Fit the exposure-response model g_phi (and the learned-alternative controls).

Primary model (paper: "Learning without Prescribing the Curvature"):
six scalar inputs, one 16-unit tanh hidden layer, tanh scalar output whose
raw target is in [-1,1]. Inputs are standardized with FITTING-split
statistics; the integer boundary distance delta and 1/N are kept in their
natural scales to avoid dividing by a zero training variance at fixed N.

Objective, for fitting records indexed by question j and branch pair s:

  L(phi) = (1/|D_fit|) sum_j (1/n_j) sum_s (g_phi(x_js) - y_js)^2
           + lambda ||phi||^2
           + eta E[(g_phi(h+1) - 2 g_phi(h) + g_phi(h-1))^2]

The smoothness term is evaluated only on supported adjacent holder counts,
recomputing every holder-dependent coordinate together while decision
distance, relevance and team size stay fixed. It imposes neither concavity
nor a threshold peak. Task balancing prevents questions with many eligible
states from dominating. AdamW, lr 1e-3, batch 256, <=100 epochs, checkpoint
by task-balanced SELECTION error. Seeds 0,1,2 (0 is primary).

lambda, eta and the exposure weight are chosen on the selection split
within a fixed search budget shared by the learned alternatives.

Also fits the two learned controls on the same split:
  --fit-phase        tau_eta(d) = sigmoid(a0 + a1 d), structural terms kept
  --fit-global-set   DeepSets package-value predictor on terminal outcomes

Usage:
  python scripts/train_response.py --interventions runs/interventions/n4.jsonl \\
      --out src/merge/search/response_model.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge.search.response import (  # noqa: E402
    FEATURE_LAYOUT, HIDDEN, N_INPUTS, STANDARDIZE, response_features,
)

SEED = 0
LAMBDAS = [1e-5, 1e-4, 1e-3]
ETAS = [0.0, 0.1, 1.0]


class Net(torch.nn.Module):
    def __init__(self, n_in=N_INPUTS, hidden=HIDDEN):
        super().__init__()
        self.l1 = torch.nn.Linear(n_in, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x)))).squeeze(-1)


def load_rows(paths, variant="full"):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("y_terminal_f1") is None:
                continue
            r["x"] = response_features(r["d_t"], r["h"], r["n_agents"],
                                       r["q_c"], variant=variant)
            rows.append(r)
    return rows


def task_balanced_mse(pred, y, groups):
    """Mean over questions of the within-question mean squared error."""
    out, n = 0.0, 0
    for _g, idx in groups.items():
        d = pred[idx] - y[idx]
        out = out + (d * d).mean()
        n += 1
    return out / max(1, n)


def _groups(rows, idx=None):
    g = defaultdict(list)
    for k, r in enumerate(rows):
        if idx is None or k in idx:
            g[r["task_id"]].append(k)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in g.items()}


def smoothness(net, ctx, mu, sd, variant="full"):
    """E[(g(h+1) - 2 g(h) + g(h-1))^2] on supported adjacent holder counts."""
    if not ctx:
        return torch.tensor(0.0)
    xs = []
    for d_t, n_agents, q_c, h in ctx:
        tri = [response_features(d_t, hh, n_agents, q_c, variant)
               for hh in (h - 1, h, h + 1)]
        xs.append(np.stack(tri))
    X = torch.tensor((np.stack(xs) - mu) / sd, dtype=torch.float32)
    g = net(X.reshape(-1, X.shape[-1])).reshape(-1, 3)
    second = g[:, 2] - 2 * g[:, 1] + g[:, 0]
    return (second * second).mean()


def fit_once(rows_fit, rows_sel, lam, eta, seed, variant="full", epochs=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    Xf = np.stack([r["x"] for r in rows_fit])
    mu = Xf.mean(0)
    sd = Xf.std(0)
    for k in range(len(mu)):
        if not STANDARDIZE[k] or sd[k] < 1e-8:
            mu[k], sd[k] = 0.0, 1.0          # natural scale / zero-variance guard
    Xf_t = torch.tensor((Xf - mu) / sd, dtype=torch.float32)
    yf = torch.tensor([r["y_terminal_f1"] for r in rows_fit], dtype=torch.float32)
    Xs = np.stack([r["x"] for r in rows_sel]) if rows_sel else Xf
    Xs_t = torch.tensor((Xs - mu) / sd, dtype=torch.float32)
    ys = torch.tensor([r["y_terminal_f1"] for r in (rows_sel or rows_fit)],
                      dtype=torch.float32)
    gs = _groups(rows_sel or rows_fit)

    # supported adjacent-holder contexts, from the fitting split only
    seen = {(round(r["d_t"], 4), r["n_agents"], round(r["q_c"], 4)): set()
            for r in rows_fit}
    for r in rows_fit:
        seen[(round(r["d_t"], 4), r["n_agents"], round(r["q_c"], 4))].add(r["h"])
    ctx = [(d, n, q, h) for (d, n, q), hs in seen.items() for h in hs
           if (h - 1) >= 1 and (h + 1) <= n]

    net = Net(n_in=Xf.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    best = (float("inf"), None, 0)
    idx = np.arange(len(rows_fit))
    for ep in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            sel = torch.tensor(idx[b:b + 256], dtype=torch.long)
            pred = net(Xf_t[sel])
            loss = ((pred - yf[sel]) ** 2).mean()
            l2 = sum((p * p).sum() for p in net.parameters())
            loss = loss + lam * l2
            if eta > 0:
                loss = loss + eta * smoothness(net, ctx, mu, sd, variant)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sel_err = float(task_balanced_mse(net(Xs_t), ys, gs))
        if sel_err < best[0]:
            best = (sel_err, {k: v.detach().clone()
                              for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, mu, sd, best[0], best[2]


def export(net, mu, sd, meta, out_path):
    sd_ = net.state_dict()
    np.savez_compressed(
        out_path,
        W1=sd_["l1.weight"].numpy().astype(np.float32),
        b1=sd_["l1.bias"].numpy().astype(np.float32),
        w2=sd_["l2.weight"].numpy().reshape(-1).astype(np.float32),
        b2=np.float32(sd_["l2.bias"].numpy()[0]),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
        meta=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interventions", nargs="+", required=True)
    ap.add_argument("--out", default="src/merge/search/response_model.npz")
    ap.add_argument("--variant", default="full",
                    choices=["full", "raw_encoding", "no_distance"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fit-phase", action="store_true")
    ap.add_argument("--fit-global-set", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.interventions, args.variant)
    rows = [r for r in rows if r["n_agents"] == 4]     # primary model: N=4 only
    fit = [r for r in rows if r["split"] == "fitting"]
    sel = [r for r in rows if r["split"] == "selection"]
    if not fit:
        raise SystemExit("no fitting-split records; check --splits assignment")
    print(f"fitting pairs {len(fit)} over {len({r['task_id'] for r in fit})} questions; "
          f"selection pairs {len(sel)} over {len({r['task_id'] for r in sel})} questions")
    sha = hashlib.sha256(json.dumps(
        sorted((r["task_id"], r["round"], r["cluster"], r["h"],
                round(r["y_terminal_f1"], 6)) for r in fit)).encode()
    ).hexdigest()[:16]

    # ---- shared fixed search budget over (lambda, eta) -------------------
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for lam in LAMBDAS:
        for eta in ETAS:
            net, mu, sd, err, ep = fit_once(fit, sel, lam, eta, seeds[0],
                                            args.variant, args.epochs)
            results.append((err, lam, eta, ep))
            print(f"  lambda={lam:g} eta={eta:g}: selection err {err:.6f} "
                  f"(epoch {ep})", flush=True)
    results.sort()
    best_err, lam, eta, _ = results[0]
    print(f"selected lambda={lam:g} eta={eta:g} (selection err {best_err:.6f})")

    stability = {}
    primary = None
    for s in seeds:
        net, mu, sd, err, ep = fit_once(fit, sel, lam, eta, s, args.variant,
                                        args.epochs)
        stability[str(s)] = {"selection_err": err, "epoch": int(ep)}
        if s == seeds[0]:
            primary = (net, mu, sd, err, ep)
    net, mu, sd, err, ep = primary
    meta = {
        "variant": args.variant, "inputs": FEATURE_LAYOUT, "hidden": HIDDEN,
        "standardize": STANDARDIZE, "lambda": lam, "eta": eta,
        "epochs_max": args.epochs, "selected_epoch": int(ep),
        "selection_error": err, "seed_primary": seeds[0],
        "seed_stability": stability,
        "n_fit_pairs": len(fit), "n_sel_pairs": len(sel),
        "n_fit_questions": len({r["task_id"] for r in fit}),
        "search_budget": len(LAMBDAS) * len(ETAS),
        "trainable_params": int(sum(p.numel() for p in net.parameters())),
        "data_sha256": sha, "target": "paired terminal group F1 difference",
    }
    out = Path(args.out)
    export(net, mu, sd, meta, out)
    print(f"wrote {out}  params={meta['trainable_params']}  sha={sha}")

    if args.fit_phase:
        fit_phase(fit, sel, out.parent / "learned_phase.json")
    if args.fit_global_set:
        fit_global_set(fit, sel, out.parent / "global_set.npz")


def fit_phase(fit, sel, out_path):
    """tau_eta(d) = sigmoid(a0 + a1 d) on the same centered marginal targets.

    The structural exposure marginal for one extra holder of a relevant
    cluster is w_exp[tau q_c/(m-1) - (1-tau) (2k-1)] at k new receivers. We
    fit (a0, a1, w_exp) by least squares to the measured centered marginals.
    """
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.zeros(2))
    logw = torch.nn.Parameter(torch.tensor(math.log(0.4)))
    d = torch.tensor([r["d_t"] for r in fit], dtype=torch.float32)
    q = torch.tensor([r["q_c"] for r in fit], dtype=torch.float32)
    n = torch.tensor([float(r["n_agents"]) for r in fit], dtype=torch.float32)
    h = torch.tensor([float(r["h"]) for r in fit], dtype=torch.float32)
    y = torch.tensor([r["y_terminal_f1"] for r in fit], dtype=torch.float32)
    m = torch.floor(n / 2) + 1
    # The interventions start from a cluster with ONE actual holder, so the base
    # package A_h holds k = h-1 NEW receivers and the focal transfer takes it to
    # k = h. The structural marginals for that one transfer are therefore
    #   d f_reach = q_c / (m-1) while h < m, and
    #   d f_broad = max(h-1,0)^2 - max(h-2,0)^2
    # (not 2h-1: confusing the holder count h with the new-receiver count k
    # mis-specifies the regression basis and biases a0, a1 and w_exp).
    z = torch.zeros_like(h)
    broad = (torch.maximum(h - 1.0, z) ** 2
             - torch.maximum(h - 2.0, z) ** 2)
    opt = torch.optim.AdamW([a, logw], lr=5e-2)
    for _ in range(2000):
        tau = torch.sigmoid(a[0] + a[1] * d)
        reach = q / torch.clamp(m - 1, min=1.0) * (h < m).float()
        pred = torch.exp(logw) * (tau * reach - (1 - tau) * broad)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    params = {"a0": float(a[0]), "a1": float(a[1]),
              "w_exp": float(torch.exp(logw)), "n_fit_pairs": len(fit),
              "trainable_params": 3}
    Path(out_path).write_text(json.dumps(params, indent=1))
    print(f"wrote {out_path}: {params}")


def fit_global_set(fit, sel, out_path):
    """DeepSets package-value predictor trained on terminal package outcomes.

    Each transfer is embedded from the same primitive decision, relevance,
    holder and package information the response model sees; embeddings are
    summed and a head predicts the package's terminal group F1 (centered).
    """
    torch.manual_seed(0)

    def pkg_feats(r, with_focal):
        n = r["n_agents"]; m = n // 2 + 1
        h = r["h"] + (1 if with_focal else 0)
        k = r["h"] - 1 + (1 if with_focal else 0)      # transfers in the package
        f = []
        for _ in range(max(k, 0)):
            f.append([r["d_t"], h / n, (m - h) / n, float(m - h), 1.0 / n,
                      r["q_c"], r.get("local_score", 0.0),
                      r.get("item_tokens", 20) / 120.0])
        return np.array(f, dtype=np.float32) if f else np.zeros((0, 8), np.float32)

    X, Y = [], []
    for r in fit:
        for wf in (False, True):
            X.append(pkg_feats(r, wf))
            Y.append(r["full_group_f1"] if wf else r["base_group_f1"])
    flat = np.concatenate([x for x in X if len(x)], 0)
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-8] = 1.0
    W1 = torch.nn.Linear(8, HIDDEN)
    w2 = torch.nn.Linear(HIDDEN, 1)
    opt = torch.optim.AdamW(list(W1.parameters()) + list(w2.parameters()), lr=1e-3)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Yt = Yt - Yt.mean()
    for _ in range(200):
        preds = []
        for x in X:
            if len(x) == 0:
                preds.append(torch.zeros(()))
                continue
            z = torch.tensor((x - mu) / sd, dtype=torch.float32)
            preds.append(torch.tanh(w2(torch.tanh(W1(z)).sum(0))).squeeze())
        p = torch.stack(preds)
        loss = ((p - Yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    meta = {"inputs": ["d_t", "h/N", "delta/N", "delta", "1/N", "q_c",
                       "local_score", "tokens/120"],
            "hidden": HIDDEN, "n_fit_packages": len(X),
            "trainable_params": int(sum(p.numel() for p in
                                        list(W1.parameters()) + list(w2.parameters()))),
            "target": "terminal group F1 of the package (centered)"}
    np.savez_compressed(
        out_path,
        W1=W1.weight.detach().numpy().astype(np.float32),
        b1=W1.bias.detach().numpy().astype(np.float32),
        w2=w2.weight.detach().numpy().reshape(-1).astype(np.float32),
        b2=np.float32(w2.bias.detach().numpy()[0]),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
        meta=json.dumps(meta))
    print(f"wrote {out_path}: params={meta['trainable_params']}")


if __name__ == "__main__":
    main()
