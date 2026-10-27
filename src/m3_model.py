"""M3: the tuned GCN of Toenshoff et al. (TMLR 2024), reimplemented in plain PyTorch Geometric.

Source config: github.com/toenshoff/LRGB @ c3500cb702dcc5bdbe02783b94f24eebb7fb52c9,
configs/LRGB-tuned/peptides-func-GCN.yaml, with graphgps/layer/gcn_conv_layer.py,
graphgps/encoder/kernel_pos_encoder.py (RWSE), graphgps/encoder/composed_encoders.py (Atom+RWSE),
graphgps/head/mlp_graph.py, graphgps/loss/multilabel_classification_loss.py,
graphgps/optimizer/extra_optimizers.py (cosine_with_warmup), graphgps/train/custom_train.py.
Deviations are listed in PREREG_M3.md.
"""
import math
import os
import random
import re
import json
import time

import numpy as np
import torch
import torch.nn as nn

CFG = {  # peptides-func-GCN.yaml
    "dim_inner": 235, "layers_mp": 6, "layers_post_mp": 3, "dropout": 0.1,
    "rwse_steps": 20, "rwse_dim": 28, "batch_size": 200, "base_lr": 1e-3, "weight_decay": 0.0,
    "num_warmup_epochs": 5, "clip_grad_norm_value": 1.0, "max_epoch_published": 250,
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def budget():
    text = open(os.path.join(ROOT, "PREREG_M3.md"), encoding="utf-8").read()
    m = re.search(r"```m3budget\s*\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit("PREREG_M3.md has no ```m3budget block")
    return json.loads(m.group(1))


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def rw_landing(edge_index, n, ksteps):
    """get_rw_landing_probs: P = D^-1 A (dense), diag(P^k) for k = 1..K, space_dim 0."""
    if edge_index.size(1) == 0:
        return torch.zeros(n, ksteps)
    A = torch.zeros(n, n, dtype=torch.float64)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(1)
    dinv = deg.pow(-1.0)
    dinv[torch.isinf(dinv)] = 0
    P = dinv[:, None] * A
    Pk, out = P.clone(), []
    for _ in range(ksteps):
        out.append(torch.diagonal(Pk))
        Pk = Pk @ P
    return torch.stack(out, 1).float()


def build_graphs(smiles, Y):
    from ogb.utils import smiles2graph
    from torch_geometric.data import Data
    graphs = []
    for i, s in enumerate(smiles):
        g = smiles2graph(s)
        ei = torch.from_numpy(g["edge_index"]).long()
        n = int(g["num_nodes"])
        graphs.append(Data(x=torch.from_numpy(g["node_feat"]).long(), edge_index=ei, num_nodes=n,
                           pestat_RWSE=rw_landing(ei, n, CFG["rwse_steps"]),
                           y=torch.tensor(Y[i], dtype=torch.float32).view(1, -1)))
    return graphs


class AtomEncoder(nn.Module):
    """torch_geometric.graphgym.models.encoder.AtomEncoder: sum of per-feature embeddings."""
    def __init__(self, emb_dim):
        super().__init__()
        from ogb.utils.features import get_atom_feature_dims
        self.embs = nn.ModuleList()
        for dim in get_atom_feature_dims():
            e = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(e.weight.data)
            self.embs.append(e)

    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embs))


class GCN(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        from torch_geometric.nn import GCNConv
        d, pe = CFG["dim_inner"], CFG["rwse_dim"]
        self.atom = AtomEncoder(d - pe)
        self.rw_norm = nn.BatchNorm1d(CFG["rwse_steps"])
        self.rw_lin = nn.Linear(CFG["rwse_steps"], pe)
        self.convs = nn.ModuleList([GCNConv(d, d, bias=True) for _ in range(CFG["layers_mp"])])
        self.act = nn.GELU()
        self.drop = nn.Dropout(CFG["dropout"])
        head = []
        for _ in range(CFG["layers_post_mp"] - 1):
            head += [nn.Dropout(CFG["dropout"]), nn.Linear(d, d), nn.GELU()]
        head += [nn.Dropout(CFG["dropout"]), nn.Linear(d, n_out)]
        self.head = nn.Sequential(*head)

    def forward(self, b):
        from torch_geometric.nn import global_mean_pool
        x = torch.cat([self.atom(b.x), self.rw_lin(self.rw_norm(b.pestat_RWSE))], 1)
        for conv in self.convs:
            x = x + self.drop(self.act(conv(x, b.edge_index)))
        return self.head(global_mean_pool(x, b.batch))


def cosine_with_warmup(opt, warmup, total):
    """transformers.get_cosine_schedule_with_warmup (num_cycles 0.5), stepped per epoch."""
    def f(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return torch.cat([torch.sigmoid(model(b)) for b in loader]).numpy()


def macro_ap(Y, S):
    """Macro AP over labels with at least one positive (the same rule as run_arm.per_label_ap)."""
    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(Y[:, k], S[:, k]) for k in range(Y.shape[1]) if Y[:, k].sum() > 0]
    return float(np.mean(aps))


def train(graphs, tr, va, te, seed, max_epoch, shuffle_labels=False, probe_epochs=None, log=print):
    """Returns a dict. With probe_epochs set: timing only, no metric is computed."""
    from torch_geometric.loader import DataLoader
    seed_all(seed)
    trg = [graphs[i] for i in tr]
    if shuffle_labels:
        perm = np.random.default_rng(0).permutation(len(trg))
        ys = [trg[j].y for j in perm]
        trg = [g.clone() for g in trg]
        for g, y in zip(trg, ys):
            g.y = y
    gen = torch.Generator().manual_seed(seed)
    tl = DataLoader(trg, batch_size=CFG["batch_size"], shuffle=True, generator=gen)
    vl = DataLoader([graphs[i] for i in va], batch_size=CFG["batch_size"])
    sl = DataLoader([graphs[i] for i in te], batch_size=CFG["batch_size"])
    Yva = np.vstack([graphs[i].y.numpy() for i in va])
    Yte = np.vstack([graphs[i].y.numpy() for i in te])
    model = GCN(Yte.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["base_lr"], weight_decay=CFG["weight_decay"])
    sch = cosine_with_warmup(opt, CFG["num_warmup_epochs"], max_epoch)
    lossf = nn.BCEWithLogitsLoss()
    n_epochs = probe_epochs or max_epoch
    hist, t_train, t_eval = [], [], []
    best = None
    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for b in tl:
            opt.zero_grad()
            out = model(b)
            lab = b.y == b.y
            loss = lossf(out[lab], b.y[lab])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_grad_norm_value"])
            opt.step()
            tot += float(loss); nb += 1
        sch.step()
        t_train.append(time.time() - t0)
        t1 = time.time()
        Sva, Ste = predict(model, vl), predict(model, sl)
        t_eval.append(time.time() - t1)
        if probe_epochs:
            log(f"probe epoch {ep}: train {t_train[-1]:.1f}s eval {t_eval[-1]:.1f}s")
            continue
        va_ap, te_ap = macro_ap(Yva, Sva), macro_ap(Yte, Ste)
        hist.append({"epoch": ep, "loss": round(tot / nb, 6), "val_ap": round(va_ap, 6), "test_ap": round(te_ap, 6)})
        if best is None or va_ap > best["val_ap"]:
            best = {"epoch": ep, "val_ap": va_ap, "test_ap": te_ap, "S_test": Ste}
        if ep % 10 == 0 or ep == n_epochs - 1:
            log(f"epoch {ep}: loss {tot / nb:.4f} val {va_ap:.4f} ({t_train[-1]:.0f}s + {t_eval[-1]:.0f}s)")
    res = {"epochs_run": n_epochs, "sec_train_epoch_mean": round(float(np.mean(t_train)), 2),
           "sec_eval_pass_mean": round(float(np.mean(t_eval)), 2),
           "n_params": int(sum(p.numel() for p in model.parameters()))}
    if probe_epochs:
        return res
    res.update({"best_epoch": best["epoch"], "val_ap_best": round(best["val_ap"], 6),
                "test_ap_at_best_val": round(best["test_ap"], 6), "history": hist,
                "S_test": best["S_test"], "Y_test": Yte})
    return res
