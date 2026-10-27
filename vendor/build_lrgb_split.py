"""W4: build and ship a corrected split for LRGB Peptides-func.

The case record is unambiguous that measuring a problem changes nothing and shipping a replacement
does. So this produces the artifact: a drop-in split file with the same shape as LRGB's own, built
so that no test sequence has a close relative in training.

Method, which is the one LRGB's split does not use:
  transitive single-linkage over ALL pairs at Levenshtein ratio 0.80, via union-find, then whole
  clusters assigned to train/val/test. Representative-based clustering breaks transitive chains
  (A~B, B~C, A!~C) and that is exactly how related sequences end up on opposite sides.

Two controls, both in the same run, because a corrected file that is not measured against the
thing it replaces is just an assertion:
  1. the ORIGINAL split re-measured here must still read 65.29%, proving the instrument is the
     same one and nothing drifted;
  2. the REBUILT split must read 0.0000.

Proportions are held at LRGB's own 10,873 / 2,331 / 2,331 so the file is a drop-in replacement and
nobody has to change their loader beyond the path.

CPU is correct: the all-pairs stage is rapidfuzz's C cdist over 12,569 unique short peptides with
a score cutoff, which prunes almost everything, and union-find is pointer chasing.
"""
import csv, gzip, hashlib, io, json, os, pickle, sys
import numpy as np
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "corrected_splits")
os.makedirs(OUT, exist_ok=True)
THRESH = 80.0
SEED = 20261017


def load_lrgb():
    raw = open(os.path.join(CACHE, "lrgb_peptides.csv.gz"), "rb").read()
    dmd5 = hashlib.md5(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode("utf-8", "replace"))))
    sraw = open(os.path.join(CACHE, "lrgb_splits.pickle"), "rb").read()
    smd5 = hashlib.md5(sraw).hexdigest()
    return rows, pickle.loads(sraw), dmd5, smd5


def twin_rate(train_seqs, test_seqs, block=256):
    tr = list(dict.fromkeys(train_seqs))
    hits = 0
    for i in range(0, len(test_seqs), block):
        sim = process.cdist(test_seqs[i:i + block], tr, scorer=fuzz.ratio,
                            score_cutoff=THRESH, workers=-1)
        hits += int((sim >= THRESH).any(axis=1).sum())
    return hits, len(test_seqs)


def union_find_clusters(seqs, block=256):
    """Transitive single linkage at 0.80. This is the step LRGB's split does not do."""
    uniq = list(dict.fromkeys(seqs))
    idx = {s: i for i, s in enumerate(uniq)}
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n = len(uniq)
    for i0 in range(0, n, block):
        sim = process.cdist(uniq[i0:i0 + block], uniq, scorer=fuzz.ratio,
                            score_cutoff=THRESH, workers=-1)
        ii, jj = np.nonzero(sim >= THRESH)
        for a, b in zip(ii, jj):
            ga, gb = i0 + int(a), int(b)
            if ga != gb:
                union(ga, gb)
        if (i0 // block) % 10 == 0:
            print(f"  clustering {min(i0 + block, n)}/{n}", flush=True)
    roots = {}
    for s in uniq:
        roots.setdefault(find(idx[s]), []).append(s)
    return roots


def main():
    rows, orig, dmd5, smd5 = load_lrgb()
    seqs = [r["peptide_seq"] for r in rows]
    print(f"# LRGB data md5={dmd5}  split md5={smd5}  rows={len(rows)}", flush=True)

    # ---- control 1: the original split, re-measured by this same instrument -----------
    o_tr = [seqs[i] for i in np.asarray(orig["train"]).ravel()]
    o_te = [seqs[i] for i in np.asarray(orig["test"]).ravel()]
    h, n = twin_rate(o_tr, o_te)
    orig_rate = round(100 * h / n, 2)
    print(f"CONTROL original split: twin>=0.80 = {h}/{n} = {orig_rate}%  "
          f"(published 65.29%)", flush=True)

    # ---- build: whole clusters to folds ---------------------------------------------
    print("# clustering all unique sequences, transitive single linkage at 0.80", flush=True)
    clusters = union_find_clusters(seqs)
    print(f"# {len(clusters)} clusters over {len(set(seqs))} unique sequences", flush=True)

    # assign clusters largest-first into the bucket furthest below quota: keeps the big
    # families from landing entirely in one split and preserves LRGB's proportions.
    seq_rows = {}
    for i, s in enumerate(seqs):
        seq_rows.setdefault(s, []).append(i)
    quota = {"train": len(np.asarray(orig["train"]).ravel()),
             "val": len(np.asarray(orig["val"]).ravel()),
             "test": len(np.asarray(orig["test"]).ravel())}
    order = sorted(clusters.values(), key=len, reverse=True)
    rng = np.random.default_rng(SEED)
    assign = {k: [] for k in quota}
    for cl in order:
        members = [i for s in cl for i in seq_rows[s]]
        deficit = {k: quota[k] - len(assign[k]) for k in quota}
        best = max(deficit, key=lambda k: deficit[k])
        assign[best].extend(members)
    for k in assign:
        rng.shuffle(assign[k])
    print("# sizes: " + ", ".join(f"{k} {len(v)}" for k, v in assign.items())
          + f"  (LRGB: {quota})", flush=True)

    # ---- gate: the rebuilt split must read 0.0000 -----------------------------------
    n_tr = [seqs[i] for i in assign["train"]]
    n_te = [seqs[i] for i in assign["test"]]
    h2, n2 = twin_rate(n_tr, n_te)
    new_rate = round(100 * h2 / n2, 4)
    print(f"GATE rebuilt split: twin>=0.80 = {h2}/{n2} = {new_rate}%", flush=True)

    trset = set(n_tr)
    exact = sum(1 for s in n_te if s in trset)
    print(f"      exact overlap = {exact}", flush=True)

    out = {k: sorted(int(i) for i in v) for k, v in assign.items()}
    path = os.path.join(OUT, "lrgb_peptides_func_singlelinkage080.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"method": "transitive single linkage (union-find) at Levenshtein ratio 0.80, "
                             "whole clusters assigned to folds",
                   "source_data_md5": dmd5, "source_split_md5": smd5,
                   "seed": SEED, "threshold": 0.80,
                   "control_original_twin_pct": orig_rate,
                   "rebuilt_twin_pct": new_rate,
                   "rebuilt_exact_overlap": exact,
                   "n_clusters": len(clusters),
                   "splits": out}, f, indent=1)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"# wrote {path}  md5={md5}", flush=True)
    print("RESULT " + json.dumps({"control_original_pct": orig_rate,
                                  "rebuilt_pct": new_rate,
                                  "rebuilt_exact": exact,
                                  "clusters": len(clusters),
                                  "sizes": {k: len(v) for k, v in assign.items()},
                                  "file_md5": md5}), flush=True)


if __name__ == "__main__":
    main()
