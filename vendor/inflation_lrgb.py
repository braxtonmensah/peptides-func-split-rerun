"""W6: the inflation number on the headline benchmark.

W1 asked whether PepBenchmark's contamination costs anything and got a null. This asks the same
question on LRGB Peptides-func, the benchmark the whole page leads with, now that W4 has produced
a corrected split to compare against.

The design is the cleanest available anywhere in this programme: the SAME 15,535 molecules, the
SAME labels, the SAME model, and two split files. One is LRGB's own, which carries 65.29%
contamination. One is the corrected one, which carries 0.0%. Nothing else differs.

PREREGISTERED HERE, before the run:
  H1  mean(original) - mean(corrected) >= +0.03 AP, with the same sign in at least 4 of 5
      resamples. That would be the first inflation figure on this benchmark.
  H0  below +0.01, or sign inconsistent. Then LRGB's contamination costs nothing measurable for
      this model, which is the same answer W1 gave on PepBenchmark, and two nulls in a row would
      make "contaminated but it costs less than you think" the honest headline of this programme.
  Between is inconclusive and gets reported as inconclusive.

Peptides-func is 10-way MULTI-LABEL, so the metric is average precision per label, macro-averaged,
which is what the benchmark itself reports. Not Spearman: using a different metric from the
benchmark would make the number unquotable against anything published.

Resampling: LRGB ships one split, so the five "seeds" are five independent rebuilds of the
corrected split with different RNG seeds, each compared against the same fixed original. That
isolates split-assignment noise, which is the thing that has to be controlled after W1 showed
seed spread swamping the effect.

Features are the same fixed descriptors as W1, deliberately: a weak model biases against finding
an effect, so an effect that survives it is not a capacity artefact.

CPU is correct: ridge over 26 descriptors on 15,535 rows is linear algebra in milliseconds; the
cost is the clustering, which is cached from W4.
"""
import csv, gzip, hashlib, io, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
sys.path.insert(0, HERE)
from inflation_pepbench import featurise, ridge_avg
from build_lrgb_split import union_find_clusters

SEEDS = [20261017, 11, 202, 3003, 44004]


def load():
    raw = open(os.path.join(CACHE, "lrgb_peptides.csv.gz"), "rb").read()
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode("utf-8", "replace"))))
    sraw = open(os.path.join(CACHE, "lrgb_splits.pickle"), "rb").read()
    return rows, pickle.loads(sraw), hashlib.md5(raw).hexdigest()


def parse_labels(rows):
    """`labels` is a stringified list of 10 binary targets."""
    Y = []
    for r in rows:
        v = [float(x) for x in r["labels"].strip("[] ").replace(",", " ").split()]
        Y.append(v)
    return np.array(Y)


def average_precision(y_true, score):
    """AP for one binary label, the standard rank-based definition."""
    if y_true.sum() == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / yt.sum())


def macro_ap(Ytr, Yte, Xtr, Xte):
    aps = []
    for k in range(Ytr.shape[1]):
        if Ytr[:, k].sum() == 0 or Yte[:, k].sum() == 0:
            continue
        pred = ridge_avg(Xtr, Ytr[:, k], Xte)
        aps.append(average_precision(Yte[:, k], pred))
    return float(np.nanmean(aps)), len(aps)


def build_corrected(seqs, clusters, quota, seed):
    seq_rows = {}
    for i, s in enumerate(seqs):
        seq_rows.setdefault(s, []).append(i)
    rng = np.random.default_rng(seed)
    order = sorted(clusters.values(), key=len, reverse=True)
    # jitter the order of equal-sized clusters so the five rebuilds genuinely differ
    order = sorted(order, key=lambda c: (-len(c), rng.random()))
    assign = {k: [] for k in quota}
    for cl in order:
        members = [i for s in cl for i in seq_rows[s]]
        best = max(quota, key=lambda k: quota[k] - len(assign[k]))
        assign[best].extend(members)
    return assign


def main():
    rows, orig, dmd5 = load()
    seqs = [r["peptide_seq"] for r in rows]
    Y = parse_labels(rows)
    X = np.array([featurise(s) for s in seqs])
    print(f"# LRGB md5={dmd5}  rows={len(rows)}  labels={Y.shape[1]}  feats={X.shape[1]}", flush=True)

    o_tr = np.asarray(orig["train"]).ravel()
    o_te = np.asarray(orig["test"]).ravel()
    quota = {"train": len(o_tr), "val": len(np.asarray(orig["val"]).ravel()), "test": len(o_te)}

    ap_o, nlab = macro_ap(Y[o_tr], Y[o_te], X[o_tr], X[o_te])
    print(f"ORIGINAL split (65.29% contaminated): macro-AP {ap_o:.4f} over {nlab} labels", flush=True)

    print("# clustering once, reused for all resamples", flush=True)
    clusters = union_find_clusters(seqs)
    print(f"# {len(clusters)} clusters", flush=True)

    aps = []
    for sd in SEEDS:
        a = build_corrected(seqs, clusters, quota, sd)
        tr, te = np.array(a["train"]), np.array(a["test"])
        ap_c, _ = macro_ap(Y[tr], Y[te], X[tr], X[te])
        aps.append(ap_c)
        print(f"  corrected seed {sd:<9} macro-AP {ap_c:.4f}  delta {ap_o - ap_c:+.4f}", flush=True)

    d = np.array([ap_o - a for a in aps])
    same = int((np.sign(d) == np.sign(d.mean())).sum())
    se = d.std(ddof=1) / np.sqrt(len(d))
    verdict = ("H1" if (d.mean() >= 0.03 and same >= 4)
               else "H0" if (d.mean() < 0.01 or same < 4) else "INCONCLUSIVE")
    print(f"\n>> INFLATION original - corrected = {d.mean():+.4f}  SE {se:.4f}  "
          f"same sign {same}/{len(d)}  => {verdict}", flush=True)
    print(f">> corrected spread sd {np.std(aps, ddof=1):.4f} vs effect {abs(d.mean()):.4f}", flush=True)
    print("RESULT " + json.dumps({"ap_original": round(ap_o, 4),
                                  "ap_corrected": [round(a, 4) for a in aps],
                                  "delta_mean": round(float(d.mean()), 4),
                                  "se": round(float(se), 4),
                                  "same_sign": same, "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
