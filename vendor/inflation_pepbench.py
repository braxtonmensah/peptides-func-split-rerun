"""W1: turn a contamination rate into an inflation, using PepBenchmark's own shipped splits.

Pre-registered in PREREG_INFLATION.md before any model was trained. Read that first; the gate and
the null are fixed there and are not negotiable after seeing the numbers.

The design exists because PepBenchmark ships four split strategies over the IDENTICAL rows with
five seeds each. So the only thing that varies between arms is which rows are held out. No
retraining from scratch, no architecture confound, no data difference.

Two deliberate choices that bias AGAINST finding an effect, so that finding one means something:
  - a weak model (ridge on fixed descriptors). Less capacity to memorise a near-twin means less to
    gain from contamination. If the effect survives a weak model it is not a capacity artefact.
  - the penalty is AVERAGED over a fixed grid, never selected per fold. Selecting at small n is a
    defect this programme has already published about; committing it here to sharpen a result
    would be indefensible.

CPU is correct: ridge on a few thousand rows of 30-odd descriptors is milliseconds of linear
algebra, and the amino-acid composition loop is branchy Python over short strings.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "pepbench")

AA = "ACDEFGHIKLMNPQRSTVWY"
# Kyte-Doolittle hydropathy, the standard scale.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}

LAMS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
STRATEGIES = ["random_split", "mmseqs2_split", "hybrid_split", "cdhit_split"]
DATASETS = ["antimicrobial_E_coli_mic", "hemolytic_hc50"]


def featurise(seq):
    s = [c for c in seq.upper() if c in AA]
    n = max(1, len(s))
    comp = [s.count(a) / n for a in AA]
    kd = [KD[c] for c in s] or [0.0]
    net = sum(CHARGE.get(c, 0) for c in s)
    arom = sum(s.count(a) for a in "FWY") / n
    # hydrophobic moment at 100 degrees, the alpha-helix periodicity
    ang = np.deg2rad(100.0) * np.arange(len(s))
    mu = float(np.hypot(np.sum(np.array(kd) * np.cos(ang)), np.sum(np.array(kd) * np.sin(ang)))) / n
    return comp + [len(s) / 50.0, net / 10.0, net / n, float(np.mean(kd)), arom, mu]


def rankdata(a):
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def ridge_avg(Xtr, ytr, Xte):
    """Averaged over the penalty grid. NOT selected: selection at small n is the defect."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    ym = ytr.mean()
    preds = []
    for lam in LAMS:
        A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        w = np.linalg.solve(A, Ztr.T @ (ytr - ym))
        preds.append(Zte @ w + ym)
    return np.mean(preds, axis=0)


def load(ds):
    seqs = [l.strip() for l in open(os.path.join(CACHE, f"{ds}__fasta.csv"),
                                    encoding="utf-8").read().replace("\r", "").split("\n")[1:] if l.strip()]
    lp = os.path.join(CACHE, f"{ds}__label.csv")
    if not os.path.exists(lp):
        import urllib.request
        url = (f"https://huggingface.co/datasets/jiahuizhang/PepBenchData/resolve/main/"
               f"PepBenchData-50/{ds}/label.csv")
        req = urllib.request.Request(url, headers={"User-Agent": "silicyte-verify/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(lp, "wb") as f:
            f.write(r.read())
    lines = open(lp, encoding="utf-8").read().replace("\r", "").split("\n")
    assert lines[0].strip() == "feature", f"unexpected label header {lines[0]!r}"
    y = np.array([float(v) for v in lines[1:] if v.strip()])
    assert len(y) == len(seqs), f"{ds}: {len(seqs)} seqs vs {len(y)} labels"
    return seqs, y


def main():
    out = {}
    for ds in DATASETS:
        seqs, y = load(ds)
        X = np.array([featurise(s) for s in seqs])
        print(f"\n# {ds}: {len(seqs)} rows, {X.shape[1]} features", flush=True)
        out[ds] = {}
        for strat in STRATEGIES:
            p = os.path.join(CACHE, f"{ds}__{strat}.json")
            if not os.path.exists(p):
                print(f"  {strat}: missing"); continue
            obj = json.load(open(p, encoding="utf-8"))
            per_seed = []
            for seed in sorted(obj):
                tr = np.asarray(obj[seed]["train"], int)
                te = np.asarray(obj[seed]["test"], int)
                pred = ridge_avg(X[tr], y[tr], X[te])
                per_seed.append(spearman(pred, y[te]))
            out[ds][strat] = per_seed
            print(f"  {strat:<16} mean {np.mean(per_seed):+.4f}  "
                  f"sd {np.std(per_seed):.4f}  seeds {['%+.3f' % v for v in per_seed]}", flush=True)

        if "random_split" in out[ds] and "cdhit_split" in out[ds]:
            r = np.array(out[ds]["random_split"]); c = np.array(out[ds]["cdhit_split"])
            n = min(len(r), len(c))
            d = r[:n] - c[:n]
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            print(f"  >> INFLATION random - cdhit = {d.mean():+.4f} "
                  f"(per-seed {['%+.3f' % v for v in d]}, SE {se:.4f}, "
                  f"same sign {int((np.sign(d) == np.sign(d.mean())).sum())}/{n})", flush=True)
            h = out[ds].get("hybrid_split")
            if h:
                hh = np.array(h)[:n]
                dh = r[:n] - hh
                print(f"  >> random - hybrid(default) = {dh.mean():+.4f} "
                      f"(per-seed {['%+.3f' % v for v in dh]})", flush=True)
    print("\nRESULT " + json.dumps({k: {s: [round(v, 4) for v in vs] for s, vs in d.items()}
                                    for k, d in out.items()}), flush=True)


if __name__ == "__main__":
    main()
