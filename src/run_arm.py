"""One matrix cell: one model on one split, plus contrast C on the shipped split.

  python src/run_arm.py --model M1 --split r0
  python src/run_arm.py --model M1 --split shipped --mode shuffle       # must-fail control
  python src/run_arm.py --model M1 --split r0 --mode determinism        # run twice, compare

M1  count ECFP (scikit-fingerprints) + one LightGBM per label, arXiv 2501.17901's code; fit on train+val.
M2  26 fixed descriptors + ridge averaged over a fixed penalty grid, vendored unchanged; fit on train.
Writes out/<model>_<split>_<mode>.json.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, RESAMPLE_SEEDS, VENDOR, write_json  # noqa: E402
from data import load_lrgb, load_splits  # noqa: E402

sys.path.insert(0, VENDOR)


def fit_rows(model, s):
    return s["train"] + s["val"] if model == "M1" else list(s["train"])


def features(model, d):
    if model == "M1":
        from skfp.fingerprints import ECFPFingerprint
        from skfp.preprocessing import MolFromSmilesTransformer
        mols = MolFromSmilesTransformer(n_jobs=-1).transform(d["smiles"])
        return np.asarray(ECFPFingerprint(count=True, n_jobs=-1, sparse=False).transform(mols))
    from inflation_pepbench import featurise   # vendored, unchanged
    return np.array([featurise(s) for s in d["seqs"]])


def fit_predict(model, Xtr, Ytr, Xte):
    """Returns an (n_test, n_labels) score matrix."""
    if model == "M1":
        from lightgbm import LGBMClassifier
        from sklearn.multioutput import MultiOutputClassifier
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        clf = MultiOutputClassifier(LGBMClassifier(
            n_estimators=500, is_unbalance=True, importance_type="gain", random_state=0, verbose=-1,
            # compute-only options for bit-identical reruns; no modelling choice changes
            n_jobs=4, deterministic=True, force_col_wise=True))
        clf.fit(Xtr, Ytr.astype(int))
        return np.column_stack([p[:, 1] for p in clf.predict_proba(Xte)])
    from inflation_pepbench import ridge_avg   # vendored, unchanged
    return np.column_stack([ridge_avg(Xtr, Ytr[:, k], Xte) for k in range(Ytr.shape[1])])


def per_label_ap(Yte, S, labels=None):
    labels = range(Yte.shape[1]) if labels is None else labels
    out = {}
    for k in labels:
        if Yte[:, k].sum() > 0:
            out[int(k)] = float(average_precision_score(Yte[:, k], S[:, k]))
    return out


def macro(ap):
    return float(np.mean(list(ap.values())))


def run_once(model, d, X, s, shuffle=False):
    tr, te = fit_rows(model, s), s["test"]
    Ytr = d["Y"][tr]
    if shuffle:
        Ytr = Ytr[np.random.default_rng(0).permutation(len(Ytr))]
    S = fit_predict(model, X[tr], Ytr, X[te])
    return S, d["Y"][te]


def contrast_c(model, Yte, S, masks):
    """Same training, shipped test set split into twinned and untwinned rows."""
    m = np.asarray(masks["vs_trainval" if model == "M1" else "vs_train"], bool)
    tw, un = np.nonzero(m)[0], np.nonzero(~m)[0]
    labels = [k for k in range(Yte.shape[1]) if Yte[tw, k].sum() > 0 and Yte[un, k].sum() > 0]
    full_tw, full_un = per_label_ap(Yte[tw], S[tw], labels), per_label_ap(Yte[un], S[un], labels)
    res = []
    n = min(len(tw), len(un))
    ap_un = macro(per_label_ap(Yte[un[:n]] if len(un) > n else Yte[un], S[un[:n]] if len(un) > n else S[un], labels))
    for sd in RESAMPLE_SEEDS:
        sub = np.sort(np.random.default_rng(sd).choice(tw, size=n, replace=False)) if len(tw) > n else tw
        ap_tw = macro(per_label_ap(Yte[sub], S[sub], labels))
        res.append({"seed": sd, "n_twinned_subset": int(len(sub)), "ap_twinned": round(ap_tw, 6),
                    "ap_untwinned": round(ap_un, 6), "diff": round(ap_tw - ap_un, 6)})
    return {"twin_reference": "train+val" if model == "M1" else "train",
            "n_twinned": int(len(tw)), "n_untwinned": int(len(un)),
            "labels_used": labels, "labels_dropped": [k for k in range(Yte.shape[1]) if k not in labels],
            "ap_twinned_all_rows": round(macro(full_tw), 6), "ap_untwinned_all_rows": round(macro(full_un), 6),
            "positive_rate_twinned": [round(float(x), 4) for x in Yte[tw].mean(0)],
            "positive_rate_untwinned": [round(float(x), 4) for x in Yte[un].mean(0)],
            "per_resample": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["M1", "M2"], required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--mode", choices=["normal", "shuffle", "determinism"], default="normal")
    a = ap.parse_args()
    t0 = time.time()
    d = load_lrgb()
    sp = load_splits()
    s = sp["splits"][a.split]
    X = features(a.model, d)
    print(f"{a.model} features {X.shape} in {time.time() - t0:.0f}s", flush=True)

    out = {"model": a.model, "split": a.split, "mode": a.mode,
           "fit_rows": len(fit_rows(a.model, s)), "test_rows": len(s["test"]), "n_features": int(X.shape[1])}
    import importlib.metadata as md
    out["versions"] = {p: md.version(p) for p in ("scikit-fingerprints", "lightgbm", "scikit-learn",
                                                 "rdkit", "numpy", "rapidfuzz")}
    S, Yte = run_once(a.model, d, X, s, shuffle=(a.mode == "shuffle"))
    apl = per_label_ap(Yte, S)
    out["ap_per_label"] = {str(k): round(v, 6) for k, v in apl.items()}
    out["macro_ap"] = round(macro(apl), 6)
    out["test_positive_rate"] = [round(float(x), 6) for x in Yte.mean(0)]
    print(f"{a.model} {a.split} {a.mode}: macro AP {out['macro_ap']:.6f} over {len(apl)} labels", flush=True)

    if a.mode == "determinism":
        S2, _ = run_once(a.model, d, X, s)
        out["macro_ap_second_run"] = round(macro(per_label_ap(Yte, S2)), 6)
        out["identical_to_6dp"] = out["macro_ap"] == out["macro_ap_second_run"]
        print(f"  second run {out['macro_ap_second_run']:.6f}  identical {out['identical_to_6dp']}", flush=True)
    if a.mode == "normal" and a.split == "shipped":
        out["contrast_c"] = contrast_c(a.model, Yte, S, sp["twin_masks"]["shipped"])
        c = out["contrast_c"]
        print(f"  contrast C: twinned {c['n_twinned']} untwinned {c['n_untwinned']} "
              f"diffs {[r['diff'] for r in c['per_resample']]}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, f"{a.model}_{a.split}_{a.mode}.json"), out)


if __name__ == "__main__":
    main()
