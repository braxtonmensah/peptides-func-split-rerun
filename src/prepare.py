"""prepare: data integrity, the leakage control, and the five corrected resamples.

Gates (any failure exits 1):
  - LRGB data and split MD5s equal the recorded values (in data.load_lrgb)
  - the scikit-fingerprints Hugging Face copy has the same SMILES, labels and split indices
  - shipped split sizes 10,873 / 2,331 / 2,331, and the corrected file's sizes are the same
  - shipped split twin_at_080 = 65.29% with 1,522 twinned test rows
  - corrected file and each of the 5 resamples = 0.00%
Writes work/splits.json (indices only, no LRGB data) and out/controls_data.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CORRECTED_MD5, CORRECTED_URL, OUT, RESAMPLE_SEEDS, VENDOR, WORK,  # noqa: E402
                    fetch, md5, write_json)
from data import load_lrgb  # noqa: E402

from rapidfuzz import fuzz, process  # noqa: E402

THRESH = 80.0
SIZES = {"train": 10873, "val": 2331, "test": 2331}


def twin_mask(train_seqs, test_seqs, block=256):
    """Per test row: has a training sequence at Levenshtein ratio >= 0.80. Same call as the
    vendored build_lrgb_split.twin_rate, kept per row so contrast C can use it."""
    tr = list(dict.fromkeys(train_seqs))
    out = np.zeros(len(test_seqs), bool)
    for i in range(0, len(test_seqs), block):
        sim = process.cdist(test_seqs[i:i + block], tr, scorer=fuzz.ratio,
                            score_cutoff=THRESH, workers=-1)
        out[i:i + block] = (sim >= THRESH).any(axis=1)
    return out


def main():
    t0 = time.time()
    fails, rep = [], {}
    d = load_lrgb()
    seqs, smiles, Y, shipped = d["seqs"], d["smiles"], d["Y"], d["shipped"]
    rep["md5"] = d["md5"]
    rep["rows"], rep["labels"] = len(seqs), int(Y.shape[1])
    print(f"LRGB rows {len(seqs)}  labels {Y.shape[1]}  md5 {d['md5']}  (match)", flush=True)

    # ---- cross-check against the scikit-fingerprints HF copy (the paper's loader) ----------
    try:
        from skfp.datasets.lrgb import load_lrgb_mol_splits, load_peptides_func
        hf_smiles, hf_y = load_peptides_func(mol_type="SMILES", data_dir=os.path.join(WORK, "skfp"))
        hf_tr, hf_va, hf_te = load_lrgb_mol_splits("Peptides-func", data_dir=os.path.join(WORK, "skfp"))
        same_smiles = list(hf_smiles) == smiles
        same_y = np.asarray(hf_y).shape == Y.shape and bool(np.array_equal(np.asarray(hf_y, float), Y))
        same_split = (sorted(hf_tr) == sorted(shipped["train"]) and sorted(hf_va) == sorted(shipped["val"])
                      and sorted(hf_te) == sorted(shipped["test"]))
        rep["skfp_crosscheck"] = {"same_smiles_same_order": same_smiles, "same_labels": same_y,
                                  "same_split_indices": same_split, "hf_rows": len(hf_smiles)}
        if not (same_smiles and same_y and same_split):
            fails.append(f"skfp HF copy differs from LRGB's files: {rep['skfp_crosscheck']}")
    except Exception as e:
        rep["skfp_crosscheck"] = {"error": repr(e)[:300]}
        fails.append(f"skfp cross-check could not run: {e!r}"[:300])
    print(f"skfp cross-check {rep['skfp_crosscheck']}", flush=True)

    # ---- sizes ------------------------------------------------------------------------------
    rep["shipped_sizes"] = {k: len(v) for k, v in shipped.items()}
    if rep["shipped_sizes"] != SIZES:
        fails.append(f"shipped sizes {rep['shipped_sizes']} != {SIZES}")
    cpath = fetch(CORRECTED_URL, os.path.join(WORK, "lrgb_peptides_func_singlelinkage080.json"))
    rep["corrected_file_md5"] = md5(cpath)
    if rep["corrected_file_md5"] != CORRECTED_MD5:
        fails.append(f"corrected file md5 {rep['corrected_file_md5']} != {CORRECTED_MD5}")
    cf = json.load(open(cpath, encoding="utf-8"))
    if cf.get("source_data_md5") != d["md5"]["data"] or cf.get("source_split_md5") != d["md5"]["split"]:
        fails.append("corrected file header MD5s do not match the LRGB files")
    corrected_file = {k: [int(i) for i in cf["splits"][k]] for k in ("train", "val", "test")}
    rep["corrected_file_sizes"] = {k: len(v) for k, v in corrected_file.items()}
    if rep["corrected_file_sizes"] != SIZES:
        fails.append(f"corrected file sizes {rep['corrected_file_sizes']} != {SIZES}")
    print(f"sizes shipped {rep['shipped_sizes']}  corrected file {rep['corrected_file_sizes']}", flush=True)

    # ---- the five resamples, vendored code, seeds 0..4 -------------------------------------
    sys.path.insert(0, VENDOR)
    from build_lrgb_split import union_find_clusters
    from inflation_lrgb import build_corrected
    print("# clustering all unique sequences, transitive single linkage at 0.80", flush=True)
    clusters = union_find_clusters(seqs)
    rep["n_clusters"] = len(clusters)
    print(f"# {len(clusters)} clusters (record: 6,339)", flush=True)
    quota = dict(SIZES)
    splits = {"shipped": shipped, "corrected_file": corrected_file}
    for sd in RESAMPLE_SEEDS:
        a = build_corrected(seqs, clusters, quota, sd)
        splits[f"r{sd}"] = {k: sorted(int(i) for i in a[k]) for k in ("train", "val", "test")}

    # ---- leakage control, every split, same job ------------------------------------------
    rep["twin"] = {}
    masks = {}
    for name, s in splits.items():
        allidx = s["train"] + s["val"] + s["test"]
        if len(allidx) != len(seqs) or len(set(allidx)) != len(seqs):
            fails.append(f"{name}: indices are not a partition of {len(seqs)} rows")
        te = [seqs[i] for i in s["test"]]
        m_tr = twin_mask([seqs[i] for i in s["train"]], te)
        m_tv = twin_mask([seqs[i] for i in s["train"] + s["val"]], te)
        masks[name] = {"vs_train": m_tr, "vs_trainval": m_tv}
        row = {"sizes": {k: len(v) for k, v in s.items()},
               "twin_at_080_pct": round(100 * m_tr.mean(), 2), "twinned_test_rows": int(m_tr.sum()),
               "twin_at_080_pct_vs_train_plus_val": round(100 * m_tv.mean(), 2),
               "twinned_test_rows_vs_train_plus_val": int(m_tv.sum())}
        rep["twin"][name] = row
        print(f"twin {name:<15} {row}", flush=True)
    if rep["twin"]["shipped"]["twin_at_080_pct"] != 65.29 or rep["twin"]["shipped"]["twinned_test_rows"] != 1522:
        fails.append(f"shipped twin control {rep['twin']['shipped']} != 65.29% / 1,522")
    for name in ["corrected_file"] + [f"r{s}" for s in RESAMPLE_SEEDS]:
        if rep["twin"][name]["twin_at_080_pct"] != 0.0 or rep["twin"][name]["twinned_test_rows"] != 0:
            fails.append(f"{name} twin control {rep['twin'][name]['twin_at_080_pct']}% != 0.00%")

    # cross-check against the record itself
    rec = json.load(open(os.path.join(VENDOR, "leakage-measurements.json"), encoding="utf-8"))
    lr = [m for m in rec["measurements"] if m.get("benchmark") == "LRGB Peptides-func"][0]
    rep["record"] = {k: lr[k] for k in ("twin_at_080_pct", "test_rows", "train_rows")}
    if (lr["twin_at_080_pct"] != rep["twin"]["shipped"]["twin_at_080_pct"] or lr["test_rows"] != 2331
            or lr["train_rows"] != 10873):
        fails.append(f"record {rep['record']} disagrees with this run")

    write_json(os.path.join(WORK, "splits.json"),
               {"splits": splits,
                "twin_masks": {n: {k: [int(x) for x in v] for k, v in m.items()} for n, m in masks.items()}})
    rep["fails"], rep["pass"] = fails, not fails
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, "controls_data.json"), rep)
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
