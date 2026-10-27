"""Load LRGB Peptides-func from LRGB's own files, MD5-checked. Imported by every job."""
import csv
import gzip
import io
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LRGB_CSV_MD5, LRGB_CSV_URL, LRGB_SPLIT_MD5, LRGB_SPLIT_URL,  # noqa: E402
                    VENDOR, WORK, fetch, md5)

sys.path.insert(0, VENDOR)


def load_lrgb():
    """Rows, labels and LRGB's shipped split. Fails unless both MD5s match the record."""
    csv_path = fetch(LRGB_CSV_URL, os.path.join(WORK, "peptide_multi_class_dataset.csv.gz"))
    spl_path = fetch(LRGB_SPLIT_URL, os.path.join(WORK, "splits_random_stratified_peptide.pickle"))
    got_csv, got_spl = md5(csv_path), md5(spl_path)
    if got_csv != LRGB_CSV_MD5 or got_spl != LRGB_SPLIT_MD5:
        raise SystemExit(f"MD5 mismatch: data {got_csv} (want {LRGB_CSV_MD5}), "
                         f"split {got_spl} (want {LRGB_SPLIT_MD5})")
    raw = open(csv_path, "rb").read()
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode("utf-8", "replace"))))
    from inflation_lrgb import parse_labels   # vendored, unchanged
    Y = parse_labels(rows)
    seqs = [r["peptide_seq"] for r in rows]
    smiles = [r["smiles"] for r in rows]
    with open(spl_path, "rb") as f:
        orig = pickle.load(f)
    shipped = {k: [int(i) for i in np.asarray(orig[k]).ravel()] for k in ("train", "val", "test")}
    return {"seqs": seqs, "smiles": smiles, "Y": Y, "shipped": shipped,
            "md5": {"data": got_csv, "split": got_spl}}


def load_splits():
    return json.load(open(os.path.join(WORK, "splits.json"), encoding="utf-8"))
