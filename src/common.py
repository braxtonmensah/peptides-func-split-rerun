"""Shared helpers: paths, pinned inputs, the prereg gate, the verdict rule."""
import hashlib
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor")
WORK = os.path.join(ROOT, "work")          # downloaded data and built splits (never committed)
OUT = os.path.join(ROOT, "out")            # per-arm JSON from the matrix
RESULTS = os.path.join(ROOT, "results")    # aggregated, published

LRGB_CSV_URL = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
LRGB_SPLIT_URL = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
LRGB_CSV_MD5 = "701eb743e899f4d793f0e13c8fa5a1b4"
LRGB_SPLIT_MD5 = "5a0114bdadc80b94fc7ae974f13ef061"

CORRECTED_URL = ("https://raw.githubusercontent.com/braxtonmensah/silicyte-site/"
                 "a147b98abe6e06978d8fdf89b16845f1ca4f0b6f/data/lrgb-corrected-v1/"
                 "lrgb_peptides_func_singlelinkage080.json")
CORRECTED_MD5 = "b5751ec51e80fa7fb5aba12016de81fb"

RESAMPLE_SEEDS = [0, 1, 2, 3, 4]
PAPER_AP = 0.7460          # arXiv 2501.17901, Table 1, count ECFP, Peptides-func AUPRC 74.60
PAPER_TOL = 0.010          # fixed in PREREG.md
SHUFFLE_TOL = 0.02         # fixed in PREREG.md
SHUFFLE_FLAG = 0.04


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "peptides-func-split-rerun/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def prereg_gate():
    """The gate as written in PREREG.md's ```gate block. Every verdict is computed from this."""
    text = open(os.path.join(ROOT, "PREREG.md"), encoding="utf-8").read()
    m = re.search(r"```gate\s*\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit("PREREG.md has no ```gate block")
    return json.loads(m.group(1))


def verdict(diffs, gate):
    """Exactly the rule in inflation_lrgb.py, with thresholds read from the prereg."""
    import numpy as np
    d = np.asarray(diffs, float)
    if len(d) != gate["n_resamples"]:
        raise SystemExit(f"expected {gate['n_resamples']} resamples, got {len(d)}")
    same = int((np.sign(d) == np.sign(d.mean())).sum())
    if d.mean() >= gate["h1_min_diff"] and same >= gate["h1_min_same_sign"]:
        v = "H1"
    elif d.mean() < gate["h0_below"] or same < gate["h1_min_same_sign"]:
        v = "H0"
    else:
        v = "INCONCLUSIVE"
    return v, same


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")
