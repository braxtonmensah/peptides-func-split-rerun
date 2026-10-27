"""One M3 job: the tuned GCN (PREREG_M3.md) on one split with one seed.

  python src/run_m3.py --mode probe                          # timing only, computes no metric
  python src/run_m3.py --split r0 --seed 0 --mode normal
  python src/run_m3.py --split shipped --seed 0 --mode shuffle
  python src/run_m3.py --split r0 --seed 0 --mode determinism  # a second, separate r0 job
Writes out/M3_<split>_s<seed>_<mode>.json (probe: out/M3_probe.json, seconds only).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, write_json  # noqa: E402
from data import load_lrgb, load_splits  # noqa: E402
import m3_model  # noqa: E402

PROBE_EPOCHS = 3


def smoke(graphs, s):
    """Exercise the full scoring path (2 epochs, 300/100/100 rows) on RANDOM labels, so that a code
    bug cannot surface after hours of training. Random labels carry no information about any split;
    the scores are not printed or saved, only whether the path ran."""
    from run_arm import contrast_c, macro, per_label_ap
    rng = np.random.default_rng(123)
    idx = s["train"][:300] + s["val"][:100] + s["test"][:100]
    fake = []
    for i in idx:
        g = graphs[i].clone()
        g.y = (np.random.default_rng(i).random((1, 10)) < 0.2).astype(np.float32)
        import torch
        g.y = torch.from_numpy(g.y)
        fake.append(g)
    tr, va, te = list(range(300)), list(range(300, 400)), list(range(400, 500))
    r = m3_model.train(fake, tr, va, te, seed=0, max_epoch=2, log=lambda m: None)
    S, Yte = r["S_test"], r["Y_test"]
    apl = per_label_ap(Yte, S)
    assert abs(macro(apl) - r["test_ap_at_best_val"]) < 1e-6
    m = rng.random(100) < 0.6
    c = contrast_c("M3", Yte, S, {"vs_train": [int(x) for x in m], "vs_trainval": [int(x) for x in m]})
    assert len(c["per_resample"]) == 5
    m3_model.train(fake, tr, va, te, seed=0, max_epoch=1, shuffle_labels=True, log=lambda m: None)
    print("smoke: full scoring path ran on random labels (scores discarded)", flush=True)
    return "ran"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="shipped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["probe", "normal", "shuffle", "determinism"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    d = load_lrgb()
    graphs = m3_model.build_graphs(d["smiles"], d["Y"])
    t_setup = time.time() - t0
    print(f"M3 graphs {len(graphs)} built in {t_setup:.0f}s", flush=True)
    import importlib.metadata as md
    versions = {p: md.version(p) for p in ("torch", "torch_geometric", "ogb", "rdkit", "numpy", "scikit-learn")}

    if a.mode == "probe":
        s = d["shipped"]
        r = m3_model.train(graphs, s["train"], s["val"], s["test"], seed=0, max_epoch=250,
                           probe_epochs=PROBE_EPOCHS, log=lambda m: print(m, flush=True))
        r.update({"mode": "probe", "setup_seconds": round(t_setup, 1), "versions": versions,
                  "cpu_count": os.cpu_count(), "note": "timing only; no metric was computed"})
        per_epoch = r["sec_train_epoch_mean"] + r["sec_eval_pass_mean"]
        r["projected_minutes"] = {str(E): round((t_setup + E * per_epoch * 1.15) / 60, 1)
                                  for E in (250, 200, 150, 120, 100, 80, 60, 50)}
        print(f"projected minutes (x1.15): {r['projected_minutes']}", flush=True)
        r["smoke_code_path_on_random_labels"] = smoke(graphs, s)
        write_json(os.path.join(OUT, "M3_probe.json"), r)
        return

    E = m3_model.budget()["max_epoch"]
    if not isinstance(E, int) or E <= 0:
        raise SystemExit("PREREG_M3.md m3budget.max_epoch is not set; run the timing probe first")
    sp = load_splits()
    s = sp["splits"][a.split]
    r = m3_model.train(graphs, s["train"], s["val"], s["test"], seed=a.seed, max_epoch=E,
                       shuffle_labels=(a.mode == "shuffle"), log=lambda m: print(m, flush=True))
    S, Yte = r.pop("S_test"), r.pop("Y_test")
    if not np.array_equal(Yte, d["Y"][s["test"]]):
        raise SystemExit("test label order mismatch")
    from run_arm import contrast_c, macro, per_label_ap
    apl = per_label_ap(Yte, S)
    out = {"model": "M3", "split": a.split, "seed": a.seed, "mode": a.mode, "max_epoch": E,
           "fit_rows": len(s["train"]), "val_rows": len(s["val"]), "test_rows": len(s["test"]),
           "ap_per_label": {str(k): round(v, 6) for k, v in apl.items()},
           "macro_ap": round(macro(apl), 6),
           "test_positive_rate": [round(float(x), 6) for x in Yte.mean(0)],
           "versions": versions, "cpu_count": os.cpu_count(), "setup_seconds": round(t_setup, 1)}
    out.update(r)
    if abs(out["macro_ap"] - out["test_ap_at_best_val"]) > 1e-6:
        raise SystemExit("macro AP recomputation disagrees with the training loop")
    print(f"M3 {a.split} seed {a.seed} {a.mode}: test macro AP {out['macro_ap']:.6f} "
          f"at epoch {out['best_epoch']} (val {out['val_ap_best']:.6f})", flush=True)
    if a.mode == "normal" and a.split == "shipped" and a.seed == 0:
        out["contrast_c"] = contrast_c("M3", Yte, S, sp["twin_masks"]["shipped"])
        c = out["contrast_c"]
        print(f"  contrast C: twinned {c['n_twinned']} untwinned {c['n_untwinned']} "
              f"diffs {[x['diff'] for x in c['per_resample']]}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, f"M3_{a.split}_s{a.seed}_{a.mode}.json"), out)


if __name__ == "__main__":
    main()
