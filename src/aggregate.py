"""aggregate: contrasts, controls, verdicts (computed from PREREG.md's gate), schema, badge, summary.

Reads out/*.json from the matrix and out/controls_data.json + out/prereg_check.json.
Writes results/summary.json, results/arms/*.json, results/badge.json and the job summary.
Exit 1 if any control fails or the output fails its schema.
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT, PAPER_AP, PAPER_TOL, RESAMPLE_SEEDS, RESULTS, ROOT,  # noqa: E402
                    SHUFFLE_FLAG, SHUFFLE_TOL, prereg_gate, verdict, write_json)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def stats(diffs):
    d = np.asarray(diffs, float)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("inf") * np.sign(d.mean())
    return float(d.mean()), sd, t


def contrast_a(model, gate, fails):
    shipped = load(f"{model}_shipped_normal.json")
    cors = [load(f"{model}_r{s}_normal.json") for s in RESAMPLE_SEEDS]
    if shipped is None or any(c is None for c in cors):
        fails.append(f"{model}: missing matrix outputs for contrast A")
        return None
    ap_s = shipped["macro_ap"]
    per = [{"resample": f"r{s}", "seed": s, "ap_shipped": ap_s, "ap_corrected": c["macro_ap"],
            "diff": round(ap_s - c["macro_ap"], 6)} for s, c in zip(RESAMPLE_SEEDS, cors)]
    diffs = [p["diff"] for p in per]
    mean_d, sd_d, t = stats(diffs)
    v, same = verdict(diffs, gate)
    cf = load(f"{model}_corrected_file_normal.json")
    aps_c = [c["macro_ap"] for c in cors]
    return {"model": model, "contrast": "A", "ap_shipped": ap_s, "per_resample": per,
            "mean": round(float(np.mean(aps_c)), 6), "sd": round(float(np.std(aps_c, ddof=1)), 6),
            "diff": round(mean_d, 6), "sd_diff": round(sd_d, 6), "t": round(t, 3), "df": len(diffs) - 1,
            "same_sign": same, "verdict": v,
            "published_corrected_file_ap_not_gated": cf["macro_ap"] if cf else None}


def contrast_c(model, gate, fails):
    shipped = load(f"{model}_shipped_normal.json")
    if shipped is None or "contrast_c" not in shipped:
        fails.append(f"{model}: missing contrast C")
        return None
    c = shipped["contrast_c"]
    diffs = [r["diff"] for r in c["per_resample"]]
    mean_d, sd_d, t = stats(diffs)
    v, same = verdict(diffs, gate)
    aps = [r["ap_twinned"] for r in c["per_resample"]]
    out = {k: c[k] for k in c if k != "per_resample"}
    out.update({"model": model, "contrast": "C", "per_resample": c["per_resample"],
                "mean": round(float(np.mean(aps)), 6), "sd": round(float(np.std(aps, ddof=1)), 6),
                "diff": round(mean_d, 6), "sd_diff": round(sd_d, 6), "t": round(t, 3), "df": len(diffs) - 1,
                "same_sign": same, "verdict": v})
    return out


def fmt_diffs(r):
    return ", ".join("%+.4f" % p["diff"] for p in r["per_resample"])


def validate(summary):
    import jsonschema
    schema = json.load(open(os.path.join(ROOT, "schema", "summary.schema.json"), encoding="utf-8"))
    jsonschema.validate(summary, schema)


def main():
    gate = prereg_gate()
    fails = []
    pre = load("prereg_check.json") or {"pass": False, "fails": ["prereg_check.json missing"]}
    if not pre.get("pass"):
        fails += [f"prereg: {f}" for f in pre.get("fails", [])]
    data = load("controls_data.json") or {"pass": False, "fails": ["controls_data.json missing"]}
    if not data.get("pass"):
        fails += [f"data: {f}" for f in data.get("fails", [])]

    A = {m: contrast_a(m, gate, fails) for m in ("M1", "M2")}
    C = {m: contrast_c(m, gate, fails) for m in ("M1", "M2")}

    # controls
    controls = {"data_and_leakage": {k: data.get(k) for k in ("md5", "skfp_crosscheck", "shipped_sizes",
                                                              "corrected_file_sizes", "corrected_file_md5",
                                                              "n_clusters", "twin", "record")}}
    controls["shuffle"] = {}
    for m in ("M1", "M2"):
        r = load(f"{m}_shipped_shuffle.json")
        if r is None:
            fails.append(f"{m}: shuffle control missing"); continue
        base = float(np.mean([r["test_positive_rate"][int(k)] for k in r["ap_per_label"]]))
        dev = {k: round(v - r["test_positive_rate"][int(k)], 4) for k, v in r["ap_per_label"].items()}
        ok = abs(r["macro_ap"] - base) <= SHUFFLE_TOL
        controls["shuffle"][m] = {"macro_ap": r["macro_ap"], "mean_positive_rate": round(base, 6),
                                  "abs_gap": round(abs(r["macro_ap"] - base), 6), "tolerance": SHUFFLE_TOL,
                                  "pass": ok, "per_label_deviation": dev,
                                  "labels_flagged_over_0.04": [k for k, v in dev.items() if abs(v) > SHUFFLE_FLAG]}
        if not ok:
            fails.append(f"{m}: label-shuffled AP {r['macro_ap']} not within {SHUFFLE_TOL} of base rate {base:.4f}")
    controls["determinism"] = {}
    for m in ("M1", "M2"):
        r = load(f"{m}_r0_determinism.json")
        if r is None:
            fails.append(f"{m}: determinism control missing"); continue
        controls["determinism"][m] = {k: r[k] for k in ("macro_ap", "macro_ap_second_run", "identical_to_6dp")}
        if not r["identical_to_6dp"]:
            fails.append(f"{m}: resample 0 not identical across two runs")
        a0 = A[m]["per_resample"][0]["ap_corrected"] if A[m] else None
        if a0 is not None and a0 != r["macro_ap"]:
            fails.append(f"{m}: r0 differs between jobs ({a0} vs {r['macro_ap']})")
    m1s = A["M1"]["ap_shipped"] if A["M1"] else None
    comparable = m1s is not None and abs(m1s - PAPER_AP) <= PAPER_TOL
    controls["paper_reproduction"] = {"paper": "arXiv 2501.17901, Table 1, count ECFP, Peptides-func AP 74.60",
                                      "paper_ap": PAPER_AP, "this_run_ap": m1s, "tolerance": PAPER_TOL,
                                      "status": "COMPARABLE" if comparable else "NOT COMPARABLE"}

    prim = A["M1"] or {}
    summary = {
        "name": "peptides-func-split-rerun",
        "primary": "M1 (count ECFP + LightGBM, arXiv 2501.17901) contrast A: shipped minus corrected macro AP",
        "per_resample": prim.get("per_resample", []), "mean": prim.get("mean"), "sd": prim.get("sd"),
        "diff": prim.get("diff"), "t": prim.get("t"), "verdict": prim.get("verdict", "INCONCLUSIVE"),
        "ap_shipped": prim.get("ap_shipped"), "paper_comparability": controls["paper_reproduction"]["status"],
        "gate": gate, "contrasts": {"A_M1": A["M1"], "A_M2": A["M2"], "C_M1": C["M1"], "C_M2": C["M2"]},
        "controls": controls,
        "provenance": {"prereg_commit": pre.get("prereg_commit"), "prereg_commit_unix": pre.get("prereg_commit_unix"),
                       "run_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
                       "repository": os.environ.get("GITHUB_REPOSITORY"),
                       "aggregated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "controls_failed": fails,
    }
    if pre.get("prereg_commit_unix") and pre["prereg_commit_unix"] >= time.time():
        fails.append("prereg commit is not older than this run")
    try:
        validate(summary)
        summary["schema_valid"] = True
    except Exception as e:
        summary["schema_valid"] = False
        fails.append(f"schema: {e}"[:300])
    # the verdict string must be the one recomputed from the gate, never a typed value
    if A["M1"] and verdict([p["diff"] for p in A["M1"]["per_resample"]], gate)[0] != summary["verdict"]:
        fails.append("verdict does not match a recomputation from the prereg gate")
    summary["controls_failed"] = fails

    os.makedirs(os.path.join(RESULTS, "arms"), exist_ok=True)
    for p in glob.glob(os.path.join(OUT, "*.json")):
        shutil.copy(p, os.path.join(RESULTS, "arms", os.path.basename(p)))
    write_json(os.path.join(RESULTS, "summary.json"), summary)

    color = {"H1": "blue", "H0": "blue", "INCONCLUSIVE": "yellow"}.get(summary["verdict"], "lightgrey")
    msg = (f"{summary['verdict']}: {summary['diff']:+.4f} AP (sd {A['M1']['sd_diff']:.4f})"
           if A["M1"] else "no result")
    if fails:
        color, msg = "lightgrey", msg + " (controls failed)"
    write_json(os.path.join(RESULTS, "badge.json"),
               {"schemaVersion": 1, "label": "Peptides-func inflation, M1", "message": msg, "color": color})

    lines = ["## Peptides-func: same model, two splits", "",
             f"Gate (PREREG.md): H1 if diff >= +{gate['h1_min_diff']} and same sign in >= "
             f"{gate['h1_min_same_sign']} of {gate['n_resamples']}; H0 if diff < +{gate['h0_below']} or sign "
             "inconsistent; else INCONCLUSIVE.", "",
             "| model | contrast | AP (shipped or twinned) | per-resample diff | mean diff | sd | t | verdict |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for key, r in summary["contrasts"].items():
        if not r:
            lines.append(f"| {key} | missing | | | | | | |"); continue
        first = r.get("ap_shipped", r.get("mean"))
        lines.append(f"| {r['model']} | {r['contrast']} | {first:.4f} | "
                     f"{fmt_diffs(r)} | {r['diff']:+.4f} | "
                     f"{r['sd_diff']:.4f} | {r['t']:.2f} | **{r['verdict']}** |")
    tw = (data.get("twin") or {})
    lines += ["", "Controls: shipped twin_at_080 "
              f"{tw.get('shipped', {}).get('twin_at_080_pct')}% ({tw.get('shipped', {}).get('twinned_test_rows')} rows); "
              "corrected file and resamples " + ", ".join(f"{k} {v.get('twin_at_080_pct')}%" for k, v in tw.items()
                                                         if k != "shipped") + ".",
              f"Paper reproduction (M1 shipped vs 0.7460, tol {PAPER_TOL}): {controls['paper_reproduction']['status']} "
              f"({m1s}).",
              "Shuffle: " + ", ".join(f"{m} AP {v['macro_ap']:.4f} vs base {v['mean_positive_rate']:.4f}"
                                      for m, v in controls["shuffle"].items()) + ".",
              "Determinism: " + ", ".join(f"{m} {v['identical_to_6dp']}" for m, v in controls["determinism"].items()) + ".",
              "", "Failures: " + ("none" if not fails else "; ".join(fails))]
    md = "\n".join(lines) + "\n"
    open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8", newline="\n").write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(md)
    print(md)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
