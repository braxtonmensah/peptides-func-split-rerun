"""aggregate_m3: contrasts A and C, controls and verdicts for arm M3, by the rules in PREREG_M3.md.

Reads out/M3_*.json (11 jobs), out/controls_data.json, out/prereg_check.json, out/m3_prereg_check.json.
Writes results/m3/summary.json, results/m3/arms/*.json, results/m3/badge.json, results/m3/summary.md.
M1/M2 files are never read or written here. Exit 1 if a failing control fails.
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, RESAMPLE_SEEDS, RESULTS, SHUFFLE_FLAG, SHUFFLE_TOL, prereg_gate, verdict, write_json  # noqa: E402

PAPER_AP, PAPER_SD, PAPER_TOL = 0.6860, 0.0050, 0.015   # arXiv 2309.00367, tuned GCN; PREREG_M3.md
DET_LABEL = 0.01
SHIPPED_SEEDS = [0, 1, 2]
R3 = os.path.join(RESULTS, "m3")


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def stats(diffs):
    d = np.asarray(diffs, float)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("inf") * np.sign(d.mean())
    return float(d.mean()), sd, t


def main():
    gate = prereg_gate()
    fails, flags = [], []
    for nm, label in (("prereg_check.json", "prereg"), ("m3_prereg_check.json", "m3 prereg"),
                      ("controls_data.json", "data")):
        r = load(nm) or {"pass": False, "fails": [f"{nm} missing"]}
        if not r.get("pass"):
            fails += [f"{label}: {f}" for f in r.get("fails", [])]
    m3pre = load("m3_prereg_check.json") or {}
    E = (m3pre.get("m3budget") or {}).get("max_epoch")
    data = load("controls_data.json") or {}

    jobs = {f"shipped_s{s}": load(f"M3_shipped_s{s}_normal.json") for s in SHIPPED_SEEDS}
    jobs.update({f"r{s}": load(f"M3_r{s}_s0_normal.json") for s in RESAMPLE_SEEDS})
    jobs["corrected_file"] = load("M3_corrected_file_s0_normal.json")
    jobs["shuffle"] = load("M3_shipped_s0_shuffle.json")
    jobs["determinism"] = load("M3_r0_s0_determinism.json")
    for k, r in jobs.items():
        if r is None:
            fails.append(f"M3 job {k} missing")
        elif r.get("max_epoch") != E or r.get("epochs_run") != E:
            fails.append(f"M3 job {k}: epochs {r.get('epochs_run')} / {r.get('max_epoch')} != budget {E}")

    A = C = None
    shipped_aps = [jobs[f"shipped_s{s}"]["macro_ap"] for s in SHIPPED_SEEDS if jobs[f"shipped_s{s}"]]
    if len(shipped_aps) == 3 and all(jobs[f"r{s}"] for s in RESAMPLE_SEEDS):
        ap_s = float(np.mean(shipped_aps))
        per = [{"resample": f"r{s}", "seed": s, "ap_shipped": round(ap_s, 6),
                "ap_corrected": jobs[f"r{s}"]["macro_ap"],
                "diff": round(ap_s - jobs[f"r{s}"]["macro_ap"], 6)} for s in RESAMPLE_SEEDS]
        diffs = [p["diff"] for p in per]
        mean_d, sd_d, t = stats(diffs)
        v, same = verdict(diffs, gate)
        aps_c = [p["ap_corrected"] for p in per]
        A = {"model": "M3", "contrast": "A", "ap_shipped": round(ap_s, 6),
             "ap_shipped_per_seed": shipped_aps, "ap_shipped_seed_sd": round(float(np.std(shipped_aps, ddof=1)), 6),
             "per_resample": per, "mean": round(float(np.mean(aps_c)), 6), "sd": round(float(np.std(aps_c, ddof=1)), 6),
             "diff": round(mean_d, 6), "sd_diff": round(sd_d, 6), "t": round(t, 3), "df": len(diffs) - 1,
             "same_sign": same, "verdict": v,
             "best_epochs": {k: jobs[k]["best_epoch"] for k in jobs if jobs[k] and k not in ("shuffle",)},
             "published_corrected_file_ap_not_gated": jobs["corrected_file"]["macro_ap"] if jobs["corrected_file"] else None}
    else:
        fails.append("M3: missing outputs for contrast A")
    s0 = jobs["shipped_s0"]
    if s0 and "contrast_c" in s0:
        c = s0["contrast_c"]
        diffs = [r["diff"] for r in c["per_resample"]]
        mean_d, sd_d, t = stats(diffs)
        v, same = verdict(diffs, gate)
        aps = [r["ap_twinned"] for r in c["per_resample"]]
        C = {k: c[k] for k in c if k != "per_resample"}
        C.update({"model": "M3", "contrast": "C", "per_resample": c["per_resample"],
                  "mean": round(float(np.mean(aps)), 6), "sd": round(float(np.std(aps, ddof=1)), 6),
                  "diff": round(mean_d, 6), "sd_diff": round(sd_d, 6), "t": round(t, 3), "df": len(diffs) - 1,
                  "same_sign": same, "verdict": v})
    else:
        fails.append("M3: missing contrast C")

    controls = {}
    r = jobs["shuffle"]
    if r:
        base = float(np.mean([r["test_positive_rate"][int(k)] for k in r["ap_per_label"]]))
        dev = {k: round(v - r["test_positive_rate"][int(k)], 4) for k, v in r["ap_per_label"].items()}
        ok = abs(r["macro_ap"] - base) <= SHUFFLE_TOL
        controls["shuffle"] = {"macro_ap": r["macro_ap"], "mean_positive_rate": round(base, 6),
                               "abs_gap": round(abs(r["macro_ap"] - base), 6), "tolerance": SHUFFLE_TOL, "pass": ok,
                               "per_label_deviation": dev,
                               "labels_flagged_over_0.04": [k for k, v in dev.items() if abs(v) > SHUFFLE_FLAG]}
        if not ok:
            fails.append(f"M3: label-shuffled AP {r['macro_ap']} not within {SHUFFLE_TOL} of base rate {base:.4f}")
    rd, r0 = jobs["determinism"], jobs["r0"]
    if rd and r0:
        gap = abs(rd["macro_ap"] - r0["macro_ap"])
        controls["determinism"] = {"r0_normal": r0["macro_ap"], "r0_determinism_job": rd["macro_ap"],
                                   "abs_diff": round(gap, 6), "identical_to_6dp": gap == 0,
                                   "cpu_note": "flag, not fail (PREREG_M3.md control 4)"}
        if gap != 0:
            flags.append(f"determinism: r0 differs between two jobs by {gap:.6f}")
        if gap > DET_LABEL:
            flags.append("determinism gap above 0.01 AP: labelled in the README")
    if A:
        comp = abs(A["ap_shipped"] - PAPER_AP) <= PAPER_TOL
        controls["paper_reproduction"] = {
            "paper": "arXiv 2309.00367 (Toenshoff et al.), tuned GCN, Peptides-func test AP 0.6860 +- 0.0050, 4 seeds",
            "paper_ap": PAPER_AP, "this_run_ap_mean_3_seeds": A["ap_shipped"], "tolerance": PAPER_TOL,
            "max_epoch": E, "published_max_epoch": 250,
            "status": "COMPARABLE" if comp else "NOT COMPARABLE"}
        if not comp:
            flags.append(f"paper reproduction: {A['ap_shipped']:.4f} vs 0.6860 (NOT COMPARABLE)")

    summary = {"name": "peptides-func-split-rerun arm M3",
               "model": "tuned GCN, Toenshoff et al. TMLR 2024, configs/LRGB-tuned/peptides-func-GCN.yaml @ c3500cb, "
                        "reimplemented in PyG (PREREG_M3.md)",
               "max_epoch": E, "gate": gate, "contrasts": {"A_M3": A, "C_M3": C}, "controls": controls,
               "verdict": A["verdict"] if A else "INCONCLUSIVE", "diff": A["diff"] if A else None,
               "flags": flags,
               "provenance": {"prereg_m3_commit": m3pre.get("prereg_m3_commit"),
                              "prereg_m3_last_change": m3pre.get("prereg_m3_last_change"),
                              "run_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
                              "repository": os.environ.get("GITHUB_REPOSITORY"),
                              "aggregated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
               "controls_failed": fails,
               "data_controls": {k: data.get(k) for k in ("md5", "skfp_crosscheck", "shipped_sizes", "twin")}}
    if A and verdict([p["diff"] for p in A["per_resample"]], gate)[0] != summary["verdict"]:
        fails.append("verdict does not match a recomputation from the prereg gate")
    os.makedirs(os.path.join(R3, "arms"), exist_ok=True)
    for p in glob.glob(os.path.join(OUT, "M3_*.json")):
        shutil.copy(p, os.path.join(R3, "arms", os.path.basename(p)))
    write_json(os.path.join(R3, "summary.json"), summary)
    msg = f"{summary['verdict']}: {A['diff']:+.4f} AP (sd {A['sd_diff']:.4f}, E={E})" if A else "no result"
    color = "lightgrey" if fails else ("yellow" if summary["verdict"] == "INCONCLUSIVE" else "blue")
    if fails:
        msg += " (controls failed)"
    write_json(os.path.join(R3, "badge.json"),
               {"schemaVersion": 1, "label": "Peptides-func inflation, M3 GCN", "message": msg, "color": color})

    lines = ["## Peptides-func arm M3: tuned GCN (Toenshoff et al.), same splits", "",
             f"Epoch budget E = {E} (published config: 250). Gate as PREREG.md.", "",
             "| contrast | AP contaminated side | AP clean side | per-resample diff | mean diff | sd | t | same sign | verdict |",
             "|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in (A, C):
        if r:
            hi = r.get("ap_shipped", r.get("mean"))
            lo = r["mean"] if r["contrast"] == "A" else r["per_resample"][0]["ap_untwinned"]
            lines.append(f"| {r['contrast']} | {hi:.4f} | {lo:.4f} | "
                         + ", ".join("%+.4f" % p["diff"] for p in r["per_resample"])
                         + f" | {r['diff']:+.4f} | {r['sd_diff']:.4f} | {r['t']:.2f} | {r['same_sign']} of 5 | **{r['verdict']}** |")
    if A:
        lines += ["", f"Shipped split per seed: {A['ap_shipped_per_seed']} (sd {A['ap_shipped_seed_sd']}); "
                      f"published corrected file {A['published_corrected_file_ap_not_gated']} (not gated)."]
    lines += ["", "Controls: " + json.dumps(controls), "", "Flags: " + ("none" if not flags else "; ".join(flags)),
              "", "Failures: " + ("none" if not fails else "; ".join(fails))]
    md = "\n".join(lines) + "\n"
    open(os.path.join(R3, "summary.md"), "w", encoding="utf-8", newline="\n").write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(md)
    print(md)
    summary["controls_failed"] = fails
    write_json(os.path.join(R3, "summary.json"), summary)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
