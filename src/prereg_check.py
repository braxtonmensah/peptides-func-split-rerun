"""prereg-check: the gate predates every result, and equals the gate in the vendored source.

1. Thresholds parsed from the docstring of vendor/inflation_lrgb.py (+0.03, 4 of 5, +0.01) equal the
   ```gate block in PREREG.md.
2. Every vendored file matches its pinned sha256, and (when the network is up) the upstream file at
   the pinned commit has the same sha256.
3. Git history: the commit that added PREREG.md is a strict ancestor of every commit, on any fetched
   branch, that touches results/. It is also an ancestor of the commit being run.
Writes out/prereg_check.json. Exit 1 on any failure.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ROOT, VENDOR, prereg_gate, sha256, write_json  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def docstring_gate():
    src = open(os.path.join(VENDOR, "inflation_lrgb.py"), encoding="utf-8").read()
    doc = src.split('"""')[1]
    h1 = re.search(r"H1\s+mean\(original\) - mean\(corrected\) >= \+([0-9.]+) AP", doc)
    ss = re.search(r"at least (\d+) of (\d+)\s+resamples", doc)
    h0 = re.search(r"H0\s+below \+([0-9.]+)", doc)
    if not (h1 and ss and h0):
        raise SystemExit("could not parse the gate out of vendor/inflation_lrgb.py")
    return {"h1_min_diff": float(h1.group(1)), "h1_min_same_sign": int(ss.group(1)),
            "n_resamples": int(ss.group(2)), "h0_below": float(h0.group(1))}


def main():
    fails, report = [], {}

    # 1. thresholds
    g_pre, g_doc = prereg_gate(), docstring_gate()
    report["gate_prereg"], report["gate_vendored_docstring"] = g_pre, g_doc
    if g_pre != g_doc:
        fails.append(f"gate mismatch: PREREG {g_pre} vs vendored docstring {g_doc}")
    print(f"gate PREREG.md      {g_pre}")
    print(f"gate inflation_lrgb {g_doc}")

    # 2. vendored files against pins
    pins = json.load(open(os.path.join(VENDOR, "PINS.json"), encoding="utf-8"))
    report["vendored"] = []
    for p in pins["files"]:
        local = sha256(os.path.join(VENDOR, p["name"]))
        row = {"name": p["name"], "pinned": p["sha256"], "local": local, "upstream": None}
        if local != p["sha256"]:
            fails.append(f"{p['name']}: local sha256 {local} != pinned {p['sha256']}")
        try:
            url = f"https://raw.githubusercontent.com/{p['repo']}/{p['commit']}/{p['path']}"
            data = urllib.request.urlopen(url, timeout=60).read()
            import hashlib
            row["upstream"] = hashlib.sha256(data).hexdigest()
            if row["upstream"] != p["sha256"]:
                fails.append(f"{p['name']}: upstream at {p['commit'][:7]} is {row['upstream']}")
        except Exception as e:  # network trouble is reported, the local pin still binds
            row["upstream"] = f"unreachable: {e}"
        report["vendored"].append(row)
        print(f"vendored {p['name']:<28} local==pin {local == p['sha256']}  upstream {str(row['upstream'])[:16]}")

    # 3. git order
    add = git("log", "--all", "--diff-filter=A", "--format=%H %ct", "--", "PREREG.md").stdout.split()
    if len(add) < 2:
        fails.append("PREREG.md has no adding commit in the fetched history (use fetch-depth: 0)")
    else:
        pre_sha, pre_ts = add[-2], int(add[-1])   # oldest add
        report["prereg_commit"], report["prereg_commit_unix"] = pre_sha, pre_ts
        head = git("rev-parse", "HEAD").stdout.strip()
        if git("merge-base", "--is-ancestor", pre_sha, head).returncode != 0:
            fails.append(f"PREREG commit {pre_sha[:7]} is not an ancestor of HEAD {head[:7]}")
        res = git("log", "--all", "--format=%H %ct", "--", "results").stdout.split()
        touched = list(zip(res[0::2], res[1::2]))
        report["results_commits"] = len(touched)
        for sha, ts in touched:
            if sha == pre_sha or git("merge-base", "--is-ancestor", pre_sha, sha).returncode != 0:
                fails.append(f"results/ touched by {sha[:7]}, which does not descend from PREREG {pre_sha[:7]}")
            if int(ts) < pre_ts:
                fails.append(f"results/ commit {sha[:7]} is older than PREREG {pre_sha[:7]}")
        print(f"PREREG.md added in {pre_sha[:7]} at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(pre_ts))}; "
              f"{len(touched)} later commit(s) touch results/, all descend from it: "
              f"{not any('results/' in f for f in fails)}")

    report["fails"] = fails
    report["pass"] = not fails
    write_json(os.path.join(OUT, "prereg_check.json"), report)
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
