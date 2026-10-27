"""m3-prereg-check: PREREG_M3.md predates every M3 result, and its budget was fixed before them.

1. The commit that added PREREG_M3.md is an ancestor of HEAD and of every commit, on any fetched
   branch, that touches results/m3; and it is older than each of them.
2. The last commit that changed PREREG_M3.md (the budget addendum) is older than every commit that
   touches results/m3.
3. The m3budget block sets an integer max_epoch (a full run refuses to start without one).
Writes out/m3_prereg_check.json. Exit 1 on any failure.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ROOT, write_json  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402


def budget():
    """The same parse as m3_model.budget(), without importing torch (this job installs nothing)."""
    text = open(os.path.join(ROOT, "PREREG_M3.md"), encoding="utf-8").read()
    m = re.search(r"```m3budget\s*\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit("PREREG_M3.md has no ```m3budget block")
    return json.loads(m.group(1))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main():
    fails, rep = [], {}
    add = git("log", "--all", "--diff-filter=A", "--format=%H %ct", "--", "PREREG_M3.md").stdout.split()
    last = git("log", "-1", "--format=%H %ct", "--", "PREREG_M3.md").stdout.split()
    if len(add) < 2 or len(last) < 2:
        raise SystemExit("PREREG_M3.md history not found (fetch-depth: 0?)")
    add_sha, add_ts = add[-2], int(add[-1])
    last_sha, last_ts = last[0], int(last[1])
    rep.update({"prereg_m3_commit": add_sha, "prereg_m3_commit_unix": add_ts,
                "prereg_m3_last_change": last_sha, "prereg_m3_last_change_unix": last_ts})
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", add_sha, head).returncode != 0:
        fails.append(f"PREREG_M3 commit {add_sha[:7]} is not an ancestor of HEAD {head[:7]}")
    res = git("log", "--all", "--format=%H %ct", "--", "results/m3").stdout.split()
    touched = list(zip(res[0::2], res[1::2]))
    rep["results_m3_commits"] = len(touched)
    for sha, ts in touched:
        if git("merge-base", "--is-ancestor", add_sha, sha).returncode != 0:
            fails.append(f"results/m3 touched by {sha[:7]}, which does not descend from PREREG_M3 {add_sha[:7]}")
        if int(ts) < add_ts or int(ts) < last_ts:
            fails.append(f"results/m3 commit {sha[:7]} is older than PREREG_M3 or its budget addendum")
    b = budget()
    rep["m3budget"] = b
    if not isinstance(b.get("max_epoch"), int) or b["max_epoch"] <= 0:
        fails.append(f"m3budget.max_epoch not set: {b}")
    print(f"PREREG_M3.md added {add_sha[:7]} {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(add_ts))}; "
          f"last change {last_sha[:7]} {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(last_ts))}; "
          f"budget {b}; {len(touched)} results/m3 commit(s)")
    rep["fails"], rep["pass"] = fails, not fails
    write_json(os.path.join(OUT, "m3_prereg_check.json"), rep)
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
