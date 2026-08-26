#!/usr/bin/env python
"""Summarize demo runs under experiments/ into a results table.

    python src/utils/summarize.py [experiments_dir]

Robust to runs that aren't finished: result.json is the "done" marker, so a run
without one is shown as `running` (trajectory.csv touched in the last 30 s) or
`incomplete` (crashed/aborted before finalize). Partial/missing files never
crash the summary — each run is parsed independently. Safe to run while a demo
is recording or back-to-back between runs.
"""
import json
import sys
import time
from pathlib import Path

RUNNING_WINDOW_S = 30.0


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _status(run):
    result = _load(run / "result.json")
    if result is not None:
        return ("done", result)
    traj = run / "trajectory.csv"
    if traj.exists() and (time.time() - traj.stat().st_mtime) < RUNNING_WINDOW_S:
        return ("running", None)
    return ("incomplete", None)


def _row(run):
    meta = _load(run / "metadata.json")
    if meta is None:
        return None  # not a run dir
    status, result = _status(run)
    r = result or {}
    found = r.get("found")
    return {
        "run": run.name,
        "target": meta.get("target") or "-",
        "status": status,
        "found": "" if found is None else ("yes" if found else "no"),
        "t_find": r.get("time_to_find_s"),
        "path_m": r.get("path_length_m"),
        "goals": r.get("n_goals"),
        "dur_s": r.get("duration_s"),
    }


def _fmt(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main(root="experiments"):
    runs = sorted(p for p in Path(root).glob("*") if p.is_dir())
    rows = [r for r in (_row(p) for p in runs) if r is not None]
    if not rows:
        print(f"no runs under {root}/")
        return

    cols = [("run", 24), ("target", 14), ("status", 11), ("found", 6),
            ("t_find", 8), ("path_m", 8), ("goals", 6), ("dur_s", 8)]
    print("  ".join(h.ljust(w) for h, w in cols))
    print("  ".join("-" * w for _, w in cols))
    for row in rows:
        print("  ".join(_fmt(row[h]).ljust(w) for h, w in cols))

    done = [r for r in rows if r["status"] == "done"]
    if done:
        n_found = sum(1 for r in done if r["found"] == "yes")
        finds = [r["t_find"] for r in done if r["found"] == "yes" and r["t_find"]]
        print(f"\n{len(done)} completed | found {n_found}/{len(done)} "
              f"({100 * n_found / len(done):.0f}%)"
              + (f" | avg time-to-find {sum(finds) / len(finds):.1f}s" if finds else ""))
    active = [r for r in rows if r["status"] != "done"]
    if active:
        print(f"{len(active)} not completed "
              f"({sum(r['status'] == 'running' for r in active)} running, "
              f"{sum(r['status'] == 'incomplete' for r in active)} incomplete)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        arg = str(Path(__file__).resolve().parent.parent.parent / "experiments")
    main(arg)
