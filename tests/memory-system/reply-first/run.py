#!/usr/bin/env python3
"""Reply first, memory write after: nothing is lost when the process ends between the two
(README.md in this folder).

1. A serving process with memory on answers one question and is made to end right after its reply
   is written (`NANO_TEST_STOP_AFTER_REPLY`), before the write's second half.
2. The memory folder is read as the process left it: the answer's episode and its journal record
   must be there (written before the reply); the journal's views and the hierarchy must not have
   it yet (that is what the stop skipped).
3. A new process opens the same folder and answers nothing: the views must now reach the journal's
   last event and the hierarchy must hold the answer.
4. Four other questions in a process left to run normally, for the times: the reply's
   server milliseconds and the write before and after the reply, from the audit rows.

Usage: python tests/memory-system/reply-first/run.py [--exe <program>]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NANO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "second-session"))
import run as ss  # noqa: E402

QUESTION = "Which code calls make_room on the residency manager?"
# Different questions for the timed calls (a repeated question is answered from the reply cache).
OTHERS = [
    "How does forking a key and value cache share its entries?",
    "Where is the memory journal's torn last line set aside?",
    "Which function hashes the cited lines of a memory anchor?",
    "Where is the run-day counter ticked?",
]


def project_dir(home):
    dirs = [d for d in (home / "memory" / "projects").iterdir() if d.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f"expected one project memory folder, found {len(dirs)}")
    return dirs[0]


def state(proj, episode):
    rec = "ep:" + episode[:16]
    episodes = (proj / "episodes.jsonl").read_text(encoding="utf-8") if (proj / "episodes.jsonl").exists() else ""
    events = []
    for seg in sorted((proj / "journal").glob("*.jsonl")):
        events += [json.loads(l) for l in seg.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = json.loads((proj / "index.json").read_text(encoding="utf-8")) if (proj / "index.json").exists() else {}
    hier = (proj / "hierarchy.json").read_text(encoding="utf-8") if (proj / "hierarchy.json").exists() else ""
    return {
        "episode": episode in episodes,
        "journal_record": any(e.get("kind") == "record" and e.get("data", {}).get("id") == rec for e in events),
        "journal_last": max((e.get("seq", 0) for e in events), default=0),
        "views_at": index.get("journal_seq"),
        "hierarchy": episode in hier,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(NANO / "scratch/lane-c/target/release/neuromorph-nano.exe"))
    ap.add_argument("--work", default=str(NANO / "scratch/lane-c/reply-first"))
    a = ap.parse_args()
    exe, work = Path(a.exe).resolve(), Path(a.work).resolve()
    home = work / "home"
    ss.make_home(home)
    # 1. Reply, then the process ends before the write's second half.
    os.environ["NANO_TEST_STOP_AFTER_REPLY"] = "1"
    srv = ss.Server(exe, home, [])
    reply, _ = srv.pack(QUESTION)
    srv.p.wait(timeout=60)
    del os.environ["NANO_TEST_STOP_AFTER_REPLY"]
    head = reply.split("\n", 1)[0]
    episode = head.rsplit("episode ", 1)[-1].strip() if "episode " in head else ""
    if not episode:
        raise SystemExit(f"no episode in the reply's first line: {head}")
    proj = project_dir(home)
    stopped = state(proj, episode)
    # 3. A new process opens the folder.
    srv = ss.Server(exe, home, [])
    srv.close()
    reopened = state(proj, episode)
    # 4. Times with the process left to run.
    since = int(time.time() * 1000)
    srv = ss.Server(exe, home, [])
    for q in OTHERS:
        srv.pack(q)
    srv.close()
    rows = sorted(ss.audit_rows(home, since), key=lambda v: v.get("time_ms", 0))
    times = [{"server_ms": r.get("ms"), "write_ms": ss.write_ms(r), "after_reply_ms": ss.after_ms(r)} for r in rows]
    ok = (
        stopped["episode"] and stopped["journal_record"] and not stopped["hierarchy"]
        and (stopped["views_at"] or 0) < stopped["journal_last"]
        and reopened["episode"] and reopened["journal_record"] and reopened["hierarchy"]
        and reopened["views_at"] == reopened["journal_last"]
    )
    report = {"exe": str(exe), "episode": episode, "after_stop": stopped, "after_reopen": reopened, "times": times, "pass": ok}
    out = NANO / "runs" / f"memory-reply-first-{int(time.time() * 1000)}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("report:", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
