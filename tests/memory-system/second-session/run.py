#!/usr/bin/env python3
"""Second-session test of the Neuromorph Nano's memory (README.md in this folder).

Session one, on a scratch Nano folder (NANO_HOME, never the real memory/): a serving process with
memory on answers fold 0 of the tuning questions; then, for each question, a scripted Claude Code
session is fed to the real `hook` command: the prompt, the pack as received, a Read and an Edit of
the question's answer lines, and a passing test (one question: a failing test, so its outcome is
refuted). The real `ledger` command settles the windows and writes the outcome rows.

Session two, a new process on a copy of that folder: the same questions, reworded forms and
questions about the same places from another angle, once with memory on and once with memory off,
the same program, back to back. Every reply is scored against the answers' anchor lines as the
bench scores them, and the report is written to runs/.

Usage: python tests/memory-system/second-session/run.py [--exe <program>] [--label <name>]
"""
import argparse
import hashlib
import json
import re
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NANO = HERE.parents[2]
ROOT = NANO / "scratch" / "frozen-p2"
FOLD = 0
REFUTED = "d05"
# The tuning sets (as the bench runs them, `scratch/lane-c2/all6.sh`): material root and question
# file. `fold0` is fold 0 of the 70 tuning questions with their reworded and other-angle forms.
SETS = {
    "closure": ("scratch/frozen-p2", "tests/dev/closure.json"),
    "dev": ("scratch/frozen-p2", "tests/dev/questions.json"),
    "cm1-r1": ("../neuromorph-mini", "scratch/lane-a/wp5/scale/cm1-r1.json"),
    "cm1-r2": ("../neuromorph-mini", "scratch/lane-a/wp5/scale/cm1-r2.json"),
    "large-code": ("tests/dev-large/material-code", "tests/dev-large/questions-code-tune.json"),
    "large-docs": ("tests/dev-large/material-docs", "tests/dev-large/questions-docs-tune.json"),
}


def load_set(name):
    """One tuning set: session one asks its answerable questions; session two asks every question
    again (answerable ones as `repeated`, the others as `decoy`). The refuted one is `d05` where the
    set holds it, else the set's first answerable question by id."""
    global ROOT, REFUTED
    root, qf = SETS[name]
    ROOT = (NANO / root).resolve()
    qs = sorted(json.loads((NANO / qf).read_text(encoding="utf-8"))["questions"], key=lambda q: q["id"])
    one = [q for q in qs if q.get("answers")]
    REFUTED = "d05" if any(q["id"] == "d05" for q in one) else one[0]["id"]
    two = [dict(q, group="repeated" if q.get("answers") else "decoy", of=q["id"], answers=q.get("answers") or []) for q in qs]
    return one, two


def load_sets():
    folds = json.loads((NANO / "tests/dev/folds.json").read_text(encoding="utf-8"))["folds"]
    src = {}
    for f in ("tests/dev/closure.json", "tests/dev/questions.json"):
        for q in json.loads((NANO / f).read_text(encoding="utf-8"))["questions"]:
            src[q["id"]] = q
    one = [q for q in src.values() if folds.get(q["id"]) == FOLD]
    one.sort(key=lambda q: q["id"])
    two = []
    for q in one:
        two.append(dict(q, group="repeated" if q["answers"] else "decoy", of=q["id"]))
    for q in json.loads((HERE / "session-two.json").read_text(encoding="utf-8"))["questions"]:
        base = src[q["of"]]
        idx = q.get("answers_of")
        answers = [base["answers"][i - 1] for i in idx] if idx else base["answers"]
        two.append(dict(q, answers=answers))
    return one, two


def locate(q):
    """Each answer's [first, last] line in the tree now, by its anchor (as the bench does)."""
    out = []
    for a in q["answers"]:
        text = (ROOT / a["file"]).read_text(encoding="utf-8", errors="replace").split("\n")
        hits = [i + 1 for i, l in enumerate(text) if a["anchor"] in l]
        # An anchor found more than once: the occurrence nearest the stated first line; not found:
        # the stated lines.
        at = min(hits, key=lambda h: abs(h - a["lines"][0])) if hits else a["lines"][0]
        span = a["lines"][1] - a["lines"][0]
        out.append((a["file"], at, at + span, a.get("match") == "file", text[at - 1] if at - 1 < len(text) else ""))
    return out


def cited(reply):
    out = []
    for l in reply.split("\n"):
        if l.startswith("@@ "):
            head, span = l[3:], True
        elif l.startswith("~~ "):
            head, span = l[3:], False
        else:
            continue
        # `path:first-last`, then a space or `;` (a header with evidence and no marks).
        m = re.match(r"(\S+?):(\d+)(?:-(\d+))?(?=[\s;]|$)", head)
        if not m:
            continue
        a = int(m.group(2))
        b = int(m.group(3)) if span and m.group(3) else a
        out.append((m.group(1).replace("\\", "/"), a, b))
    return out


def hit(ans, c):
    return any(f == c[0] and (whole or (c[1] <= last and c[2] >= first)) for f, first, last, whole, _ in ans)


def score(q, ans, reply):
    c = cited(reply)
    return {
        "id": q["id"],
        "group": q["group"],
        "first": bool(c) and hit(ans, c[0]),
        "top3": any(hit(ans, x) for x in c[:3]),
        "pack": any(hit(ans, x) for x in c),
        "all": bool(ans) and all(any(f == x[0] and (w or (x[1] <= l and x[2] >= a)) for x in c) for f, a, l, w, _ in ans),
        "tokens": len(reply) // 4,
        "withheld": reply.split("\n", 1)[0].find(" withheld") >= 0,
        "low": any(l.startswith("confidence: low") for l in reply.split("\n")),
        "recalled_first": next((l for l in reply.split("\n") if l.startswith("@@ ")), "").find("recalled:") >= 0,
    }


class Server:
    """One `serve` process over MCP stdio."""

    def __init__(self, exe, home, extra):
        env = dict(os.environ, NANO_HOME=str(home), NANO_MODELS=str(NANO / "weights"))
        env.pop("CLAUDE_PROJECT_DIR", None)
        self.p = subprocess.Popen(
            [str(exe), "serve", "--root", str(ROOT), *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=str(home),
        )
        self.n = 0
        self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "second-session", "version": "1"}})
        self.p.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.p.stdin.flush()

    def call(self, method, params):
        self.n += 1
        self.p.stdin.write((json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n").encode())
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit(f"server closed during {method}")
            v = json.loads(line)
            if v.get("id") == self.n:
                return v

    def pack(self, question):
        t = time.perf_counter()
        v = self.call("tools/call", {"name": "nm_context_pack", "arguments": {"question": question, "form": "full"}})
        secs = time.perf_counter() - t
        content = v.get("result", {}).get("content", [])
        text = "".join(c.get("text", "") for c in content) if content else json.dumps(v)
        return text, secs

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=120)


def hook(exe, home, event, payload):
    env = dict(os.environ, NANO_HOME=str(home), CLAUDE_PROJECT_DIR=str(ROOT), NANO_MODELS=str(NANO / "weights"))
    payload = dict(payload, hook_event_name=event, cwd=str(ROOT))
    r = subprocess.run([str(exe), "hook", event], input=json.dumps(payload).encode(), capture_output=True, env=env, cwd=str(home))
    if r.returncode != 0:
        raise SystemExit(f"hook {event}: {r.stderr.decode(errors='replace')}")
    time.sleep(0.003)


def run_cmd(exe, home, args):
    env = dict(os.environ, NANO_HOME=str(home), NANO_MODELS=str(NANO / "weights"))
    env.pop("CLAUDE_PROJECT_DIR", None)
    r = subprocess.run([str(exe), *args], capture_output=True, env=env, cwd=str(home))
    return r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")


def make_home(home, lane="served"):
    if home.exists():
        shutil.rmtree(home)
    (home / "memory").mkdir(parents=True)
    for f in ("nano.json", "rules.md"):
        shutil.copy(NANO / f, home / f)
    shutil.copytree(NANO / "designs", home / "designs")
    for card in (NANO / "crates").glob("*/part.json"):
        (home / "crates" / card.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy(card, home / "crates" / card.parent.name / "part.json")
    (home / "crates/nmcore").mkdir(parents=True, exist_ok=True)
    shutil.copy(NANO / "crates/nmcore/Cargo.toml", home / "crates/nmcore/Cargo.toml")
    if lane == "off":
        # The meaning lane runs by default only with memory on (its vectors are kept in the memory
        # folder); "off" keeps it out of both arms, so the arms differ in memory alone.
        (home / "settings.json").write_text(json.dumps({"dense_cache": "off"}) + "\n", encoding="utf-8")


def session_one(exe, home, questions, dump):
    """Session one: answers with memory on, then the scripted Claude Code sessions."""
    srv = Server(exe, home, [])
    replies = {}
    for q in questions:
        if not q["answers"]:
            continue
        s = f"one-{q['id']}"
        hook(exe, home, "SessionStart", {"session_id": s, "source": "startup"})
        hook(exe, home, "UserPromptSubmit", {"session_id": s, "prompt_id": "p1", "prompt": f"Work on this: {q['question']}"})
        replies[q["id"]], _ = srv.pack(q["question"])
        (dump / f"one-{q['id']}.txt").write_text(replies[q["id"]], encoding="utf-8")
        hook(exe, home, "PostToolUse", {
            "session_id": s, "tool_name": "mcp__neuromorph-nano__nm_context_pack", "tool_use_id": f"{s}-pack",
            "tool_input": {"question": q["question"], "form": "full"}, "tool_response": [{"type": "text", "text": replies[q["id"]]}],
        })
        crates = set()
        for i, (f, first, last, _, line) in enumerate(locate(q)):
            path = str(ROOT / f)
            hook(exe, home, "PostToolUse", {
                "session_id": s, "tool_name": "Read", "tool_use_id": f"{s}-read-{i}", "tool_input": {"file_path": path},
                "tool_response": {"type": "text", "file": {"filePath": path, "startLine": first, "numLines": last - first + 1}},
            })
            edit = {"file_path": path, "old_string": line, "new_string": line + " "}
            hook(exe, home, "PreToolUse", {"session_id": s, "tool_name": "Edit", "tool_use_id": f"{s}-edit-{i}", "tool_input": edit})
            hook(exe, home, "PostToolUse", {"session_id": s, "tool_name": "Edit", "tool_use_id": f"{s}-edit-{i}", "tool_input": edit, "tool_response": {"filePath": path}})
            parts = f.split("/")
            crates.add(parts[1] if parts[0] == "crates" and len(parts) > 1 else "nmcore")
        cmd = "cargo test -p " + " -p ".join(sorted(crates))
        if q["id"] == REFUTED:
            hook(exe, home, "PostToolUseFailure", {"session_id": s, "tool_name": "Bash", "tool_use_id": f"{s}-test", "tool_input": {"command": cmd},
                                                   "error": "Exit code 101\ntest result: FAILED. 3 passed; 1 failed; 0 ignored"})
        else:
            hook(exe, home, "PostToolUse", {"session_id": s, "tool_name": "Bash", "tool_use_id": f"{s}-test", "tool_input": {"command": cmd},
                                            "tool_response": {"stdout": "test result: ok. 4 passed; 0 failed; 0 ignored", "stderr": "", "interrupted": False}})
        hook(exe, home, "Stop", {"session_id": s, "stop_hook_active": False})
        hook(exe, home, "SessionEnd", {"session_id": s, "reason": "other"})
    srv.close()
    ledger = run_cmd(exe, home, ["ledger", "--project", str(ROOT)])
    outcomes = run_cmd(exe, home, ["outcomes", "--project", str(ROOT)])
    listed = run_cmd(exe, home, ["outcomes", "--project", str(ROOT), "--json"]).strip().splitlines()
    known = json.loads(listed[-1]) if listed and listed[-1].startswith("[") else []
    return replies, ledger, outcomes, known


def tree_hash(d):
    h = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def audit_rows(home, since):
    p = home / "memory" / "audit.jsonl"
    rows = []
    if p.exists():
        for l in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                v = json.loads(l)
            except ValueError:
                continue
            if v.get("tool") == "nm_context_pack" and v.get("state") == "done" and v.get("time_ms", 0) >= since:
                rows.append(v)
    return rows


def recall_ms(row):
    """Memory's recall in one call (ms), from its audit row's steps: the memory read steps, added;
    None when none ran. The outcome evidence step (`nmmem.evidence#m0`) runs inside a read step
    and is not added again."""
    total, ran = 0.0, False
    for st in row.get("steps", []):
        read = st.get("part") == "nmmem.gate#m0" and any(str(c).startswith("read:") for c in st.get("cells", []))
        if read:
            total += float(st.get("us", st.get("ms", 0) * 1000)) / 1000.0
            ran = True
    return total if ran else None


def write_ms(row):
    """Memory's write on the reply's path in one call (ms); None when none ran."""
    for st in row.get("steps", []):
        cells = [str(c) for c in st.get("cells", [])]
        if st.get("part") == "nmmem.gate#m0" and not any(c.startswith("read:") for c in cells) and "after-reply" not in cells:
            return float(st.get("us", st.get("ms", 0) * 1000)) / 1000.0
    return None


def after_ms(row):
    """Memory's write after the reply (ms), noted in the next call's row; None when none ran."""
    for st in row.get("steps", []):
        if st.get("part") == "nmmem.gate#m0" and "after-reply" in [str(c) for c in st.get("cells", [])]:
            return float(st.get("us", st.get("ms", 0) * 1000)) / 1000.0
    return None


def memory_marks(reply, refuted_places, refuted_ids=()):
    """What a reply shows of memory: spans tagged as recalled (and whether each one's lines are the
    tree's lines now), memory notes, a refuted outcome shown as refuted, and a refuted place served
    as a recalled span."""
    lines = reply.split("\n")
    recalled, fresh, served_refuted, overlap_other = 0, 0, 0, 0
    for i, l in enumerate(lines):
        if not (l.startswith("@@ ") and "recalled:" in l):
            continue
        recalled += 1
        c = cited(l)
        if not c:
            continue
        path, a, b = c[0]
        body = lines[i + 1:i + 1 + (b - a + 1)]
        now = (ROOT / path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")[a - 1:b]
        fresh += int(body == now)
        overlap = any(path == p and a <= e and s <= b for p, s, e in refuted_places)
        # Served by memory against the rule: a recalled span tagged with a refuted outcome. A span
        # over the same lines tagged with another, verified outcome is that outcome's evidence.
        by_refuted = any(r in l for r in refuted_ids)
        served_refuted += int(by_refuted)
        overlap_other += int(overlap and not by_refuted)
    return {
        "recalled": recalled,
        "recalled_fresh": fresh,
        "memory_notes": sum(1 for l in lines if l.startswith("~~ memory:")),
        "refuted_shown": any("refuted earlier" in l for l in lines),
        "refuted_served": served_refuted,
        "refuted_lines_by_verified": overlap_other,
    }


def session_two(exe, home, questions, extra, dump, arm):
    since = int(time.time() * 1000)
    srv = Server(exe, home, extra)
    rows = []
    for q in questions:
        reply, secs = srv.pack(q["question"])
        (dump / f"two-{arm}-{q['id']}.txt").write_text(reply, encoding="utf-8")
        r = score(q, locate(q), reply)
        r.update(secs=secs, reply=reply, of=q["of"])
        rows.append(r)
    srv.close()
    # One audit row per call, in call order (with memory off the row holds no question).
    audits = sorted(audit_rows(home, since), key=lambda v: v.get("time_ms", 0))
    for i, r in enumerate(rows):
        a = audits[i] if len(audits) == len(rows) else None
        r["server_ms"] = a.get("ms") if a else None
        r["recall_ms"] = recall_ms(a) if a else None
        r["write_ms"] = write_ms(a) if a else None
        r["after_ms"] = after_ms(a) if a else None
    return rows


def table(rows, groups):
    out = []
    for g in groups:
        rs = [r for r in rows if r["group"] in g]
        if not rs:
            continue
        secs = [r["secs"] for r in rs]
        out.append({
            "group": "+".join(g), "n": len(rs),
            "first": sum(r["first"] for r in rs), "top3": sum(r["top3"] for r in rs), "pack": sum(r["pack"] for r in rs),
            "all": sum(r["all"] for r in rs), "tokens": round(statistics.mean(r["tokens"] for r in rs)),
            "median_s": round(statistics.median(secs), 3), "mean_s": round(statistics.mean(secs), 3),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(NANO / "scratch/lane-c/target/release/neuromorph-nano.exe"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--work", default=str(NANO / "scratch/lane-c/wp13"))
    ap.add_argument("--lane", default="served", choices=["served", "off"], help="the meaning lane: as served (on with memory on only) or off in both arms")
    ap.add_argument("--on-args", default="", help="extra serve arguments for the memory-on arm, space separated")
    ap.add_argument("--set", default="fold0", choices=["fold0", *SETS], help="fold 0 with reworded and angle forms, or one whole tuning set")
    a = ap.parse_args()
    exe = Path(a.exe).resolve()
    work = Path(a.work).resolve() / a.label
    dump = work / "replies"
    if work.exists():
        shutil.rmtree(work)
    dump.mkdir(parents=True)
    one, two = load_sets() if a.set == "fold0" else load_set(a.set)
    home1 = work / "home-one"
    make_home(home1, a.lane)
    t0 = time.time()
    _, ledger, outcomes, known = session_one(exe, home1, one, dump)
    refuted_places = []
    for k in known:
        if k.get("standing") == "refuted":
            for p in k.get("places", []):
                path, at = p["place"].rsplit(":", 1)
                s, e = (at.split("-") + [at])[:2]
                refuted_places.append((path, int(s), int(e)))
    refuted_ids = [l.split()[0] for l in outcomes.splitlines() if "REFUTED" in l and l.strip().startswith("oc:")]
    (work / "ledger.txt").write_text(ledger, encoding="utf-8")
    (work / "outcomes.txt").write_text(outcomes, encoding="utf-8")
    home_on, home_off = work / "home-on", work / "home-off"
    shutil.copytree(home1, home_on)
    shutil.copytree(home1, home_off)
    on = session_two(exe, home_on, two, a.on_args.split(), dump, "on")
    project_off = home_off / "memory" / "projects"
    before_off = tree_hash(project_off)
    off = session_two(exe, home_off, two, ["--memory", "off"], dump, "off")
    after_off = tree_hash(project_off)
    off_written = sorted(k for k in set(before_off) | set(after_off) if before_off.get(k) != after_off.get(k))
    groups = [("repeated",), ("reworded",), ("angle",), ("repeated", "reworded", "angle")]
    report = {
        "label": a.label, "exe": str(exe), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "root": str(ROOT), "set": a.set, "fold": FOLD if a.set == "fold0" else None, "refuted": REFUTED, "on_args": a.on_args, "lane": a.lane, "seconds": round(time.time() - t0, 1),
        "outcomes": outcomes, "on": table(on, groups), "off": table(off, groups),
        "off_memory_files_changed": off_written,
    }
    # Per question, memory on against off, and what memory showed.
    per = []
    refuted_served = []
    for r1, r0 in zip(on, off):
        m = memory_marks(r1["reply"], refuted_places, refuted_ids)
        if r1["of"] == REFUTED:
            refuted_served.append({"id": r1["id"], "shown": m["refuted_shown"], "served_by_memory": m["refuted_served"], "same_lines_by_verified": m["refuted_lines_by_verified"]})
        per.append({"id": r1["id"], "group": r1["group"], "on": [int(r1["first"]), int(r1["top3"]), int(r1["pack"]), int(r1["all"])],
                    "off": [int(r0["first"]), int(r0["top3"]), int(r0["pack"]), int(r0["all"])], "tokens": [r1["tokens"], r0["tokens"]],
                    "secs": [round(r1["secs"], 3), round(r0["secs"], 3)], "recall_ms": r1["recall_ms"], **m})
    report["per_question"] = per
    rec = [r["recall_ms"] for r in on if r["recall_ms"] is not None]
    report["recall_ms"] = {"n": len(rec), "median": round(statistics.median(rec), 2) if rec else None, "max": round(max(rec), 2) if rec else None}
    wr = [r["write_ms"] for r in on if r["write_ms"] is not None]
    report["write_ms"] = {"n": len(wr), "median": round(statistics.median(wr), 2) if wr else None, "max": round(max(wr), 2) if wr else None}
    af = [r["after_ms"] for r in on if r.get("after_ms") is not None]
    report["after_reply_ms"] = {"n": len(af), "median": round(statistics.median(af), 2) if af else None, "max": round(max(af), 2) if af else None}
    srv_on = [r["server_ms"] for r in on if r["server_ms"] is not None]
    srv_off = [r["server_ms"] for r in off if r["server_ms"] is not None]
    report["server_ms_median"] = {"on": statistics.median(srv_on) if srv_on else None, "off": statistics.median(srv_off) if srv_off else None}
    report["refuted_check"] = refuted_served
    report["refuted_places"] = refuted_places
    report["outcomes_verified"] = sum(1 for k in known if k.get("standing") == "verified")
    report["outcomes_refuted"] = sum(1 for k in known if k.get("standing") == "refuted")
    by_group = {}
    for p in per:
        g = by_group.setdefault(p["group"], {"replies": 0, "with_recall": 0, "recalled_spans": 0, "fresh": 0})
        g["replies"] += 1
        g["with_recall"] += int(p["recalled"] > 0)
        g["recalled_spans"] += p["recalled"]
        g["fresh"] += p["recalled_fresh"]
    report["recall_by_group"] = by_group
    # Decoys (no answer in the material): replies withheld or marked low, and replies whose first
    # cited span is a recalled one, per arm.
    dec = [(r1, r0) for r1, r0 in zip(on, off) if r1["group"] == "decoy"]
    report["decoys"] = {"n": len(dec), **{f"{k}_{arm}": sum(int(x[i][k]) for x in dec) for k in ("withheld", "low", "recalled_first") for i, arm in ((0, "on"), (1, "off"))}}
    out = NANO / "runs" / f"memory-second-session-{a.label}-{int(time.time() * 1000)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {out}")
    for arm in ("on", "off"):
        for t in report[arm]:
            print(f"{arm:3} {t['group']:<25} n {t['n']:2}  first {t['first']:2}  top3 {t['top3']:2}  pack {t['pack']:2}  all {t['all']:2}  tokens {t['tokens']:5}  median {t['median_s']:.3f} s")
    print("recall ms:", report["recall_ms"], " write ms (reply path):", report["write_ms"], " after reply:", report["after_reply_ms"], " server ms median:", report["server_ms_median"])
    print("memory off, files changed in the project memory:", off_written or "none")
    print("outcomes after session one: verified", report["outcomes_verified"], "refuted", report["outcomes_refuted"])
    print("recall by group:", report["recall_by_group"])
    print("refuted outcome:", report["refuted_check"])
    print("decoys:", report["decoys"])


if __name__ == "__main__":
    sys.exit(main())
