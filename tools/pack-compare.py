"""Compare the two context pack formats of the Neuromorph Nano on the same questions.

Runs `neuromorph-nano serve --root <repo> --memory off` over MCP stdio, asks each question in the
`cited` and `full` forms twice, once with `format: "json"` (the earlier JSON reply) and once in the
plain-text default, and records for each reply its size (characters / 4) and the set of
(file, line) pairs it carries. A pair counts only when the line's text in the reply equals that
line of the file on disk, so a format cannot score a line it does not really carry. The text
format must carry every pair the JSON format carries, for every question and form.

Usage: python tools/pack-compare.py [--exe <neuromorph-nano.exe>] [--root <repo>] [--old-exe <exe>]
       [--questions <file, one question per line>]
`--old-exe` also checks that the JSON format of `--exe` is byte-identical to the reply of an
earlier build that had only the JSON format. Exit code 1 when any pair is lost.
"""
import argparse
import json
import pathlib
import subprocess
import sys

QUESTIONS = [
    "Where are rows appended to the audit log?",
    "What is the default port of the interface server?",
    "How does setup merge an existing .mcp.json file?",
    "Which function parses the delivery form argument of a context pack?",
    "Who calls withhold_uncovered?",
    "Why does the reducer skip a packet that is too large for the token budget?",
    "What value is the relevance floor FLOOR set to?",
    "Which crates does nmapi depend on?",
    "How is the reply to an MCP tools/call request built?",
    "What happens when a control is set to a value it does not allow?",
    "Where is the token estimate of a packet computed?",
    "Which check makes sure a decoy question is withheld?",
    "How does the interface server refuse writes from another origin?",
    "What does the launcher do with live copies that are still running?",
]


class Server:
    def __init__(self, exe, root, cwd):
        self.p = subprocess.Popen([str(exe), "serve", "--root", str(root), "--memory", "off"], cwd=str(cwd),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.n = 0
        self.rpc("initialize", {})

    def rpc(self, method, params):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.p.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server closed its output")
        return json.loads(line.decode("utf-8"))

    def pack(self, args):
        r = self.rpc("tools/call", {"name": "nm_context_pack", "arguments": args})
        res = r["result"]
        if res.get("isError"):
            raise SystemExit("tool error: " + res["content"][0]["text"])
        return res["content"][0]["text"]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


class Files:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.cache = {}

    def line(self, path, n):
        if path not in self.cache:
            f = self.root / path
            try:
                self.cache[path] = f.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                self.cache[path] = None
        lines = self.cache[path]
        if lines is None or n < 1 or n > len(lines):
            return None
        return lines[n - 1].rstrip("\r")


def place(source):
    if ":" not in source:
        return None
    path, _, line = source.rpartition(":")
    return (path, int(line)) if line.isdigit() and int(line) > 0 else None


def carried(files, path, first, lines):
    """(file, line) pairs whose text in the reply equals the file's line."""
    out = []
    for i, text in enumerate(lines):
        real = files.line(path, first + i)
        if real is not None and real.rstrip() == text.rstrip():
            out.append((path, first + i))
    return out


def pairs_json(body, files):
    v = json.loads(body)
    out = []
    for p in v.get("packets", []):
        at = place(p.get("source", ""))
        if at:
            out += carried(files, at[0], at[1], p.get("text", "").split("\n"))
    return out


def pairs_text(body, files):
    lines = body.split("\n")
    out = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("@@ "):
            head = l[3:].split(" ")[0]
            path, _, rng = head.rpartition(":")
            a, _, b = rng.partition("-")
            a, b = int(a), int(b)
            out += carried(files, path, a, lines[i + 1:i + 1 + (b - a + 1)])
            i += 1 + (b - a + 1)
        elif l.startswith("~~ "):
            head = l[3:]
            n = 1
            if " lines)" in head:
                n = int(head.rsplit("(", 1)[1].split(" ")[0])
            at = place(head.split(" ")[0])
            if at:
                out += carried(files, at[0], at[1], lines[i + 1:i + 1 + n])
            i += 1 + n
        else:
            i += 1
    return out


def main():
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(here / "target" / "release" / "neuromorph-nano.exe"))
    ap.add_argument("--root", default=str(here))
    ap.add_argument("--old-exe")
    ap.add_argument("--questions")
    a = ap.parse_args()
    qs = QUESTIONS
    if a.questions:
        qs = [q.strip() for q in pathlib.Path(a.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
    files = Files(a.root)
    new = Server(a.exe, a.root, here)
    old = Server(a.old_exe, a.root, here) if a.old_exe else None
    lost_any = False
    totals = {}
    print("form  | old tok | new tok | saving | old pairs | new pairs | dup old | lost | question")
    for q in qs:
        for form in ("cited", "full"):
            j = new.pack({"question": q, "form": form, "format": "json"})
            t = new.pack({"question": q, "form": form})
            if old is not None:
                o = old.pack({"question": q, "form": form})
                if o != j:
                    print(f"  json format differs from the earlier build on [{form}] {q}")
                    lost_any = True
            pj, pt = pairs_json(j, files), pairs_text(t, files)
            sj, st = set(pj), set(pt)
            lost = sj - st
            lost_any |= bool(lost)
            tj, tt = len(j) / 4, len(t) / 4
            agg = totals.setdefault(form, [0, 0, 0, 0])
            agg[0] += tj
            agg[1] += tt
            agg[2] += len(sj)
            agg[3] += len(lost)
            save = 100 * (1 - tt / tj) if tj else 0
            print(f"{form:5} | {tj:7.0f} | {tt:7.0f} | {save:5.1f}% | {len(sj):9} | {len(st):9} | {len(pj) - len(sj):7} | {len(lost):4} | {q}")
            for x in sorted(lost)[:5]:
                print(f"      lost {x[0]}:{x[1]}")
    new.close()
    if old is not None:
        old.close()
    for form, (tj, tt, pairs, lost) in totals.items():
        print(f"{form}: json {tj:.0f} tokens, text {tt:.0f} tokens, saving {100 * (1 - tt / tj):.1f}%; pairs {pairs}, lost {lost}")
    print("every pair kept" if not lost_any else "LOSS or json mismatch found")
    sys.exit(1 if lost_any else 0)


if __name__ == "__main__":
    main()
