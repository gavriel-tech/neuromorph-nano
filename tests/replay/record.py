"""Build a selection recording from a bench diagnosis (`bench --mode board --paths stack --why-r`).

usage: python -I record.py --diag <diag.txt> --questions <questions.json> --root <material>
                           [--plain <plain board report>] --name <set name> --out <recording.json>
       python -I record.py --dump <candidates.jsonl> [--plain <plain board report>] --name <set name> --out <recording.json>

A recording holds, per question, what reaches G2's selection: the request's parts (text, role,
settle mass against the content-free mass) and relations, the budget, each candidate piece (path,
line range, tokens, its place in G1's order, and per part: the drones' part chance `r`, the settle
chance `s` G2 uses with `--r settle`, the epistemic tag, the share of the part's stems it holds,
the three holdings `h` and the content-free pose mass `p`, accepted or not), the answer ranges
(anchors resolved as the bench does), and the served choice (G2's set in its order, its stop
reason, the gate, the place of the first answer in what went up, every range up, the answer's
place in G1's hand-over, the ranges in the windows). With `--plain`, the plain served path's
marks per question from a board report of the same program (`--paths plain`).

The diagnosis prints only part of G2's candidates: the chosen set, the answer pieces and the four
strongest per part. Every other candidate is missing, and a printed piece's value on a part where
it is not among the four is unknown (recorded as null). `source: "diag"` marks such a recording;
the bench's `--dump-candidates` (scratch/replay/PATCH.md) writes the full form, `source: "dump"`.
Standard library only.
"""
import json
import re
import sys
from pathlib import Path

DOC_EXT = {"md", "markdown", "mdx", "txt", "text", "rst", "adoc", "asciidoc", "org", "csv", "tsv", "tab", "html", "htm", "xhtml", "xml",
           "json", "jsonl", "ndjson", "ipynb", "yaml", "yml", "log", "pdf", "rtf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt",
           "ods", "odp", "epub"}
DOC_NAMES = {"readme", "license", "licence", "changelog", "notice", "authors", "copying", "todo", "install"}


def is_document(path):
    """`nmtype::is_document`, the same rule."""
    name = path.split(":")[0].rsplit("/", 1)[-1].lower()
    ext = name.rsplit(".", 1)[1] if "." in name else ""
    return ext in DOC_EXT or name in DOC_NAMES


def file_kind(path):
    """code, document, notes (internal notes and records), config, test."""
    p = path.lower()
    if p.startswith(("internal/", "runs/", "private/")):
        return "notes"
    if is_document(p):
        return "config" if p.endswith((".json", ".jsonl", ".yaml", ".yml", ".toml")) else "document"
    if p.endswith(".toml"):
        return "config"
    if "/tests/" in p or p.startswith("tests/"):
        return "test"
    return "code"


def args_after(argv, flag):
    if flag not in argv:
        return []
    out = []
    for x in argv[argv.index(flag) + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def decode(b):
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def load_questions(qfile, root):
    """The bench's loader: each answer range moved to where its anchor line is now."""
    d = json.loads(Path(qfile).read_text(encoding="utf-8"))
    out = []
    for q in d["questions"]:
        answers = []
        for a in q.get("answers", []):
            span = a["lines"][1] - a["lines"][0]
            f = Path(root) / a["file"]
            first, note = None, ""
            if f.is_file():
                hits = [i + 1 for i, l in enumerate(decode(f.read_bytes()).splitlines()) if a["anchor"] in l]
                if len(hits) == 1:
                    first = hits[0]
                else:
                    note = f"anchor occurs {len(hits)} times"
            else:
                note = "file not in the material"
            answers.append({"file": a["file"], "first": first if first else a["lines"][0], "last": (first if first else a["lines"][0]) + span,
                            "whole_file": a.get("match") == "file", "resolved": first is not None, "note": note})
        out.append({"id": q["id"], "kind": q["kind"], "question": q["question"], "answers": answers})
    return out


CELL = re.compile(r"^(-?[\d.]+|nan)/(-?[\d.]+|nan)/(\w+)/(-?[\d.]+|nan)(?:/(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)/p(-?[\d.]+|-?inf))?(/na)?$")
PIECE = re.compile(r"^(\*?)(.+?):(\d+)-(\d+) t(\d+)(?: g(\d+|-))?(?: set ([01]))? \[(.*)\]$")


def cell(text):
    m = CELL.match(text.strip())
    if not m:
        return None
    f = lambda x: float(x) if x not in (None, "nan") else None
    return {"r": f(m.group(1)), "s": f(m.group(2)), "tag": m.group(3), "share": f(m.group(4)),
            "h": [f(m.group(5)), f(m.group(6)), f(m.group(7))] if m.group(5) is not None else None,
            "p": f(m.group(8)) if m.group(8) is not None else None, "accepted": m.group(9) is None}


def piece(text):
    m = PIECE.match(text.strip())
    if not m:
        return None
    return {"star": m.group(1) == "*", "path": m.group(2), "first": int(m.group(3)), "last": int(m.group(4)), "tokens": int(m.group(5)),
            "g1": None if m.group(6) in (None, "-") else int(m.group(6)), "in_set": None if m.group(7) is None else m.group(7) == "1",
            "cells": [cell(c) for c in m.group(8).split()] if m.group(8).strip() else []}


HEAD = re.compile(r"^(\S+)\s+(\S+)\s+stack u(\S+) e([01]) (\S+) w(\d+)/(\d+) h(\S+)(?: g2 (\S+) set (\d+) up (\d+))?")
PART = re.compile(r"k(\d+) (.*?) \((\w+), mass (-?[\d.]+|-?inf) against (-?[\d.]+|-?inf) nats\)")


def num(x):
    return None if x == "-" else int(x)


def parse_diag(text):
    """Per question id: the header marks and the diagnosis lines, parsed."""
    out, cur = {}, None
    on = False
    for line in text.splitlines():
        if line.startswith("per question:"):
            on = True
            continue
        if not on or not line.strip() or line.startswith("written:"):
            continue
        if not line.startswith(" "):
            m = HEAD.match(line)
            if not m:
                cur = None
                continue
            cur = {"id": m.group(1), "kind": m.group(2), "up_place": num(m.group(3)), "every_up": m.group(4) == "1", "window": m.group(5),
                   "ranges_in_windows": [int(m.group(6)), int(m.group(7))], "hand_place": num(m.group(8)), "gate": m.group(9),
                   "set_size": int(m.group(10)) if m.group(10) else 0, "up_size": int(m.group(11)) if m.group(11) else 0,
                   "parts": [], "relations": [], "set": [], "stop": None, "ans": [], "top": {}}
            out[cur["id"]] = cur
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("parts:"):
            body, _, rel = s[len("parts:"):].rpartition("; relations:")
            for m in PART.finditer(body):
                fl = lambda x: float(x)
                cur["parts"].append({"text": m.group(2), "role": m.group(3), "mass": fl(m.group(4)), "against": fl(m.group(5))})
            cur["relations"] = [r.strip() for r in rel.split(",") if r.strip()]
        elif s.startswith("set ("):
            m = re.match(r"set \(([^)]*)\):\s*(.*)$", s)
            cur["stop"] = m.group(1)
            cur["set"] = [p for p in (piece(x) for x in m.group(2).split(" | ")) if p] if m.group(2).strip() else []
        elif s.startswith("ans "):
            p = piece(s[4:])
            if p:
                cur["ans"].append(p)
        elif s.startswith("lost "):
            continue
        else:
            m = re.match(r"k(\d+) (.*?): positive (\d+), first answer at (\S+):\s*(.*)$", s)
            if m:
                k = int(m.group(1))
                cur["top"][k] = {"positive": int(m.group(3)), "first_answer": num(m.group(4)),
                                 "pieces": [p for p in (piece(x) for x in m.group(5).split(" | ")) if p] if m.group(5).strip() else []}
    return out


def parse_plain(text):
    """Plain path marks per question from a board report (`per question:` lines)."""
    rows, on = {}, False
    for line in text.splitlines():
        if line.startswith("per question:"):
            on = True
            continue
        if not on or not line.strip() or line.startswith((" ", "written:")):
            continue
        m = re.search(r"\bplain ([01]{5})([wl]*)", line)
        if m:
            b = m.group(1)
            rows[line.split()[0]] = {"first": b[0] == "1", "top3": b[1] == "1", "pack": b[2] == "1", "every": b[3] == "1", "cited": b[4] == "1",
                                     "withheld": "w" in m.group(2), "low": "l" in m.group(2)}
    return rows


def covers(c, a):
    return c["path"] == a["file"] and (a["whole_file"] or (c["first"] <= a["last"] and c["last"] >= a["first"]))


def build(diag, questions, plain, name, root, program, budget):
    recs = []
    for q in questions:
        d = diag.get(q["id"])
        if d is None:
            continue
        n_p = len(d["parts"])
        cands = {}

        def add(p, k=None):
            key = f"{p['path']}:{p['first']}-{p['last']}"
            c = cands.get(key)
            if c is None:
                c = cands[key] = {"key": key, "path": p["path"], "first": p["first"], "last": p["last"], "tokens": p["tokens"], "g1": p["g1"],
                                  "kind": file_kind(p["path"]), "document": is_document(p["path"]), "cells": [None] * n_p}
            if c["g1"] is None and p["g1"] is not None:
                c["g1"] = p["g1"]
            if k is None:
                for i, x in enumerate(p["cells"][:n_p]):
                    c["cells"][i] = x
            elif p["cells"] and k < n_p:
                c["cells"][k] = p["cells"][0]
            return key

        set_keys = [add(p) for p in d["set"]]
        for p in d["ans"]:
            add(p)
        for k, t in d["top"].items():
            for p in t["pieces"]:
                add(p, k)
        for c in cands.values():
            c["answer"] = any(covers(c, a) for a in q["answers"])
            c["in_cands"] = c["g1"] is not None
            c["partial"] = any(x is None for x in c["cells"])
            tags = [x["tag"] for x in c["cells"] if x]
            order = ["defines", "uses", "tests", "documents", "mentions"]
            c["role"] = min(tags, key=order.index) if tags and all(t in order for t in tags) else (tags[0] if tags else None)
        answer_pieces = [f"{p['path']}:{p['first']}-{p['last']}" for p in d["ans"]]
        rec = {
            "id": q["id"], "kind": q["kind"], "question": q["question"], "answers": q["answers"],
            "request": {"parts": d["parts"], "relations": d["relations"]},
            "budget": budget, "band": None,
            "served": {"set": set_keys, "stop": d["stop"], "gate": d["gate"], "set_size": d["set_size"], "up_size": d["up_size"],
                       "up_place": d["up_place"], "every_up": d["every_up"], "hand_place": d["hand_place"], "window": d["window"],
                       "ranges_in_windows": d["ranges_in_windows"], "up": None},
            "answer_pieces": answer_pieces,
            "top": {str(k): {"positive": t["positive"], "first_answer": t["first_answer"]} for k, t in d["top"].items()},
            "candidates": sorted(cands.values(), key=lambda c: (c["g1"] is None, c["g1"] or 0, c["key"])),
        }
        if plain and q["id"] in plain:
            rec["plain"] = plain[q["id"]]
        recs.append(rec)
    return {"source": "diag", "name": name, "program": program, "root": str(root), "budget": budget, "questions": recs}


ROLE_ORDER = ["defines", "uses", "tests", "documents", "mentions"]


def from_dump(lines, plain, name):
    """A recording from the bench's `--dump-candidates` lines (scratch/replay/PATCH.md): every candidate."""
    recs = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        cands = d.get("candidates", [])
        for c in cands:
            c["kind"] = file_kind(c["path"])
            c["document"] = is_document(c["path"])
            c["answer"] = any(covers(c, a) for a in d["answers"])
            c["in_cands"] = c.get("g1") is not None
            c["partial"] = False
            tags = [x["tag"] for x in c["cells"] if x]
            c["role"] = min(tags, key=ROLE_ORDER.index) if tags and all(x in ROLE_ORDER for x in tags) else (tags[0] if tags else None)
        served = d.get("served", {})
        by = {c["key"]: c for c in cands}
        up = served.get("up", [])
        hit = lambda k: k in by and by[k]["answer"]
        g1_order = sorted((c for c in cands if c["in_cands"]), key=lambda c: c["g1"])
        hand = [c for c in g1_order if c.get("handover")]
        ranges_in = sum(1 for a in d["answers"] if any(covers(c, a) and c.get("window") is not None for c in cands))
        every = lambda keys: bool(d["answers"]) and all(any(covers(by[k], a) for k in keys if k in by) for a in d["answers"])
        rec = {
            "id": d["id"], "kind": d["kind"], "question": d["question"], "answers": d["answers"],
            "request": d.get("request", {"parts": [], "relations": []}), "budget": d.get("budget", 4000), "band": d.get("band"),
            "facts": d.get("facts", []), "rel_pairs": d.get("rel_pairs", []),
            "served": {"set": served.get("set", []), "stop": served.get("stop"), "gate": served.get("gate"), "set_size": len(served.get("set", [])),
                       "up_size": len(up), "up_place": next((i + 1 for i, k in enumerate(up) if hit(k)), None), "every_up": every(up),
                       "hand_place": next((i + 1 for i, c in enumerate(hand) if c["answer"]), None), "window": None,
                       "ranges_in_windows": [ranges_in, len(d["answers"])], "up": up, "rivals": served.get("rivals", []),
                       "pack_cited": served.get("pack_cited", []), "every_hand": every([c["key"] for c in hand])},
            "answer_pieces": [c["key"] for c in cands if c["answer"]],
            "top": {},
            "candidates": cands,
        }
        if plain and d["id"] in plain:
            rec["plain"] = plain[d["id"]]
        recs.append(rec)
    return {"source": "dump", "name": name, "program": None, "root": None, "budget": 4000, "questions": recs}


def main(argv):
    dump_f = (args_after(argv, "--dump") or [None])[0]
    if dump_f:
        out = (args_after(argv, "--out") or [None])[0]
        plain_f = (args_after(argv, "--plain") or [None])[0]
        plain = parse_plain(Path(plain_f).read_text(encoding="utf-8", errors="replace")) if plain_f else None
        rec = from_dump(Path(dump_f).read_text(encoding="utf-8").splitlines(), plain, (args_after(argv, "--name") or ["set"])[0])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        print(f"{out}: {len(rec['questions'])} questions from a dump")
        return 0
    diag_f = (args_after(argv, "--diag") or [None])[0]
    qfile = (args_after(argv, "--questions") or [None])[0]
    root = (args_after(argv, "--root") or [None])[0]
    out = (args_after(argv, "--out") or [None])[0]
    name = (args_after(argv, "--name") or ["set"])[0]
    if not (diag_f and qfile and root and out):
        print(__doc__)
        return 2
    text = Path(diag_f).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"program ([0-9a-f]{64})", text)
    plain_f = (args_after(argv, "--plain") or [None])[0]
    plain = parse_plain(Path(plain_f).read_text(encoding="utf-8", errors="replace")) if plain_f else None
    rec = build(parse_diag(text), load_questions(qfile, root), plain, name, root, m.group(1) if m else None, 4000)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
    n = len(rec["questions"])
    partial = sum(c["partial"] for q in rec["questions"] for c in q["candidates"])
    total = sum(len(q["candidates"]) for q in rec["questions"])
    unresolved = [q["id"] for q in rec["questions"] if any(not a["resolved"] for a in q["answers"])]
    print(f"{out}: {n} questions, {total} candidates recorded ({partial} with a part value unknown); answers unresolved: {' '.join(unresolved) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
