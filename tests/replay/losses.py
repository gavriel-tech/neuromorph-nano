"""Per lost question: where the loss shows, where it starts, the answer piece against the piece that
beat it, and a cause label; tables by starting stage and by cause, largest first (guide section 13).

usage: python -I losses.py --rec <recording.json> [...] [--target first|pack] [--folds <folds.json> --part tune]
                           [--list] [--checks drone-check.json]

A question is lost when the recorded served choice misses the target: `first` (default, a right
range cited first) or `pack` (every answer range in the pack). Where it shows: the first stage the
recorded run lost the answer at (windows: an answer range in no window; G1: not every range among
G2's candidates, G1's hand-over and queue; G2: not every range in the chosen set; first: in the set
but a rival written first). Where it starts: `origins.start` (the answer forced in by stage).

The piece that beat it: for target first, the piece written first; for target pack, the set piece
with the largest settle chance on the part the answer piece holds best. The answer piece is the
answer piece with the largest settle chance on any part.

Cause labels, first that applies:
  file missing from intake  an answer file not in the material, or no piece holds an answer range;
  request parts wrong       the hand reading (drone-check.json) found the parts wrong, the request's
                            kind does not fit the question, or every part is below its content-free mass;
  relation not followed     a lost answer range whose pieces hold no word of the request (settle
                            chance under 0.02 on every part): only a relation reaches it (target pack);
  notes or docs beat code   the answer is in code and the piece that beat it is a document or notes;
  answer split across pieces an answer range spread over two pieces or more, each weaker than the beater;
  room taken by rivals      G2 stopped at the budget with the answer among its candidates, valued, not taken;
  word mismatch             the answer piece's settle chance is below the beater's on the part it holds
                            best, or it holds under half of every part's stems;
  other (G1's order)        target first: the answer piece is in the set and settles its best part at
                            least as well as the piece written first, but G1's order puts that piece first;
  other                     none of these.
Standard library only.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def lib(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = lib("replay")
O = lib("origins")
REC = lib("record")
CODE_KINDS = ("code", "test", "config")


def lost(q, target):
    return not O.survives(q, q["served"]["set"], target)


def shows(q, target):
    n = O.natural(q)
    if not n["windows"]:
        return "windows"
    if not n["cands"]:
        return "G1 (hand-over and queue)"
    if not n["set"]:
        return "G2 selection"
    if target == "first" and not O.first_ok(q, q["served"]["set"]):
        return "first (rival written first)"
    return "pack"


def answer_piece(q):
    a = [c for c in q["candidates"] if c["answer"]]
    return max(a, key=lambda c: (max(c["S"]) if c["S"] else 0.0, c["in_cands"])) if a else None


def beater(q, target, ans):
    by = O.by_key(q)
    set_p = [by[k] for k in q["served"]["set"] if k in by]
    ordered = sorted(set_p, key=lambda c: (c["g1"] if c["g1"] is not None else 10**9, c["key"]))
    if target == "first":
        return next((c for c in ordered[:1] if not c["answer"]), None)
    if ans is None or not ans["S"]:
        return next((c for c in ordered if not c["answer"]), None)
    k = max(range(len(ans["S"])), key=lambda i: ans["S"][i])
    rivals = [c for c in set_p if not c["answer"]]
    return max(rivals, key=lambda c: c["S"][k]) if rivals else None


def lost_ranges(q):
    by = O.by_key(q)
    set_p = [by[k] for k in q["served"]["set"] if k in by]
    return [a for a in q["answers"] if not any(R.covers(c, a) for c in set_p)]


def cause(q, target, ans, beat, judged, root):
    if O.intake(q, root):
        return "file missing from intake"
    j = judged.get(q["id"])
    fits = O.FITS.get(q["kind"])
    parts = q["request"]["parts"]
    if (j is not None and not j["parts"]) or (fits and O.request_kind(q["question"]) not in fits) or (parts and all(p["mass"] <= p["against"] for p in parts)):
        return "request parts wrong"
    if target == "pack":
        for a in lost_ranges(q):
            hold = [c for c in q["candidates"] if c["answer"] and R.covers(c, a)]
            if hold and all(max(c["S"] or [0.0]) < 0.02 for c in hold) and len(q["answers"]) > 1:
                return "relation not followed"
    answer_in_code = all(REC.file_kind(a["file"]) in CODE_KINDS for a in q["answers"])
    if beat is not None and answer_in_code and beat["kind"] in ("document", "notes"):
        return "notes or docs beat code"
    bs = lambda c: max(c["S"]) if c and c["S"] else 0.0
    if ans is not None and beat is not None:
        k = max(range(len(ans["S"])), key=lambda i: ans["S"][i]) if ans["S"] else 0
        weaker = (ans["S"][k] if ans["S"] else 0.0) < (beat["S"][k] if beat["S"] else 0.0)
    else:
        weaker = True
    sp = O.split(q)
    if sp and weaker:
        hold = [c for a in sp for c in q["candidates"] if c["answer"] and R.covers(c, a)]
        if all(bs(c) < bs(beat) for c in hold):
            return "answer split across pieces"
    if target == "pack" and q["served"]["stop"] == "budget" and ans is not None and ans["in_cands"] and bs(ans) > 0 and ans["key"] not in q["served"]["set"]:
        return "room taken by rivals"
    if ans is None:
        return "other"
    if weaker or all(x < 0.5 for x in ans["share"]):
        return "word mismatch"
    if target == "first" and ans["key"] in q["served"]["set"]:
        return "other (G1's order)"
    return "other"


def fmt(c, n):
    if c is None:
        return "-"
    s = ",".join(f"{x:.2f}" for x in c["S"][:n])
    return f"{c['key']} [{c['kind']}, {c['role']}, t{c['tokens']}, g{c['g1'] if c['g1'] is not None else '-'}, s {s}]"


def report(qs, target, judged, roots, listing):
    ans_q = [q for q in qs if q["answers"]]
    rows = []
    for q in ans_q:
        if not lost(q, target):
            continue
        a = answer_piece(q)
        b = beater(q, target, a)
        st, fc, fs = O.start(q, target)
        rows.append({"id": q["id"], "kind": q["kind"], "shows": shows(q, target), "starts": st,
                     "cause": cause(q, target, a, b, judged, roots.get(q["_set"])), "answer": a, "beater": b, "n": len(q["request"]["parts"])})
    out = [f"target {target}: lost {len(rows)} of {len(ans_q)} questions with answers"]

    def tab(title, key):
        out.append("")
        out.append(title)
        g = {}
        for r in rows:
            g.setdefault(r[key], []).append(r["id"])
        for k, v in sorted(g.items(), key=lambda x: -len(x[1])):
            out.append(f"  {k:<34} {len(v):>3}  {' '.join(v)}")
    tab("where it shows", "shows")
    tab("where it starts", "starts")
    tab("cause", "cause")
    out.append("")
    out.append("starting stage by cause (counts)")
    causes = sorted({r["cause"] for r in rows}, key=lambda c: -sum(1 for r in rows if r["cause"] == c))
    starts = sorted({r["starts"] for r in rows}, key=lambda s: -sum(1 for r in rows if r["starts"] == s))
    out.append("  " + " " * 34 + "".join(f"{c[:12]:>13}" for c in causes))
    for s in starts:
        out.append(f"  {s:<34}" + "".join(f"{sum(1 for r in rows if r['starts'] == s and r['cause'] == c):>13}" for c in causes))
    if listing:
        out.append("")
        out.append("per question: id kind | shows | starts | cause; answer piece; beaten by")
        for r in rows:
            out.append(f"{r['id']:<6} {r['kind']:<9} | {r['shows']} | {r['starts']} | {r['cause']}")
            out.append(f"         answer   {fmt(r['answer'], r['n'])}")
            out.append(f"         beaten by {fmt(r['beater'], r['n'])}")
    return out


def main(argv):
    recs = R.args_after(argv, "--rec")
    if not recs:
        print(__doc__)
        return 2
    targets = R.args_after(argv, "--target") or ["first"]
    folds, part = (R.args_after(argv, "--folds") or [None])[0], (R.args_after(argv, "--part") or [None])[0]
    qs = R.load(recs, folds, part)
    jf = (R.args_after(argv, "--checks") or [str(HERE / "drone-check.json")])[0]
    judged = {x["id"]: x for x in json.loads(Path(jf).read_text(encoding="utf-8"))["judged"]} if Path(jf).is_file() else {}
    roots = O.roots_of(recs)
    for t in targets:
        for l in report(qs, t, judged, roots, "--list" in argv):
            print(l)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
