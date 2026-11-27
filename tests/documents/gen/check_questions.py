"""Check ../questions/dev.json and ../questions/held-out.json against the corpus.

  python check_questions.py          check both files; prints every problem and a count
  python check_questions.py --show   also print the first and last line of every answer span

Checks: JSON shape, ids and kinds, every file exists under corpus/, every anchor occurs on exactly
one line of its file (UTF-16 and Windows-1252 notes are decoded properly), the anchor's line is
lines[0], lines[1] >= lines[0] and lies within the file, match is "lines" or "file", compare and
multihop questions have two or more answers, decoys have none, no question id repeats.
Standard library only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_outlines import CORPUS, read_lines  # noqa: E402

QDIR = os.path.join(HERE, "..", "questions")
KINDS = {"define", "lookup", "procedure", "explain", "compare", "summarise", "locate", "table",
         "temporal", "multihop", "decoy"}
SETS = [("dev.json", "x"), ("held-out.json", "h")]


def check(name, prefix, show):
    probs = []
    try:
        doc = json.load(open(os.path.join(QDIR, name), encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return ["%s: not valid JSON: %s" % (name, e)]
    if set(doc) != {"note", "questions"} or not isinstance(doc["note"], str) \
            or not isinstance(doc["questions"], list):
        return ["%s: top level must be exactly {note, questions}" % name]
    seen = set()
    for qn in doc["questions"]:
        qid = qn.get("id", "?")
        where = "%s %s" % (name, qid)
        if set(qn) != {"id", "kind", "question", "answers"}:
            probs.append("%s: keys must be id, kind, question, answers" % where)
            continue
        if not (isinstance(qid, str) and qid.startswith(prefix) and qid[1:].isdigit()):
            probs.append("%s: id must look like %s01" % (where, prefix))
        if qid in seen:
            probs.append("%s: id repeats" % where)
        seen.add(qid)
        if qn["kind"] not in KINDS:
            probs.append("%s: unknown kind %r" % (where, qn["kind"]))
        if not isinstance(qn["question"], str) or not qn["question"].strip():
            probs.append("%s: empty question" % where)
        answers = qn["answers"]
        if qn["kind"] == "decoy" and answers:
            probs.append("%s: a decoy must have no answers" % where)
        if qn["kind"] != "decoy" and not answers:
            probs.append("%s: no answers" % where)
        if qn["kind"] in ("compare", "multihop") and len(answers) < 2:
            probs.append("%s: needs two answers" % where)
        for a in answers:
            if set(a) != {"file", "anchor", "lines", "match"}:
                probs.append("%s: answer keys must be file, anchor, lines, match" % where)
                continue
            path = os.path.join(CORPUS, a["file"])
            if "\\" in a["file"] or not os.path.isfile(path):
                probs.append("%s: no such file %s" % (where, a["file"]))
                continue
            lines = read_lines(a["file"])
            hits = [i + 1 for i, l in enumerate(lines) if a["anchor"] in l]
            if len(hits) != 1:
                probs.append("%s: anchor %r is on %d lines of %s" % (where, a["anchor"], len(hits),
                                                                      a["file"]))
                continue
            lo_hi = a["lines"]
            if not (isinstance(lo_hi, list) and len(lo_hi) == 2
                    and all(isinstance(x, int) for x in lo_hi)):
                probs.append("%s: lines must be [first, last]" % where)
                continue
            first, last = lo_hi
            if first != hits[0]:
                probs.append("%s: anchor is on line %d, lines[0] is %d" % (where, hits[0], first))
            if last < first or last > len(lines):
                probs.append("%s: bad span %s (file has %d lines)" % (where, lo_hi, len(lines)))
            if a["match"] not in ("lines", "file"):
                probs.append("%s: match must be lines or file" % where)
            if show:
                print("%s [%s] %s:%d-%d" % (where, qn["kind"], a["file"], first, last))
                print("    > " + lines[first - 1][:110])
                if last != first:
                    print("    < " + lines[last - 1][:110])
    return probs


def main():
    show = "--show" in sys.argv
    total = 0
    for name, prefix in SETS:
        probs = check(name, prefix, show)
        for p in probs:
            print(p)
        print("%s: %d problem(s)" % (name, len(probs)))
        total += len(probs)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
