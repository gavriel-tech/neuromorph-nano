"""Where a failure starts (guide section 13), from a recording.

usage: python -I origins.py --rec <recording.json> [...] [--target first|pack] [--folds <folds.json> --part tune]
                            [--checks drone-check.json]

(a) The answer forced in by stage. For each question with answers, the answer pieces are placed by
hand into G2's candidates (G1's hand-over and queue; a piece with no G1 place goes last, as the
bench patch places it) and G2's selection is replayed (`replay.served`); then into G2's chosen set
(the pack is written from the set in G1's order). The earliest stage after which the answer
survives names where the loss starts:
  windows or G1: survives once forced into the candidates, and was not among them (windows: some
    answer range was in no window; G1: in the windows, not handed over or queued). Splitting these
    two needs the live program (`--force-answer windows`, scratch/replay/PATCH.md);
  G2 selection: lost when forced into the candidates, survives when forced into the set;
  pack order (G1's order): lost even when forced into the set (target first only: the set is
    written in G1's order and a rival comes first).
Target `first` (default): a right range is cited first; `pack`: every answer range is in the pack.

(b) Stage input checks, each a pass or fail count with question ids: the request's drone (the kind
the request is read as, ported from `nmpod::intake::Request::read`, against the question's kind;
parts, from the hand reading in drone-check.json), pieces (an answer range split across pieces, or
swamped: under a quarter of the piece holding it), intake (answer file missing, or no piece of it),
evidence (the answer pieces' best settle chance and their place among G2's candidates per part, by
file kind and material), after the pack (every answer range in the set but a rival first).
Standard library only.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def lib(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = lib("replay")


def args_after(argv, flag):
    return R.args_after(argv, flag)


# ---- the request's kind, as `nmpod::intake` reads it (jobs empty, as the bench asks) ----

def _words(q, keep_us):
    return [w.lower() for w in re.split(r"[^0-9A-Za-z_]+" if keep_us else r"[^0-9A-Za-z]+", q) if w]


def asks_callers(q):
    w = _words(q, False)
    pair = lambda a, b: any(w[i] == a and w[i + 1] == b for i in range(len(w) - 1))
    return any(x in w for x in ("calls", "caller", "callers", "invoked", "invokes")) or pair("called", "from") or pair("called", "by") or pair("started", "from")


def wording_kind(q):
    w = _words(q, True)
    starts = lambda a, b: len(w) > 1 and w[0] == a and w[1] in b
    if asks_callers(q):
        return "relation"
    if any(x in w for x in ("value", "default", "limit", "threshold", "maximum", "minimum", "timeout", "budget", "constant")) or starts("how", ("many", "deeply", "long", "much")):
        return "value"
    if "why" in w or starts("how", ("does", "do", "is", "are")) or "explain" in w:
        return "explain"
    if (starts("where", ("is", "are")) and any(x in w for x in ("defined", "implemented", "declared"))) or \
            starts("which", ("function", "type", "struct", "enum", "method", "trait", "routine", "module")) or \
            starts("what", ("function", "type", "struct", "enum", "method", "trait", "routine")):
        return "name"
    return "behaviour"


def request_kind(q):
    low = q.lower()
    words = _words(low, False)
    first = words[0] if words else ""
    st = low.lstrip().startswith
    wording = wording_kind(q)
    absent = st("is there") or st("are there") or st("is any") or st("does any") or " exist" in low
    trace_words = any(x in low for x in ("trace", "call path", "call chain", "end to end", "flow from"))
    if absent:
        return "absent-check"
    if trace_words or wording == "relation":
        return "trace"
    if wording == "value" or (first == "how" and len(words) > 1 and words[1] in ("large", "far", "big", "small", "often", "old")):
        return "value"
    if wording == "explain" or (wording == "behaviour" and first not in ("where", "which")):
        return "explain"
    return "locate"


# The request kinds that fit each question kind (dev kinds and large kinds).
FITS = {"locate": {"locate"}, "name": {"locate"}, "file": {"locate"}, "value": {"value"}, "explain": {"explain"},
        "behaviour": {"explain", "locate"}, "trace": {"trace", "explain"}, "relation": {"trace"},
        "documents": {"explain", "locate", "value", "trace"}, "decoy": None}


# ---- natural stages and the answer forced in ----

def by_key(q):
    return {c["key"]: c for c in q["candidates"]}


def every(q, keys):
    k = by_key(q)
    ps = [k[x] for x in keys if x in k]
    return bool(q["answers"]) and all(any(R.covers(c, a) for c in ps) for a in q["answers"])


def first_ok(q, keys):
    k = by_key(q)
    ps = sorted((k[x] for x in keys if x in k), key=lambda c: (c["g1"] if c["g1"] is not None else 10**9, c["key"]))
    return bool(ps) and any(R.covers(ps[0], a) for a in q["answers"])


def natural(q):
    """Where the answer is, stage by stage, in the recorded run."""
    s = q["served"]
    a, b = s["ranges_in_windows"]
    ans = [c for c in q["candidates"] if c["answer"]]
    in_c = bool(q["answers"]) and all(any(R.covers(c, x) and c["in_cands"] for c in ans) for x in q["answers"])
    some_c = any(c["in_cands"] for c in ans)
    return {"windows": b > 0 and a == b, "windows_some": a > 0, "cands": in_c, "cands_some": some_c,
            "set": every(q, s["set"]), "up": bool(s["every_up"]), "pack": every(q, s["set"]), "first": s["up_place"] == 1}


def survives(q, keys, target):
    return first_ok(q, keys) if target == "first" else every(q, keys)


def forced_cands(q):
    """G2's selection replayed with the answer pieces among its candidates."""
    cands = R.eligible(q)
    have = {c["key"] for c in cands}
    for c in q["candidates"]:
        if c["answer"] and c["key"] not in have:
            d = {k: v for k, v in c.items() if k != "answer"}
            d["g1"] = 10**6 + len(cands)
            cands.append(d)
    return R.served(q, cands, q["budget"])


def start(q, target):
    """Where the loss starts for one question: (label, forced into candidates survives, forced into set survives)."""
    nat = natural(q)
    s = q["served"]
    if survives(q, s["set"], target):
        return "no loss", True, True
    fc_keys = forced_cands(q)
    fc = survives_forced(q, fc_keys, target)
    fs = survives_forced(q, list(dict.fromkeys(s["set"] + [c["key"] for c in q["candidates"] if c["answer"]])), target)
    if not fs:
        # The set is written in G1's order: with no G1 place the answer cannot lead, so the loss is
        # where it lost its place (the windows or G1); with one, a rival G1 placed earlier leads.
        if not nat["cands_some"]:
            return ("G1 (hand-over and queue)" if nat["windows_some"] else "windows or G1"), fc, fs
        return "pack order (G1's order)", fc, fs
    if not fc:
        return "G2 selection", fc, fs
    if not nat["cands"]:
        return ("G1 (hand-over and queue)" if nat["windows"] else "windows or G1"), fc, fs
    return "replay differs (relation or fact columns)", fc, fs


def survives_forced(q, keys, target):
    """As `survives`, with a forced piece that has no G1 place written last."""
    return survives(q, keys, target)


# ---- stage input checks ----

def swamped(q):
    out = []
    for a in q["answers"]:
        holders = [c for c in q["candidates"] if c["answer"] and R.covers(c, a)]
        rl = a["last"] - a["first"] + 1
        if holders and not a["whole_file"] and all((c["last"] - c["first"] + 1) > 4 * rl for c in holders):
            out.append(a)
    return out


def split(q):
    return [a for a in q["answers"] if not a["whole_file"] and sum(1 for c in q["candidates"] if c["answer"] and R.covers(c, a)) >= 2]


def intake(q, root):
    out = []
    for a in q["answers"]:
        if not a.get("resolved", True) or (root and not (Path(root) / a["file"]).is_file()):
            out.append((a["file"], "file missing"))
        elif not any(c["answer"] and R.covers(c, a) for c in q["candidates"]):
            out.append((a["file"], "no piece holds the range"))
    return out


def best_s(c):
    return max(c["S"]) if c["S"] else 0.0


def answer_rank(q):
    """The best place of an answer piece among G2's candidates over the parts (from the diagnosis)."""
    ps = [v["first_answer"] for v in q.get("top", {}).values() if v.get("first_answer")]
    if not ps and q.get("_source") == "dump":
        for k in range(len(q["request"]["parts"])):
            order = sorted((c for c in q["candidates"] if c["in_cands"] and c["S"][k] > 0), key=lambda c: -c["S"][k])
            p = next((i + 1 for i, c in enumerate(order) if c["answer"]), None)
            if p:
                ps.append(p)
    return min(ps) if ps else None


def checks(qs, roots, judged):
    lines = []
    ans = [q for q in qs if q["answers"]]
    kind_bad = [q["id"] for q in qs if FITS.get(q["kind"]) and request_kind(q["question"]) not in FITS[q["kind"]]]
    kinds_seen = {}
    for q in qs:
        kinds_seen.setdefault((q["kind"], request_kind(q["question"])), []).append(q["id"])
    lines.append(f"request's drone, kind: pass {sum(1 for q in qs if FITS.get(q['kind'])) - len(kind_bad)}, fail {len(kind_bad)}: {' '.join(kind_bad) or '-'}")
    lines.append("  question kind -> request kind: " + "; ".join(f"{a} -> {b} {len(v)}" for (a, b), v in sorted(kinds_seen.items())))
    weak = [q["id"] for q in ans if q["request"]["parts"] and all(p["mass"] <= p["against"] for p in q["request"]["parts"])]
    some_weak = [q["id"] for q in ans if any(p["mass"] <= p["against"] for p in q["request"]["parts"])]
    lines.append(f"request's drone, parts below their content-free mass: every part {len(weak)} ({' '.join(weak) or '-'}); some part {len(some_weak)}")
    if judged:
        ids = {q["id"] for q in qs}
        j = [x for x in judged if x["id"] in ids]
        if j:
            pf = [x["id"] for x in j if not x["parts"]]
            kf = [x["id"] for x in j if not x["kind"]]
            lines.append(f"request's drone, hand-read {len(j)}: parts right {len(j) - len(pf)}, wrong {len(pf)} ({' '.join(pf) or '-'}); kind right {len(j) - len(kf)}, wrong {len(kf)} ({' '.join(kf) or '-'})")
    sp = [q["id"] for q in ans if split(q)]
    sw = [q["id"] for q in ans if swamped(q)]
    lines.append(f"pieces: answer range split across pieces {len(sp)} ({' '.join(sp) or '-'}); swamped (range under a quarter of its piece) {len(sw)} ({' '.join(sw) or '-'}); pass {len(ans) - len(set(sp) | set(sw))}")
    miss = [(q["id"], x) for q in ans for x in intake(q, roots.get(q["_set"]))]
    nowin = [q["id"] for q in ans if not natural(q)["windows"]]
    lines.append(f"intake: answer file missing or with no piece {len(miss)} ({' '.join(f'{i}:{f} ({w})' for i, (f, w) in miss) or '-'}); pass {len(ans) - len({i for i, _ in miss})}")
    lines.append(f"  (an answer range in no window: {len(nowin)} {' '.join(nowin) or '-'})")
    # Evidence by file kind and material.
    groups = {}
    for q in ans:
        for c in q["candidates"]:
            if not c["answer"]:
                continue
            g = (q["_set"].replace("-tune", ""), c["kind"])
            groups.setdefault(g, []).append((q, c))
    lines.append("evidence, answer pieces by material and file kind: n, median best settle chance, median best rival's, share where the answer piece is first on some part, share accepted on every part it holds")
    for g, v in sorted(groups.items()):
        bs = sorted(best_s(c) for _, c in v)
        rv = []
        for q, c in v:
            r = [best_s(x) for x in q["candidates"] if not x["answer"] and x["in_cands"]]
            rv.append(max(r) if r else 0.0)
        rv.sort()
        top = sum(1 for q, c in v if answer_rank(q) == 1)
        acc = sum(1 for _, c in v if all(x["accepted"] for x in c["cells"] if x and (x["s"] or 0) > 0))
        lines.append(f"  {g[0]:<11} {g[1]:<9} n {len(v):>3}  answer s {bs[len(bs) // 2]:.3f}  rival s {rv[len(rv) // 2]:.3f}  first on a part {top}/{len(v)}  accepted {acc}/{len(v)}")
    after = [q for q in ans if natural(q)["set"] and not natural(q)["first"]]
    lines.append(f"after the pack: every answer range in the set but a rival first {len(after)}: " + " ".join(
        f"{q['id']}(u{q['served']['up_place']})" for q in after))
    return lines


def roots_of(files):
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        out[d.get("name", Path(f).stem)] = d.get("root")
    return out


def main(argv):
    recs = args_after(argv, "--rec")
    if not recs:
        print(__doc__)
        return 2
    target = (args_after(argv, "--target") or ["first"])[0]
    folds, part = (args_after(argv, "--folds") or [None])[0], (args_after(argv, "--part") or [None])[0]
    qs = R.load(recs, folds, part)
    jf = (args_after(argv, "--checks") or [str(HERE / "drone-check.json")])[0]
    judged = json.loads(Path(jf).read_text(encoding="utf-8"))["judged"] if Path(jf).is_file() else []
    ans = [q for q in qs if q["answers"]]
    table = {}
    for q in ans:
        lab, fc, fs = start(q, target)
        table.setdefault(lab, []).append((q["id"], fc, fs))
    print(f"(a) the answer forced in by stage, target {target}, {len(ans)} questions with answers")
    for lab, v in sorted(table.items(), key=lambda x: (x[0] == "no loss", -len(x[1]))):
        print(f"  {lab:<44} {len(v):>3}  {' '.join(i for i, _, _ in v) if lab != 'no loss' else ''}")
    print()
    print("(b) stage input checks")
    for l in checks(qs, roots_of(recs), judged):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
