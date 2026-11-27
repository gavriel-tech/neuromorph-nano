"""The dictionary on the large set's tuning fold (code), from the dump's candidates.

Each candidate is rescored with the exam's scorer (rarity-weighted share of the request's recorded
stems the piece holds, rarity over the material's 30-line pieces) without and with dictionary
expansion (dictionary built from the whole material). Per answerable question: the best answer
piece's rank among the candidates before and after, the rivals a link raised above it, and a paired
permutation test on reciprocal rank. Also the yardstick: R@3 without expansion and its interval.

`python -I tests/replay/dict/tune.py [--alpha 0.01] [--correction bh] [--list]`
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lexicon", HERE / "lexicon.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

DUMP = L.ROOT / "scratch" / "replay" / "dump-p1-large-code" / "candidates.jsonl"
TUNE = L.ROOT / "tests" / "dev-large" / "questions-code-tune.json"


# The word mismatch losses of the large tuning fold, code part (build plan, "Step 4: the loss table",
# target first, 2026-11-17 03:05 UTC; doc19 is a document question and not in this dump).
MISMATCH = set("loc06 val04 exp05 trc06 loc10 loc12 trc09 val14 val15 val17 exp15 trc16 trc17 loc19 loc23 val19 val21 exp22 exp24".split())


def covers(c, a):
    return c["path"] == a["file"] and (a["whole_file"] or (c["first"] <= a["last"] and c["last"] >= a["first"]))


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.01
    corr = argv[argv.index("--correction") + 1] if "--correction" in argv else "bh"
    tune_ids = {q["id"] for q in json.loads(TUNE.read_text(encoding="utf-8"))["questions"]}
    qs = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if q["id"] in tune_ids and q["answers"]]
    files = L.material_files()
    ls = L.line_stems(files)
    _, sets = L.material_pieces(files, ls)
    idf = L.rarity(sets)
    t0 = time.perf_counter()
    items = [it for rel, lines in files.items() for it in L.doc_items(rel, lines)]
    links, info = L.build_dictionary(L.pairs(files, items), alpha=alpha, correction=corr, strata="--strata" in argv)
    print(f"dictionary on the whole material: {info['links']} links over {info['linked_words']} words, {time.perf_counter() - t0:.1f} s")

    def stems_range(path, first, last):
        s = set()
        for x in range(first - 1, min(last, len(ls.get(path, [])))):
            s |= ls[path][x]
        return s

    rows, missing = [], []
    for q in qs:
        stems = []
        for p in q["request"]["parts"]:
            for w in p["stems"]:
                if w not in stems:
                    stems.append(w)
        cands = q["candidates"]
        csets = [stems_range(c["path"], c["first"], c["last"]) for c in cands]
        ans_idx = [i for i, c in enumerate(cands) if any(covers(c, a) for a in q["answers"])]
        ans_text = set()
        for a in q["answers"]:
            if a["file"] in ls:
                ans_text |= stems_range(a["file"], a["first"], a["last"]) if not a["whole_file"] else stems_range(a["file"], 1, len(ls[a["file"]]))
        shared = any(w in ans_text for w in stems)
        if not ans_idx:
            missing.append(f"{q['id']} ({'shared' if shared else 'no stem shared'})")
            continue
        s0 = [L.score(stems, s, idf) for s in csets]
        s1 = [L.score(stems, s, idf, links) for s in csets]
        r0, r1 = L.mid_rank(s0, ans_idx), L.mid_rank(s1, ans_idx)
        b0, b1 = max(s0[i] for i in ans_idx), max(s1[i] for i in ans_idx)
        aset = set(ans_idx)
        raised = sorted({cands[i]["key"] for i in range(len(cands)) if i not in aset and s0[i] <= b0 and s1[i] > b1})
        rows.append({"id": q["id"], "kind": q["kind"], "n": len(cands), "shared": shared, "r0": r0, "r1": r1,
                     "raised": raised, "stems": stems,
                     "g1": min((cands[i].get("g1") or 10 ** 6) for i in ans_idx)})

    def line(name, sub):
        if not sub:
            return
        a = [r["r0"] for r in sub]
        b = [r["r1"] for r in sub]
        n = len(sub)
        f = lambda v, k: sum(x <= k for x in v) / n
        m0, m1 = sum(1 / x for x in a) / n, sum(1 / x for x in b) / n
        p = L.perm_test([1 / x for x in a], [1 / x for x in b])
        print(f"{name:34}{n:>4}  R@1 {f(a, 1):.3f}->{f(b, 1):.3f}  R@3 {f(a, 3):.3f}->{f(b, 3):.3f}  R@10 {f(a, 10):.3f}->{f(b, 10):.3f}"
              f"  MRR {m0:.3f}->{m1:.3f} (p {p:.4f})  better {sum(y < x for x, y in zip(a, b))} worse {sum(y > x for x, y in zip(a, b))}"
              f"  rivals raised {sum(len(r['raised']) for r in sub)}")

    print(f"answerable {len(qs)}; answer among the candidates {len(rows)}; not among them {len(missing)} {missing}")
    line("all answerable with a candidate", rows)
    line("no request stem in the answer", [r for r in rows if not r["shared"]])
    line("shares a stem", [r for r in rows if r["shared"]])
    line("word mismatch losses (19)", [r for r in rows if r["id"] in MISMATCH])
    line("all other answerable", [r for r in rows if r["id"] not in MISMATCH])
    for k in ("locate", "value", "explain", "trace"):
        line(f"kind {k}", [r for r in rows if r["kind"] == k])
    k3 = sum(r["r0"] <= 3 for r in rows)
    lo, hi = L.wilson(k3, len(rows))
    print(f"yardstick: tuning R@3 without expansion {k3}/{len(rows)} = {k3 / len(rows):.3f}, 95% Wilson [{lo:.3f}, {hi:.3f}], width {hi - lo:.3f}")
    if "--list" in argv:
        print("per question: id kind candidates shared | rank before -> after | rivals raised (first 3) | G1 place of answer")
        for r in rows:
            print(f"  {r['id']:6} {r['kind']:8} {r['n']:4} {'shared' if r['shared'] else 'NONE  '} | {r['r0']:6.1f} -> {r['r1']:6.1f} | "
                  f"{len(r['raised'])} {r['raised'][:3]} | g1 {r['g1']}")


if __name__ == "__main__":
    main(sys.argv[1:])
