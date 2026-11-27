"""The organism writes its own exam, and the dictionary is tested on it across halves.

Every documented item gives a question (its doc comment's first sentence, backticked and
identifier-shaped words and the item's own name parts taken out; four content words or more) and
an answer (the item's line range). Files are split in two halves, seeded and stratified by crate;
the dictionary is built from one half's files and tested on the other half's exam, both ways.
Candidates are the material's 30-line pieces; each question's own doc comment lines are masked
out of the pieces while it is scored, so the comment cannot answer itself.

`python -I tests/replay/dict/exam.py [--alpha 0.01] [--correction bh] [--json out.json]`
"""
import importlib.util
import json
import random
import re
import sys
import time
from array import array
from operator import add
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lexicon", HERE / "lexicon.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SENT = re.compile(r"(?<=[.!?])\s+")


def first_sentence(doc):
    t = re.sub(r"\s+", " ", doc).strip()
    return SENT.split(t, 1)[0] if t else ""


def make_exam(files, items):
    exam = []
    for it in items:
        s = first_sentence(it["doc"])
        own = {L.stem(p) for p in L.name_parts(it["name"])}
        q = []
        for w in L.content_stems(L.plain(s)):
            if w not in own and w not in q:
                q.append(w)
        if len(q) >= 4:
            exam.append(dict(it, question=s, stems=q))
    return exam


def split_files(files):
    by = defaultdict(list)
    for rel in sorted(files):
        by[L.stratum(rel)].append(rel)
    rng = random.Random(L.SEED)
    half = {}
    flip = 0
    for k in sorted(by):
        fs = by[k][:]
        rng.shuffle(fs)
        for rel in fs:
            half[rel] = "AB"[flip]
            flip ^= 1
    return half


class Index:
    def __init__(self, files):
        self.files = files
        self.ls = L.line_stems(files)
        self.keys, self.sets = L.material_pieces(files, self.ls)
        self.P = len(self.keys)
        self.idf = L.rarity(self.sets)
        self.post = defaultdict(set)
        for i, s in enumerate(self.sets):
            for w in s:
                self.post[w].add(i)
        self.where = {}
        for i, (rel, a, b) in enumerate(self.keys):
            self.where[(rel, (a - 1) // L.PIECE)] = i

    def pieces_over(self, rel, first, last):
        """Piece indices overlapping 0-based lines first..last of a file."""
        return sorted({self.where[(rel, x // L.PIECE)] for x in range(first, last + 1) if (rel, x // L.PIECE) in self.where})

    def masked(self, rel, lines_out):
        """Piece index -> stem set with the given 0-based lines left out."""
        out = {}
        drop = set(lines_out)
        for i in self.pieces_over(rel, min(lines_out), max(lines_out)):
            _, a, b = self.keys[i]
            s = set()
            for x in range(a - 1, b):
                if x not in drop:
                    s |= self.ls[rel][x]
            out[i] = s
        return out


class Expander:
    """Per request stem w, the expansion credit of every piece: the best strength x
    min(rarity w, rarity c) over w's linked parts c the piece holds (cached)."""

    def __init__(self, ix, links):
        self.ix, self.links, self.cache = ix, links, {}

    def vec(self, w):
        v = self.cache.get(w)
        if v is None:
            v = array("f", bytes(4 * self.ix.P))
            r = self.ix.idf(w)
            for c, s in self.links.get(w, {}).items():
                val = s * min(r, self.ix.idf(c))
                for p in self.ix.post.get(c, ()):
                    if val > v[p]:
                        v[p] = val
            self.cache[w] = v
        return v


def score_all(ix, stems, exp, masked):
    P = ix.P
    tot = [0.0] * P
    den = 0.0
    for w in stems:
        r = ix.idf(w)
        den += r
        hit = ix.post.get(w, ())
        if exp is not None and w in exp.links:
            col = exp.vec(w).tolist()
            for p in hit:
                col[p] = r
            tot = list(map(add, tot, col))
        else:
            for p in hit:
                tot[p] += r
    for p, s in masked.items():
        tot[p] = L.score(stems, s, ix.idf, exp.links if exp is not None else None) * den
    return [t / den for t in tot] if den else tot


def summarize(ranks):
    n = len(ranks)
    return {"n": n, "r@1": sum(r <= 1 for r in ranks) / n, "r@3": sum(r <= 3 for r in ranks) / n,
            "r@10": sum(r <= 10 for r in ranks) / n, "mrr": sum(1 / r for r in ranks) / n}


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.01
    corr = argv[argv.index("--correction") + 1] if "--correction" in argv else "bh"
    files = L.material_files()
    t0 = time.perf_counter()
    ix = Index(files)
    items = [it for rel, lines in files.items() for it in L.doc_items(rel, lines)]
    exam = make_exam(files, items)
    half = split_files(files)
    print(f"index: {ix.P} pieces of {L.PIECE} lines, {time.perf_counter() - t0:.1f} s")
    print(f"exam: {len(exam)} questions from {len(items)} documented items; by file kind {dict(Counter(q['kind'] for q in exam))}")
    print(f"halves: files A {sum(v == 'A' for v in half.values())}, B {sum(v == 'B' for v in half.values())}; "
          f"questions A {sum(half[q['file']] == 'A' for q in exam)}, B {sum(half[q['file']] == 'B' for q in exam)}")
    rows = []
    for build_on, test_on in (("A", "B"), ("B", "A")):
        t1 = time.perf_counter()
        its = [it for it in items if half[it["file"]] == build_on]
        links, info = L.build_dictionary(L.pairs(files, its), alpha=alpha, correction=corr, strata="--strata" in argv)
        print(f"dictionary from half {build_on}: {info['links']} links over {info['linked_words']} words ({time.perf_counter() - t1:.1f} s)")
        exp = Expander(ix, links)
        for q in exam:
            if half[q["file"]] != test_on:
                continue
            ans = ix.pieces_over(q["file"], q["first"], q["last"])
            m = ix.masked(q["file"], q["doc_lines"])
            s0 = score_all(ix, q["stems"], None, m)
            s1 = score_all(ix, q["stems"], exp, m)
            shared = any(w in ix.sets[a] if a not in m else w in m[a] for a in ans for w in q["stems"])
            rows.append({"id": f"{q['file']}:{q['first'] + 1}", "half": test_on, "kind": q["kind"],
                         "question": q["question"], "stems": q["stems"], "shared": shared,
                         "r0": L.mid_rank(s0, ans), "r1": L.mid_rank(s1, ans)})
    print(f"scored {len(rows)} questions, {time.perf_counter() - t0:.1f} s in all")
    r0 = [r["r0"] for r in rows]
    r1 = [r["r1"] for r in rows]
    a, b = summarize(r0), summarize(r1)
    print(f"{'':28}{'n':>6}{'R@1':>8}{'R@3':>8}{'R@10':>8}{'MRR':>8}")
    for name, sel in (("all", lambda r: True), ("half B (dict A)", lambda r: r["half"] == "B"),
                      ("half A (dict B)", lambda r: r["half"] == "A"),
                      ("no stem shared with answer", lambda r: not r["shared"]), ("shares a stem", lambda r: r["shared"])):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        x, y = summarize([r["r0"] for r in sub]), summarize([r["r1"] for r in sub])
        for tag, z in (("without", x), ("with", y)):
            print(f"{name + ' ' + tag:28}{z['n']:>6}{z['r@1']:>8.3f}{z['r@3']:>8.3f}{z['r@10']:>8.3f}{z['mrr']:>8.3f}")
        p = L.perm_test([1 / r["r0"] for r in sub], [1 / r["r1"] for r in sub])
        up = sum(r["r1"] < r["r0"] for r in sub)
        dn = sum(r["r1"] > r["r0"] for r in sub)
        print(f"{'':28}reciprocal rank change {y['mrr'] - x['mrr']:+.4f}, permutation p {p:.4f}; rank better {up}, worse {dn}")
    k3 = sum(r <= 3 for r in r0)
    lo, hi = L.wilson(k3, len(r0))
    print(f"yardstick: exam R@3 without expansion {k3}/{len(r0)} = {k3 / len(r0):.3f}, 95% Wilson [{lo:.3f}, {hi:.3f}], width {hi - lo:.3f}")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps({"alpha": alpha, "correction": corr, "rows": rows}), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
