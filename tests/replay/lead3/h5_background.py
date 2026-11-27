"""H5: each piece's own background level, read off the organism's own exam.

Background questions: the exam's questions (tests/replay/dict/exam.py make_exam over the frozen code
material), turned into request parts the way the request drone makes them: the ported part former
(common.drone_parts, the drone's frame word list and run rules) over the plain words of the comment's
first sentence (lexicon.plain: backticked and identifier-shaped tokens out), each word stemmed by the
nmlex stem port, the item's own name parts removed (as the exam builder does), empty parts dropped.
A seeded sample (seed 20261117) of N questions.

Scorer (the same at the background and at the real question): a piece's evidence is the mean over the
request's parts of log((eps + hold_k) / eps), the conjunction rule written as a log likelihood ratio
against the floor eps; ranking by it within a question is the same as ranking by the conjunction.
At a background question hold_k is the rarity-weighted share of part k's stems the piece holds (the
exam's rarity over the material's 30-line pieces), the question's own comment lines masked out of any
piece that overlaps them. At the real question hold_k is the dump's recorded hold (cells[k].h[2]); a
check line shows how closely the Python hold reproduces it.

Background level of a piece (each candidate of the dump present in the material): CSLS form = mean of
its 10 highest background scores; background-mean form = mean over all N. Corrected evidence at the
real question = mean part LLR - background level ("csls2": the CSLS paper's own form 2 s - r, i.e.
the level at half weight). Ranked alone, and fused with file mass in place of
the conjunction (the best ordering's form).

`python -I tests/replay/lead3/h5_background.py`
"""
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as M  # noqa: E402

L = M.load_lexicon()
X = M.load_exam_module()


def build():
    t0 = time.perf_counter()
    files = L.material_files()
    ix = X.Index(files)
    items = [it for rel, lines in files.items() for it in L.doc_items(rel, lines)]
    exam = X.make_exam(files, items)
    print(f"material: {ix.P} pieces, exam {len(exam)} questions ({time.perf_counter() - t0:.1f} s)")
    return files, ix, exam


def exam_parts(q):
    own = {L.stem(p) for p in L.name_parts(q["name"])}
    parts = []
    for st in M.drone_parts(L.plain(q["question"]), L.stem):
        st = [w for w in st if w not in own and L.keep(w)]
        if st:
            parts.append(st)
    return parts


def piece_sets(ix, keys):
    """key -> stem set from the material lines first..last (1-based)."""
    out = {}
    for key, (path, a, b) in keys.items():
        if path in ix.ls:
            s = set()
            for x in range(a - 1, min(b, len(ix.ls[path]))):
                s |= ix.ls[path][x]
            out[key] = s
    return out


def background(ix, exam_sample, cand, eps):
    """Per candidate key, the list of its background scores (nonzero only) over the sample."""
    keys = list(cand)
    pos = {k: i for i, k in enumerate(keys)}
    post = defaultdict(list)
    for k, s in cand.items():
        for w in s:
            post[w].append(pos[k])
    byfile = defaultdict(list)
    for k in keys:
        byfile[k.rpartition(":")[0]].append(k)
    scores = [[] for _ in keys]
    for q, parts in exam_sample:
        drop = set(q["doc_lines"])
        masked = {}
        for k in byfile.get(q["file"], ()):
            path, a, b = span(k)
            if any((x in drop) for x in range(a - 1, b)):
                s = set()
                for x in range(a - 1, min(b, len(ix.ls[path]))):
                    if x not in drop:
                        s |= ix.ls[path][x]
                masked[pos[k]] = s
        tot = defaultdict(float)
        K = len(parts)
        for st in parts:
            den = sum(ix.idf(w) for w in st)
            h = defaultdict(float)
            for w in st:
                r = ix.idf(w) / den
                for p in post.get(w, ()):
                    h[p] += r
            for p, s in masked.items():
                h[p] = sum(ix.idf(w) for w in st if w in s) / den
            for p, v in h.items():
                if v > 0:
                    tot[p] += math.log((eps + v) / eps) / K
        for p, v in tot.items():
            if v > 0:
                scores[p].append(v)
    return {k: scores[pos[k]] for k in keys}


def span(key):
    path, _, rng = key.rpartition(":")
    a, _, b = rng.partition("-")
    return path, int(a), int(b or a)


def levels(bg, N, topk):
    csls, mean = {}, {}
    for k, v in bg.items():
        v = sorted(v, reverse=True)
        top = v[:topk] + [0.0] * max(0, topk - len(v))
        csls[k] = sum(top) / topk
        mean[k] = sum(v) / N
    return csls, mean


def llr(c, K, eps, py=None):
    if py is not None:
        return sum(math.log((eps + py[k]) / eps) for k in range(K)) / K
    return sum(math.log((eps + M.hold(c, k)) / eps) for k in range(K)) / K


def order_corr(C, K, eps, lev):
    return sorted(C, key=lambda c: (-round(llr(c, K, eps) - (lev.get(c["key"], 0.0) if lev else 0.0), 9), -c["ssum"]))


def main():
    files, ix, exam = build()
    rows, Q, D = M.load()
    keys = {}
    for q, a, C, K in Q:
        for c in C:
            keys[c["key"]] = (c["path"], c["first"], c["last"])
    cand = piece_sets(ix, keys)
    print(f"dump candidate pieces: {len(keys)} distinct, {len(cand)} present in the frozen material")
    # check: Python hold against the dump's recorded hold
    diffs = []
    for q, a, C, K in Q:
        P = q["request"]["parts"]
        for c in C:
            s = cand.get(c["key"])
            if s is None:
                continue
            for k in range(min(K, len(c["cells"]))):
                st = P[k]["stems"]
                den = sum(ix.idf(w) for w in st) or 1
                diffs.append(abs(sum(ix.idf(w) for w in st if w in s) / den - M.hold(c, k)))
    diffs.sort()
    print(f"Python hold vs dump hold: {len(diffs)} cells, within 0.02: {sum(d <= .02 for d in diffs) / len(diffs):.3f}, "
          f"within 0.1: {sum(d <= .1 for d in diffs) / len(diffs):.3f}, median abs diff {diffs[len(diffs) // 2]:.4f}")
    rng = random.Random(M.SEED)
    order = list(range(len(exam)))
    rng.shuffle(order)
    ex_all = [(exam[i], exam_parts(exam[i])) for i in order]
    ex_all = [(q, p) for q, p in ex_all if p]
    npart = [len(p) for _, p in ex_all]
    print(f"exam questions with parts: {len(ex_all)}; parts per question mean {sum(npart) / len(npart):.2f} "
          f"(dump: {sum(K for _, _, _, K in Q) / len(Q):.2f})")
    G1 = M.per_question(Q, M.order_g1)
    BEST = M.per_question(Q, M.order_best)
    out = [M.fmt_summary("G1 order (served)", G1), M.fmt_summary("best: conj fused with file mass (k10)", BEST)]
    pr = out.append
    cache = {}

    def run(N, eps, topk, form):
        if (N, eps) not in cache:
            t = time.perf_counter()
            cache[(N, eps)] = background(ix, ex_all[:N], cand, eps)
            print(f"  background N {N} eps {eps}: {time.perf_counter() - t:.1f} s", flush=True)
        csls, mean = levels(cache[(N, eps)], N, topk)
        lev = csls if form.startswith("csls") else mean
        if form.endswith("2"):  # the source's own form, 2 s - r(piece): the level enters at half weight
            lev = {k: v / 2 for k, v in lev.items()}
        A = M.per_question(Q, lambda q, a, C, K: order_corr(C, K, eps, None))
        B = M.per_question(Q, lambda q, a, C, K: order_corr(C, K, eps, lev))
        FA = M.per_question(Q, lambda q, a, C, K: M.rrf(C, [order_corr(C, K, eps, None), M.order_filemass(C)], 10))
        FB = M.per_question(Q, lambda q, a, C, K: M.rrf(C, [order_corr(C, K, eps, lev), M.order_filemass(C)], 10))
        return A, B, FA, FB, lev

    configs = [(1000, 0.2, 10, "csls"), (1000, 0.2, 10, "mean"),
               (500, 0.2, 10, "csls"), (2000, 0.2, 10, "csls"),
               (1000, 0.2, 5, "csls"), (1000, 0.2, 20, "csls"),
               (1000, 0.1, 10, "csls"), (1000, 0.4, 10, "csls"),
               (1000, 0.2, 10, "csls2"), (500, 0.2, 10, "csls2"), (2000, 0.2, 10, "csls2"),
               (1000, 0.2, 5, "csls2"), (1000, 0.2, 20, "csls2"), (1000, 0.1, 10, "csls2"), (1000, 0.4, 10, "csls2"),
               (500, 0.2, 10, "mean"), (2000, 0.2, 10, "mean"), (1000, 0.1, 10, "mean"), (1000, 0.4, 10, "mean")]
    main_lev = {}
    for N, eps, topk, form in configs:
        A, B, FA, FB, lev = run(N, eps, topk, form)
        name = f"{form} N {N} eps {eps}" + (f" top {topk}" if form.startswith("csls") else "")
        main_lev[(N, eps, topk, form)] = lev
        pr(f"--- {name}")
        pr(M.fmt_summary("  conj (LLR) alone, before", A))
        pr(M.fmt_summary("  conj (LLR) alone, after", B))
        pr(M.fmt_perm("after vs before", A, B))
        pr(M.fmt_summary("  fused with file mass, before (= best form)", FA))
        pr(M.fmt_summary("  fused with file mass, after", FB))
        pr(M.fmt_perm("after vs best", BEST, FB) + "   " + M.fmt_perm("after vs G1", G1, FB).strip())
        if (N, eps, topk) == (1000, 0.2, 10) and form in ("csls", "mean", "csls2"):
            pr("  rows by kind, fused after against best:")
            pr(M.kind_table(FB, BEST))
            ra = sum(p["rivals"] for p in A)
            rb = sum(p["rivals"] for p in B)
            up = sum(1 for x, y in zip(A, B) if y["rivals"] < x["rivals"])
            dn = sum(1 for x, y in zip(A, B) if y["rivals"] > x["rivals"])
            pr(f"  rivals ranked above the answer (conj alone): {ra} before, {rb} after; fewer on {up} questions, more on {dn}")
    # what the background level looks like: answer pieces against the pieces that beat them
    lev = main_lev[(1000, 0.2, 10, "csls")]
    ans_lv, riv_lv = [], []
    for q, a, C, K in Q:
        O = order_corr(C, K, 0.2, None)
        r = next((i for i, c in enumerate(O) if c["isans"]), None)
        if r is None:
            continue
        ans_lv += [lev.get(c["key"], 0) for c in C if c["isans"]]
        riv_lv += [lev.get(c["key"], 0) for c in O[:r]]
    pr(f"CSLS level (N 1000, eps 0.2, top 10): answer pieces mean {sum(ans_lv) / len(ans_lv):.3f} (n {len(ans_lv)}), "
       f"pieces ranked above the answer by conj mean {sum(riv_lv) / max(1, len(riv_lv)):.3f} (n {len(riv_lv)})")
    top = sorted(lev.items(), key=lambda kv: -kv[1])[:8]
    pr("highest CSLS levels: " + "; ".join(f"{k} {v:.3f}" for k, v in top))
    text = "\n".join(out)
    print(text)
    (M.ROOT / "scratch" / "lead3" / "h5_out.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
