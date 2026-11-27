"""Shared pieces for the lead's three mechanisms on the replay (H5, H7, H8).

Reads the large code tuning dump the way the lead's scripts do (scratch/lead-design/h2.py to h7.py):
answer pieces by overlap, per-range coverage, the joint evidence rule "conj" (product over parts of
eps + hold anywhere, eps 0.2), file mass (sum of the pieces' settle chances per file), the lead's best
ordering (reciprocal rank fusion of conj and file mass, k 10) and G1's served order. Also the
measures (first, top 3, top 10, every range in the first 10, mean range reciprocal rank, by kind) and
the paired sign flip test (10,000 flips, seed 20261117). Standard library only; `python -I`.
Written 2026-11-17 for the lead's mechanisms test.
"""
import collections
import importlib.util
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DUMP = ROOT / "scratch" / "replay" / "dump-p1-large-code" / "candidates.jsonl"
SEED = 20261117
EPS = 0.2


def load_lexicon():
    spec = importlib.util.spec_from_file_location("lexicon", ROOT / "tests" / "replay" / "dict" / "lexicon.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_exam_module():
    spec = importlib.util.spec_from_file_location("exam", ROOT / "tests" / "replay" / "dict" / "exam.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ov(c, a):
    return c["path"] == a["file"] and c["first"] <= a["last"] and a["first"] <= c["last"]


def load():
    """(rows, Q, D): all dump rows; answerable (q, ans, C, K); decoy rows (q, C, K)."""
    rows = [json.loads(l) for l in open(DUMP, encoding="utf-8")]
    Q, D = [], []
    for q in rows:
        ans = [a for a in q["answers"] if a.get("resolved")]
        C = q["candidates"]
        K = len(q["request"]["parts"])
        for c in C:
            c["isans"] = any(ov(c, a) for a in ans)
            c["cov"] = [i for i, a in enumerate(ans) if ov(c, a)]
            s = [x["s"] for x in c["cells"]] + [0] * K
            c["ssum"] = sum(s[:K])
            c["g"] = c["g1"] if c["g1"] is not None else 10 ** 6
        if ans:
            Q.append((q, ans, C, K))
        else:
            D.append((q, C, K))
    return rows, Q, D


NULL_CELL = {"h": [0, 0, 0], "p": 0, "s": 0, "r": 0, "share": 0, "accepted": False}


def cell(c, k):
    return c["cells"][k] if k < len(c["cells"]) else NULL_CELL


def hold(c, k):
    return max(0.0, cell(c, k)["h"][2])


def conj(c, K, eps=EPS):
    return math.exp(sum(math.log(eps + hold(c, k)) for k in range(K)))


def filemass(C):
    fs = collections.defaultdict(float)
    for c in C:
        fs[c["path"]] += c["ssum"]
    return fs


def ranks_of(O):
    return {id(c): i for i, c in enumerate(O)}


def rrf(C, orders, k0=10):
    """Reciprocal rank fusion of full orders of C."""
    R = [ranks_of(O) for O in orders]
    return sorted(C, key=lambda c: -sum(1 / (k0 + r[id(c)]) for r in R))


def order_conj(C, K, eps=EPS):
    return sorted(C, key=lambda c: (-round(conj(c, K, eps), 9), -c["ssum"]))


def order_filemass(C):
    fs = filemass(C)
    return sorted(C, key=lambda c: -c["ssum"] * fs[c["path"]])


def order_best(q, a, C, K):
    """The lead's best ordering: conj fused with file mass (k 10), mean range RR 0.408."""
    return rrf(C, [order_conj(C, K), order_filemass(C)], 10)


def order_g1(q, a, C, K):
    return sorted(C, key=lambda c: c["g"])


def range_rr(O, ans):
    s = 0.0
    for ai in range(len(ans)):
        ri = next((i for i, c in enumerate(O) if ai in c["cov"]), None)
        s += 0 if ri is None else 1 / (ri + 1)
    return s / len(ans)


def per_question(Q, order):
    out = []
    for q, ans, C, K in Q:
        O = order(q, ans, C, K)
        r = next((i for i, c in enumerate(O) if c["isans"]), None)
        out.append({"id": q["id"], "kind": q["kind"], "rank": r, "rr": range_rr(O, ans),
                    "every10": all(any(ai in c["cov"] for c in O[:10]) for ai in range(len(ans))),
                    "rivals": r if r is not None else len(C)})
    return out


def summary(P):
    n = len(P)
    return {"first": sum(p["rank"] == 0 for p in P), "top3": sum(p["rank"] is not None and p["rank"] < 3 for p in P),
            "top10": sum(p["rank"] is not None and p["rank"] < 10 for p in P), "every10": sum(p["every10"] for p in P),
            "mrr": sum(p["rr"] for p in P) / n, "rivals": sum(p["rivals"] for p in P), "n": n}


def fmt_summary(name, P):
    s = summary(P)
    return (f"{name:52s} first {s['first']:2d} top3 {s['top3']:2d} top10 {s['top10']:2d} every-range-in-10 {s['every10']:2d} "
            f"mean range RR {s['mrr']:.3f} rivals above answer {s['rivals']}")


def by_kind(P):
    k = collections.defaultdict(list)
    for p in P:
        k[p["kind"]].append(p)
    return {kk: summary(v) for kk, v in sorted(k.items())}


def perm(A, B, n=10000, seed=SEED):
    """Paired sign flip test on mean(B - A) of per-question mean range RR; two-sided."""
    d = [b["rr"] - a["rr"] for a, b in zip(A, B)]
    obs = sum(d) / len(d)
    rnd = random.Random(seed)
    ge = 0
    for _ in range(n):
        s = sum(x if rnd.random() < .5 else -x for x in d) / len(d)
        ge += abs(s) >= abs(obs) - 1e-12
    return {"diff": obs, "better": sum(1 for x in d if x > 1e-9), "worse": sum(1 for x in d if x < -1e-9), "p": (ge + 1) / (n + 1)}


def fmt_perm(name, A, B):
    t = perm(A, B)
    return f"  {name}: diff {t['diff']:+.3f}, better {t['better']} worse {t['worse']}, p {t['p']:.4f}"


def kind_table(P, Pref):
    lines = []
    a, b = by_kind(Pref), by_kind(P)
    for k in b:
        x, y = a[k], b[k]
        lines.append(f"    {k:8s} n {y['n']:2d}  first {x['first']}->{y['first']}  top3 {x['top3']}->{y['top3']}  "
                     f"every10 {x['every10']}->{y['every10']}  mrr {x['mrr']:.3f}->{y['mrr']:.3f}")
    return "\n".join(lines)


# --- the request drone's part forming, ported (nmpod intake.rs Request::read, clause_groups) -------
# Ported for questions written as plain text (the exam's): runs of adjacent content words, a frame word
# ends a run; a run of one plain word joins the plain run after it, else the one before. Clause cuts
# are taken at `, and`/`, or`/`;`/`:` and at `and`/`or` between two phrases of two content words or
# more (as the drone does); trace cuts are not used (exam questions have no kind).

FRAME = set("""a an the of in on at by for with without from to into onto through over under between as and or but if then
than so that this these those it its it's they them their there what where which who whom whose how why when whether is are
was were be been being do does did done has have had can could may might must shall should will would code function functions
method methods file files line lines defined declared implemented located set computed happen happens trace call calls called
caller callers invoked invokes find show get gets each any all some one's own still again also just including include includes
happening what's exactly way ways many much large far long often one""".split())
OPENERS = set("where which what how when why who whether is are does do did may must can could should will would has have including then".split())


def drone_parts(text, stem):
    raw = text.split()
    toks = []
    for i, r in enumerate(raw):
        ends = r.endswith((",", ";", ":")) or (r.endswith("?") and i + 1 < len(raw)) or (r.endswith(".") and i + 1 < len(raw))
        w = re.sub(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$", "", r)
        if w.endswith("'s"):
            w = w[:-2]
        if w:
            toks.append([w, ends])
        elif toks:
            toks[-1][1] |= ends
    content = lambda w: any(ch.isalpha() for ch in w) and w.lower() not in FRAME
    count = lambda a, b: sum(1 for t in toks[a:b] if content(t[0]))
    clauses, s = [], 0
    for i in range(len(toks)):
        w = toks[i][0].lower()
        comma_before = i > 0 and toks[i - 1][1]
        cj = w in ("and", "or")
        nxt = next((j + 1 for j in range(i + 1, len(toks)) if toks[j][1]), len(toks))
        opener = lambda j: j < len(toks) and toks[j][0].lower() in OPENERS
        cut = (comma_before and (cj or opener(i))) or (cj and opener(i + 1)) or (cj and count(s, i) >= 2 and count(i + 1, nxt) >= 2)
        if cut and i > s:
            clauses.append((s, i))
            s = i
    clauses.append((s, len(toks)))
    groups = []
    for a, b in clauses:
        runs, cur = [], []
        for t in toks[a:b]:
            if content(t[0]):
                cur.append(t[0])
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        out, i = [], 0
        while i < len(runs):
            r = runs[i]
            if len(r) == 1:
                if i + 1 < len(runs):
                    out.append(r + runs[i + 1])
                    i += 2
                    continue
                if out:
                    out[-1].extend(r)
                    i += 1
                    continue
            out.append(r)
            i += 1
        groups.extend(out)
    parts = []
    for g in groups:
        st = []
        for w in g:
            x = stem(w.lower())
            if x not in st:
                st.append(x)
        parts.append(st)
    return parts
