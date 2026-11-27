"""H8: an honest gate by split conformal calibration on the organism's own exam.

Scorer (the same on the exam and the tuning dump): per piece, llr = sum over the request's parts of
log((eps + hold_k) / eps), hold_k the rarity-weighted share of part k's stems the piece holds (the
exam's rarity over the material's 30-line pieces). Two mass forms over a question's candidate set:
- "conj": weight proportional to prod(eps + hold_k) = eps^K exp(llr) (the conjunction rule itself;
  a piece holding nothing keeps the floor);
- "excess": weight exp(llr) - 1 (the floor taken away; a piece holding nothing weighs 0).
Normalised to sum 1 over the candidate set. On the exam the candidate set is the N highest pieces of
the 12,108 (N = the tuning dump's median candidate count, with its lower and upper quartile as
sensitivity), each question's own comment lines masked (as the exam builder does); on the tuning dump
it is the dump's own candidates, holds recomputed by the same scorer from the dump's part stems.
Exam parts: the ported request drone part former over the plain words of the comment's first
sentence, own name parts removed (as in h5_background.py).

Nonconformity of a calibration question: the normalised mass taken from the top, pieces scoring at or
above the true piece included (ties count against), before the true piece is inside; infinite when the
true piece is outside the candidate set. Threshold: the ceil((n + 1) 0.9)-th smallest (finite-sample
correction). A live set: every candidate whose cumulative mass (ties included) is at most the
threshold. Gate: settled when the set fits the room (12 pieces), low otherwise; the pack is the set
when settled, else the first 12. Truth: the pack holds every answer range; decoys are the missing side.
Brier: settled read as 0.9; low read as the exam coverage the first 12 would reach (the share of exam
nonconformity scores at or below the first 12's mass).

`python -I tests/replay/lead3/h8_gate.py`
"""
import bisect
import heapq
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as M  # noqa: E402
import h5_background as H5  # noqa: E402

L, X = H5.L, H5.X
EPSS = (0.1, 0.2, 0.4)
ALPHA = 0.1
NMAX = 500


def llr_all(ix, parts, masked, post, epss):
    """Per eps, dict piece -> llr (pieces with any hold only)."""
    h_parts = []
    for st in parts:
        den = sum(ix.idf(w) for w in st)
        h = defaultdict(float)
        for w in st:
            r = ix.idf(w) / den
            for p in post.get(w, ()):
                h[p] += r
        for p, s in masked.items():
            v = sum(ix.idf(w) for w in st if w in s) / den
            if v > 0:
                h[p] = v
            else:
                h.pop(p, None)
        h_parts.append(h)
    out = {}
    for eps in epss:
        tot = defaultdict(float)
        for h in h_parts:
            for p, v in h.items():
                if v > 0:
                    tot[p] += math.log((eps + v) / eps)
        out[eps] = tot
    return out


def weights(llrs, form):
    if form == "conj":
        return [math.exp(x) for x in llrs]
    return [math.expm1(x) for x in llrs]


def nonconf(top, true_llr, N, form):
    """top: llr values sorted descending (at least N, floors padded with 0)."""
    t = top[:N]
    if true_llr is None:
        return math.inf
    gt = sum(1 for x in t if x > true_llr + 1e-12)
    if gt >= N:
        return math.inf
    w = weights(t, form)
    tot = sum(w)
    if tot <= 0:
        return 1.0
    return sum(wi for x, wi in zip(t, w) if x >= true_llr - 1e-12) / tot


def qhat(scores, alpha=ALPHA):
    s = sorted(scores)
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else s[k - 1]


def exam_records(ix, exam_rows, epss):
    post = ix.post
    recs = []
    t0 = time.perf_counter()
    for qi, (q, parts) in enumerate(exam_rows):
        masked = ix.masked(q["file"], q["doc_lines"]) if q["doc_lines"] else {}
        ans = ix.pieces_over(q["file"], q["first"], q["last"])
        by = llr_all(ix, parts, masked, post, epss)
        r = {"half": None, "K": len(parts), "file": q["file"]}
        for eps in epss:
            tot = by[eps]
            top = heapq.nlargest(NMAX, tot.values())
            top += [0.0] * (NMAX - len(top))
            true = max((tot.get(a, 0.0) for a in ans), default=None)
            r[eps] = (top, true)
        recs.append(r)
        if qi % 1000 == 999:
            print(f"  exam scored {qi + 1} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return recs


def dump_llr(ix, q, C, eps, cand):
    out = []
    for c in C:
        s = cand.get(c["key"], set())
        v = 0.0
        for p in q["request"]["parts"]:
            st = p["stems"]
            den = sum(ix.idf(w) for w in st) or 1.0
            h = sum(ix.idf(w) for w in st if w in s) / den
            v += math.log((eps + h) / eps)
        out.append(v)
    return out


def main():
    files, ix, exam = H5.build()
    rows, Q, D = M.load()
    counts = sorted(len(q["candidates"]) for q in rows)
    n_med, n_lo, n_hi = counts[len(counts) // 2], counts[len(counts) // 4], counts[(3 * len(counts)) // 4]
    print(f"dump candidate counts: quartiles {n_lo} / {n_med} / {n_hi}")
    exam_rows = [(q, H5.exam_parts(q)) for q in exam]
    exam_rows = [(q, p) for q, p in exam_rows if p]
    half = X.split_files(files)
    recs = exam_records(ix, exam_rows, EPSS)
    for r, (q, _) in zip(recs, exam_rows):
        r["half"] = half[q["file"]]
    keys = {}
    for q in rows:
        for c in q["candidates"]:
            keys[c["key"]] = (c["path"], c["first"], c["last"])
    cand = H5.piece_sets(ix, keys)
    out = []
    pr = out.append
    pr(f"exam: {len(recs)} calibration questions; dump: {len(Q)} answerable, {len(D)} decoys; "
       f"candidate count quartiles {n_lo} / {n_med} / {n_hi}")
    pr("served gate (comprehension.py --nano): settled holds 8 of 15, tie 2 of 6, absent 23 of 50; right decisions 0.549; "
       "Brier 0.444 against 0.249 for the base rate")

    def tuning(eps, form, qh, room, order="score"):
        res = []
        qh0 = qh
        for q in rows:
            qh = qh0(q["id"]) if callable(qh0) else qh0
            ans = [a for a in q["answers"] if a.get("resolved")]
            C = q["candidates"]
            v = dump_llr(ix, q, C, eps, cand)
            idx = sorted(range(len(C)), key=lambda i: (-v[i], -C[i]["ssum"]))
            vs = [v[i] for i in idx]
            w = weights(vs, form)
            tot = sum(w) or 1.0
            # cumulative mass with ties included
            cum = []
            acc = 0.0
            j = 0
            while j < len(vs):
                k = j
                while k < len(vs) and abs(vs[k] - vs[j]) <= 1e-12:
                    k += 1
                acc += sum(w[j:k])
                cum += [acc / tot] * (k - j)
                j = k
            size = sum(1 for x in cum if x <= qh + 1e-12) if qh != math.inf else len(C)
            S = [C[i] for i in idx[:size]]
            P = S if size <= room else [C[i] for i in idx[:room]]
            m_room = cum[min(room, len(cum)) - 1]
            any_in = bool(ans) and any(M.ov(c, a) for c in S for a in ans)
            all_in = bool(ans) and all(any(M.ov(c, a) for c in S) for a in ans)
            held = bool(ans) and all(any(M.ov(c, a) for c in P) for a in ans)
            # nonconformity of the tuning question itself (answerable only)
            nc = None
            if ans:
                tv = [v[i] for i in range(len(C)) if C[i]["isans"]]
                if tv:
                    t = max(tv)
                    nc = sum(wi for x, wi in zip(vs, w) if x >= t - 1e-12) / tot
                else:
                    nc = math.inf
            res.append({"id": q["id"], "kind": q["kind"], "decoy": not ans, "size": size, "settled": size <= room,
                        "any": any_in, "all": all_in, "held": held, "m_room": m_room, "nc": nc,
                        "served": len(q["served"]["pack_cited"]), "top1": cum[0]})
        return res

    def report(name, eps, form, N, room, detail=False, alpha=ALPHA, loo=False):
        cal = [nonconf(r[eps][0], r[eps][1], N, form) for r in recs]
        qh = qhat(cal, alpha)
        if loo:  # calibrate on the tuning questions themselves, each left out of its own threshold
            own = {r["id"]: r["nc"] for r in tuning(eps, form, math.inf, room) if not r["decoy"]}
            qh = lambda i: qhat([v for k, v in own.items() if k != i], alpha)
            cal = sorted(own.values())
        # internal check on the exam: calibrate on one half of files, cover the other
        hs = []
        for a, b in (("A", "B"), ("B", "A")):
            qa = qhat([c for c, r in zip(cal, recs) if r["half"] == a], alpha)
            tb = [c for c, r in zip(cal, recs) if r["half"] == b]
            hs.append(sum(1 for c in tb if c <= qa) / len(tb))
        res = tuning(eps, form, qh, room)
        A = [r for r in res if not r["decoy"]]
        Dd = [r for r in res if r["decoy"]]
        cs = sorted(cal)
        if loo:
            hs = [float("nan"), float("nan")]
        pr(f"--- {name}: threshold {(qh('') if callable(qh) else qh):.4f}; exam coverage across halves {hs[0]:.3f} / {hs[1]:.3f}; exam infinite {sum(1 for c in cal if c == math.inf)}")
        pr(f"  tuning: set holds an answer piece {sum(r['any'] for r in A)} of {len(A)} ({sum(r['any'] for r in A) / len(A):.3f}), "
           f"every range {sum(r['all'] for r in A)} ({sum(r['all'] for r in A) / len(A):.3f})")
        sz = sorted(r["size"] for r in res)
        sv = sorted(r["served"] for r in res)
        pr(f"  set size median {sz[len(sz) // 2]}, mean {sum(sz) / len(sz):.1f}, quartiles {sz[len(sz) // 4]} / {sz[(3 * len(sz)) // 4]}; "
           f"served pack pieces median {sv[len(sv) // 2]}, mean {sum(sv) / len(sv):.1f}; decoys' set size median {sorted(r['size'] for r in Dd)[len(Dd) // 2]}")
        st = [r for r in res if r["settled"]]
        lo = [r for r in res if not r["settled"]]
        sh = sum(r["held"] for r in st)
        lh = sum(r["held"] for r in lo)
        right = sum(1 for r in st if r["held"]) + sum(1 for r in lo if not r["held"])
        ys = [1.0 if r["held"] else 0.0 for r in res]
        ps = []
        for r in res:
            if r["settled"]:
                ps.append(1 - alpha)
            else:
                ps.append(bisect.bisect_right(cs, r["m_room"] + 1e-12) / len(cs))
        brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)
        base = sum(ys) / len(ys)
        bb = sum((base - y) ** 2 for y in ys) / len(ys)
        pr(f"  gate (room {room}): settled holds {sh} of {len(st)} (decoys settled {sum(r['decoy'] for r in st)}), "
           f"low holds {lh} of {len(lo)}; right decisions {right / len(res):.3f}; Brier {brier:.3f} against {bb:.3f} for the base rate {base:.3f}")
        if detail:
            tn = sorted(r["nc"] for r in A)
            q4 = lambda s, f: s[min(len(s) - 1, int(f * len(s)))]
            pr(f"  alike check, nonconformity quartiles and 90th: exam {q4(cs, .25):.3f} / {q4(cs, .5):.3f} / {q4(cs, .75):.3f} / {q4(cs, .9):.3f}; "
               f"tuning {q4(tn, .25):.3f} / {q4(tn, .5):.3f} / {q4(tn, .75):.3f} / {q4(tn, .9):.3f}")
            # two-sample Kolmogorov-Smirnov statistic and its asymptotic p
            allv = sorted(set(cs + tn))
            ks = max(abs(bisect.bisect_right(cs, x) / len(cs) - bisect.bisect_right(tn, x) / len(tn)) for x in allv if x != math.inf)
            ne = len(cs) * len(tn) / (len(cs) + len(tn))
            lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * ks
            pks = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
            pr(f"  KS statistic {ks:.3f}, p {max(0.0, min(1.0, pks)):.2g}; exam top-1 mass mean "
               f"{sum(weights([r[eps][0][0]], form)[0] / max(1e-12, sum(weights(r[eps][0][:N], form))) for r in recs) / len(recs):.3f}, "
               f"tuning {sum(r['top1'] for r in res) / len(res):.3f}")
            ek = [r["K"] for r in recs]
            pr(f"  parts per question: exam {sum(ek) / len(ek):.2f}, tuning {sum(len(q['request']['parts']) for q in rows) / len(rows):.2f}")
            # rank of the true piece by the scorer
            er = []
            for r in recs:
                top, t = r[eps]
                er.append(math.inf if t is None else 1 + sum(1 for x in top[:N] if x > t + 1e-12))
            tr = []
            for q, a, C, K in Q:
                v = dump_llr(ix, q, C, eps, cand)
                tv = [v[i] for i in range(len(C)) if C[i]["isans"]]
                tr.append(math.inf if not tv else 1 + sum(1 for x in v if x > max(tv) + 1e-12))
            for k in (1, 3, 10, 50):
                pr(f"    true piece within the first {k:2d}: exam {sum(1 for x in er if x <= k) / len(er):.3f}, tuning {sum(1 for x in tr if x <= k) / len(tr):.3f}")
            kinds = defaultdict(list)
            for r in A:
                kinds[r["kind"]].append(r)
            pr("  coverage by kind (every range): " + ", ".join(f"{k} {sum(x['all'] for x in v)}/{len(v)}" for k, v in sorted(kinds.items())))
        return res

    for a in (0.3, 0.25, 0.2):
        report(f"excess mass, eps 0.2, N {n_med}, room 12, coverage {1 - a:.2f} (exam allows at most 0.77)", 0.2, "excess", n_med, 12, alpha=a)
    for eps in EPSS:
        report(f"leave-one-out on the tuning questions themselves, excess mass, eps {eps}, room 12, coverage 0.90", eps, "excess", n_med, 12, loo=True)
    report(f"leave-one-out on the tuning questions themselves, excess mass, eps 0.2, room 12, coverage 0.80", 0.2, "excess", n_med, 12, loo=True, alpha=0.2)
    for form in ("excess", "conj"):
        report(f"{form} mass, eps 0.2, N {n_med}, room 12", 0.2, form, n_med, 12, detail=True)
        for eps in (0.1, 0.4):
            report(f"{form} mass, eps {eps}, N {n_med}, room 12", eps, form, n_med, 12)
        for N in (n_lo, n_hi):
            report(f"{form} mass, eps 0.2, N {N}, room 12", 0.2, form, N, 12)
        for room in (8, 16):
            report(f"{form} mass, eps 0.2, N {n_med}, room {room}", 0.2, form, n_med, room)
    text = "\n".join(out)
    print(text)
    (M.ROOT / "scratch" / "lead3" / "h8_out.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
