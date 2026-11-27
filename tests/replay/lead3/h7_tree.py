"""H7: one root close to every part of the request (distinct-root form of the group Steiner tree).

Graph over a question's dump candidates: nodes are candidates; edges are each candidate's typed
`edges` (Call, DefUse, TypeUse, Test, Names; all targets lie inside the candidate set) plus same-file
neighbours (the previous and next candidate of the same file by first line). Edges are taken both
ways. deg(v) = number of distinct neighbours of v. Edge cost u -> v = w x log2(1 + deg(u)) (the node
left is the node passed through; Kacholia et al. 2005 form), no reward for degree anywhere. Variant
"entered": cost w x log2(1 + deg(v)).

For each request part k, a cheapest-first expansion (Dijkstra) where every candidate starts at
-log(eps + hold_k) (eps 0.2 as in the conjunction rule; a candidate holding nothing starts at
-log eps). So with no edges a node's tree cost is exactly minus the log of the conjunction score;
an edge lets a node borrow a part from a neighbour at the edge's cost. Tree cost of v = sum over parts
of dist_k(v). Roots ranked by tree cost (ties: conjunction, then settle sum); the order written is
each root followed by the nodes on its paths (per part, the predecessor chain from the root back to
its source), each node once.

`python -I tests/replay/lead3/h7_tree.py`
"""
import heapq
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as M  # noqa: E402


def graph(C, typed=True, samefile=True):
    key = {c["key"]: i for i, c in enumerate(C)}
    nb = defaultdict(set)
    if typed:
        for i, c in enumerate(C):
            for e in c["edges"]:
                j = key.get(e["to"])
                if j is not None and j != i:
                    nb[i].add(j)
                    nb[j].add(i)
    if samefile:
        by = defaultdict(list)
        for i, c in enumerate(C):
            by[c["path"]].append(i)
        for L in by.values():
            L.sort(key=lambda i: C[i]["first"])
            for x, y in zip(L, L[1:]):
                nb[x].add(y)
                nb[y].add(x)
    return nb


def tree_order(C, K, w, eps=M.EPS, charge="left", typed=True, samefile=True, return_cost=False, paths=True):
    n = len(C)
    nb = graph(C, typed, samefile)
    deg = [len(nb[i]) for i in range(n)]
    pen = [math.log2(1 + d) for d in deg]
    dist = []
    pred = []
    for k in range(K):
        d = [-math.log(eps + M.hold(C[i], k)) for i in range(n)]
        p = [-1] * n
        h = [(d[i], i) for i in range(n)]
        heapq.heapify(h)
        done = [False] * n
        while h:
            du, u = heapq.heappop(h)
            if done[u] or du > d[u]:
                continue
            done[u] = True
            for v in nb[u]:
                cst = w * (pen[u] if charge == "left" else pen[v])
                if du + cst < d[v] - 1e-12:
                    d[v] = du + cst
                    p[v] = u
                    heapq.heappush(h, (d[v], v))
        dist.append(d)
        pred.append(p)
    cost = [sum(dist[k][i] for k in range(K)) for i in range(n)]
    roots = sorted(range(n), key=lambda i: (round(cost[i], 9), -round(M.conj(C[i], K, eps), 9), -C[i]["ssum"]))
    seen, O = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            O.append(C[r])
        for k in range(K if paths else 0):
            v = pred[k][r]
            while v != -1:
                if v not in seen:
                    seen.add(v)
                    O.append(C[v])
                v = pred[k][v]
    if return_cost:
        return O, cost, roots
    return O


def main():
    rows, Q, D = M.load()
    G1 = M.per_question(Q, M.order_g1)
    BEST = M.per_question(Q, M.order_best)
    out = []
    pr = out.append
    pr(M.fmt_summary("G1 order (served)", G1))
    pr(M.fmt_summary("best: conj fused with file mass (k10)", BEST))
    CJ = M.per_question(Q, lambda q, a, C, K: M.order_conj(C, K))
    pr(M.fmt_summary("conj alone (the tree with no edges)", CJ))
    # graph facts
    degs, ncomp = [], []
    for q, a, C, K in Q:
        nb = graph(C)
        degs += [len(nb[i]) for i in range(len(C))]
    degs.sort()
    pr(f"graph: candidates {len(degs)}, degree median {degs[len(degs) // 2]}, 90th {degs[int(.9 * len(degs))]}, max {degs[-1]}, "
       f"isolated {sum(1 for d in degs if d == 0)}")
    variants = []
    for w in (0.25, 0.5, 1.0):
        variants.append((f"tree w {w}", dict(w=w)))
    variants += [("tree w 0.5, cost on node entered", dict(w=0.5, charge="entered")),
                 ("tree w 0.5, typed edges only", dict(w=0.5, samefile=False)),
                 ("tree w 0.5, same-file edges only", dict(w=0.5, typed=False)),
                 ("tree w 0.5, roots only (no path nodes written)", dict(w=0.5, paths=False)),
                 ("tree w 0.5, eps 0.1", dict(w=0.5, eps=0.1)),
                 ("tree w 0.5, eps 0.4", dict(w=0.5, eps=0.4))]
    results = {}
    for name, kw in variants:
        T = M.per_question(Q, lambda q, a, C, K, kw=kw: tree_order(C, K, **kw))
        F = M.per_question(Q, lambda q, a, C, K, kw=kw: M.rrf(C, [tree_order(C, K, **kw), M.order_best(q, a, C, K)], 10))
        F3 = M.per_question(Q, lambda q, a, C, K, kw=kw: M.rrf(C, [tree_order(C, K, **kw), M.order_filemass(C)], 10))
        results[name] = (T, F, F3)
        for tag, P in (("alone", T), ("fused with best (k10)", F), ("in place of conj, fused with file mass", F3)):
            pr(M.fmt_summary(f"{name}: {tag}", P))
            pr(M.fmt_perm("vs best", BEST, P) + "   " + M.fmt_perm("vs G1", G1, P).strip()
               + ("   " + M.fmt_perm("vs conj alone", CJ, P).strip() if tag == "alone" else ""))
    # trace questions apart, and rows by kind for the main w
    for name in ("tree w 0.25", "tree w 0.5", "tree w 1.0"):
        T, F, F3 = results[name]
        for tag, P in (("alone", T), ("fused with best", F)):
            tr = [p for p in P if p["kind"] == "trace"]
            trb = [p for p in BEST if p["kind"] == "trace"]
            trg = [p for p in G1 if p["kind"] == "trace"]
            pr(f"trace questions ({len(tr)}), {name} {tag}: every range in first 10 {sum(p['every10'] for p in tr)} "
               f"(best {sum(p['every10'] for p in trb)}, G1 {sum(p['every10'] for p in trg)}); mean range RR "
               f"{sum(p['rr'] for p in tr) / len(tr):.3f} (best {sum(p['rr'] for p in trb) / len(trb):.3f})")
    pr("rows by kind, tree w 0.5 alone against best:")
    pr(M.kind_table(results["tree w 0.5"][0], BEST))
    pr("rows by kind, tree w 0.5 fused with best against best:")
    pr(M.kind_table(results["tree w 0.5"][1], BEST))
    # multi-range questions: does the tree bring the second range up?
    multi = [i for i, (q, a, C, K) in enumerate(Q) if len(a) > 1]
    for name in ("tree w 0.5",):
        T, F, _ = results[name]
        pr(f"questions with 2+ answer ranges ({len(multi)}): every range in first 10: best {sum(BEST[i]['every10'] for i in multi)}, "
           f"{name} alone {sum(T[i]['every10'] for i in multi)}, fused {sum(F[i]['every10'] for i in multi)}; "
           f"mean range RR best {sum(BEST[i]['rr'] for i in multi) / len(multi):.3f}, alone {sum(T[i]['rr'] for i in multi) / len(multi):.3f}, "
           f"fused {sum(F[i]['rr'] for i in multi) / len(multi):.3f}")
    # how often the best root is joined by a borrowed part (the tree differs from the plain conjunction)
    borrowed = 0
    for q, a, C, K in Q:
        O, cost, roots = tree_order(C, K, 0.5, return_cost=True)
        r = roots[0]
        own = -sum(math.log(M.EPS + M.hold(C[r], k)) for k in range(K))
        borrowed += cost[r] < own - 1e-9
    pr(f"w 0.5: first root borrows at least one part through an edge on {borrowed} of {len(Q)} questions")
    text = "\n".join(out)
    print(text)
    (M.ROOT / "scratch" / "lead3" / "h7_out.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
