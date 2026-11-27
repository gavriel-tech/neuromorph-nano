"""Replay G2's selection on a recording with a selection rule written as a small Python function.

usage: python -I replay.py --rec <recording.json> [<recording.json> ...] --rule <rule> [--rule <rule> ...]
                           [--against recorded|served] [--band default|zero|spread|fit] [--folds <folds.json> --part tune]
                           [--list] [--repro] [--pooled [--base recorded|<rule>] [--root <set>=<material> ...]]

<rule> is `served` (G2's greedy over the request's coverage F, as `nmbrain::g2::select`),
`defs_first` or `cover_fill` (the two examples below), or `<file.py>:<function>`: a function
`rule(q, cands, budget)` returning the chosen piece keys in pack order (or a dict with `set` and,
optionally, `up`). `q` is the question's record (request parts, relations, kind), `cands` G2's
candidates (each a dict: `key`, `path`, `tokens`, `g1` place in G1's order, `kind`, `document`,
`role`, `S` settle chance per part, `R` the drones' part chance per part, `tags` per part, `share`
per part), `budget` the room in tokens. The answer is never visible to a rule.

Per question it computes: sent up (every answer range in what goes up: the set, then G1's order
filled to the budget), in pack (every answer range in the chosen set), right first, top 3 (on the
set's order) and tokens (the set's), and compares each rule with the recorded served choice
(`--against recorded`, default) or with the served rule's replay (`--against served`) by
`tests/dev-large/compare.py`'s exact McNemar test. `--repro` reports how closely the served rule
reproduces the recorded set. Standard library only; seconds per run.

`--pooled` adds the yardstick over every question of every recording together, against `--base`
(default the recorded choice; `served` for the served rule's replay on the same candidates): the
continuous score (per answer range one over the place, in the written order, of the first piece
holding it, 0 when none; averaged over the question's ranges) with a paired permutation test
(10,000 sign flips, fixed seed), McNemar on each binary measure, mean pack tokens and pack
precision (the share of the written lines inside an answer range), with rows per set, per question
kind and per match class (`lexical` when a request stem occurs in the answer ranges' text, from
`--root <set>=<material>`; `mismatch` when none does), and the pass rule's verdict (see `pooled`).
A dump's recorded set can hold pieces that are not among the dumped candidates; the recorded arm
rebuilds them from their keys with estimated tokens, and a rule can only choose among the dumped
candidates, so `--base served` is the like-for-like comparison.

Limits of a recording parsed from the bench's diagnosis (`source: diag`): only the chosen set, the
answer pieces and the four strongest per part are known; G2's relation and caller-fact columns are
not recorded; the noise band is not printed (`--band default`: 0 on a diag recording, which
reproduces the recorded sets best, see `band_of`; `spread`: the median absolute deviation over the
known candidates; `zero`; `fit`: per question the band that best reproduces the recorded set, a
reference only); the pack's "needed with it" pieces (typed relations) are not
modelled, so in pack counts the chosen set only. A `source: dump` recording has none of these limits.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLE_ORDER = ["defines", "uses", "tests", "documents", "mentions"]


def load_compare():
    spec = importlib.util.spec_from_file_location("compare", HERE.parent / "dev-large" / "compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def args_after(argv, flag):
    out, on = [], False
    for x in argv:
        if x == flag:
            on = True
            continue
        if on and x.startswith("--"):
            on = False
        if on:
            out.append(x)
    return out


def load(files, folds=None, part=None):
    qs = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for q in d["questions"]:
            q["_source"] = d.get("source", "diag")
            q["_set"] = d.get("name", Path(f).stem)
            n = len(q["request"]["parts"])
            for c in q["candidates"]:
                cells = c["cells"]
                c["S"] = [(x["s"] or 0.0) if x else 0.0 for x in cells] + [0.0] * (n - len(cells))
                c["R"] = [(x["r"] or 0.0) if x else 0.0 for x in cells] + [0.0] * (n - len(cells))
                c["tags"] = [x["tag"] if x else None for x in cells]
                c["share"] = [(x["share"] or 0.0) if x else 0.0 for x in cells]
            qs.append(q)
    if folds and part:
        fd = json.loads(Path(folds).read_text(encoding="utf-8"))["folds"]
        qs = [q for q in qs if fd.get(q["id"], part) == part]
    return qs


def covers(c, a):
    return c["path"] == a["file"] and (a["whole_file"] or (c["first"] <= a["last"] and c["last"] >= a["first"]))


def mad(v):
    def med(x):
        x = sorted(x)
        n = len(x)
        return 0.0 if n == 0 else (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2)
    m = med(v)
    return med([abs(x - m) for x in v])


def eligible(q):
    """G2's candidates (G1's hand-over and queue) as a rule sees them: no answer flags."""
    hide = {"answer"}
    # A dump can list one piece twice (it is in G1's hand-over and again in its queue, same place):
    # each piece is a candidate once, or the greedy could take it twice.
    seen, out = set(), []
    for c in q["candidates"]:
        if c.get("in_cands") and c["key"] not in seen:
            seen.add(c["key"])
            out.append({k: v for k, v in c.items() if k not in hide})
    return out


def columns(q, cands, r_of):
    """G2's columns for a recording that holds them (a dump): the word parts, one column per caller
    fact piece (worth one over the relation's fact pieces), and the relations settled through the
    set (`rel_pairs`). A diag recording has neither: word parts only."""
    facts = q.get("facts") or []
    fact_cols = [(f["relation"], p, 1.0 / max(1, len(f["pieces"]))) for f in facts for p in f["pieces"]]
    fact_rel = {f["relation"] for f in facts}
    kept = [k for k in range(len(q["request"].get("relations", []))) if k not in fact_rel]
    pairs = {}
    for p_, q_, k in q.get("rel_pairs") or []:
        if k in kept:
            pairs.setdefault((p_, q_), []).append(kept.index(k))

    def r_full(c):
        return list(r_of(c)) + [w if c["key"] == p else 0.0 for (_, p, w) in fact_cols]
    return r_full, len(kept) if pairs else 0, pairs


def greedy(cands, r_of, budget, band, q=None):
    """`nmbrain::g2::select`: the budgeted greedy on F (word parts, caller-fact columns and relations
    settled through the set when the recording holds them), stopping at the room or inside the noise
    band, and the better of its set and the best single piece."""
    pairs, n_rel = {}, 0
    if q is not None:
        r_of, n_rel, pairs = columns(q, cands, r_of)
    nw = max((len(r_of(c)) for c in cands), default=0)
    n = nw + n_rel
    rho = {c["key"]: max(r_of(c), default=0.0) for c in cands}
    miss = [1.0] * n
    taken, used = [], 0
    fits = [c for c in cands if c["tokens"] <= budget]
    single = {c["key"]: sum(r_of(c)) for c in fits}
    left = list(fits)
    stop = "nothing left"
    place = {id(c): i for i, c in enumerate(cands)}

    def rel_terms(x):
        out = []
        for s in taken:
            for k in pairs.get((x["key"], s["key"]), []):
                out.append((nw + k, rho[s["key"]]))
            for k in pairs.get((s["key"], x["key"]), []):
                out.append((nw + k, rho[x["key"]]))
        return out

    def g(c):
        word = sum(m * x for m, x in zip(miss, r_of(c)))
        if not pairs:
            return word
        m2 = list(miss)
        for k, r in rel_terms(c):
            m2[k] *= 1.0 - min(1.0, max(0.0, r))
        return word + sum(miss[k] - m2[k] for k in range(nw, n))
    while left:
        # The greedy takes the largest gain per token; ties go to the earlier candidate.
        best = max(left, key=lambda c: (g(c) / max(1, c["tokens"]), -place[id(c)]))
        left.remove(best)
        if used + best["tokens"] > budget:
            stop = "budget"
            continue
        gain = g(best)
        if gain <= band:
            stop = "noise band"
            continue
        terms = rel_terms(best)
        miss = [m * (1.0 - min(1.0, max(0.0, x))) for m, x in zip(miss, r_of(best))] + miss[nw:]
        for k, r in terms:
            miss[k] *= 1.0 - min(1.0, max(0.0, r))
        taken.append(best)
        used += best["tokens"]
    value = sum(1.0 - m for m in miss)
    if single:
        b = max(fits, key=lambda c: (single[c["key"]], -place[id(c)]))
        if single[b["key"]] > value:
            return [b], "best single piece"
    return taken, stop


def band_of(q, values):
    """G2's noise band: the median absolute deviation of the positive single values over all of G2's
    candidates (a dump recording, or the band it recorded). A diag recording holds only the strong
    tail of the candidates, whose spread is far above the true band (most of G2's candidates are
    near 0, so its band is small): there the band is taken as 0."""
    if q.get("band") is not None:
        return q["band"]
    if q.get("_source") == "dump":
        return mad([v for v in values if v > 0])
    return 0.0


def band_known(q, cands):
    return band_of(q, [sum(c["S"]) for c in cands])


def band_spread(q, cands):
    """The median absolute deviation over the candidates the recording holds, whatever its source."""
    return mad([s for s in (sum(c["S"]) for c in cands) if s > 0])


def served(q, cands, budget, band=None):
    """The served rule: G2's greedy on the settle chance, the set written in G1's order."""
    b = band_known(q, cands) if band is None else band
    taken, _ = greedy(cands, lambda c: c["S"], budget, b, q)
    return [c["key"] for c in sorted(taken, key=lambda c: (c["g1"] if c["g1"] is not None else 10**9, c["key"]))]


def served_gain_order(q, cands, budget):
    """The served set, written in the greedy's order of taking (largest gain per token first) instead
    of G1's order: a one-line probe of where the pack's first place is decided."""
    taken, _ = greedy(cands, lambda c: c["S"], budget, band_known(q, cands), q)
    return [c["key"] for c in taken]


ROLE_W = {"defines": 1.0, "uses": 0.5, "tests": 0.4, "documents": 0.5, "mentions": 0.2, None: 0.2}


def defs_first(q, cands, budget):
    """Example 1, definitions above mentions (research note 7.1): each part's chance weighted by the
    piece's tag on that part (defines 1, uses 0.5, documents 0.5, tests 0.4, mentions 0.2), the same
    greedy, then the set ordered definitions first, uses next, documents and mentions last."""
    w = lambda c: [s * ROLE_W.get(t, 0.2) for s, t in zip(c["S"], c["tags"])]
    taken, _ = greedy(cands, w, budget, band_of(q, [sum(w(c)) for c in cands]))
    rank = lambda c: ROLE_ORDER.index(c["role"]) if c["role"] in ROLE_ORDER else len(ROLE_ORDER)
    return [c["key"] for c in sorted(taken, key=lambda c: (rank(c), -sum(w(c))))]


def cover_fill(q, cands, budget):
    """Example 2, a covering set by parts, then fill by score (research note 7.3): for each part in
    turn (strongest holder first), the candidate with the largest chance on it, preferring one that
    also holds other open parts; a part counts as covered by a piece holding at least half the
    part's best chance. Then the rest by summed chance per token within the room, above the band."""
    n = len(q["request"]["parts"])
    best = [max((c["S"][k] for c in cands), default=0.0) for k in range(n)]
    open_ = [k for k in range(n) if best[k] > 0]
    chosen, used = [], 0
    while open_:
        k = max(open_, key=lambda k: best[k])
        held = lambda c: sum(1 for j in open_ if best[j] > 0 and c["S"][j] >= 0.5 * best[j])
        pick = max((c for c in cands if c not in chosen and c["tokens"] + used <= budget), key=lambda c: (c["S"][k], held(c), -c["tokens"]), default=None)
        if pick is None or pick["S"][k] <= 0:
            open_.remove(k)
            continue
        chosen.append(pick)
        used += pick["tokens"]
        open_ = [j for j in open_ if pick["S"][j] < 0.5 * best[j]]
    band = band_known(q, cands)
    for c in sorted((c for c in cands if c not in chosen), key=lambda c: -sum(c["S"]) / max(1, c["tokens"])):
        if sum(c["S"]) > band and used + c["tokens"] <= budget:
            chosen.append(c)
            used += c["tokens"]
    return [c["key"] for c in chosen]


BUILTIN = {"served": served, "defs_first": defs_first, "cover_fill": cover_fill, "served_gain_order": served_gain_order}


def get_rule(spec):
    if spec in BUILTIN:
        return spec, BUILTIN[spec]
    path, _, fn = spec.partition(":")
    s = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return spec, getattr(m, fn or "rule")


def fill_up(q, chosen):
    """What goes up: the chosen set, then G1's order filled to the budget. A G1 place the recording
    does not hold is counted at the known candidates' mean tokens (diag recordings)."""
    by_rank = {c["g1"]: c for c in q["candidates"] if c.get("g1") is not None}
    known = [c["tokens"] for c in by_rank.values()] or [300]
    mean = sum(known) / len(known)
    up = list(chosen)
    keyset = set(chosen)
    tok = {c["key"]: c["tokens"] for c in q["candidates"]}
    used = sum(tok.get(k, 0) for k in chosen)
    budget = q["budget"]
    top = max(by_rank) if by_rank else 0
    # Ranks beyond the largest known one: the candidates go on (up_size says how many went up).
    for rank in range(1, top + 1):
        c = by_rank.get(rank)
        t = c["tokens"] if c else mean
        if c and c["key"] in keyset:
            continue
        if used + t <= budget:
            used += t
            if c:
                up.append(c["key"])
                keyset.add(c["key"])
    return up


def recip_rank(pieces, ans):
    """The continuous score: per answer range, one over the place (in the written pack order) of the
    first piece holding it, 0 when no piece holds it; averaged over the question's ranges."""
    if not ans:
        return None
    total = 0.0
    for a in ans:
        place = next((i + 1 for i, c in enumerate(pieces) if covers(c, a)), None)
        total += 1.0 / place if place else 0.0
    return total / len(ans)


def pack_lines(pieces, ans):
    """Pack precision's two counts: the pack's written lines (the union of the pieces' line ranges per
    file) and how many of them fall inside an answer range."""
    by_file = {}
    for c in pieces:
        by_file.setdefault(c["path"], set()).update(range(c["first"], c["last"] + 1))
    written = sum(len(v) for v in by_file.values())
    inside = 0
    for path, lines in by_file.items():
        hold = set()
        for a in ans:
            if a["file"] != path:
                continue
            if a["whole_file"]:
                hold = lines
                break
            hold |= set(range(a["first"], a["last"] + 1))
        inside += len(lines & hold)
    return inside, written


def rebuilt(q, k):
    """A piece named by its key that is not among the recorded candidates (a dump's served set can
    hold such pieces): path and lines from the key, tokens estimated from the question's candidates."""
    per_line = sorted(c["tokens"] / max(1, c["last"] - c["first"] + 1) for c in q["candidates"]) or [13.0]
    tpl = per_line[len(per_line) // 2]
    path, _, rng = k.rpartition(":")
    a, _, b = rng.partition("-")
    try:
        a, b = int(a), int(b)
    except ValueError:
        return None
    return {"key": k, "path": path, "first": a, "last": b, "tokens": round(tpl * (b - a + 1)), "g1": None}


def measures(q, chosen, up=None):
    tok = {c["key"]: c for c in q["candidates"]}
    pieces = [p for p in (tok.get(k) or rebuilt(q, k) for k in chosen) if p is not None]
    if up is None:
        up = fill_up(q, chosen)
    up_p = [p for p in (tok.get(k) or rebuilt(q, k) for k in up) if p is not None]
    ans = q["answers"]
    hit = lambda c: any(covers(c, a) for a in ans)
    every = lambda ps: bool(ans) and all(any(covers(c, a) for c in ps) for a in ans)
    inside, written = pack_lines(pieces, ans)
    return {
        "first": bool(pieces) and hit(pieces[0]) if ans else None,
        "top3": any(hit(c) for c in pieces[:3]) if ans else None,
        "pack": every(pieces) if ans else None,
        "up": every(up_p) if ans else None,
        "tokens": sum(c["tokens"] for c in pieces),
        "n": len(pieces),
        "rr": recip_rank(pieces, ans),
        "lines_in": inside,
        "lines": written,
    }


def recorded(q):
    """The recorded served choice's measures."""
    s = q["served"]
    tok = {c["key"]: c for c in q["candidates"]}
    if q.get("_source") == "dump" and s.get("up"):
        # A dump's set can hold pieces that are not among the dumped candidates (G2's pool is wider
        # than G1's hand-over and queue). They are rebuilt from their keys, tokens estimated from the
        # question's candidates (tokens per line), and the set is taken in its written order (the
        # head of what went up).
        ordered = [tok.get(k) or rebuilt(q, k) for k in s["up"][: len(s["set"])]]
        pieces = ordered
    else:
        pieces = [tok[k] for k in s["set"] if k in tok]
        ordered = sorted(pieces, key=lambda c: (c["g1"] if c["g1"] is not None else 10**9, c["key"]))
    ans = q["answers"]
    every = bool(ans) and all(any(covers(c, a) for c in pieces) for a in ans)
    u = s["up_place"]
    inside, written = pack_lines(ordered, ans)
    return {
        "first": (u == 1) if ans else None,
        "top3": (u is not None and u <= 3) if ans else None,
        "pack": every if ans else None,
        "up": s["every_up"] if ans else None,
        "tokens": sum(c["tokens"] for c in pieces),
        "n": len(pieces),
        "order": [c["key"] for c in ordered],
        "rr": recip_rank(ordered, ans),
        "lines_in": inside,
        "lines": written,
    }


def fit_band(q, cands):
    """The band (among the gains the greedy meets) that best reproduces the recorded set: a reference."""
    target = set(q["served"]["set"])
    singles = sorted({round(sum(c["S"]), 6) for c in cands} | {0.0})
    best, best_j = 0.0, -1.0
    for b in [0.0] + [x - 1e-6 for x in singles if x > 0]:
        got = set(served(q, cands, q["budget"], b))
        j = len(got & target) / max(1, len(got | target))
        if j > best_j:
            best, best_j = b, j
    return best


def run_rule(q, fn, band_mode="default"):
    cands = eligible(q)
    if fn is served and band_mode != "default":
        b = {"zero": lambda: 0.0, "spread": lambda: band_spread(q, cands), "fit": lambda: fit_band(q, cands)}[band_mode]()
        out = served(q, cands, q["budget"], b)
    else:
        out = fn(q, cands, q["budget"])
    if isinstance(out, dict):
        return list(out.get("set", [])), out.get("up")
    return list(out), None


def table(qs, arms, against, cmp):
    """Counts per measure and the paired test of every arm against `against`."""
    lines = []
    ans = [q for q in qs if q["answers"]]
    lines.append(f"questions {len(qs)} ({len(ans)} with answers); against {against}")
    lines.append(f"{'arm':<26} {'first':>6} {'top3':>6} {'pack':>6} {'up':>6} {'tok med':>8}   paired vs {against}: measure B-A (0->1/1->0, p)")
    base = arms[against]
    for name, res in arms.items():
        cnt = {m: sum(bool(res[q['id']][m]) for q in ans) for m in ("first", "top3", "pack", "up")}
        toks = sorted(res[q["id"]]["tokens"] for q in qs)
        tmed = toks[len(toks) // 2] if toks else 0
        tests = []
        if name != against:
            for m in ("first", "top3", "pack", "up"):
                up = sum(1 for q in ans if not base[q["id"]][m] and res[q["id"]][m])
                down = sum(1 for q in ans if base[q["id"]][m] and not res[q["id"]][m])
                tests.append(f"{m} {up - down:+d} ({up}/{down}, p {cmp.mcnemar(up, down):.3f})")
        lines.append(f"{name:<26} {cnt['first']:>6} {cnt['top3']:>6} {cnt['pack']:>6} {cnt['up']:>6} {tmed:>8}   {'; '.join(tests)}")
    return lines


def repro(qs, band_mode):
    """How closely the served rule reproduces the recorded set and its measures."""
    exact, jac, agree = 0, [], {m: 0 for m in ("first", "top3", "pack", "up")}
    first_same = 0
    ans = [q for q in qs if q["answers"]]
    for q in qs:
        got, _ = run_rule(q, served, band_mode)
        rec = q["served"]["set"]
        exact += set(got) == set(rec)
        jac.append(len(set(got) & set(rec)) / max(1, len(set(got) | set(rec))))
        ro = recorded(q)["order"]
        first_same += bool(got) and bool(ro) and got[0] == ro[0]
        if q["answers"]:
            a, b = measures(q, got), recorded(q)
            for m in agree:
                agree[m] += a[m] == b[m]
    n = len(qs)
    return (f"served rule, band {band_mode}: same set {exact}/{n}, mean overlap (Jaccard) {sum(jac) / max(1, n):.3f}, same first piece {first_same}/{n}; "
            f"per-question agreement on {len(ans)} answered: " + ", ".join(f"{m} {v}" for m, v in agree.items()))


def text_stems(text):
    """The stems of a text as the drones form them (`nmlex`: split at non-word characters, the parts
    of each identifier, each part's stem)."""
    import re
    out = set()
    for w in re.split(r"[^0-9A-Za-z_]+", text):
        for p in name_parts(w):
            out.add(stem(p))
    return out


def name_parts(name):
    """`nmlex::name_parts`: split at underscores, dashes, colons and lower-to-upper changes, lower case."""
    out, cur, prev_lower = [], "", False
    for ch in name:
        if ch in "_-:":
            if cur:
                out.append(cur)
            cur, prev_lower = "", False
            continue
        if ch.isupper() and prev_lower and cur:
            out.append(cur)
            cur = ""
        prev_lower = ch.islower() or ch.isdigit()
        cur += ch.lower()
    if cur:
        out.append(cur)
    return out


def stem(w):
    """`nmlex::stem`: one common ending off, then a final e, each only when three letters stay."""
    s = w.lower()
    for suf in ("ings", "ing", "ers", "er", "ed", "es", "s", "ly"):
        if s.endswith(suf) and len(s) - len(suf) >= 3:
            s = s[: -len(suf)]
            break
    if s.endswith("e") and len(s) - 1 >= 3:
        s = s[:-1]
    return s


_FILES = {}


def file_lines(root, path):
    key = (str(root), path)
    if key not in _FILES:
        try:
            _FILES[key] = (Path(root) / path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            _FILES[key] = None
    return _FILES[key]


def match_class(q, root):
    """`lexical` when some request stem occurs in the answer ranges' text, `mismatch` when none does,
    `unknown` when the material is not given or an answer file cannot be read."""
    if not q["answers"] or root is None:
        return "unknown"
    req = {s for p in q["request"]["parts"] for s in p.get("stems", [])}
    text = []
    for a in q["answers"]:
        lines = file_lines(root, a["file"])
        if lines is None:
            return "unknown"
        text.extend(lines if a["whole_file"] else lines[a["first"] - 1: a["last"]])
    return "lexical" if req & text_stems("\n".join(text)) else "mismatch"


BINARY = ("first", "top3", "pack", "up")


def perm_test(d, flips=10000, seed=1):
    """Paired permutation test on the mean of per-question differences: random sign flips, fixed
    seed, two-sided, p = (as extreme + 1) / (flips + 1)."""
    import random
    if not d:
        return 1.0
    obs = abs(sum(d))
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    hit = 0
    for _ in range(flips):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        if abs(s) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (flips + 1)


def group_row(name, qs, a, b, cmp, flips):
    """One row of the pooled table: mean score A and B with its permutation p, each binary measure's
    counts and McNemar p, mean tokens, pack precision (pooled lines and per-question median)."""
    ans = [q for q in qs if q["answers"]]
    if not ans:
        return None
    ra = [a[q["id"]]["rr"] for q in ans]
    rb = [b[q["id"]]["rr"] for q in ans]
    p = perm_test([y - x for x, y in zip(ra, rb)], flips)
    row = {"name": name, "n": len(ans), "rrA": sum(ra) / len(ans), "rrB": sum(rb) / len(ans), "p": p}
    for m in BINARY:
        up = sum(1 for q in ans if not a[q["id"]][m] and b[q["id"]][m])
        down = sum(1 for q in ans if a[q["id"]][m] and not b[q["id"]][m])
        row[m] = (sum(bool(a[q["id"]][m]) for q in ans), sum(bool(b[q["id"]][m]) for q in ans), up, down, cmp.mcnemar(up, down))
    row["tokA"] = sum(a[q["id"]]["tokens"] for q in ans) / len(ans)
    row["tokB"] = sum(b[q["id"]]["tokens"] for q in ans) / len(ans)

    def prec(r):
        li = sum(r[q["id"]]["lines_in"] for q in ans)
        lw = sum(r[q["id"]]["lines"] for q in ans)
        per = sorted(r[q["id"]]["lines_in"] / r[q["id"]]["lines"] if r[q["id"]]["lines"] else 0.0 for q in ans)
        return li, lw, (li / lw if lw else 0.0), per[len(per) // 2]
    row["precA"], row["precB"] = prec(a), prec(b)
    return row


def falls(row):
    """A group falls when its score falls with p at most 0.05 or a binary measure falls with McNemar p at most 0.05."""
    if row["rrB"] < row["rrA"] and row["p"] <= 0.05:
        return True
    return any(row[m][3] > row[m][2] and row[m][4] <= 0.05 for m in BINARY)


def pooled(qs, arms, base, cmp, roots, flips=10000):
    """The pooled paired report of every arm against `base` and the pass rule's verdict.

    Pass rule (fixed before any rule was measured): a rule is recommended when (1) the pooled mean
    score rises with permutation p at most 0.05, (2) pooled "every range in the pack" is not lower,
    (3) pooled mean pack tokens are not higher by more than 10 percent, (4) pooled pack precision is
    not lower, and (5) no single set, question kind or match class falls with its own p at most 0.05
    (score by permutation, or any binary measure by McNemar)."""
    for q in qs:
        q["_match"] = match_class(q, roots.get(q["_set"]))
    lines = []
    a = arms[base]
    for name, b in arms.items():
        if name == base:
            continue
        lines.append(f"== {name} against {base}")
        lines.append(f"{'group':<22} {'n':>3} {'score A':>8} {'score B':>8} {'p':>7}  " + "  ".join(f"{m + ' A>B (+/-, p)':<24}" for m in BINARY) + f"  {'tok A':>6} {'tok B':>6}  precision A, B (lines in/written, median)")
        groups = [("pooled", qs)]
        for key, label in (("_set", "set"), ("kind", "kind"), ("_match", "match")):
            for v in sorted({q[key] for q in qs if q["answers"]}):
                groups.append((f"{label} {v}", [q for q in qs if q[key] == v]))
        rows = []
        for gname, gq in groups:
            r = group_row(gname, gq, a, b, cmp, flips)
            if r is None:
                continue
            rows.append(r)
            bins = "  ".join(f"{r[m][0]:>3}>{r[m][1]:<3} (+{r[m][2]}/-{r[m][3]}, {r[m][4]:.3f})".ljust(24) for m in BINARY)
            pa, pb = r["precA"], r["precB"]
            lines.append(f"{gname:<22} {r['n']:>3} {r['rrA']:>8.4f} {r['rrB']:>8.4f} {r['p']:>7.4f}  {bins}  {r['tokA']:>6.0f} {r['tokB']:>6.0f}  "
                         f"{pa[2]:.4f} ({pa[0]}/{pa[1]}, {pa[3]:.4f}), {pb[2]:.4f} ({pb[0]}/{pb[1]}, {pb[3]:.4f}){'  FALLS' if falls(r) else ''}")
        P = rows[0]
        checks = {
            "score rises, p<=0.05": P["rrB"] > P["rrA"] and P["p"] <= 0.05,
            "every range in pack not lower": P["pack"][1] >= P["pack"][0],
            "tokens not >10% higher": P["tokB"] <= 1.10 * P["tokA"],
            "precision not lower": P["precB"][2] >= P["precA"][2],
            "no set, kind or match class falls": not any(falls(r) for r in rows[1:]),
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"
        lines.append(f"pass rule: {verdict} (" + "; ".join(f"{k} {'yes' if v else 'no'}" for k, v in checks.items()) + ")")
        lines.append("")
    return lines


def main(argv):
    recs = args_after(argv, "--rec")
    rules = args_after(argv, "--rule") or ["served", "defs_first", "cover_fill"]
    if not recs:
        print(__doc__)
        return 2
    against = (args_after(argv, "--against") or ["recorded"])[0]
    band_mode = (args_after(argv, "--band") or ["default"])[0]
    folds, part = (args_after(argv, "--folds") or [None])[0], (args_after(argv, "--part") or [None])[0]
    qs = load(recs, folds, part)
    cmp = load_compare()
    arms = {"recorded": {q["id"]: recorded(q) for q in qs}}
    for spec in rules:
        name, fn = get_rule(spec)
        res = {}
        for q in qs:
            chosen, up = run_rule(q, fn, band_mode)
            res[q["id"]] = measures(q, chosen, up)
        arms[name] = res
    if against == "served" and "served" not in arms:
        arms["served"] = {q["id"]: measures(q, *run_rule(q, served, band_mode)) for q in qs}
    sources = sorted({q["_source"] for q in qs})
    print(f"recordings: {', '.join(recs)} (source {', '.join(sources)})")
    for l in table(qs, arms, against, cmp):
        print(l)
    if "--pooled" in argv:
        roots = dict(x.split("=", 1) for x in args_after(argv, "--root"))
        base = (args_after(argv, "--base") or ["recorded"])[0]
        print()
        for l in pooled(qs, arms, base, cmp, roots):
            print(l)
    if "--repro" in argv:
        for b in ("default", "spread", "fit"):
            print(repro(qs, b))
    if "--list" in argv:
        print()
        names = list(arms)
        print("id      kind      " + "  ".join(f"{n[:12]:<12}" for n in names) + "   (first top3 pack up)")
        for q in qs:
            if not q["answers"]:
                continue
            row = []
            for n in names:
                r = arms[n][q["id"]]
                row.append(f"{''.join('1' if r[m] else '0' for m in ('first', 'top3', 'pack', 'up')):<12}")
            print(f"{q['id']:<7} {q['kind']:<9} " + "  ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
