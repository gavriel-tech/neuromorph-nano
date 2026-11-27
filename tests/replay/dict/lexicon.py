"""The project's own dictionary and the self-made exam: the shared pieces.

One tokenizer (identifier parts split on case and underscores, stemmed as `nmlex::stem` does, so
stems agree with the request drone's recorded stems), one doc comment pairer, one dictionary
builder, one scorer. Standard library only; prototype for the lead, written 2026-11-17.
"""
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MATERIAL = ROOT / "tests" / "dev-large" / "material-code"
PIECE = 30
SEED = 20261117

# --- tokenizer (ports of nmlex::stem, nmlex::name_parts, nmpod intake stems_of) -------------

SUFFIXES = ("ings", "ing", "ers", "er", "ed", "es", "s", "ly")


def stem(w):
    s = w.lower()
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) - len(suf) >= 3:
            s = s[: -len(suf)]
            break
    if s.endswith("e") and len(s) - 1 >= 3:
        s = s[:-1]
    return s


def name_parts(name):
    out, cur, prev_lower = [], "", False
    for c in name:
        if c in "_-:":
            if cur:
                out.append(cur)
                cur = ""
            prev_lower = False
            continue
        if c.isupper() and prev_lower and cur:
            out.append(cur)
            cur = ""
        prev_lower = c.islower() or c.isdigit()
        cur += c.lower()
    if cur:
        out.append(cur)
    return out


WORD = re.compile(r"[A-Za-z0-9_]+")


def keep(s):
    return len(s) >= 2 and not s.isdigit() and any(ch.isalpha() for ch in s)


def stems_of(text):
    """Stems of every identifier part of `text` (the one tokenizer used everywhere)."""
    out = []
    for w in WORD.findall(text):
        for p in name_parts(w):
            s = stem(p)
            if keep(s):
                out.append(s)
    return out


# nmlex::STOP, the request drone's own stop list; used only to count an exam sentence's content
# words the way the request drone would (never in the dictionary).
STOP = set("""what where which who whom does did the and for are how why when this that with from into its
use uses used using defined define defines definition crate crates function functions file files struct
type module called name named there any depend depends dependency dependencies on is in of to a an do
has have list show find all repo repository code about say says mean means work works happen get set can
should would will not you our their it be by at as or if value constant line trait enum directly normal
workspace emma-labs tested recently changed""".split())


def content_stems(text):
    """The request drone's key terms of plain text, stemmed: words of 3 letters or more, not STOP."""
    out = []
    for w in re.split(r"[^A-Za-z0-9_\-.]+", text):
        w = w.strip(".-?").lower()
        if len(w) >= 3 and w not in STOP and not w.isdigit():
            for p in name_parts(w):
                s = stem(p)
                if keep(s):
                    out.append(s)
    return out


# --- material -------------------------------------------------------------------------------

TEXT_KINDS = None  # every file is read as text; undecodable bytes are replaced


def material_files(root=MATERIAL):
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            try:
                files[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
    return files


def stratum(rel):
    parts = rel.split("/")
    if parts[0] == "crates" and len(parts) > 2:
        return "crates/" + parts[1]
    return parts[0] if len(parts) > 1 else "(top)"


def file_kind(rel):
    ext = rel.rsplit(".", 1)[-1].lower() if "." in rel.rsplit("/", 1)[-1] else ""
    return {"rs": "rust", "ts": "ts/tsx", "tsx": "ts/tsx", "js": "ts/tsx", "mjs": "ts/tsx",
            "cu": "cuda/c", "cuh": "cuda/c", "h": "cuda/c", "hpp": "cuda/c", "c": "cuda/c",
            "py": "python", "md": "markdown"}.get(ext, "other")


def comment_line(line):
    t = line.lstrip()
    return ((t.startswith("#") and not t.startswith(("#[", "#!", "#define", "#include")))
            or t.startswith(("//", "/*", "*", "--", ";", "<!--", '"""', "'''")))


# --- doc comment pairing -------------------------------------------------------------------

NAME_RES = [
    re.compile(r"\b(?:fn|struct|enum|trait|const|static|type|mod|union|class|interface|function|let|var)\s+([A-Za-z_]\w*)"),
    re.compile(r"\bmacro_rules!\s*([A-Za-z_]\w*)"),
    re.compile(r"\bimpl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:pub(?:\([\w:]+\))?\s+)?([A-Za-z_]\w*)\s*[:(=,{]"),
]


SHAPES = (("fn", ("fn", "function", "def")), ("type", ("struct", "class", "interface", "enum", "union", "type")),
          ("const", ("const", "static", "let", "var")), ("block", ("impl", "trait", "mod", "macro_rules")))


def item_shape(line):
    """The item's syntactic shape from its declaration keyword (a parse, not a word list)."""
    ws = set(re.findall(r"[A-Za-z_]+", line))
    for shape, kws in SHAPES:
        if ws & set(kws):
            return shape
    return "field"


def item_name(line):
    for r in NAME_RES:
        m = r.search(line)
        if m:
            return m.group(1)
    m = WORD.search(line)
    return m.group(0) if m else ""


def brace_end(lines, start):
    """Last line (0-based) of the item that starts at `start`: to the matching brace, or to the
    first `;` or `,` that ends a declaration without a body."""
    depth, opened = 0, False
    for i in range(start, min(len(lines), start + 4000)):
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[i])
        line = line.split("//", 1)[0]
        for ch in line:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth <= 0:
                    return i
            elif ch in ";" and not opened and depth == 0:
                return i
        if not opened and depth == 0 and line.rstrip().endswith(","):
            return i
    return start


def strip_doc(line):
    t = line.strip()
    for p in ("///", "//!", "/**", "*/", "*", "//"):
        if t.startswith(p):
            t = t[len(p):]
            break
    if t.endswith("*/"):
        t = t[:-2]
    return t.strip()


def doc_items(rel, lines):
    """Every documented item of a file: dict(file, kind, doc_lines (0-based list), doc text,
    name, item first/last (0-based, inclusive))."""
    kind = file_kind(rel)
    out = []
    n = len(lines)
    if kind in ("rust", "ts/tsx", "cuda/c"):
        i = 0
        while i < n:
            t = lines[i].lstrip()
            block = None
            if t.startswith("///") and not t.startswith("////"):
                j = i
                while j < n and lines[j].lstrip().startswith("///"):
                    j += 1
                block = (i, j)
            elif t.startswith("/**") and not t.startswith("/**/"):
                j = i
                while j < n and "*/" not in lines[j]:
                    j += 1
                block = (i, min(j + 1, n))
            if block is None:
                i += 1
                continue
            a, b = block
            k = b
            while k < n and (lines[k].lstrip().startswith(("#[", "@")) or not lines[k].strip()):
                if not lines[k].strip() and k > b:
                    break
                k += 1
            if k < n and lines[k].strip() and not comment_line(lines[k]):
                end = brace_end(lines, k)
                doc = " ".join(strip_doc(lines[x]) for x in range(a, b))
                out.append({"file": rel, "kind": kind, "doc_lines": list(range(a, b)), "doc": doc,
                            "name": item_name(lines[k]), "shape": item_shape(lines[k]), "first": k, "last": end})
            i = b
    elif kind == "python":
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)(?:def|class)\s+([A-Za-z_]\w*)", line)
            if not m or i + 1 >= n:
                continue
            j = i + 1
            while j < n and not lines[j].rstrip().endswith(":") and j < i + 10:
                j += 1
            d = j + 1
            if d >= n or not lines[d].lstrip().startswith(('"""', "'''")):
                continue
            q = lines[d].lstrip()[:3]
            e = d if lines[d].count(q) >= 2 else next((x for x in range(d + 1, n) if q in lines[x]), d)
            ind = len(m.group(1))
            end = e
            for x in range(e + 1, n):
                if lines[x].strip() and len(lines[x]) - len(lines[x].lstrip()) <= ind:
                    break
                end = x
            doc = " ".join(lines[x].strip().strip(q) for x in range(d, e + 1))
            out.append({"file": rel, "kind": kind, "doc_lines": list(range(d, e + 1)), "doc": doc,
                        "name": m.group(2), "shape": "fn", "first": i, "last": end})
    elif kind == "markdown":
        i = 0
        while i < n:
            if lines[i].lstrip().startswith("```"):
                j = i + 1
                while j < n and not lines[j].lstrip().startswith("```"):
                    j += 1
                p = i - 1
                while p >= 0 and not lines[p].strip():
                    p -= 1
                if p >= 0 and i - p <= 2:
                    a = p
                    while a - 1 >= 0 and lines[a - 1].strip() and not lines[a - 1].lstrip().startswith(("#", "```", "|")):
                        a -= 1
                    para = [x for x in range(a, p + 1) if not lines[x].lstrip().startswith(("#", "```", "|"))]
                    if para and j > i + 1:
                        out.append({"file": rel, "kind": kind, "doc_lines": para,
                                    "doc": " ".join(lines[x].strip() for x in para), "name": "", "shape": "block",
                                    "first": i + 1, "last": j - 1})
                i = j + 1
            else:
                i += 1
    return out


BACKTICK = re.compile(r"`[^`]*`")
IDENT_SHAPED = re.compile(r"\b\w*(?:_\w|[a-z][A-Z])\w*\b|\b\w+::\w+\b|\b\w+\(\)")


def plain(text):
    """Plain words of a comment: backticked spans and identifier-shaped tokens taken out."""
    return IDENT_SHAPED.sub(" ", BACKTICK.sub(" ", text))


def code_stems(lines, first, last):
    """Identifier part stems of an item's body, comment lines left out."""
    out = set()
    for x in range(first, last + 1):
        if not comment_line(lines[x]):
            out.update(stems_of(lines[x]))
    return out


def pairs(files, items):
    """(W, C) per documented item: the comment's plain word stems, the item's name and body
    identifier part stems."""
    out = []
    for it in items:
        lines = files[it["file"]]
        w = set(stems_of(plain(it["doc"])))
        c = code_stems(lines, it["first"], it["last"])
        c.update(stem(p) for p in name_parts(it["name"]) if keep(stem(p)))
        if w and c:
            out.append((w, c, it.get("shape", "field")))
    return out


# --- the dictionary ------------------------------------------------------------------------

def binom_tail(k, n, p, logf):
    """P(X >= k), X ~ Bin(n, p), summed from k up until the terms vanish."""
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total, x = 0.0, k
    while x <= n:
        t = math.exp(logf[n] - logf[x] - logf[n - x] + x * lp + (n - x) * lq)
        total += t
        if x > n * p and t < total * 1e-12:
            break
        x += 1
    return min(1.0, total)


def poisson_tail(k, mu):
    """P(X >= k), X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    t = math.exp(-mu)
    cdf = t
    for x in range(1, k):
        t *= mu / x
        cdf += t
    return max(0.0, 1.0 - cdf) if cdf < 0.999999 else _poisson_upper(k, mu)


def _poisson_upper(k, mu):
    t = math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if mu > 0 else 0.0
    total, x = 0.0, k
    while t > 0 and (t > total * 1e-12 or x < k + 2):
        total += t
        x += 1
        t *= mu / x
    return min(1.0, total)


def build_dictionary(pair_list, alpha=0.01, correction="bh", strata=False):
    """Links w -> {c: strength}. A pair is tested by the one-sided test of its joint count against
    independence: binomial X ~ Bin(n_w, n_c / N), p = P(X >= n_wc); with strata=True the expected
    joint count is summed within item shapes (fn, type, const, block, field), mu = sum over shapes
    of n_w,k n_c,k / N_k, and the tail is Poisson(mu), so a part that merely marks an item's shape
    (`fn`, `let`) is not linked to the words that describe that shape. Kept at level alpha after
    the correction for the number of pairs tested ("none", "bh": Benjamini-Hochberg false discovery
    rate, "bonferroni"). Strength = (lower bound of P(c|w) - P0) / (1 - P0), P0 the chance
    expected under independence (n_c / N, or mu / n_w with strata), the lower bound the one-sided
    Wilson bound at the same alpha: the share of w's items holding c beyond what c's own spread
    predicts, counted only as far as the evidence supports it. Self links (c == w) left out."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha)
    N = len(pair_list)
    nw, nc, joint = Counter(), Counter(), Counter()
    nwk, nck, Nk = Counter(), Counter(), Counter()
    for w, c, shape in pair_list:
        nw.update(w)
        nc.update(c)
        Nk[shape] += 1
        if strata:
            nwk.update((a, shape) for a in w)
            nck.update((b, shape) for b in c)
        for a in w:
            for b in c:
                if a != b:
                    joint[(a, b)] += 1
    logf = [0.0] * (N + 2)
    for i in range(1, N + 2):
        logf[i] = logf[i - 1] + math.log(i)
    shapes = sorted(Nk)

    def p0(a, b):
        if strata:
            mu = sum(nwk[(a, k)] * nck[(b, k)] / Nk[k] for k in shapes)
            return mu / nw[a]
        return nc[b] / N

    tested = []
    for (a, b), k in joint.items():
        n, pc = nw[a], p0(a, b)
        if k <= n * pc:
            pv = 1.0
        elif strata:
            pv = poisson_tail(k, n * pc)
        else:
            pv = binom_tail(k, n, pc, logf)
        tested.append((pv, a, b, k))
    m = len(tested)
    if correction == "bonferroni":
        cut = alpha / m
    elif correction == "bh":
        tested.sort()
        cut = 0.0
        for i, (pv, _, _, _) in enumerate(tested, 1):
            if pv <= alpha * i / m:
                cut = pv
    else:
        cut = alpha
    links = defaultdict(dict)
    for pv, a, b, k in tested:
        if pv > cut or pv > alpha:
            continue
        n, pc = nw[a], p0(a, b)
        ph = k / n
        lb = (ph + z * z / (2 * n) - z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        s = (lb - pc) / (1.0 - pc) if pc < 1.0 else 0.0
        if s > 0:
            links[a][b] = s
    return dict(links), {"pairs": N, "words": len(nw), "parts": len(nc), "pairs_tested": m,
                         "p_cut": cut, "links": sum(len(v) for v in links.values()),
                         "linked_words": len(links)}


# --- pieces and scorer -----------------------------------------------------------------------

def line_stems(files):
    return {rel: [set(stems_of(l)) for l in lines] for rel, lines in files.items()}


def material_pieces(files, ls):
    """The material's 30-line pieces: (file, first, last) 1-based, and each piece's stem set."""
    keys, sets = [], []
    for rel, lines in files.items():
        for a in range(0, max(1, len(lines)), PIECE):
            b = min(len(lines), a + PIECE)
            s = set()
            for x in range(a, b):
                s |= ls[rel][x]
            keys.append((rel, a + 1, max(a + 1, b)))
            sets.append(s)
    return keys, sets


def rarity(sets):
    df = Counter()
    for s in sets:
        df.update(s)
    N = len(sets)
    return lambda w: math.log((N + 1) / (df.get(w, 0) + 0.5))


def score(qstems, pset, idf, links=None):
    """Rarity-weighted share of the question's stems the piece holds. With links, a stem w the
    piece lacks is credited by its strongest linked part present (strength times the rarer-capped
    rarity min(rarity of w, rarity of c), so a part spread over most pieces adds little);
    w itself always counts in full and is never replaced."""
    num = den = 0.0
    for w in qstems:
        r = idf(w)
        den += r
        if w in pset:
            num += r
        elif links and w in links:
            best = 0.0
            for c, s in links[w].items():
                if c in pset:
                    v = s * min(r, idf(c))
                    if v > best:
                        best = v
            num += best
    return num / den if den else 0.0


def mid_rank(scores, target_idx):
    """Rank of the best target piece, ties shared (1 + strictly greater + half the other ties)."""
    best = max(scores[i] for i in target_idx)
    gt = sum(1 for s in scores if s > best)
    eq = sum(1 for s in scores if s == best) - 1
    return 1 + gt + eq / 2.0


def perm_test(a, b, n=10000, seed=SEED):
    """Paired sign flip test on mean(b - a), two-sided."""
    d = [y - x for x, y in zip(a, b)]
    obs = abs(sum(d))
    rng = random.Random(seed)
    hit = 0
    for _ in range(n):
        if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n + 1)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
