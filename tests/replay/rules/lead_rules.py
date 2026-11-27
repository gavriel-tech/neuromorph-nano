"""The lead's selection rules R0 to R5, replayed on a recording (`tests/replay/replay.py --rule
tests/replay/rules/lead_rules.py:<name>`). Standard library only.

Every number a rule uses comes from the material (rarities, co-occurrence, the share of pieces that
score as high), from the request's own candidates, or from the constant k of R2 and R3, shown at
three values. The material is read from disk once per run and held in memory only.

Material: the files the bench reads (a git root: `git ls-files -co --exclude-standard`; otherwise a
walk that skips hidden folders, `target`, `node_modules` and the root `.gitignore`'s plain names),
text files under 1 MB, cut into 30-line pieces (the drones' cut for code). Units for co-occurrence:
the spans between definition name lines (a Markdown heading, a TOML table or key, and so on).

Rules (each returns the chosen keys in written order):
- R1: aspects from the material. The request's stems (the union of the drone's parts' stems), each
  weighted by its rarity over the material's pieces (inverse document frequency). Two stems join one
  aspect when they co-occur inside the same unit more often than independence predicts (one-sided
  binomial test, p at most 0.05); joined pairs form groups (connected components); every other stem
  is its own aspect. An aspect's weight is its stems' rarity sum, scaled so the aspects' weights sum
  to the number of aspects. A piece's match score on an aspect is the rarity sum of the aspect's
  stems it holds. Its chance is rank-calibrated against the material: 1 minus the share of all
  material pieces that score at least as high on that aspect (0 for a score of 0). Selection: the
  budgeted greedy on F (guide 12.1) with G2's relation and fact columns as recorded, the band as
  G2's (the median absolute deviation of the positive single values), the set written in G1's order.
- R2(k): as R1, a stem on a definition name line counts k times; a comment, doc comment or string
  occurrence counts once, as does an ordinary code line.
- R3(k): as R2, with credit spread like Aider: a stem on a definition line gives k x rarity / d,
  d the number of files defining it; any other occurrence gives rarity x (file rarity: ln(Nf / f) /
  ln(Nf), f the number of files holding it).
- R4: partners through typed edges. For each relation the request's drone names, a relation aspect
  (weight 1, as G2's relation columns): a candidate settles it when a piece already in the set links
  to it by a dumped edge and its own text confirms the link (it defines an identifier the anchor's
  code uses, or uses one the anchor defines); its chance is the anchor's best chance times the
  rarity-weighted share of the anchor's linked identifiers it confirms. The partner is written right
  after its anchor.
- R5: the set written in the greedy's order of taking (largest gain in F per token first), each
  partner right after its anchor.
- R0: R1's form with hand parts (`scratch/lead-replay/hand-parts.json`) as the aspects, where they
  exist. Diagnostic only.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
NANO = HERE.parent.parent.parent

ROOTS = {
    "large-code": NANO / "tests/dev-large/material-code",
    "large-docs": NANO / "tests/dev-large/material-docs",
    "dev": NANO / "scratch/frozen-p2",
    "closure": NANO / "scratch/frozen-p2",
    "cm1-r1": NANO.parent / "neuromorph-mini",
    "cm1-r2": NANO.parent / "neuromorph-mini",
}
HAND = NANO / "scratch/lead-replay/hand-parts.json"
PIECE = 30


def _replay():
    spec = importlib.util.spec_from_file_location("replay_lead", HERE.parent / "replay.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RP = _replay()
stem, name_parts = RP.stem, RP.name_parts

WORD = re.compile(r"[0-9A-Za-z_]+")
C_LIKE = {"rs", "c", "h", "cc", "cpp", "hpp", "cu", "cuh", "js", "jsx", "ts", "tsx", "go", "java", "kt", "swift", "cs", "css", "scss", "wgsl", "glsl", "hlsl"}
HASH = {"py", "sh", "bash", "toml", "yaml", "yml", "ps1", "rb", "cmake", "ini", "cfg", "conf", "mk", "gitignore"}
DOCS = {"md", "markdown", "mdx", "txt", "text", "rst", "adoc", "org", "html", "htm"}
DEF_RE = {
    "rs": re.compile(r"^\s*(pub(\([^)]*\))?\s+)?((async|const|unsafe|default|extern(\s+\"[^\"]*\")?)\s+)*(fn|struct|enum|trait|type|const|static|mod|impl|union)\b|^\s*macro_rules!"),
    "py": re.compile(r"^\s*(async\s+)?(def|class)\s"),
    "js": re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?(function|class|interface|type|enum|const|let|var)\s"),
    "c": re.compile(r"^\s*(typedef|struct|enum|union|#\s*define|template|class|__global__|__device__|static\s+\w)"),
    "toml": re.compile(r"^\s*(\[|[A-Za-z0-9_.\"-]+\s*=)"),
    "json": re.compile(r"^\s*\"[^\"]+\"\s*:"),
    "yaml": re.compile(r"^\s*[A-Za-z0-9_-]+\s*:"),
    "md": re.compile(r"^#{1,6}\s"),
}
LANG = {"rs": "rs", "py": "py", "js": "js", "jsx": "js", "ts": "js", "tsx": "js", "c": "c", "h": "c", "cc": "c", "cpp": "c", "hpp": "c", "cu": "c", "cuh": "c",
        "toml": "toml", "json": "json", "yaml": "yaml", "yml": "yaml", "md": "md", "markdown": "md", "mdx": "md"}
DEF_NAME = re.compile(r"\b(?:fn|struct|enum|trait|type|const|static|mod|union|def|class|function|interface|let|var|macro_rules!)\s+([A-Za-z_][A-Za-z0-9_]*)")


def stems_of(text):
    out = set()
    for w in WORD.findall(text):
        for p in name_parts(w):
            out.add(stem(p))
    return out


def split_line(line, ext, state):
    """A code line's code part and its comment and string parts. `state` carries an open block comment."""
    if ext in DOCS:
        return "", line, state
    hash_c = ext in HASH
    code, other, i, n = [], [], 0, len(line)
    while i < n:
        if state["block"]:
            j = line.find("*/", i)
            if j < 0:
                other.append(line[i:])
                return "".join(code), " ".join(other), state
            other.append(line[i:j])
            i = j + 2
            state["block"] = False
            continue
        ch = line[i]
        if not hash_c and line.startswith("//", i):
            other.append(line[i + 2:])
            break
        if hash_c and ch == "#" and ext not in ("rs",):
            other.append(line[i + 1:])
            break
        if not hash_c and line.startswith("/*", i):
            state["block"] = True
            i += 2
            continue
        if ch == '"' or (ch == "'" and ext in ("py", "js", "jsx", "ts", "tsx", "sh", "toml", "yaml", "yml")):
            j = i + 1
            while j < n and line[j] != ch:
                j += 2 if line[j] == "\\" else 1
            other.append(line[i + 1:j])
            i = j + 1
            continue
        code.append(ch)
        i += 1
    return "".join(code), " ".join(other), state


class FileInfo:
    """Per line: (stems on a definition name line's code part, stems on other code, stems in comments
    and strings), the identifiers defined and used on code lines."""

    def __init__(self, path, lines):
        ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        lang = LANG.get(ext)
        dre = DEF_RE.get(lang) if lang else None
        state = {"block": False}
        self.lines = []
        self.defs = []
        self.ids = []
        self.is_def = []
        for line in lines:
            code, other, state = split_line(line, ext, state)
            d = bool(dre and dre.search(line)) if lang != "md" else bool(dre.search(line))
            if lang == "md" and d:
                cs, os_ = stems_of(line), set()
                self.lines.append((cs, set(), os_))
            else:
                cs = stems_of(code)
                self.lines.append((cs if d else set(), set() if d else cs, stems_of(other)))
            self.is_def.append(d)
            names = set(DEF_NAME.findall(code)) if d else set()
            if d and lang in ("toml", "json", "yaml"):
                m = re.match(r"^\s*\"?([A-Za-z0-9_.-]+)", line)
                if m:
                    names.add(m.group(1))
            self.defs.append(names)
            self.ids.append(set(WORD.findall(code)) - names)
        # R8, the role "describes": the comment lines directly above a definition name line (doc
        # comments and plain comments, attribute lines passed over), not a file's inner notes (`//!`).
        self.desc = [set() for _ in self.lines]
        for i, d in enumerate(self.is_def):
            if not d or lang == "md":
                continue
            j = i - 1
            while j >= 0:
                raw = lines[j].strip()
                if raw.startswith("#[") or raw.startswith("@"):
                    j -= 1
                    continue
                dl, cl, ol = self.lines[j]
                if raw and not dl and not cl and ol and not raw.startswith("//!") and not raw.startswith("/*!"):
                    self.desc[j] = ol
                    j -= 1
                    continue
                break

    def piece(self, a, b):
        """stem -> 2 when it occurs on a definition name line in lines a..b (1-based), 3 when only in a
        comment describing a definition (R8), else 1."""
        out = {}
        lo = max(0, a - 1)
        for i, (d, c, o) in enumerate(self.lines[lo: b]):
            for s in d:
                out[s] = 2
            for s in self.desc[lo + i]:
                if out.get(s, 0) < 2:
                    out[s] = 3
            for s in c | o:
                out.setdefault(s, 1)
        return out

    def piece_ids(self, a, b):
        defs, used = set(), set()
        for i in range(max(0, a - 1), min(b, len(self.lines))):
            defs |= self.defs[i]
            used |= self.ids[i]
        return defs, used


def material_files(root):
    root = Path(root)
    try:
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=60)
        top = Path(r.stdout.strip()).resolve() if r.returncode == 0 else None
    except OSError:
        top = None
    if top is not None and top == root.resolve():
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"], capture_output=True, timeout=300)
        names = [x.decode("utf-8", "replace") for x in r.stdout.split(b"\0") if x]
        return [n for n in names if not any(part.startswith(".") for part in n.split("/")[:-1])]
    # The root `.gitignore`'s plain names: anchored ones (a leading `/`) at the root only, the others
    # at any depth; patterns with wildcards or inner slashes are not read.
    anchored, anywhere = set(), set()
    gi = root / ".gitignore"
    if gi.exists():
        for l in gi.read_text(encoding="utf-8", errors="replace").splitlines():
            l = l.strip()
            if not l or l.startswith("#") or l.startswith("!"):
                continue
            body = l.strip("/")
            if not body or any(ch in body for ch in "*?[/"):
                continue
            (anchored if l.startswith("/") else anywhere).add(body)
    out = []
    for dp, dns, fns in os.walk(root):
        top = Path(dp).resolve() == root.resolve()
        dns[:] = [d for d in dns if not d.startswith(".") and d not in ("target", "node_modules") and d not in anywhere and not (top and d in anchored)]
        for f in fns:
            if f in anywhere or (top and f in anchored):
                continue
            out.append(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    return out


class Material:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        for name in material_files(root):
            p = self.root / name
            try:
                if p.stat().st_size > 1_000_000:
                    continue
                raw = p.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:8192]:
                continue
            self.files[name] = FileInfo(name, raw.decode("utf-8", "replace").splitlines())
        self.pieces = []
        self.df = {}
        self.index = {}
        file_has, file_def = {}, {}
        units, name_units = [], []
        for name, fi in self.files.items():
            n = len(fi.lines)
            for a in range(1, max(n, 1) + 1, PIECE):
                pc = fi.piece(a, a + PIECE - 1)
                k = len(self.pieces)
                self.pieces.append(pc)
                for s in pc:
                    self.df[s] = self.df.get(s, 0) + 1
                    self.index.setdefault(s, []).append(k)
            has, dfs = set(), set()
            cur = set()
            for i, (d, c, o) in enumerate(fi.lines):
                if fi.is_def[i]:
                    ns = stems_of(" ".join(fi.defs[i])) if fi.defs[i] else (d if name.rsplit(".", 1)[-1].lower() in ("md", "markdown", "mdx") else set())
                    if ns:
                        name_units.append(ns)
                if fi.is_def[i] and cur:
                    units.append(cur)
                    cur = set()
                cur |= d | c | o
                has |= d | c | o
                dfs |= d
            if cur:
                units.append(cur)
            for s in has:
                file_has[s] = file_has.get(s, 0) + 1
            for s in dfs:
                file_def[s] = file_def.get(s, 0) + 1
        self.N = max(1, len(self.pieces))
        self.Nf = max(2, len(self.files))
        self.file_has, self.file_def = file_has, file_def
        self.units = {"span": units, "name": name_units}
        self.unit_index = {}
        for kind, us in self.units.items():
            ix = {}
            for k, u in enumerate(us):
                for s in u:
                    ix.setdefault(s, set()).add(k)
            self.unit_index[kind] = ix
        self.dist_cache = {}
        self.id_df = None

    def idf(self, s):
        d = self.df.get(s, 0)
        return math.log(self.N / d) if d else 0.0

    def fr(self, s):
        f = self.file_has.get(s, 0)
        return math.log(self.Nf / f) / math.log(self.Nf) if f else 0.0

    def credit(self, s, flag, mode, k):
        """One stem's credit in a piece: flag 2 on a definition name line, 3 in a describing comment
        (counted as the name line in modes `r2d` and `r3d`, R8), 1 elsewhere."""
        i = self.idf(s)
        if mode == "r1":
            return i
        named = flag == 2 or (flag == 3 and mode.endswith("d"))
        if mode.startswith("r2"):
            return i * (k if named else 1.0)
        mention = i * self.fr(s)
        if named:
            return max(k * i / max(1, self.file_def.get(s, 1)), mention)
        return mention

    def score(self, pc, aspect, mode, k):
        return sum(self.credit(s, pc[s], mode, k) for s in aspect if s in pc)

    def dist(self, aspect, mode, k):
        """The positive scores of all material pieces on an aspect, sorted (for rank calibration)."""
        key = (tuple(sorted(aspect)), mode, k)
        if key not in self.dist_cache:
            ids = set()
            for s in aspect:
                ids.update(self.index.get(s, ()))
            self.dist_cache[key] = sorted(self.score(self.pieces[i], aspect, mode, k) for i in ids)
        return self.dist_cache[key]

    def tail(self, aspect, mode, k, s):
        """The share of material pieces scoring at least s on the aspect."""
        import bisect
        if s <= 0:
            return 1.0
        d = self.dist(aspect, mode, k)
        return (len(d) - bisect.bisect_left(d, s - 1e-12)) / self.N

    def cooccur_p(self, a, b, unit="span"):
        """One-sided binomial test that stems a and b share a unit more often than independence predicts.
        `span`: the span from one definition name line to the next; `name`: the defined name itself
        (the parts of the identifier, or a heading's words)."""
        n = len(self.units[unit])
        ua, ub = self.unit_index[unit].get(a, set()), self.unit_index[unit].get(b, set())
        if not ua or not ub or n == 0:
            return 1.0
        x = len(ua & ub)
        p = (len(ua) / n) * (len(ub) / n)
        if x <= n * p:
            return 1.0
        return binom_tail(n, x, p)

    def piece_of(self, path, a, b):
        fi = self.files.get(path)
        return fi.piece(a, b) if fi else {}

    def ids_of(self, path, a, b):
        fi = self.files.get(path)
        return fi.piece_ids(a, b) if fi else (set(), set())

    def ident_idf(self, name):
        if self.id_df is None:
            self.id_df = {}
            for fi in self.files.values():
                n = len(fi.lines)
                for a in range(0, n, PIECE):
                    seen = set()
                    for i in range(a, min(n, a + PIECE)):
                        seen |= fi.ids[i] | fi.defs[i]
                    for x in seen:
                        self.id_df[x] = self.id_df.get(x, 0) + 1
        d = self.id_df.get(name, 0)
        return math.log(self.N / d) if d else 0.0


def binom_tail(n, x, p):
    """P(X >= x) for X ~ Binomial(n, p), summed in log space from x up until the terms vanish."""
    if p <= 0:
        return 0.0 if x > 0 else 1.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total, k = 0.0, x
    lg = math.lgamma(n + 1)
    while k <= n:
        t = math.exp(lg - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
        total += t
        if t < total * 1e-12:
            break
        k += 1
    return min(1.0, total)


_MAT = {}


def material(setname):
    root = ROOTS.get(setname)
    if root is None:
        return None
    if setname not in _MAT:
        key = str(Path(root).resolve())
        same = next((m for m in _MAT.values() if m is not None and str(m.root.resolve()) == key), None)
        _MAT[setname] = same or Material(root)
    return _MAT[setname]


def request_stems(q):
    out = []
    for p in q["request"]["parts"]:
        for s in p.get("stems", []):
            if s not in out:
                out.append(s)
    return out


def group_aspects(mat, stems, unit="span"):
    """Stems joined when they co-occur in the material's units beyond independence (p at most 0.05)."""
    stems = [s for s in stems if mat.idf(s) > 0]
    parent = {s: s for s in stems}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            if mat.cooccur_p(a, b, unit) <= 0.05:
                parent[find(a)] = find(b)
    groups = {}
    for s in stems:
        groups.setdefault(find(s), []).append(s)
    return list(groups.values())


def aspects_for(q, mat, how):
    if how == "parts":
        asp = [[s for s in p.get("stems", []) if mat.idf(s) > 0] for p in q["request"]["parts"]]
        return [a for a in asp if a]
    if how == "hand":
        hp = hand_parts().get(q["id"])
        if hp:
            asp = [[s for s in stems_of(t) if mat.idf(s) > 0] for t in hp]
            return [a for a in asp if a]
    return group_aspects(mat, request_stems(q), "name" if how == "name" else "span")


_HAND = None


def hand_parts():
    global _HAND
    if _HAND is None:
        _HAND = json.loads(HAND.read_text(encoding="utf-8"))["parts"] if HAND.exists() else {}
    return _HAND


def chances(q, cands, mode="r1", k=1.0, how="name", cal="tail"):
    """Per candidate the rank-calibrated chance on each aspect, and the aspects' weights. `tail`: 1 minus
    the share t of material pieces scoring as high; `noise`: 1 minus n x t (n the number of candidates,
    so n x t is how many candidates would score as high by chance), floored at 0."""
    mat = material(q["_set"])
    if mat is None:
        return None
    asp = aspects_for(q, mat, how)
    if not asp:
        return None
    raw = [sum(mat.idf(s) for s in a) for a in asp]
    tot = sum(raw)
    w = [len(asp) * x / tot for x in raw] if tot > 0 else [1.0] * len(asp)
    r = {}
    for c in cands:
        pc = mat.piece_of(c["path"], c["first"], c["last"])
        t = [mat.tail(a, mode, k, mat.score(pc, a, mode, k)) for a in asp]
        r[c["key"]] = [1.0 - x for x in t] if cal == "tail" else [max(0.0, 1.0 - len(cands) * x) for x in t]
    return asp, w, r


def held_above_noise(q, cands, mode="r1", k=1.0, how="name"):
    """Per aspect the keys holding it above the content-free noise level: tail share below one over the
    number of candidates (fewer than one candidate would score as high by chance)."""
    got = chances(q, cands, mode, k, how)
    if got is None:
        return None
    asp, w, r = got
    n = max(1, len(cands))
    return asp, {c["key"]: [x > 1.0 - 1.0 / n for x in r[c["key"]]] for c in cands}


def partner_evidence(mat, anchor, cand):
    """The share (rarity-weighted) of the anchor's linked identifiers the candidate confirms: names the
    anchor uses that the candidate defines, and names the anchor defines that the candidate uses."""
    a_def, a_use = mat.ids_of(anchor["path"], anchor["first"], anchor["last"])
    c_def, c_use = mat.ids_of(cand["path"], cand["first"], cand["last"])
    links = (a_use & c_def) | (a_def & c_use)
    if not links:
        return 0.0
    pool = a_use | a_def
    tot = sum(mat.ident_idf(x) for x in pool)
    return sum(mat.ident_idf(x) for x in links) / tot if tot > 0 else 0.0


def select(q, cands, budget, r, w, partners=False, rel_all=False, recorded_band=False, band_zero=False):
    """The budgeted greedy on F: word aspects weighted, G2's fact and relation columns as recorded
    (weight 1 each), and with `partners` the relation aspects settled by edge partners. Returns the
    picks in order of taking, each with the anchor it partners (or None), and the stop reason."""
    nw = len(w)
    facts = q.get("facts") or []
    fact_cols = [(p, 1.0 / max(1, len(f["pieces"]))) for f in facts for p in f["pieces"]]
    fact_rel = {f["relation"] for f in facts}
    kept = [k for k in range(len(q["request"].get("relations", []))) if k not in fact_rel]
    pairs = {}
    for p_, q_, k in q.get("rel_pairs") or []:
        if k in kept:
            pairs.setdefault((p_, q_), []).append(kept.index(k))
    n_rel = len(kept) if pairs else 0
    n_part = (max(1, len(q["request"].get("relations", []))) if rel_all else len(q["request"].get("relations", []))) if partners else 0
    W = list(w) + [1.0] * len(fact_cols) + [1.0] * n_rel + [1.0] * n_part
    base_f = nw + len(fact_cols)
    base_p = base_f + n_rel
    rv = {c["key"]: list(r[c["key"]]) + [fw if c["key"] == p else 0.0 for (p, fw) in fact_cols] for c in cands}
    rho = {c["key"]: max(r[c["key"]], default=0.0) for c in cands}
    mat = material(q["_set"]) if partners and n_part else None
    by_key = {c["key"]: c for c in cands}
    edges = {}
    if mat is not None:
        for c in cands:
            for e in c.get("edges", []):
                if e["kind"] in ("Call", "DefUse", "TypeUse") and e["to"] in by_key:
                    edges.setdefault(c["key"], set()).add(e["to"])
                    edges.setdefault(e["to"], set()).add(c["key"])
    ev_cache = {}

    def ev(a, b):
        if (a, b) not in ev_cache:
            ev_cache[(a, b)] = partner_evidence(mat, by_key[a], by_key[b])
        return ev_cache[(a, b)]
    miss = [1.0] * len(W)
    taken, anchor_of, used = [], {}, 0
    fits = [c for c in cands if c["tokens"] <= budget]
    left = list(fits)
    place = {id(c): i for i, c in enumerate(cands)}
    stop = "nothing left"

    def terms(x):
        out = []
        for s in taken:
            for kk in pairs.get((x["key"], s["key"]), []):
                out.append((base_f + kk, rho[s["key"]], None))
            for kk in pairs.get((s["key"], x["key"]), []):
                out.append((base_f + kk, rho[x["key"]], None))
        if mat is not None:
            best, who = 0.0, None
            for s in taken:
                if s["key"] in edges.get(x["key"], ()):
                    v = rho[s["key"]] * ev(s["key"], x["key"])
                    if v > best:
                        best, who = v, s["key"]
            if best > 0:
                for j in range(n_part):
                    out.append((base_p + j, best, who))
        return out

    def gain(c):
        g = sum(W[i] * miss[i] * x for i, x in enumerate(rv[c["key"]]))
        m2 = {}
        for i, x, _ in terms(c):
            m2[i] = m2.get(i, miss[i]) * (1.0 - min(1.0, max(0.0, x)))
        return g + sum(W[i] * (miss[i] - m) for i, m in m2.items())
    values = [sum(W[i] * x for i, x in enumerate(rv[c["key"]])) for c in fits]
    band = 0.0 if band_zero else (q["band"] if recorded_band and q.get("band") is not None else RP.mad([v for v in values if v > 0]))
    gains = []
    while left:
        best = max(left, key=lambda c: (gain(c) / max(1, c["tokens"]), -place[id(c)]))
        left.remove(best)
        if used + best["tokens"] > budget:
            stop = "budget"
            continue
        g = gain(best)
        if g <= band:
            stop = "noise band"
            continue
        tt = terms(best)
        for i, x in enumerate(rv[best["key"]]):
            miss[i] *= 1.0 - min(1.0, max(0.0, x))
        who = None
        for i, x, a in tt:
            miss[i] *= 1.0 - min(1.0, max(0.0, x))
            if a is not None:
                who = a
        taken.append(best)
        gains.append(g)
        anchor_of[best["key"]] = who
        used += best["tokens"]
    value = sum(W[i] * (1.0 - m) for i, m in enumerate(miss))
    if fits:
        single = {c["key"]: v for c, v in zip(fits, values)}
        b = max(fits, key=lambda c: (single[c["key"]], -place[id(c)]))
        if single[b["key"]] > value:
            return [b], {b["key"]: None}, "best single piece"
    return taken, anchor_of, stop


def write(taken, anchor_of, order):
    """The written order: G1's order or the order of taking; a partner right after its anchor."""
    if order == "g1":
        base = sorted(taken, key=lambda c: (c["g1"] if c["g1"] is not None else 10**9, c["key"]))
    else:
        base = list(taken)
    keys = [c["key"] for c in base]
    partners = [c["key"] for c in base if anchor_of.get(c["key"])]
    out = [k for k in keys if k not in partners]
    for p in partners:
        a = anchor_of[p]
        out.insert(out.index(a) + 1 if a in out else len(out), p)
    return out


def make(mode="r1", k=1.0, how="name", partners=False, order="g1", served_r=False, rel_all=False, cal="tail", band_zero=False):
    def rule(q, cands, budget):
        if served_r:
            n = len(q["request"]["parts"])
            r = {c["key"]: list(c["S"]) for c in cands}
            w = [1.0] * n
        else:
            got = chances(q, cands, mode, k, how, cal)
            if got is None:
                return RP.served(q, cands, budget)
            _, w, r = got
        taken, anchor_of, _ = select(q, cands, budget, r, w, partners, rel_all, served_r, band_zero)
        return write(taken, anchor_of, order)
    return rule


# The served chances through this greedy: must equal `served` (a check of the greedy).
served_check = make(served_r=True)
# R1 to R3 alone: written in G1's order, as served.
r1 = make("r1")
# R1 with the units read literally (spans between definition lines): every request joins one aspect.
r1_span = make("r1", how="span")
r2_k2 = make("r2", 2.0)
r2_k5 = make("r2", 5.0)
r2_k10 = make("r2", 10.0)
r3_k2 = make("r3", 2.0)
r3_k5 = make("r3", 5.0)
r3_k10 = make("r3", 10.0)
# R1's scoring on the drone's parts as aspects (grouping off), to separate the two halves of R1.
r1_parts = make("r1", how="parts")
# R0: hand parts where they exist.
r0 = make("r1", how="hand")
# R4: partners on the served chances, and on R1's.
r4_served = make(partners=True, served_r=True)
r4_served_all = make(partners=True, served_r=True, rel_all=True)
r4_r1 = make("r1", partners=True)
# R5: written in the order of taking.
r5_served = make(order="take", served_r=True)
r5_r1 = make("r1", order="take")
r5_r2_k5 = make("r2", 5.0, order="take")
r5_r3_k5 = make("r3", 5.0, order="take")
# The noise calibration (1 minus the expected number of candidates scoring as high by chance).
r1z = make("r1", cal="noise")
r2z_k5 = make("r2", 5.0, cal="noise")
r3z_k5 = make("r3", 5.0, cal="noise")
r1z_parts = make("r1", how="parts", cal="noise")
# R8: describing comments count as the definition's name line.
r2d_k2 = make("r2d", 2.0)
r2d_k5 = make("r2d", 5.0)
r2d_k10 = make("r2d", 10.0)
r3d_k5 = make("r3d", 5.0)


def hold(c, k):
    cells = c.get("cells") or []
    x = cells[k] if k < len(cells) and cells[k] else None
    return (x.get("h") or [0, 0, 0])[2] if x else 0.0


def r6_scores(q, cands, eps=None):
    """R6, joint evidence: the sum over the request's parts of log((e_k + h_k) / e_k), h_k the share of
    part k the piece holds anywhere (the cell's third hold value), e_k the content-free base rate of
    holding part k (the mean hold over the candidates); with `eps` a fixed e for every part."""
    n = len(q["request"]["parts"])
    out = {c["key"]: 0.0 for c in cands}
    for k in range(n):
        hs = [hold(c, k) for c in cands]
        e = eps if eps is not None else (sum(hs) / len(hs) if hs else 0.0)
        if e <= 0:
            continue
        for c, h in zip(cands, hs):
            out[c["key"]] += math.log((e + h) / e)
    return out


def ssum(c):
    return sum(c["S"])


def r6_order(q, cands, eps=None):
    sc = r6_scores(q, cands, eps)
    return sorted(cands, key=lambda c: (-round(sc[c["key"]], 9), -ssum(c), c["key"]))


def r7_order(q, cands, k0=10, eps=None):
    """R7: reciprocal rank fusion of R6's order and the order by summed settle chance times file mass
    (the sum over the file's candidates of their summed settle chances)."""
    fm = {}
    for c in cands:
        fm[c["path"]] = fm.get(c["path"], 0.0) + ssum(c)
    a = {c["key"]: i for i, c in enumerate(r6_order(q, cands, eps))}
    b = {c["key"]: i for i, c in enumerate(sorted(cands, key=lambda c: (-ssum(c) * fm[c["path"]], c["key"])))}
    return sorted(cands, key=lambda c: (-(1.0 / (k0 + a[c["key"]]) + 1.0 / (k0 + b[c["key"]])), c["key"]))


def fill(order, budget):
    """The order written to the room: each piece that still fits, in order."""
    out, used = [], 0
    for c in order:
        if used + c["tokens"] <= budget:
            out.append(c["key"])
            used += c["tokens"]
    return out


def reorder_served(order_fn):
    """The served set unchanged, written in another order."""
    def rule(q, cands, budget):
        chosen = set(RP.served(q, cands, budget))
        return [c["key"] for c in order_fn(q, cands) if c["key"] in chosen]
    return rule


def ranked_fill(order_fn):
    def rule(q, cands, budget):
        return fill(order_fn(q, cands), budget)
    return rule


# R6 and R7, as an order of the served set, and as the ranking written to the room.
r6_served_order = reorder_served(r6_order)
r6_e05_served_order = reorder_served(lambda q, c: r6_order(q, c, 0.05))
r6_e2_served_order = reorder_served(lambda q, c: r6_order(q, c, 0.2))
r6_e4_served_order = reorder_served(lambda q, c: r6_order(q, c, 0.4))
r7_k10_served_order = reorder_served(lambda q, c: r7_order(q, c, 10))
r7_k60_served_order = reorder_served(lambda q, c: r7_order(q, c, 60))
r6_fill = ranked_fill(r6_order)
r6_e2_fill = ranked_fill(lambda q, c: r6_order(q, c, 0.2))
r7_k10_fill = ranked_fill(lambda q, c: r7_order(q, c, 10))
r7_k60_fill = ranked_fill(lambda q, c: r7_order(q, c, 60))
r7_k10_e2_fill = ranked_fill(lambda q, c: r7_order(q, c, 10, 0.2))
# Band 0: the greedy runs to the room (any positive gain), to separate the scoring from G2's stop,
# whose band (the spread of single values) is large on rank-calibrated chances.
r1_b0 = make("r1", band_zero=True)
r1_parts_b0 = make("r1", how="parts", band_zero=True)
r2_k2_b0 = make("r2", 2.0, band_zero=True)
r2_k5_b0 = make("r2", 5.0, band_zero=True)
r2_k10_b0 = make("r2", 10.0, band_zero=True)
r2d_k5_b0 = make("r2d", 5.0, band_zero=True)
r3_k5_b0 = make("r3", 5.0, band_zero=True)
r3d_k5_b0 = make("r3d", 5.0, band_zero=True)
r0_b0 = make("r1", how="hand", band_zero=True)
r4_r1_b0 = make("r1", partners=True, band_zero=True)
r5_r1_b0 = make("r1", order="take", band_zero=True)
r5_r2d_k5_b0 = make("r2d", 5.0, order="take", band_zero=True)
r5_r3_k5_b0 = make("r3", 5.0, order="take", band_zero=True)
served_b0 = make(served_r=True, band_zero=True)


def reorder_recorded(order_fn):
    """The recorded served set (the bench's own choice) unchanged, written in another order. Pieces of
    the set that are not among the dumped candidates carry no cells: they keep their recorded order
    after the ranked ones."""
    def rule(q, cands, budget):
        rec = q["served"].get("up", [])[: len(q["served"]["set"])] or q["served"]["set"]
        keys = set(rec)
        ranked = [c["key"] for c in order_fn(q, cands) if c["key"] in keys]
        return ranked + [k for k in rec if k not in ranked]
    return rule


def recorded_order(q, cands, budget):
    """The recorded served set in its written order (a check: equals the recorded arm)."""
    return q["served"].get("up", [])[: len(q["served"]["set"])] or q["served"]["set"]


r6_rec_order = reorder_recorded(r6_order)
r6_e05_rec_order = reorder_recorded(lambda q, c: r6_order(q, c, 0.05))
r6_e2_rec_order = reorder_recorded(lambda q, c: r6_order(q, c, 0.2))
r6_e4_rec_order = reorder_recorded(lambda q, c: r6_order(q, c, 0.4))
r7_k10_rec_order = reorder_recorded(lambda q, c: r7_order(q, c, 10))
r7_k60_rec_order = reorder_recorded(lambda q, c: r7_order(q, c, 60))
r7_k10_e2_rec_order = reorder_recorded(lambda q, c: r7_order(q, c, 10, 0.2))
settle_rec_order = reorder_recorded(lambda q, c: sorted(c, key=lambda x: (-ssum(x), x["key"])))
