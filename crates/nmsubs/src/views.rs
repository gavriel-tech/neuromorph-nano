//! The eight understanding views over one store (guide 5): global, local, structural,
//! semantic, temporal, operational, epistemic, objective. Every packet carries the substrate
//! version and a cited source. Operational reads the verified build and Step Check records
//! the build tools log (since 2026-11-09; before, it had none). Two job views were added on
//! 2026-11-09: `verification` (where a named item is tested) and `dependency` (where it is
//! used, and who depends on its crate). Epistemic still reports what the store has never
//! seen (absent key terms) until the uncertainty map exists.
use crate::{Substrate, cos};
use std::collections::BTreeMap;
use nmtype::{Category, Hit, Packet, Provenance, Stance, Status, SubstrateView};

/// The eight views, in guide order.
pub const VIEWS: [&str; 8] = ["global", "local", "structural", "semantic", "temporal", "operational", "epistemic", "objective"];
/// Job views beside the eight (2026-11-09): read by the Verification and Dependency readers.
pub const JOB_VIEWS: [&str; 5] = ["verification", "dependency", "constraint", "interface", "uncertainty"];
/// Relevance a table row carries when a question asks for a column's extreme or empty values:
/// the row is chosen by the question's own column and extreme, which its text does not repeat.
pub const TABLE_ROW_RELEVANCE: f32 = 0.8;
const ORIGIN: &str = "nmsubs.views#g0";
/// Hex digits of the content hash a stored citation keeps (64 bits: enough to tell a changed
/// span from an unchanged one).
pub const CITE_HASH_HEX: usize = 16;

/// What a remembered citation reads back as in the current tree.
#[derive(Clone, Debug)]
pub enum Reread {
    /// The cited lines, read now (at their recorded place or where they moved).
    Fresh(Box<Packet>),
    /// The file is there but no span holds the recorded content hash: the code changed.
    Stale { path: String, start: usize, end: usize },
    /// The file or the cited text is gone.
    Gone,
}

/// A stored span `lines a..b` with its optional `h:<hash>`.
struct Span<'a> {
    start: usize,
    end: usize,
    hash: Option<&'a str>,
}

/// Parses `lines a..b` and `lines a..b h:<hex>`; anything else is not a span.
fn parse_span(said: &str) -> Option<Span<'_>> {
    let rest = said.strip_prefix("lines ")?;
    let mut words = rest.split_whitespace();
    let (a, b) = words.next()?.split_once("..")?;
    let (start, end): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
    let hash = words.next().and_then(|w| w.strip_prefix("h:")).filter(|h| !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit()));
    (start >= 1 && end >= start).then_some(Span { start, end, hash })
}

/// SHA-256 of a span's lines joined by line feeds (the same digest memory records keep).
fn span_hash(lines: &[&str]) -> String {
    nmhash::sha256_bytes(lines.join("\n").as_bytes())
}
/// Dense lane chunks a territory reports at most (2026-11-10).
pub const DENSE_K: usize = 4;
/// Definitions the question names in its own words that the structural view adds.
pub const NAMED_K: usize = 8;
/// Files the question's words find first whose definitions are looked through, and how many
/// definitions each gives at most.
pub const FOUND_FILES: usize = 4;
pub const PER_FILE: usize = 3;
/// Hops a call path is followed, and the most hops it adds.
pub const TRACE_DEPTH: usize = 3;
pub const TRACE_K: usize = 8;

/// True for a question that asks for a call path (`trace`, `call path`, `call chain`, `calls
/// ... from ... to`).
fn is_trace(q: &str) -> bool {
    let l = q.to_lowercase();
    l.contains("trace") || l.contains("call path") || l.contains("call chain") || l.contains("call graph")
}

/// The evidence prefix that ties a wrapper's callee to the wrapper (`called_by`); the reducer
/// reads it.
pub const CALLED_BY: &str = "called-by";
/// Code lines a function body may hold and still be a thin wrapper (`delegate`).
pub const WRAPPER_LINES: usize = 2;
/// Lines naming an item that the dependency view reports per name.
pub const USES_K: usize = 8;

/// True for a line that is a comment (doc or plain) in the common comment forms.
fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    ["//", "/*", "*", "#", "--", ";"].iter().any(|m| t.starts_with(m)) && !t.starts_with("#[")
}

/// True when `line` calls `name`: the name stands alone (no identifier character before it) and
/// an opening parenthesis follows it.
fn calls(line: &str, name: &str) -> bool {
    line.match_indices(name).any(|(at, _)| {
        let before = line.get(..at).and_then(|s| s.chars().next_back());
        let after = line.get(at + name.len()..).and_then(|s| s.chars().next());
        !before.is_some_and(|c| c.is_alphanumeric() || c == '_') && after == Some('(')
    })
}

/// A definition's name with the type or trait it is a member of (`Limits::default`), else its name.
fn qualified(s: &crate::repo::Symbol) -> String {
    if s.parent.is_empty() { s.name.clone() } else { format!("{}::{}", s.parent, s.name) }
}

use nmlex::{name_parts, stem};

/// Evidence prefix of a code piece's fields: read with the piece by the judge, not shown in the
/// reply (the shown heading is the `in ` line).
pub const PIECE: &str = "piece:";
/// Marks the `PIECE` line that holds the file's field.
pub const FILE_FIELD: &str = "file ";

/// A code piece's fields, scored with every part of it: its own (name path and signature) and
/// its file's (the file's notes, shared by its pieces), lower-cased; both empty for documents
/// and `CHUNK`-line windows, the own field empty for notes, imports and loose blocks (named by
/// their place, not by what they define), the file field empty for the notes piece itself.
#[must_use]
pub fn piece_fields(c: &crate::Chunk) -> (String, String) {
    if c.piece.name.is_empty() {
        return (String::new(), String::new());
    }
    let own = if matches!(c.kind, "notes" | "imports" | "block") {
        String::new()
    } else if c.piece.signature.is_empty() {
        c.piece.name.to_lowercase()
    } else {
        format!("{} {}", c.piece.name, c.piece.signature).to_lowercase()
    };
    let file = if c.kind == "notes" { String::new() } else { c.piece.file_notes.to_lowercase() };
    (own, file)
}

/// A chunk packet's evidence: its span, then for a code piece its fields (`PIECE` lines, the
/// file's marked `FILE_FIELD`, each word once) and, when the piece has a heading (the signature
/// for a part after the first, the type for a member), that heading shown as `in <heading>`.
#[must_use]
pub fn piece_evidence(c: &crate::Chunk, span: String) -> Vec<String> {
    let mut ev = vec![span];
    let (own, file) = piece_fields(c);
    let mut seen = std::collections::BTreeSet::new();
    let mut words = |t: &str| -> String {
        t.split(|ch: char| !(ch.is_alphanumeric() || ch == '_')).filter(|w| !w.is_empty() && seen.insert((*w).to_owned())).collect::<Vec<_>>().join(" ")
    };
    let own = words(&own);
    let file = words(&file);
    if !own.is_empty() {
        ev.push(format!("{PIECE}{own}"));
    }
    if !file.is_empty() {
        ev.push(format!("{PIECE}{FILE_FIELD}{file}"));
    }
    if !c.heading.is_empty() && !c.piece.name.is_empty() {
        ev.push(format!("in {}", c.heading));
    }
    ev
}
/// Confidence of the lane's first fused place, the fall per place, and its least value.
pub const DENSE_TOP: f32 = 0.95;
pub const DENSE_STEP: f32 = 0.02;
pub const DENSE_FLOOR: f32 = 0.5;

impl Substrate {
    fn packet(&self, kind: &str, key: &str, text: &str, source: &str, status: Status) -> Packet {
        Packet::new(kind, key, text, Provenance::new(source, ORIGIN, self.version, nmtype::now_ms()))
            .status(status)
            .stance(Stance::Objective)
            .category(Category::Fact)
    }
    /// Names a question mentions: identifiers as written, plus its rare key terms.
    fn mentioned(q: &str) -> Vec<String> {
        let mut out = nmlex::names(q);
        for t in nmlex::key_terms(q) {
            if !out.iter().any(|x| x.eq_ignore_ascii_case(&t)) {
                out.push(t);
            }
        }
        out
    }
    /// Definition packets for a name: `symbol:<name>@<file>` per definition, and for a
    /// constant or static also `value:<name>` with its value, so two different values of
    /// one name meet under one key.
    fn definitions(&self, name: &str) -> Vec<Packet> {
        self.definitions_of(self.symbols.iter().filter(|s| s.name.eq_ignore_ascii_case(name)), "symbol")
    }
    /// The lines of a definition as they are served: from the doc comments and attributes above
    /// it to its last line, cut at `WHOLE` unit budgets (its head and the start of its body).
    fn definition_lines(&self, s: &crate::repo::Symbol) -> Option<(usize, usize, String)> {
        let f = self.files.iter().find(|f| f.path == s.file)?;
        let lines: Vec<&str> = f.text.lines().collect();
        if s.line == 0 || s.line > lines.len() {
            return None;
        }
        let hash = matches!(f.decoder, "python" | "powershell" | "shell" | "ruby");
        let level = |i: usize| lines.get(i - 1).map_or(0, |l| l.len() - l.trim_start().len());
        let mut start = s.line;
        while start > 1 && lines.get(start - 2).is_some_and(|l| crate::units::leads(l, hash)) && level(start - 1) == level(s.line) {
            start -= 1;
        }
        let last = s.end.clamp(s.line, lines.len());
        let cap = crate::units::WHOLE * self.unit_budget.max(1);
        let mut end = start;
        let mut size = 0;
        for i in start..=last {
            size += lines.get(i - 1).map_or(0, |l| l.chars().filter(|c| !c.is_whitespace()).count());
            end = i;
            if size > cap && i > s.line {
                break;
            }
        }
        Some((start, end, lines.get(start - 1..end)?.join("\n")))
    }
    /// Definition packets for symbols (`definitions`): each carries the definition's own lines,
    /// so it is judged by what it says and served as a span of the file. `kind` is `symbol` for a
    /// definition the question names, `definition` for one found by a shared word (it orders as
    /// a line window does), `trace` for a hop on a call path.
    fn definitions_of<'a>(&self, symbols: impl Iterator<Item = &'a crate::repo::Symbol>, kind: &str) -> Vec<Packet> {
        let mut out = Vec::new();
        for s in symbols {
            let (text, src, span) = match self.definition_lines(s) {
                Some((a, b, text)) => (text, format!("{}:{a}", s.file), format!("{}:{a} lines {a}..{b}", s.file)),
                None => {
                    let src = format!("{}:{}", s.file, s.line);
                    (format!("{} {} defined at {src}", s.kind, s.name), src.clone(), src)
                }
            };
            out.push(
                self.packet(kind, &format!("symbol:{}@{}", qualified(s), s.file), &text, &src, Status::Observed)
                    .evidence(vec![span, format!("{}:{} {}", s.file, s.line, s.text)]),
            );
            let src = format!("{}:{}", s.file, s.line);
            if (s.kind == "const" || s.kind == "static")
                && let Some((_, v)) = s.text.split_once('=')
            {
                let value = v.trim().trim_end_matches(';').trim();
                out.push(
                    self.packet("symbol", &format!("value:{}", s.name), &format!("{} = {value}", s.name), &src, Status::Observed)
                        .evidence(vec![format!("{src} {}", s.text)]),
                );
            }
        }
        out
    }
    /// Definitions whose names the question spells out in its own words: every part of the name
    /// (split at underscores and case changes, compared by stem) is a word of the question, or
    /// at least two parts are. Ranked by the rarity of the matched parts in the tree, the first
    /// `NAMED_K`; names the question writes as identifiers are left to `definitions`.
    fn named_definitions(&self, q: &str) -> Vec<&crate::repo::Symbol> {
        // The question's words, and the parts of the identifiers it writes (`nm_context_pack`
        // names `context` and `pack`).
        let mut words: Vec<String> = nmlex::words(q).iter().filter(|w| w.len() > 1 || w.chars().all(|c| c.is_ascii_alphabetic())).map(|w| stem(w)).collect();
        words.extend(nmlex::names(q).iter().flat_map(|n| name_parts(n)).filter(|p| p.len() > 1).map(|p| stem(&p)));
        let asked: Vec<String> = nmlex::names(q).iter().map(|n| n.to_lowercase()).collect();
        let mut scored: Vec<(f32, &crate::repo::Symbol)> = self
            .symbols
            .iter()
            .filter(|s| !asked.contains(&s.name.to_lowercase()))
            .filter_map(|s| {
                let parts = name_parts(&s.name);
                if parts.is_empty() {
                    return None;
                }
                let matched: Vec<&String> = parts.iter().filter(|p| words.contains(&stem(p))).collect();
                let all = matched.len() == parts.len();
                if !(all || matched.len() >= 2) {
                    return None;
                }
                let rarity: f32 = matched.iter().map(|p| self.idf(p).max(0.1)).sum();
                #[expect(clippy::cast_precision_loss, reason = "name parts are few")]
                let share = matched.len() as f32 / parts.len() as f32;
                // Each matched part counts: a name made of several of the question's words says more
                // than one common word does.
                #[expect(clippy::cast_precision_loss, reason = "name parts are few")]
                let parts_named = matched.len() as f32;
                Some((rarity * share * parts_named, s))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.file.cmp(&b.1.file)).then(a.1.line.cmp(&b.1.line)));
        scored.into_iter().take(NAMED_K).map(|(_, s)| s).collect()
    }
    /// The call path from `seeds` (definitions the question points at): the functions each one
    /// calls, followed `TRACE_DEPTH` hops, in the order met, at most `TRACE_K`. A called name
    /// resolves to its definition in the caller's file, else in the caller's folder; a name
    /// defined only further away is not followed (a module's call path stays in its module).
    fn call_path(&self, seeds: Vec<&crate::repo::Symbol>) -> Vec<&crate::repo::Symbol> {
        let mut by_name: BTreeMap<&str, Vec<&crate::repo::Symbol>> = BTreeMap::new();
        for s in self.symbols.iter().filter(|s| s.kind == "fn" || s.kind == "function" || s.kind == "method") {
            by_name.entry(s.name.as_str()).or_default().push(s);
        }
        let dir = |f: &str| f.rsplit_once('/').map_or(String::new(), |(d, _)| d.to_owned());
        let mut out: Vec<&crate::repo::Symbol> = Vec::new();
        let mut ring: Vec<&crate::repo::Symbol> = seeds.into_iter().filter(|s| by_name.contains_key(s.name.as_str())).collect();
        let mut seen: Vec<(String, usize)> = ring.iter().map(|s| (s.file.clone(), s.line)).collect();
        for _ in 0..TRACE_DEPTH {
            let mut next = Vec::new();
            for caller in &ring {
                let Some((_, _, body)) = self.definition_lines(caller) else { continue };
                for id in crate::refgraph::identifiers(&body) {
                    let Some(defs) = by_name.get(id.as_str()) else { continue };
                    let pick = defs
                        .iter()
                        .find(|d| d.file == caller.file)
                        .or_else(|| defs.iter().find(|d| dir(&d.file) == dir(&caller.file)));
                    let Some(d) = pick.copied() else { continue };
                    if seen.iter().any(|(f, l)| *f == d.file && *l == d.line) {
                        continue;
                    }
                    seen.push((d.file.clone(), d.line));
                    out.push(d);
                    next.push(d);
                    if out.len() >= TRACE_K {
                        return out;
                    }
                }
            }
            ring = next;
        }
        out
    }
    /// The function a thin wrapper hands its work to: when the body of function `s` (after its
    /// signature) is at most `WRAPPER_LINES` code lines and calls a function defined in its own
    /// file (else its folder), that function. A wrapper's doc comment names the topic while the
    /// deciding lines sit in the function it calls.
    fn delegate(&self, s: &crate::repo::Symbol) -> Option<&crate::repo::Symbol> {
        if !matches!(s.kind.as_str(), "fn" | "function" | "method") {
            return None;
        }
        let f = self.files.iter().find(|f| f.path == s.file)?;
        let lines: Vec<&str> = f.text.lines().collect();
        let text = lines.get(s.line.checked_sub(1)?..s.end.clamp(s.line, lines.len()))?.join("\n");
        let body = match text.split_once('{') {
            Some((_, b)) => b.to_owned(),
            None => text.lines().skip(1).collect::<Vec<_>>().join("\n"),
        };
        let code = body.lines().map(str::trim).filter(|l| !l.is_empty() && !is_comment(l) && !l.chars().all(|c| c == '}')).count();
        if code == 0 || code > WRAPPER_LINES {
            return None;
        }
        let dir = |p: &str| p.rsplit_once('/').map_or(String::new(), |(d, _)| d.to_owned());
        let is_fn = |d: &crate::repo::Symbol| matches!(d.kind.as_str(), "fn" | "function" | "method");
        crate::refgraph::identifiers(&body).into_iter().filter(|id| *id != s.name && calls(&body, id)).find_map(|id| {
            let mut fns = self.symbols.iter().filter(|d| d.name == id && is_fn(d));
            fns.clone().find(|d| d.file == s.file).or_else(|| fns.find(|d| dir(&d.file) == dir(&s.file)))
        })
    }
    /// The definition packet of the function the thin wrapper `s` calls (`delegate`), carrying
    /// the evidence line `called-by:<wrapper's source> <callee's first line>`, so the reducer can
    /// keep it with the wrapper (`nmbrain::synthesis`); the reply shows no such line, as the
    /// callee's first line is in its text. Empty when `s` is not a thin wrapper.
    fn called_by(&self, s: &crate::repo::Symbol, kind: &str) -> Vec<Packet> {
        let Some(d) = self.delegate(s) else { return Vec::new() };
        let at = self.definition_lines(s).map_or(s.line, |(a, _, _)| a);
        let by = format!("{CALLED_BY}:{}:{at} {}", s.file, d.text);
        self.definitions_of(std::iter::once(d), kind).into_iter().map(|p| p.evidence(vec![by.clone()])).collect()
    }
    /// Definitions inside the files the question's words find first (the files of the best
    /// `FOUND_FILES` lexical pieces) whose names share a part with the question: a file's notes
    /// often name the topic while the deciding lines sit in a definition further down whose name
    /// carries one of the question's words (`project_root` for a question about the project
    /// folder). At most `PER_FILE` per file, rarest matched parts first.
    fn definitions_in_found_files(&self, q: &str) -> Vec<&crate::repo::Symbol> {
        let words: Vec<String> = nmlex::key_terms(q).iter().map(|w| stem(w)).collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut files: Vec<&str> = Vec::new();
        for (_, _, c) in self.score_pieces(q, &[], self.chunks.iter().enumerate()) {
            let Some(f) = self.files.get(c.file).filter(|f| f.is_code()) else { continue };
            if !files.contains(&f.path.as_str()) {
                files.push(f.path.as_str());
            }
            if files.len() >= FOUND_FILES {
                break;
            }
        }
        let mut out = Vec::new();
        for f in files {
            let mut scored: Vec<(f32, &crate::repo::Symbol)> = self
                .symbols
                .iter()
                .filter(|s| s.file == f)
                .filter_map(|s| {
                    let w: f32 = name_parts(&qualified(s)).iter().filter(|p| p.len() >= 3 && words.contains(&stem(p))).map(|p| self.idf(p).max(0.1)).sum();
                    (w > 0.0).then_some((w, s))
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.line.cmp(&b.1.line)));
            out.extend(scored.into_iter().take(PER_FILE).map(|(_, s)| s));
        }
        out
    }
    /// The `k` nearest chunks inside `paths` (empty: everywhere), ordered by a blend of the
    /// embedding cosine (0.4) and the rarity-weighted share of the question's key terms the
    /// chunk holds (0.6), times the source weight, so code orders before the plan and note
    /// documents that describe it (context pack fix, 2026-11-08: the exact embedder counts
    /// common words, so a long chunk repeating them outranked the file that names the topic).
    fn semantic(&self, q: &str, paths: &[String], k: usize) -> Vec<Packet> {
        let lexical = self.score_pieces(q, paths, self.chunks.iter().enumerate());
        let chunk_packet = |c: &crate::Chunk, origin: &str, sc: f32, place: usize| -> Option<Packet> {
            let f = self.files.get(c.file)?.path.as_str();
            let src = format!("{f}:{}", c.start);
            // Lane chunks carry `dense:<place>:` in their key (a tentacle's distillation replaces
            // the origin and raises the confidence; the key survives), so the Core Brain admits
            // and orders them by their fused place.
            let key = if origin == crate::dense::ORIGIN { format!("dense:{place:02}:{src}") } else { format!("chunk:{src}") };
            Some(
                Packet::new("chunk", &key, &c.text, Provenance::new(&src, origin, self.version, nmtype::now_ms()))
                    .status(Status::Inferred)
                    .stance(Stance::Objective)
                    .category(Category::Fact)
                    .confidence(sc)
                    .evidence(piece_evidence(c, format!("{src} lines {}..{}", c.start, c.end))),
            )
        };
        let mut out = Vec::new();
        // The meaning lane as candidates only (`dense::CANDIDATES_ONLY`, 2026-11-17): the nearest
        // chunks by meaning that this territory holds and its own lexical head does not, as
        // ordinary candidates with their lexical score; the evidence orders them.
        // At the first start's partial coverage the lane's places cannot be compared across pieces
        // with and without a vector, so its nearest pieces come as candidates (`partial_now`).
        let candidates_only =
            crate::dense::CANDIDATES_ONLY.load(std::sync::atomic::Ordering::Relaxed) || self.lane().is_some_and(|l| l.partial_now(self, q));
        if candidates_only && let Some(r) = self.lane().and_then(|l| l.meaning_ranks(self, q)) {
            let inside = self.inside(paths);
            let head: std::collections::BTreeSet<usize> = lexical.iter().take(k).map(|(_, i, _)| *i).collect();
            let score: BTreeMap<usize, f32> = lexical.iter().map(|(s, i, _)| (*i, *s)).collect();
            let mut held: Vec<(usize, usize)> = r
                .place
                .iter()
                .filter(|(i, _)| !head.contains(*i) && self.chunks.get(**i).is_some_and(|c| inside.get(c.file).copied().unwrap_or(false)))
                .map(|(i, p)| (*p, *i))
                .collect();
            held.sort_unstable();
            out.extend(held.into_iter().take(DENSE_K).filter_map(|(_, i)| chunk_packet(self.chunks.get(i)?, ORIGIN, score.get(&i).copied().unwrap_or(0.0), 0)));
        }
        // The dense lane (2026-11-10): the question's fused ranking, computed once and shared by
        // every tentacle; this territory reports the ranked chunks it holds, best place first,
        // with a confidence from the place (`DENSE_TOP` at place 0, falling by `DENSE_STEP`).
        if let Some(r) = self.lane().filter(|_| !candidates_only).and_then(|l| l.ranks(self, q)) {
            let inside = self.inside(paths);
            let mut held: Vec<(usize, usize)> = r
                .place
                .iter()
                .filter(|(i, _)| self.chunks.get(**i).is_some_and(|c| inside.get(c.file).copied().unwrap_or(false)))
                .map(|(i, p)| (*p, *i))
                .collect();
            held.sort_unstable();
            // Step 1l, change B: with the shortlist derived per request, a territory reports every
            // shortlisted piece it holds; otherwise `DENSE_K` (or the number named for a measurement).
            let t = crate::dense::tune();
            let per_territory = if t.derive_top { usize::MAX } else { t.dense_k.unwrap_or(DENSE_K) };
            #[expect(clippy::cast_precision_loss, reason = "places are small")]
            out.extend(held.into_iter().take(per_territory).filter_map(|(p, i)| {
                chunk_packet(self.chunks.get(i)?, crate::dense::ORIGIN, (DENSE_TOP - DENSE_STEP * p as f32).max(DENSE_FLOOR), p)
            }));
        }
        let have: Vec<String> = out.iter().map(|p: &Packet| p.provenance().source.clone()).collect();
        // `k` is counted in judge windows: at least `k` chunks, then more while their tokens fit
        // in `k` of the judge's windows, so a reader brings as much of the material whatever the
        // size of the pieces it is cut into.
        let room = k.saturating_mul(crate::piece::JUDGE_WINDOW);
        let mut used = 0_usize;
        let mut taken = 0_usize;
        for (sc, _, c) in lexical {
            if taken >= k && used + c.tokens > room {
                break;
            }
            let Some(p) = chunk_packet(c, ORIGIN, sc, 0) else { continue };
            if have.iter().any(|h| *h == p.provenance().source) {
                continue;
            }
            used += c.tokens;
            taken += 1;
            out.push(p);
        }
        out
    }
    /// Which files lie inside `paths` (empty: all), once per file.
    fn inside(&self, paths: &[String]) -> Vec<bool> {
        self.files.iter().map(|f| paths.is_empty() || paths.iter().any(|p| f.path.starts_with(p.as_str()))).collect()
    }
    /// The dense lane, when bound.
    fn lane(&self) -> Option<std::sync::Arc<crate::dense::DenseLane>> {
        self.dense.read().ok()?.clone()
    }
    /// Chunks inside `paths` with their lexical score, best first: a blend of the exact
    /// embedding cosine (0.4) and the rarity-weighted share of the question's key terms the
    /// chunk holds (0.6), times the source weight, so code orders before the plan and note
    /// documents that describe it (context pack fix, 2026-11-08: the exact embedder counts
    /// common words, so a long chunk repeating them outranked the file that names the topic).
    fn lexical_scores(&self, q: &str, paths: &[String]) -> Vec<(f32, usize)> {
        self.score_pieces(q, paths, self.chunks.iter().enumerate()).into_iter().map(|(s, i, _)| (s, i)).collect()
    }
    /// `pieces` (an order number and a chunk) inside `paths` with their lexical score, best
    /// first, ties in order number (as `lexical_scores`).
    fn score_pieces<'a>(
        &self,
        q: &str,
        paths: &[String],
        pieces: impl Iterator<Item = (usize, &'a crate::Chunk)>,
    ) -> Vec<(f32, usize, &'a crate::Chunk)> {
        let qv = self.embed(q);
        let terms = nmlex::key_terms(q);
        let idf: Vec<f32> = terms.iter().map(|t| self.idf(t)).collect();
        let total: f32 = idf.iter().sum();
        // Document chunks are scored with the document terms (code words and numbers kept) over
        // their heading path and text; code chunks exactly as before.
        let dterms = nmlex::key_terms_doc(q);
        let didf: Vec<f32> = dterms.iter().map(|t| self.idf(t)).collect();
        let dtotal: f32 = didf.iter().sum();
        let doc_kind = nmlex::doc_kind(q).is_some();
        let inside = self.inside(paths);
        let mut scored: Vec<(f32, usize, &crate::Chunk)> = pieces
            .filter_map(|(i, c)| {
                if !inside.get(c.file).copied().unwrap_or(false) {
                    return None;
                }
                let file = self.files.get(c.file)?;
                let f = file.path.as_str();
                if !file.is_code() {
                    let head = c.heading.to_lowercase();
                    let cover = if dtotal > 0.0 {
                        dterms.iter().zip(&didf).filter(|(t, _)| c.low.contains(t.as_str()) || head.contains(t.as_str())).map(|(_, w)| w).sum::<f32>()
                            / dtotal
                    } else {
                        0.0
                    };
                    return Some(((0.4 * cos(&qv, &c.vec) + 0.6 * cover) * nmtype::source_weight_q(f, doc_kind), i, c));
                }
                // A piece is scored with its fields, in every part: its own and its file's.
                let (own, file) = piece_fields(c);
                let cover = if total > 0.0 {
                    terms
                        .iter()
                        .zip(&idf)
                        .filter(|(t, _)| c.low.contains(t.as_str()) || own.contains(t.as_str()) || file.contains(t.as_str()))
                        .map(|(_, w)| w)
                        .sum::<f32>()
                        / total
                } else {
                    0.0
                };
                Some(((0.4 * cos(&qv, &c.vec) + 0.6 * cover) * nmtype::source_weight(f), i, c))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored
    }
    /// Chunk indices of the whole tree in lexical order (the dense lane's second order).
    #[must_use]
    pub fn lexical_order(&self, q: &str) -> Vec<usize> {
        self.lexical_scores(q, &[]).into_iter().map(|(_, i)| i).collect()
    }
    /// Read a remembered citation fresh from the current tree. Two shapes are read:
    /// - `path:line lines a..b h:<hash>`, the shape memory stores: the span is looked up by its
    ///   content hash, at its recorded place first, else at the nearest place holding the same
    ///   lines (a moved span is found at its new numbers); a hash found nowhere is stale.
    /// - `path:line text` (episodes written before content hashes): the nearest line holding the
    ///   cited text, so a repeated line resolves to the cited one; a text that is gone is gone.
    ///
    /// A span without a hash (`path:line lines a..b`) and a bare `path:line` read the lines at
    /// their recorded place. The packet observes the current file; its kind is `recall`, so the
    /// Core Brain can admit it by the recall that chose it.
    #[must_use]
    pub fn reread_state(&self, cite: &str) -> Reread {
        let (loc, said) = cite.split_once(' ').unwrap_or((cite, ""));
        let Some((path, line)) = loc.rsplit_once(':') else { return Reread::Gone };
        let Ok(line) = line.parse::<usize>() else { return Reread::Gone };
        let Some(f) = self.files.iter().find(|f| f.path == path) else { return Reread::Gone };
        let lines: Vec<&str> = f.text.lines().collect();
        let said = said.trim();
        if let Some(span) = parse_span(said) {
            let len = span.end - span.start + 1;
            let window = |s: usize| lines.get(s - 1..s - 1 + len);
            let found = match span.hash {
                None => window(span.start).map(|_| span.start),
                Some(h) => {
                    let matches = |s: usize| window(s).is_some_and(|p| span_hash(p).starts_with(h));
                    if matches(span.start) {
                        Some(span.start)
                    } else if let Some(s) =
                        (1..=(lines.len() + 1).saturating_sub(len)).filter(|s| matches(*s)).min_by_key(|s| s.abs_diff(span.start))
                    {
                        Some(s)
                    } else {
                        return Reread::Stale { path: path.to_owned(), start: span.start, end: span.end };
                    }
                }
            };
            let Some(s) = found else { return Reread::Gone };
            let part = window(s).unwrap_or(&[]);
            let src = format!("{path}:{s}");
            let text = if len == 1 { part.first().map_or("", |l| l.trim()).to_owned() } else { part.join("\n") };
            let evidence = if len == 1 { format!("{src} {text}") } else { format!("{src} lines {s}..{}", s + len - 1) };
            return Reread::Fresh(Box::new(
                self.packet("recall", &format!("grep:{src}"), &text, &src, Status::Observed).evidence(vec![evidence]),
            ));
        }
        let at = if said.is_empty() {
            line.checked_sub(1).filter(|i| *i < lines.len())
        } else {
            // Nearest line holding the cited text, so a repeated line resolves to the cited one.
            lines
                .iter()
                .enumerate()
                .filter(|(_, l)| l.trim() == said)
                .min_by_key(|(i, _)| i.abs_diff(line.saturating_sub(1)))
                .map(|(i, _)| i)
        };
        let Some((i, text)) = at.and_then(|i| lines.get(i).map(|l| (i, l.trim()))) else { return Reread::Gone };
        let src = format!("{path}:{}", i + 1);
        Reread::Fresh(Box::new(
            self.packet("recall", &format!("grep:{src}"), text, &src, Status::Observed).evidence(vec![format!("{src} {text}")]),
        ))
    }
    /// The fresh read of a remembered citation, when it still reads back (`reread_state`).
    #[must_use]
    pub fn reread(&self, cite: &str) -> Option<Packet> {
        match self.reread_state(cite) {
            Reread::Fresh(p) => Some(*p),
            Reread::Stale { .. } | Reread::Gone => None,
        }
    }
    /// The shape memory stores for a citation: `path:line lines a..b h:<hash>`, where the hash is
    /// the first `CITE_HASH_HEX` hex digits of the SHA-256 of the span's lines as the tree holds
    /// them now. A single-line citation (`path:line text`) becomes the one-line span
    /// `line..line`. No code text is kept; a citation whose file or lines are not in the tree
    /// keeps only its place.
    #[must_use]
    pub fn cite_ref(&self, cite: &str) -> String {
        let (loc, said) = cite.split_once(' ').unwrap_or((cite, ""));
        let Some((path, line)) = loc.rsplit_once(':') else { return loc.to_owned() };
        let Ok(line) = line.parse::<usize>() else { return loc.to_owned() };
        let (start, end) = parse_span(said.trim()).map_or((line, line), |s| (s.start, s.end));
        let hash = self.files.iter().find(|f| f.path == path).and_then(|f| {
            let lines: Vec<&str> = f.text.lines().collect();
            lines.get(start.checked_sub(1)?..end).map(span_hash)
        });
        match hash {
            Some(h) => format!("{loc} lines {start}..{end} h:{}", h.get(..CITE_HASH_HEX).unwrap_or(&h)),
            None => loc.to_owned(),
        }
    }
    /// Operational records (build errors, fixes, Step Check verdicts) for what the question
    /// names, newest first, fix records before the rest (M2 revision 2, 2026-11-09): a file path
    /// in the question selects that file's records; else a crate it names selects that crate's;
    /// else records sharing one of its rarer terms (rarity at least 1.0, build words left out); a question
    /// about builds in general and nothing more gets the newest records. Before, words of any
    /// path ("crates", "src") matched every record.
    fn operational(&self, q: &str) -> Vec<Packet> {
        let ops = self.ops.read().map(|g| g.clone()).unwrap_or_default();
        let paths: Vec<String> = q
            .split(|c: char| c.is_whitespace() || c == '`' || c == ',' || c == '?' || c == '(' || c == ')')
            .map(|w| w.trim_end_matches(['.', ':', ';']))
            .filter(|w| w.contains('/') || w.ends_with(".rs") || w.ends_with(".toml") || nmlex::lang::lang_of(w).is_some())
            .map(str::to_owned)
            .collect();
        let crates: Vec<String> = Self::mentioned(q).iter().filter_map(|n| self.crate_named(n)).map(|c| format!("{}/", c.dir)).collect();
        let general = [
            "build", "error", "errors", "compile", "fail", "failed", "check", "broke", "clippy", "fix", "fixed", "before", "change",
            "about",
        ];
        let terms: Vec<String> = nmlex::key_terms(q).into_iter().filter(|t| !general.contains(&t.as_str()) && self.idf(t) >= 1.0).collect();
        let only_general = nmlex::key_terms(q).iter().all(|t| general.contains(&t.as_str()));
        let file_of = |r: &crate::OpRecord| r.source.rsplit_once(':').map_or(r.source.clone(), |(f, _)| f.to_owned());
        let mut hits: Vec<&crate::OpRecord> = ops
            .iter()
            .filter(|r| {
                let f = file_of(r);
                if !paths.is_empty() {
                    paths.iter().any(|p| f == *p || f.ends_with(p.as_str()) || p.ends_with(f.as_str()))
                } else if !crates.is_empty() {
                    crates.iter().any(|c| f.starts_with(c.as_str()))
                } else if only_general {
                    true
                } else {
                    let low = r.text.to_lowercase();
                    terms.iter().any(|t| low.contains(t.as_str()))
                }
            })
            .collect();
        hits.sort_by_key(|r| u8::from(!r.text.contains("fixed by")));
        hits.into_iter()
            .take(8)
            .map(|r| {
                self.packet(
                    "build",
                    &format!("build:{}:{}", r.source, nmhash::sha256_bytes(r.text.as_bytes()).get(..12).unwrap_or("")),
                    &r.text,
                    &r.source,
                    Status::Observed,
                )
                .category(Category::Temporal)
                .evidence(vec![format!("{} {}", r.source, r.text)])
            })
            .collect()
    }
    /// Identifiers the question names, else nothing (tests and uses are looked up by name).
    fn names_in(q: &str) -> Vec<String> {
        nmlex::names(q).into_iter().filter(|n| n.len() > 2).take(3).collect()
    }
    /// Where a named item is tested: lines naming it in test files (`/tests/`, and the test file
    /// names of other languages: `*.test.ts`, `test_*.py`) or after a file's first test marker
    /// (`#[cfg(test)]`, a section `fn check(`, and the language table's markers).
    fn verification(&self, q: &str) -> Vec<Packet> {
        let mut out = Vec::new();
        for name in Self::names_in(q) {
            let mut n = 0;
            for f in &self.files {
                let lines: Vec<&str> = f.text.lines().collect();
                let other = nmlex::lang::by_id(f.decoder).filter(|l| l.id != "rust");
                let start = if f.path.contains("/tests/") || (other.is_some() && nmlex::lang::is_test_path(&f.path)) {
                    0
                } else if let Some(l) = other {
                    lines.iter().position(|x| nmlex::lang::has_test_mark(l, x)).unwrap_or(lines.len())
                } else {
                    lines.iter().position(|l| l.contains("#[cfg(test)]") || l.contains("fn check(")).unwrap_or(lines.len())
                };
                for (i, l) in lines.iter().enumerate().skip(start) {
                    if n < 6 && l.contains(name.as_str()) {
                        let src = format!("{}:{}", f.path, i + 1);
                        out.push(
                            self.packet("test", &format!("test:{src}"), l.trim(), &src, Status::Observed)
                                .evidence(vec![format!("{src} {}", l.trim())]),
                        );
                        n += 1;
                    }
                }
            }
        }
        out
    }
    /// Where a named item is used (lines naming it other than its definitions), and the crates
    /// that depend on the crate defining it. A call question (`what calls`, `callers`, `called
    /// from`) asks for call sites: when the item has any, only those are reported, the calls the
    /// question qualifies as written (`decide::second_look(`) if there are such, else the calls
    /// outside tests, else the calls in tests; the crate's dependents are a coarser answer and are
    /// left out.
    fn dependency(&self, q: &str) -> Vec<Packet> {
        let mut out = Vec::new();
        let call_q = nmlex::words(q).iter().any(|w| matches!(w.to_lowercase().as_str(), "calls" | "called" | "caller" | "callers"));
        for name in Self::names_in(q) {
            let defs: Vec<(String, usize)> = self.symbols.iter().filter(|s| s.name == name).map(|s| (s.file.clone(), s.line)).collect();
            for (file, _) in defs.iter().filter(|_| !call_q) {
                if let Some(c) = self.crates.iter().find(|c| file.starts_with(&format!("{}/", c.dir))) {
                    let src = format!("{}:1", c.manifest);
                    let text =
                        format!("{name} is defined in crate {}, which is depended on by [{}]", c.name, self.dependents(&c.name).join(", "));
                    out.push(
                        self.packet("deps", &format!("deps:{}:{name}", c.name), &text, &src, Status::Observed).evidence(vec![src.clone()]),
                    );
                }
            }
            // Every line naming it other than its definitions, ranked: a call the question
            // qualifies as written, a call outside tests, a call in tests, another code line, a
            // comment or prose; file order within each rank.
            let qualified: Vec<String> = q
                .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
                .filter(|w| w.contains("::") && w.ends_with(&format!("::{name}")))
                .map(|w| format!("{w}("))
                .collect();
            let mut found: Vec<(u8, &str, usize, &str)> = Vec::new();
            for f in &self.files {
                let lines: Vec<&str> = f.text.lines().collect();
                let tests = if f.path.contains("/tests/") {
                    0
                } else {
                    lines.iter().position(|l| l.contains("#[cfg(test)]") || l.contains("fn check(")).unwrap_or(lines.len())
                };
                for (i, l) in lines.iter().enumerate() {
                    if l.contains(name.as_str()) && !defs.iter().any(|(df, dl)| *df == f.path && *dl == i + 1) {
                        let rank = if !f.is_code() || is_comment(l) {
                            4
                        } else if !calls(l, &name) {
                            3
                        } else if i >= tests {
                            2
                        } else if qualified.iter().any(|w| l.contains(w.as_str())) {
                            0
                        } else {
                            1
                        };
                        found.push((rank, f.path.as_str(), i + 1, l));
                    }
                }
            }
            found.sort_by_key(|x| x.0);
            if call_q && let Some(best) = found.first().map(|x| x.0).filter(|r| *r <= 2) {
                found.retain(|x| x.0 == best);
            }
            for (_, path, line, l) in found.into_iter().take(USES_K) {
                let src = format!("{path}:{line}");
                out.push(
                    self.packet("uses", &format!("uses:{src}"), l.trim(), &src, Status::Observed).evidence(vec![format!("{src} {}", l.trim())]),
                );
            }
        }
        out
    }
    /// Rules and lints that bear on the question (PLAN 4c `constraint`, 2026-11-10): lines under
    /// `## rules` in `rules.md` and the `[workspace.lints.*]` entries of the root `Cargo.toml`
    /// that hold a key term of the question, best coverage first.
    fn constraint(&self, q: &str) -> Vec<Packet> {
        let terms = nmlex::key_terms(q);
        let mut out: Vec<(usize, Packet)> = Vec::new();
        for f in self.files.iter().filter(|f| f.path == "rules.md" || f.path == "Cargo.toml") {
            let mut in_section = false;
            for (i, l) in f.text.lines().enumerate() {
                let t = l.trim();
                if f.path == "rules.md" {
                    if t.starts_with("## ") {
                        in_section = t == "## rules";
                        continue;
                    }
                    if !in_section || !t.starts_with("- ") {
                        continue;
                    }
                } else {
                    if t.starts_with('[') {
                        in_section = t.starts_with("[workspace.lints");
                        continue;
                    }
                    if !in_section || t.is_empty() {
                        continue;
                    }
                }
                let low = t.to_lowercase();
                let hits = terms.iter().filter(|w| low.contains(w.as_str())).count();
                if hits > 0 {
                    let src = format!("{}:{}", f.path, i + 1);
                    let text: String = t.chars().take(400).collect();
                    out.push((
                        hits,
                        self.packet("rule", &format!("rule:{src}"), &text, &src, Status::Observed)
                            .category(Category::Constraint)
                            .evidence(vec![format!("{src} {text}")]),
                    ));
                }
            }
        }
        out.sort_by_key(|x| std::cmp::Reverse(x.0));
        out.into_iter().take(6).map(|(_, p)| p).collect()
    }
    /// The public interface of a crate or file the question names (PLAN 4c `interface`,
    /// 2026-11-10): the defining lines of its `pub` items, up to twelve.
    fn interface(&self, q: &str) -> Vec<Packet> {
        let mut scopes: Vec<String> =
            Self::mentioned(q).iter().filter_map(|n| self.crate_named(n)).map(|c| format!("{}/", c.dir)).collect();
        scopes.extend(
            q.split_whitespace()
                .map(|w| w.trim_matches(|c: char| c == '`' || c == '?' || c == ',' || c == '.'))
                .filter(|w| w.ends_with(".rs") || nmlex::lang::lang_of(w).is_some())
                .map(str::to_owned),
        );
        scopes.dedup();
        let mut out = Vec::new();
        for s in self.symbols.iter().filter(|s| scopes.iter().any(|p| s.file.starts_with(p.as_str()) || s.file.ends_with(p.as_str())) && is_public(s)) {
            if out.len() >= 12 {
                break;
            }
            let src = format!("{}:{}", s.file, s.line);
            out.push(
                self.packet("interface", &format!("interface:{src}"), s.text.trim(), &src, Status::Observed)
                    .evidence(vec![format!("{src} {}", s.text.trim())]),
            );
        }
        out
    }
    /// What the tree cannot answer for the question (PLAN 4c `uncertainty`, 2026-11-10): key
    /// terms no file holds, and names it uses that no definition in the tree carries, each an
    /// open question for the caller.
    fn uncertainty(&self, q: &str) -> Vec<Packet> {
        let mut out: Vec<Packet> = nmlex::key_terms(q)
            .into_iter()
            .filter(|t| !self.known(t))
            .map(|t| {
                self.packet(
                    "finding",
                    &format!("unknown:{t}"),
                    &format!("no file in the substrate contains `{t}`; is it named differently here?"),
                    "unit:nmsubs.views",
                    Status::Unknown,
                )
                .stance(Stance::Unknown)
            })
            .collect();
        for n in nmlex::names(q) {
            if n.len() > 2 && !self.symbols.iter().any(|s| s.name == n) && self.crate_named(&n).is_none() {
                out.push(
                    self.packet(
                        "finding",
                        &format!("undefined:{n}"),
                        &format!("`{n}` is named in the question but nothing in the substrate defines it"),
                        "unit:nmsubs.views",
                        Status::Unknown,
                    )
                    .stance(Stance::Unknown),
                );
            }
        }
        out
    }
    /// One view for a query.
    #[must_use]
    pub fn view_of(&self, name: &str, q: &str) -> Vec<Packet> {
        match name {
            // Rows of the tables (CSV, TSV) that hold a column's extreme or empty values the
            // question asks about (`nmdoc::csv::select`); each row is cited with its reason.
            "table" => {
                let mut out = Vec::new();
                for f in self.files.iter().filter(|f| matches!(f.decoder, "csv" | "tsv")) {
                    let d = if f.decoder == "tsv" { '\t' } else { ',' };
                    let lines: Vec<&str> = f.text.lines().collect();
                    for (line, why) in nmdoc::csv::select(&f.text, d, q).into_iter().take(12) {
                        let text = lines.get(line - 1).copied().unwrap_or("").trim().to_owned();
                        let src = format!("{}:{line}", f.path);
                        out.push(
                            self.packet("row", &format!("row:{src}"), &text, &src, Status::Observed)
                                .evidence(vec![format!("{src} {text} ({why})")])
                                .confidence(TABLE_ROW_RELEVANCE),
                        );
                    }
                }
                out
            }
            "global" => vec![self.packet(
                "finding",
                "workspace",
                &format!(
                    "{} crates, {} files, {} symbols, {} chunks",
                    self.crates.len(),
                    self.files.len(),
                    self.symbols.len(),
                    self.chunks.len()
                ),
                "crate:workspace",
                Status::Inferred,
            )],
            "local" => {
                let terms: Vec<String> = Self::mentioned(q).iter().map(|t| t.to_lowercase()).collect();
                let mut out = Vec::new();
                for t in terms.iter().take(3) {
                    for h in self.grep(std::slice::from_ref(t), &[], 3) {
                        let src = format!("{}:{}", h.file, h.line);
                        out.push(
                            self.packet("grep", &format!("grep:{src}"), &h.text, &src, Status::Observed)
                                .evidence(vec![format!("{src} {}", h.text)]),
                        );
                    }
                }
                out
            }
            "structural" => {
                let mut out = Vec::new();
                for n in Self::mentioned(q) {
                    if let Some(c) = self.crate_named(&n) {
                        let src = format!("{}/Cargo.toml:1", c.dir);
                        let text = format!(
                            "crate {} depends on [{}]; depended on by [{}]",
                            c.name,
                            c.deps.join(", "),
                            self.dependents(&c.name).join(", ")
                        );
                        out.push(
                            self.packet("deps", &format!("deps:{}", c.name), &text, &src, Status::Observed).evidence(vec![src.clone()]),
                        );
                    }
                    out.extend(self.definitions(&n));
                }
                let named = self.named_definitions(q);
                let found = self.definitions_in_found_files(q);
                // A call path question: the definitions it points at start the path, then the
                // functions they call; all of them are hops of the path.
                let trace = is_trace(q);
                let (seed_kind, found_kind) = if trace { ("trace", "trace") } else { ("symbol", "definition") };
                let hops = if trace { self.call_path(named.iter().chain(&found).copied().collect()) } else { Vec::new() };
                // A thin wrapper among the definitions brings the function it calls, of the same
                // kind (a call path question follows calls already).
                let mut named_calls: Vec<Packet> = Vec::new();
                let mut found_calls: Vec<Packet> = Vec::new();
                if !trace {
                    let asked = Self::mentioned(q);
                    let written = self.symbols.iter().filter(|s| asked.iter().any(|n| s.name.eq_ignore_ascii_case(n)));
                    named_calls.extend(written.chain(named.iter().copied()).flat_map(|s| self.called_by(s, seed_kind)));
                    found_calls.extend(found.iter().flat_map(|s| self.called_by(s, found_kind)));
                }
                let mut seen: Vec<String> = Vec::new();
                if !trace {
                    seen.extend(out.iter().map(|p| p.key().to_owned()));
                }
                let packets = self
                    .definitions_of(named.into_iter(), seed_kind)
                    .into_iter()
                    .chain(named_calls)
                    .chain(self.definitions_of(found.iter().copied(), found_kind))
                    .chain(found_calls)
                    .chain(self.definitions_of(hops.into_iter(), "trace"));
                for p in packets {
                    if !seen.iter().any(|h| h == p.key()) {
                        seen.push(p.key().to_owned());
                        out.push(p);
                    }
                }
                out
            }
            "semantic" => self.semantic(q, &[], 4),
            "operational" => self.operational(q),
            "verification" => self.verification(q),
            "dependency" => self.dependency(q),
            "constraint" => self.constraint(q),
            "interface" => self.interface(q),
            "uncertainty" => self.uncertainty(q),
            "temporal" => {
                let terms = nmlex::key_terms(q);
                // Files ranked by how many of the question's terms their path holds (2026-11-09:
                // the first four in path order were taken, so a file matching every term could be
                // cut by files matching one), then the twelve best.
                let mut hits: Vec<(usize, &String, &Vec<String>)> = self
                    .history
                    .iter()
                    .map(|(f, cs)| (terms.iter().filter(|t| t.len() > 3 && f.to_lowercase().contains(t.as_str())).count(), f, cs))
                    .filter(|(n, _, _)| *n > 0)
                    .collect();
                hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
                hits.into_iter()
                    .take(12)
                    .map(|(_, f, cs)| {
                        // A working-tree entry (`- ...`, 2026-11-09) is not a commit: it is cited as
                        // `git:working-tree` and the commit count leaves it out.
                        let latest = cs.first().cloned().unwrap_or_default();
                        let commits = cs.iter().filter(|c| !c.starts_with("- ")).count();
                        let sha = if latest.starts_with("- ") {
                            "working-tree".to_owned()
                        } else {
                            latest.split(' ').next().unwrap_or("").to_owned()
                        };
                        let latest = latest.trim_start_matches("- ");
                        self.packet(
                            "history",
                            &format!("history:{f}"),
                            &format!("{f}: {commits} commits; latest {latest}"),
                            &format!("git:{sha}"),
                            Status::Observed,
                        )
                        .category(Category::Temporal)
                        .evidence(vec![format!("git:{sha} {f}: {latest}")])
                    })
                    .collect()
            }
            "epistemic" => nmlex::key_terms(q)
                .into_iter()
                .filter(|t| !self.known(t))
                .map(|t| {
                    self.packet(
                        "finding",
                        &format!("unknown:{t}"),
                        &format!("no file in the substrate contains `{t}`"),
                        "unit:nmsubs.views",
                        Status::Unknown,
                    )
                    .stance(Stance::Unknown)
                })
                .collect(),
            "objective" => {
                let terms = nmlex::key_terms(q);
                let covered = terms.iter().filter(|t| self.known(t.as_str())).count();
                vec![
                    self.packet(
                        "plan",
                        "objective",
                        &format!("objective terms [{}]; {covered} of {} seen in the substrate", terms.join(", "), terms.len()),
                        "unit:nmsubs.views",
                        Status::Inferred,
                    )
                    .category(Category::Instruction),
                ]
            }
            _ => Vec::new(),
        }
    }
}

impl SubstrateView for Substrate {
    fn version(&self) -> u64 {
        self.version
    }
    fn files(&self) -> Vec<String> {
        self.files.iter().map(|f| f.path.clone()).collect()
    }
    fn view(&self, name: &str, query: &str) -> Vec<Packet> {
        self.view_of(name, query)
    }
    /// Lines holding every term, best placed first: a code line inside a function's body, then a
    /// code line inside another definition, then the rest (file notes, comments, lines outside
    /// any definition, documents); within each, one line per definition before a second one from
    /// the same, then file order. A question in prose meets a file's notes and usage text first in
    /// file order, while the lines that decide sit in function bodies.
    fn grep(&self, terms: &[String], paths: &[String], limit: usize) -> Vec<Hit> {
        let terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();
        if terms.is_empty() || limit == 0 {
            return Vec::new();
        }
        let mut ranked: Vec<(u8, usize, usize, Hit)> = Vec::new();
        let mut per_def: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (fi, (f, low)) in self
            .files
            .iter()
            .zip(&self.low)
            .enumerate()
            .filter(|(_, (f, _))| paths.is_empty() || paths.iter().any(|p| f.path.starts_with(p.as_str())))
        {
            if !terms.iter().all(|t| low.contains(t.as_str())) {
                continue;
            }
            let defs: Vec<&crate::repo::Symbol> = if f.is_code() { self.symbols.iter().filter(|s| s.file == f.path).collect() } else { Vec::new() };
            // Lower-casing never adds or removes a line break, so the lines stay paired.
            for (i, (line, l)) in f.text.lines().zip(low.lines()).enumerate() {
                if !terms.iter().all(|t| l.contains(t.as_str())) {
                    continue;
                }
                let n = i + 1;
                let inner = |fns: bool| {
                    defs.iter()
                        .filter(|s| s.line <= n && n <= s.end && (!fns || (s.line < n && matches!(s.kind.as_str(), "fn" | "function" | "method"))))
                        .min_by_key(|s| s.end - s.line)
                };
                let (class, def) = match (is_comment(line), inner(true), inner(false)) {
                    (false, Some(d), _) => (0, d.line),
                    (false, None, Some(d)) => (1, d.line),
                    _ => (2, 0),
                };
                let k = per_def.entry((fi, def)).or_default();
                let nth = if class == 2 { 0 } else { *k };
                *k += 1;
                ranked.push((class, nth, ranked.len(), Hit { file: f.path.clone(), line: n, text: line.trim().to_owned() }));
            }
        }
        ranked.sort_by_key(|x| (x.0, x.1, x.2));
        ranked.into_iter().take(limit).map(|x| x.3).collect()
    }
    fn semantic_in(&self, query: &str, paths: &[String], k: usize) -> Vec<Packet> {
        self.semantic(query, paths, k)
    }
    fn symbols(&self, name: &str) -> Vec<Hit> {
        self.symbols.iter().filter(|s| s.name == name).map(|s| Hit { file: s.file.clone(), line: s.line, text: s.text.clone() }).collect()
    }
}

/// Self-check of the `views` sub-component on a two-file fixture: all eight views answer,
/// a constant defined twice with two values meets under one `value:` key, an absent term
/// is reported as unknown.
/// Whether a definition belongs to its file's public interface: `pub` in Rust, `export` in
/// JavaScript and TypeScript (or a member of an exported class), a top-level name without a
/// leading `_` in Python, any definition in a C or C++ header, a PowerShell function named in
/// `Export-ModuleMember` or any function of a script.
fn is_public(s: &crate::repo::Symbol) -> bool {
    let t = s.text.trim_start();
    match nmlex::lang::lang_of(&s.file).map(|l| l.id) {
        None | Some("rust") => t.starts_with("pub "),
        Some("typescript" | "javascript") => t.starts_with("export ") || (!s.parent.is_empty() && !t.starts_with("private ") && !t.starts_with('#')),
        Some("python") => !s.name.starts_with('_') && (s.parent.is_empty() || s.kind == "method"),
        Some("c" | "cpp" | "cuda") => {
            let ext = s.file.rsplit_once('.').map_or("", |(_, e)| e);
            matches!(ext, "h" | "hpp" | "hh" | "hxx" | "cuh") || (s.kind == "function" && !t.starts_with("static "))
        }
        Some(_) => true,
    }
}

pub(crate) fn check() -> Result<(), String> {
    let embed: crate::Embed = std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]);
    let s = crate::from_files(&[("a/x.rs", "pub const LIMIT: u8 = 4;\nfn go() {}\n"), ("b/y.rs", "const LIMIT: u8 = 9;\n")], embed);
    for v in VIEWS {
        let _ = s.view_of(v, "What is LIMIT?");
    }
    let vals: Vec<Packet> = s.view_of("structural", "What is LIMIT?").into_iter().filter(|p| p.key() == "value:LIMIT").collect();
    if vals.len() != 2 || vals[0].text() == vals[1].text() {
        return Err(format!("nmsubs.views: value packets {:?}", vals.iter().map(Packet::text).collect::<Vec<_>>()));
    }
    if !s.view_of("epistemic", "where is zebrafish").iter().any(|p| p.key() == "unknown:zebrafish") {
        return Err("nmsubs.views: absent term not reported".into());
    }
    if s.grep(&["limit".into()], &["b/".into()], 9).len() != 1 {
        return Err("nmsubs.views: grep inside a path".into());
    }
    // 2026-11-10 segments: a Rust item with its doc comment stays one segment; segments cover
    // the file without gaps.
    let src_lines = [
        "//! head",
        "",
        "/// doc of a",
        "pub fn a() {",
        "    let x = 1;",
        "    let y = 2;",
        "    let z = 3;",
        "}",
        "/// doc of b",
        "#[must_use]",
        "pub fn b() -> u8 {",
        "    1",
        "}",
        "",
        "",
        "",
    ];
    let segs = crate::segments("x.rs", &src_lines, &[4, 11]);
    if segs.first().map(|s| s.0) != Some(1)
        || segs.last().map(|s| s.1) != Some(src_lines.len())
        || !segs.iter().any(|s| s.0 == 9)
        || segs.windows(2).any(|w| w[1].0 != w[0].1 + 1)
    {
        return Err(format!("nmsubs.views: segments {segs:?}"));
    }
    // 2026-11-10 views: a rule bearing on the question, a public item of a named crate file, an
    // undefined name.
    let r = crate::from_files(
        &[
            ("rules.md", "# R\n## rules\n- Never delete records without a yes.\n- No em dashes.\n"),
            ("a/src/x.rs", "pub fn open_door() {}\nfn hidden() {}\n"),
        ],
        std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]),
    );
    if r.view_of("constraint", "may I delete records").len() != 1
        || r.view_of("interface", "public items of a/src/x.rs").len() != 1
        || !r.view_of("uncertainty", "where is close_window").iter().any(|p| p.key() == "undefined:close_window")
    {
        return Err("nmsubs.views: constraint, interface or uncertainty view".into());
    }
    // M1 re-read: a cited line found again (even at a moved number), a vanished text gives nothing.
    let moved = s.reread("a/x.rs:9 fn go() {}").map(|p| p.provenance().source.clone());
    if moved.as_deref() != Some("a/x.rs:2") || s.reread("a/x.rs:1 fn gone() {}").is_some() || s.reread("c/z.rs:1").is_some() {
        return Err(format!("nmsubs.views: reread {moved:?}"));
    }
    // Stored citations keep a content hash, never code: unchanged reads back, moved is found at
    // its new place, changed is stale.
    let embed = || std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]);
    let before = crate::from_files(&[("a/h.rs", "fn one() {}\nfn two() {}\nfn three() {}\n")], embed());
    let cite = before.cite_ref("a/h.rs:2 fn two() {}");
    let span = before.cite_ref("a/h.rs:1 lines 1..2");
    let moved_tree = crate::from_files(&[("a/h.rs", "// head\nfn one() {}\nfn two() {}\nfn three() {}\n")], embed());
    let changed_tree = crate::from_files(&[("a/h.rs", "fn one() {}\nfn two_renamed() {}\nfn three() {}\n")], embed());
    let fresh = matches!(before.reread_state(&cite), Reread::Fresh(ref p) if p.text() == "fn two() {}");
    let moved_ok = matches!(moved_tree.reread_state(&cite), Reread::Fresh(ref p) if p.provenance().source == "a/h.rs:3");
    let span_ok = matches!(before.reread_state(&span), Reread::Fresh(ref p) if p.text() == "fn one() {}\nfn two() {}");
    let stale = matches!(changed_tree.reread_state(&cite), Reread::Stale { start: 2, end: 2, .. });
    if cite.contains("fn ")
        || !cite.starts_with("a/h.rs:2 lines 2..2 h:")
        || cite.len() != "a/h.rs:2 lines 2..2 h:".len() + CITE_HASH_HEX
        || !fresh
        || !moved_ok
        || !span_ok
        || !stale
    {
        return Err(format!("nmsubs.views: hashed citation {cite} {span} fresh {fresh} moved {moved_ok} span {span_ok} stale {stale}"));
    }
    // 2026-11-09 job views: a use outside the definition, a test line after the marker, a
    // build record found by a shared term.
    let t = crate::from_files(
        &[("a/x.rs", "pub fn go_far() {}\nfn check() {\n    go_far();\n}\n"), ("b/y.rs", "fn other() { go_far(); }\n")],
        std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]),
    );
    let uses = t.view_of("dependency", "where is go_far used");
    let tests = t.view_of("verification", "where is go_far tested");
    if !uses.iter().any(|p| p.provenance().source == "b/y.rs:1")
        || uses.iter().any(|p| p.provenance().source == "a/x.rs:1")
        || !tests.iter().any(|p| p.provenance().source == "a/x.rs:3")
    {
        return Err("nmsubs.views: dependency or verification view".into());
    }
    // Call sites first: a call in code before a comment naming the item; a call is the name
    // standing alone with an opening parenthesis after it.
    let c = crate::from_files(
        &[("a/m.rs", "// see go_far for the rule\nfn one() { go_far(); }\n"), ("b/n.rs", "fn go_far() {}\n")],
        std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]),
    );
    let first = c.view_of("dependency", "what calls go_far").into_iter().find(|p| p.kind() == "uses").map(|p| p.provenance().source.clone());
    if first.as_deref() != Some("a/m.rs:2") || !calls("x = go_far(1)", "go_far") || calls("ago_far(", "go_far") || calls("go_far.x", "go_far") {
        return Err(format!("nmsubs.views: call sites first ({first:?})"));
    }
    // A thin wrapper brings the function it calls; a function with a body of its own does not.
    let w = crate::from_files(
        &[("a/w.rs", "/// Doc of the wrapper.\npub fn outer(x: u8) -> u8 {\n    inner(x)\n}\nfn inner(x: u8) -> u8 {\n    let y = x + 1;\n    let z = y * 2;\n    other(z)\n}\nfn other(x: u8) -> u8 { x }\n")],
        std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0]),
    );
    let sym = |n: &str| w.symbols.iter().find(|s| s.name == n);
    if sym("outer").and_then(|s| w.delegate(s)).map(|d| d.name.as_str()) != Some("inner") || sym("inner").and_then(|s| w.delegate(s)).is_some() {
        return Err("nmsubs.views: thin wrapper and the function it calls".into());
    }
    if let Ok(mut g) = t.ops.write() {
        g.push(crate::OpRecord {
            source: "a/x.rs:3".into(),
            text: "build error E0425 at a/x.rs:3 in package xcrate (check run b1): cannot find go_far".into(),
        });
    }
    if t.view_of("operational", "which build errors in xcrate").len() != 1 {
        return Err("nmsubs.views: operational view".into());
    }
    // M2: a failed Step Check and the later passing one on the same file become a fix record.
    let dir = nmtype::check_dir("nmsubs-operational").map_err(|e| e.to_string())?;
    let log = dir.join("build-outcomes.jsonl");
    let rec = |verdict: &str, added: &str| {
        format!("{{\"event\":\"stepcheck\",\"files\":[\"a/x.rs\"],\"verdict\":\"{verdict}\",\"failed_stage\":\"check\",\"detail\":\"E0308 mismatched types\",\"added\":[\"a/x.rs: {added}\"]}}
")
    };
    std::fs::write(&log, rec("fail", "let n: u32 = \\\"a\\\";") + &rec("pass", "let n: u32 = 7;")).map_err(|e| e.to_string())?;
    t.load_operational(&log);
    let fixes: Vec<String> =
        t.ops.read().map(|g| g.iter().filter(|r| r.text.contains("fixed by")).map(|r| r.text.clone()).collect()).unwrap_or_default();
    if fixes.len() != 1 || !fixes.first().is_some_and(|f| f.contains("E0308") && f.contains("let n: u32 = 7;")) {
        return Err(format!("nmsubs.views: fix record {fixes:?}"));
    }
    Ok(())
}
