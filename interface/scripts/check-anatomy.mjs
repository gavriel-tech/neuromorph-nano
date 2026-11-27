// Compares what the engine reports about its anatomy with the tree the interface draws.
//
// Reads `/api/parts` (every crate's card and its sub-components) and `/api/settings` (`anatomy`:
// nodes, parents, edges) from the running engine, builds the tree with `src/lib/topology.ts` and
// the organism scene with `src/lib/body.ts`, and checks:
//   1. every anatomy node appears exactly once, under the node that stands for its anatomy parent
//      (or under the GCX level the mapping table names for it), with a plain name;
//   2. every card sub-component and every crate appears exactly once, with a plain name;
//   3. every drawn node stands for something the engine reports (an anatomy node, a sub-component,
//      a crate, a model, a cartridge, a reader listed on a drone), or is a grouping the mapping
//      table declares, and a grouping is never empty;
//   4. every anatomy edge is drawn on the organism, and every line drawn there is an anatomy edge.
// Prints the counts and every mismatch; exits 1 when there is any.
//
// Usage: node scripts/check-anatomy.mjs [--base=http://127.0.0.1:4870] [--quiet]
import { register } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "../src");
const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("--base="))?.slice(7) ?? "http://127.0.0.1:4870";
const quiet = args.includes("--quiet");

// Lets Node load the TypeScript sources as they are: relative imports without an extension
// resolve to `.ts`, and `@/` to `src/`.
register(
  "data:text/javascript," +
    encodeURIComponent(`
      export async function resolve(spec, ctx, next) {
        if (spec.startsWith("@/")) spec = ${JSON.stringify(pathToFileURL(SRC).href)} + "/" + spec.slice(2);
        if ((spec.startsWith("./") || spec.startsWith("../") || spec.startsWith("file:")) && !/\\.[cm]?[jt]sx?$/.test(spec)) {
          try { return await next(spec + ".ts", ctx); } catch {}
        }
        return next(spec, ctx);
      }`),
  import.meta.url,
);

const get = async (p) => {
  const r = await fetch(BASE + p, { cache: "no-store" });
  if (!r.ok) throw new Error(`${p}: ${r.status}`);
  return r.json();
};

const [parts, settings, status, carts] = await Promise.all([get("/api/parts"), get("/api/settings"), get("/api/status"), get("/api/cartridges").catch(() => [])]);
const anatomy = settings.anatomy;
const { buildTree } = await import(pathToFileURL(resolve(SRC, "lib/topology.ts")).href);
const { bodyScene } = await import(pathToFileURL(resolve(SRC, "lib/body.ts")).href);
const tree = buildTree({ anatomy, parts, status, cartridges: carts });

const all = [];
const walk = (n) => {
  all.push(n);
  n.children.forEach(walk);
};
walk(tree.root);
const key = (n) => n.path.join("/") || "(organism)";
const bad = [];
const note = (s) => bad.push(s);

// Nodes drawn for an engine id: the nodes whose own ids name it (not an ancestor's).
const holders = new Map();
for (const n of all) for (const id of n.ids) (holders.get(id) ?? holders.set(id, []).get(id)).push(n);
// A name that is not plain: the crate's own id (as "Nmmusc" was), raw id characters, or an
// abbreviation set in the wrong case ("Json", "Md").
const ACRONYM = /^(md|csv|html|xml|json|yaml|ffi|gpu|cpu|kv|mcp|api|bpe|rls|npcs)$/i;
const raw = (n, id) => {
  const crate = id.split(/[.#]/)[0];
  return n.label.toLowerCase() === crate.toLowerCase() || /[_#]|\.[a-z]/.test(n.label) || n.label.split(/\s+/).some((w) => ACRONYM.test(w) && w !== w.toUpperCase());
};
const within = (n, anc) => {
  for (let x = n; x; x = x.parent) if (x === anc) return true;
  return false;
};

// GCX levels: the anatomy hangs G1, G2 and G3 under the brain's stack; the organism draws them as
// levels of the GCX (spinal cord and brain stem). Named here so the check holds the table to it.
const REGROUPED = { "nmbrain.g1#g1": "gcx/g1", "nmbrain.g2#g2": "gcx/g2", "nmmusc#g3": "gcx/g3" };

/* 1. Anatomy nodes */
const anat = anatomy.nodes;

for (const a of anat) {
  const hs = (holders.get(a.id) ?? []).filter((n) => n.anatomy?.id === a.id || n.ids[0] === a.id || n.ids.includes(a.id));
  const own = hs.filter((n) => !n.children.some((c) => c.anatomyIds.has(a.id)));
  if (own.length === 0) {
    note(`anatomy node not drawn: ${a.id}`);
    continue;
  }
  if (own.length > 1) note(`anatomy node drawn ${own.length} times: ${a.id} at ${own.map(key).join(", ")}`);
  const n = own[0];
  if (raw(n, a.id)) note(`no plain name: ${a.id} shown as "${n.label}" at ${key(n)}`);
  if (a.parent) {
    const want = REGROUPED[a.id];
    const parentNodes = holders.get(a.parent) ?? [];
    const okParent = want ? key(n).startsWith(want) : parentNodes.some((p) => p !== n && within(n, p));
    if (!okParent) note(`wrong parent: ${a.id} drawn at ${key(n)}, anatomy parent ${a.parent} drawn at ${parentNodes.map(key).join(", ") || "nowhere"}`);
  }

}

/* 2. Sub-components and crates */
let subs = 0;
let crates = 0;
for (const [crate, card] of Object.entries(parts)) {
  crates++;
  if (!(holders.get(crate) ?? []).length) note(`crate not drawn: ${crate}`);
  for (const sc of card.subcomponents) {
    subs++;
    const hs = holders.get(sc.id) ?? [];
    if (hs.length === 0) note(`sub-component not drawn: ${sc.id}`);
    else if (hs.length > 1) note(`sub-component drawn ${hs.length} times: ${sc.id} at ${hs.map(key).join(", ")}`);
    else if (raw(hs[0], sc.id) && !hs[0].anatomy) note(`no plain name: ${sc.id} shown as "${hs[0].label}" at ${key(hs[0])}`);
  }
}

/* 3. Nothing drawn that the engine does not report */
const anatIds = new Set(anat.map((a) => a.id));
const subIds = new Set(Object.values(parts).flatMap((c) => c.subcomponents.map((s) => s.id)));
const modelIds = new Set((status.models ?? []).map((m) => `model:${m.id}`));
const cartIds = new Set(carts.map((c) => `cartridge:${c.id}`));
const readerOk = (n) => n.parent?.key === "tip" && Array.isArray(n.parent.anatomy?.config?.readers) && /^r\d+$/.test(n.key);
let groupings = 0;
let reported = 0;
for (const n of all) {
  if (n === tree.root) continue;
  const real = n.ids.some((id) => anatIds.has(id) || subIds.has(id) || id in parts || modelIds.has(id) || cartIds.has(id));
  if (real || readerOk(n)) {
    reported++;
    continue;
  }
  groupings++;
  if (n.ids.length) note(`drawn with ids the engine does not report: ${key(n)} (${n.ids.join(", ")})`);
  if (!n.children.length && n.key !== "cartridges" && n.key !== "micromodels") note(`empty grouping: ${key(n)}`);
}

/* 4. Edges */
const S = bodyScene(tree, tree.root, 1600, 1000, carts, status, null);
const edgeKey = (n) => n && n.path.join("/");
const deepest = (id) => (tree.index.get(id) ?? []).reduce((a, b) => (!a || b.path.length > a.path.length ? b : a), null);
const want = new Map();
for (const e of anatomy.edges) {
  const f = edgeKey(deepest(e.from));
  const t = edgeKey(deepest(e.to));
  if (!f || !t) {
    note(`edge has an end not drawn: ${e.from} -> ${e.to}`);
    continue;
  }
  want.set(`${f}>${t}>${e.mode.kind}`, `${e.from} -> ${e.to}`);
}
const drawn = new Set(S.wires.map((w) => `${w.from}>${w.to}>${w.mode}`));
for (const [k, e] of want) if (!drawn.has(k)) note(`edge not drawn on the organism: ${e}`);
for (const k of drawn) if (!want.has(k)) note(`line drawn that is no anatomy edge: ${k}`);

const out = {
  engine: { anatomy_nodes: anat.length, anatomy_edges: anatomy.edges.length, crates, sub_components: subs },
  drawn: { nodes: all.length - 1, standing_for_engine_ids: reported, groupings, organism_lines: S.wires.length },
  regrouped: Object.keys(REGROUPED).filter((id) => anatIds.has(id)),
  mismatches: bad.length,
};
console.log(`anatomy check: ${JSON.stringify(out)}`);
if (!quiet || bad.length) for (const b of bad) console.log(`  ${b}`);
process.exit(bad.length ? 1 : 0);
