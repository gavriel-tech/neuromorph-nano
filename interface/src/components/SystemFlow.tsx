"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel"): the chain and the glossary
// now live in the Cartridges side panel (Components, Glossary). A step that points at another view
// of the Cartridges page switches it in place (the address names it, so the page opens it).
// 2026-11-26 (Luis Jake Gabriel III: "is there documentation and interface controls for the process
// we used to load our base cartridges and adapter / filter cartridges and how the mapping flow works
// ... We need to make this a coherent system that works ... organized and intuitive to use"): the
// whole chain on one page, in the order it is built, each step with its state now (read from the
// engine, nothing assumed), where it is controlled in the interface, and the exact lab command where
// a step is done with the lab tools; then one glossary of every word the pages use.

import Link from "next/link";
import { Pill } from "@/components/ui";
import { setHash } from "@/lib/hash";
import { useApi } from "@/lib/api";

type Bases = { models?: { model: string; serving: string; versions: { key: string; serving: boolean }[] }[] };
type Downloads = { models?: { id: string; state: string }[]; cartridges?: { key: string; present: boolean }[] };
type Maps = { roles?: { role: string; serving_base?: string | null; maps: { state?: string; kind: string; purpose?: string }[] }[] };
type Surgery = { roles?: Record<string, { job_filters?: { enabled?: boolean }[] }> };
type Learned = { acting: number; total: number };
type Library = { catalogue?: { stacks?: unknown[]; active?: Record<string, string> } };
type Sources = { sources?: { official: boolean }[] };
type Prefs = { in_force?: number };
type Distil = { rows_total?: number; cartridges?: unknown[] };

function Step({ n, title, what, state, tone = "muted", to, label, command }: { n: number; title: string; what: string; state: string; tone?: "verified" | "attention" | "muted"; to?: string; label?: string; command?: string }) {
  return (
    <li className="grid grid-cols-[1.75rem_minmax(0,1fr)] gap-x-2 border-b border-line/60 py-3" data-e2e="flow-step">
      <span className="mono pt-0.5 text-[11px] text-faint">{String(n).padStart(2, "0")}</span>
      <div className="grid min-w-0 gap-1">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="mono text-[12px] uppercase tracking-[0.12em] text-text">{title}</span>
          <Pill tone={tone}>{state}</Pill>
        </div>
        <p className="read text-[11.5px] leading-relaxed text-muted">{what}</p>
        <div className="mono flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[10.5px]">
          {to &&
            (to.startsWith("/cartridges/#") ? (
              <button type="button" className="link" onClick={() => setHash(to.slice("/cartridges/".length), false)}>
                {label ?? "open"}
              </button>
            ) : (
              <Link href={to} className="link">
                {label ?? "open"}
              </Link>
            ))}
          {command && (
            <span className="min-w-0 text-faint">
              lab command: <code className="break-all text-muted">{command}</code>
            </span>
          )}
        </div>
      </div>
    </li>
  );
}

const TERMS: [string, string][] = [
  ["MicroModel", "one of the four small models the Nano runs: Embedder, Reranker, Reader, Writer."],
  ["Upstream open weights", "optional Apache-2.0 Qwen3 open weights the base cartridges were built from. Neuromorph Nano runs standalone on its base cartridges (plus small tokenizer/config JSONs); upstream open weights are optional for fallback (open mode), output-parity reference checks, parameter mapping, and cartridge surgery. Never altered."],
  ["Parameter map", "a measurement of a MicroModel's weights (which channels and weights matter for which job). Maps are what cuts and filters are computed from; serving reads only job filters."],
  ["Surgery", "the lab step that makes a new base cartridge from the one before it and the open weights, restoring the weights that matter most (nano-lab run pmap surgery)."],
  ["Base cartridge", "the mapped and compressed weights a MicroModel runs on (embedder-base, reranker-base, reader-base, writer-base), one version serving at a time."],
  ["Compensation cartridge", "small gains and biases that add back what one base cartridge version leaves out; it fits only that version."],
  ["Hardware cartridge", "settings for one machine (precision per model, batch sizes, kernels)."],
  ["Job filter", "a gain and bias per channel at a MicroModel's sockets for one kind of job, built over the serving base and kept only if it passes the held-out split gate; plugged in, it is read while serving."],
  ["Lens", "a project adapter fitted from one project's own material; it reshapes the question's vector toward that project and loads only for it."],
  ["Judge read-out", "a project adapter that adds a read-out to the reranker's close calls between code pieces."],
  ["Learning unit", "a registered low-rank adapter beside one node of a MicroModel, fitted from checked outcomes."],
  ["Learned unit", "anything learned that can be plugged in or out: lenses, read-outs, learned pieces and learning units, each in the compartment (project) it was fitted in."],
  ["Preference cartridge", "your way of working: rules you state in words, values learned from what you keep or undo, or both; it stacks on top."],
  ["Distillation", "learning from teacher outputs or curated open-model data, loaded as parameters into a learned cartridge of the kind you choose (adapter, filter and more), through the held-out gate."],
  ["Stack", "the cartridges plugged in together for one use, in the order base, hardware, adapter, learned, then preference."],
  ["Library", "a store of one kind of thing (the cartridge library, the parameter map library, the model library) on this machine, or an online cartridge library on Hugging Face."],
];

const STATES: [string, string][] = [
  ["serving", "used by the engine right now"],
  ["plugged in", "switched on; used whenever the engine serves"],
  ["in the cartridge library", "on disk, not plugged in"],
  ["on trial", "recorded, never applied, until its evidence lets it serve"],
  ["current / stale / history / experiment", "a map made on the serving base, on an older one, for an earlier version only, or in a comparison"],
];

/** Components that make, keep and serve parameters, with their state and where each is controlled. */
export default function SystemFlow() {
  const bases = useApi<Bases>("/api/cartridges/bases", 30000);
  const dl = useApi<Downloads>("/api/downloads", 30000);
  const maps = useApi<Maps>("/api/parameter-maps", 60000);
  const surgery = useApi<Surgery>("/api/surgery", 30000);
  const learned = useApi<Learned>("/api/learned-units", 30000);
  const lib = useApi<Library>("/api/cartridges/library", 30000);
  const sources = useApi<Sources>("/api/cartridge-sources", 60000);
  const prefs = useApi<Prefs>("/api/preferences", 60000);
  const distil = useApi<Distil>("/api/training/distil?limit=0", 60000);

  const open = dl.data?.models ?? [];
  const openIn = open.filter((m) => m.state === "installed").length;
  const allMaps = (maps.data?.roles ?? []).flatMap((r) => r.maps);
  const count = (s: string) => allMaps.filter((m) => m.state === s).length;
  const serving = (bases.data?.models ?? []).filter((b) => b.serving && b.serving !== "open weights");
  const jf = Object.values(surgery.data?.roles ?? {}).flatMap((r) => r.job_filters ?? []);
  const stacks = lib.data?.catalogue?.stacks?.length ?? 0;
  const inUse = Object.keys(lib.data?.catalogue?.active ?? {}).length;
  const third = (sources.data?.sources ?? []).filter((x) => !x.official).length;
  const distilled = distil.data?.cartridges?.length ?? 0;

  return (
    <div className="min-w-0">
      <p className="mono mb-2 text-[10.5px] text-faint">component · state · control</p>
      <ol className="grid" data-e2e="system-flow">
        <Step n={1} title="Upstream open weights" what="Qwen3 weights, Apache-2.0. Source of each MicroModel's tokenizer and configuration; reference for every base cartridge check; input to mapping." state={dl.data ? `${openIn} of ${open.length} on disk` : "reading"} tone={openIn === open.length && open.length ? "verified" : "muted"} to="/models/#downloads" label="Downloads" />
        <Step n={2} title="Parameter map library" what="Per-MicroModel measurements of weight importance, labelled with base, purpose and outcome. Failed maps are quarantined." state={maps.data ? `${count("current")} current · ${count("stale")} stale · ${count("history")} history · ${count("experiment")} experiments` : "reading"} tone={count("stale") ? "attention" : "verified"} to="/parameters/" label="Parameters" command="neuromorph-nano lab run pmap build-base <role> <share> gpu" />
        <Step n={3} title="Base cartridges" what="Each MicroModel's serving weights, built from maps, checked against the open weights. One version serves; others are kept." state={bases.data ? serving.map((b) => b.serving).join(" · ") || "none serving" : "reading"} tone={serving.length === (bases.data?.models ?? []).length && serving.length ? "verified" : "attention"} to="/cartridges/#bases" label="Base versions" />
        <Step n={4} title="Learning sources" what="Verified agent outcomes, stated preferences, teacher data. Healthy data only." state={`${prefs.data?.in_force ?? 0} stated rules in force · ${distil.data?.rows_total ?? 0} teacher pairs`} tone="verified" to="/learning/" label="Learning" />
        <Step n={5} title="Learned cartridges" what="Job filters, adapters, lenses, learning units, distilled and preference cartridges. One held-out gate; fit checked per MicroModel." state={surgery.data && learned.data ? `${jf.filter((f) => f.enabled !== false).length} of ${jf.length} job filters plugged · ${learned.data.acting} of ${learned.data.total} learned units acting · ${distilled} distilled` : "reading"} tone="verified" to="/cartridges/#build" label="Build" />
        <Step n={6} title="Cartridge parameters" what="Parameters of any cartridge, base cartridge or open weights, by tensor, layer, head or element, beside the maps of the same layers." state={jf.length ? `${jf.length} job filters and every learned cartridge` : "reading"} tone="verified" to="/parameters/" label="Parameters" />
        <Step n={7} title="Cartridge library and stacks" what="Cartridges on this machine, per MicroModel deck; stacks per use, ordered base, hardware, adapter, learned, preference." state={lib.data ? `${stacks} stacks · ${inUse} uses served` : "reading"} tone="verified" to="/cartridges/#stacks" label="Stacks" />
        <Step n={8} title="Serving the agent" what="Context Packets with citations and the preferences in force; primes; Writer notes. No outbound traffic from the Nano." state="live" tone="verified" to="/" label="Live Activity" />
        <Step n={9} title="Online cartridge libraries" what="Official and third-party cartridge libraries on Hugging Face; imported files. Same checks as local." state={sources.data ? `official${third ? ` + ${third} third party` : ""}` : "reading"} tone="verified" to="/models/#downloads" label="Cartridge libraries" />
      </ol>
    </div>
  );
}

/** Every word the pages use, and the states, the same on every page. */
export function Glossary() {
  return (
      <div className="grid gap-6" data-e2e="glossary">
        <dl className="grid gap-2">
          <span className="eyebrow">What each word means</span>
          {TERMS.map(([k, v]) => (
            <div key={k} className="grid gap-x-4 border-b border-line/40 pb-1.5">
              <dt className="mono text-[11px] text-text">{k}</dt>
              <dd className="read text-[11.5px] leading-relaxed text-muted">{v}</dd>
            </div>
          ))}
        </dl>
        <dl className="grid content-start gap-2">
          <span className="eyebrow">States, the same on every page</span>
          {STATES.map(([k, v]) => (
            <div key={k} className="grid gap-0.5 border-b border-line/40 pb-1.5">
              <dt className="mono text-[11px] text-text">{k}</dt>
              <dd className="read text-[11.5px] leading-relaxed text-muted">{v}</dd>
            </div>
          ))}
          <p className="read mt-2 text-[11px] text-faint">
            The full build guide, with every command used, is docs/build-flow.md in the Nano&apos;s folder.
          </p>
        </dl>
      </div>
  );
}
