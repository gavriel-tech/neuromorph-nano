"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel ... the right menu are libraries
// and or discription tools etc."): what the Cartridges side panel holds besides the library: the base
// cartridge versions with the upgrade choice, every cartridge's spec sheet, the kinds of cartridges,
// and the page's own description.

import Link from "next/link";
import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import MicroModelName from "@/components/MicroModel";
import { KIND_SWATCH, type KartKind } from "@/components/CartridgeBay";
import type { BasesPayload, Note } from "@/components/CartridgeDecks";
import { Empty, KV, Pill, Placeholder, Segmented, Switch, Tick } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi, type Cartridge, type LibraryMark } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { cartridgeWords, domainShort, familyOf, type Family } from "@/lib/evidence";
import { maskName } from "@/lib/privacy";

const REFRESH = ["/api/cartridges/bases", "/api/surgery", "/api/cartridges", "/api/cartridges/library", "/api/live", "/api/learned-units", "/api/lenses"];

/** One change through the engine's API, then the views that show it refreshed and a note. */
export async function change(path: string, body: unknown, done: string, onNote: (n: Note) => void, refresh: string[] = REFRESH): Promise<boolean> {
  try {
    await post<unknown>(path, body);
    for (const r of [...refresh, "/api/audit"]) refreshAll(r);
    onNote({ text: done, tone: "verified" });
    return true;
  } catch (e) {
    onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    return false;
  }
}

const markOf = (c: Cartridge): LibraryMark | undefined => c.library ?? c.book;
const scopeShort = (s: string) => domainShort(s, maskName);
const small = "press mono border-b px-0.5 text-[10px] uppercase tracking-[0.08em] disabled:opacity-50";

/** What an upgrade does with older base cartridge versions (`library.upgrade`). */
export function UpgradePolicy({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<BasesPayload>("/api/cartridges/bases", 10000);
  const [busy, setBusy] = useState(false);
  const policy = q.data?.policy === "replace" ? "replace" : "keep";
  return (
    <span className="mono flex flex-wrap items-center gap-2 text-[10.5px] text-faint" data-e2e="base-upgrade-policy" title="What an upgrade does with older base cartridge versions; a version you chose is always kept">
      <span>new versions</span>
      <Segmented
        label="When new base cartridges arrive"
        values={["keep", "replace"]}
        value={policy}
        busy={busy}
        render={(p) => (p === "keep" ? "keep both" : "replace old")}
        onPick={async (p) => {
          setBusy(true);
          await change("/api/controls", { key: "library.upgrade", value: p }, p === "keep" ? "Upgrades keep new base cartridges beside the older versions. Logged in Audit, where it can be undone." : "Upgrades replace older base cartridges once the new version is usable (a version you chose is kept). Logged in Audit, where it can be undone.", onNote, ["/api/cartridges/bases"]);
          setBusy(false);
        }}
      />
    </span>
  );
}

/** The base cartridge versions of every MicroModel (or one): what serves, the choice, use a
 *  version, remove an old one. */
export function BaseVersions({ onNote, only }: { onNote: (n: Note) => void; only?: string }) {
  const q = useApi<BasesPayload>("/api/cartridges/bases", 10000);
  const [busy, setBusy] = useState<string | null>(null);
  const run = async (id: string, body: unknown, done: string) => {
    setBusy(id);
    await change("/api/cartridges/bases", body, done, onNote, ["/api/cartridges/bases", "/api/cartridges/library", "/api/cartridges"]);
    setBusy(null);
  };
  const d = q.data;
  if (q.error && !d) return <Empty title="Not reported by this engine build">{q.error}</Empty>;
  if (d?.error) return <Empty title="The cartridge library could not be opened">{d.error}</Empty>;
  if (!d) return <Placeholder className="h-24" what="base cartridges" />;
  if (!d.models?.length) return <Empty title="No base cartridge in the cartridge library">Without one a MicroModel serves its upstream open weights (open mode). Get them on Models, Downloads.</Empty>;
  return (
    <>
    {/* 2026-11-26: new base versions are built by the lab, inside the program (Parameters, Build). */}
    <p className="mono mb-3 text-[10px] text-faint" data-e2e="base-build-link">
      A new version is built on{" "}
      <Link href="/parameters/#add" className="link">
        Parameters, Build
      </Link>{" "}
      and kept beside these; choose what serves here.
    </p>
    <ul className="grid gap-4" data-e2e="base-versions">
      {d.models
        .filter((m) => !only || m.model === only)
        .map((m) => (
          <li key={m.model} className="grid gap-2 border-b border-line/60 pb-3">
            {!only && <MicroModelName model={m.model} layout="stack" nameClass="text-[12px] uppercase tracking-[0.16em] text-text" />}
            <div className="flex flex-wrap items-center gap-2">
              <span className="mono text-[10.5px] text-faint" title="Applies at the next start of the engine">serve</span>
              <Segmented
                label={`What serves ${m.model}`}
                values={["auto", "open", ...(m.choice !== "auto" && m.choice !== "open" ? [m.choice] : [])]}
                value={m.choice}
                busy={busy !== null}
                render={(c) => (c === "auto" ? "newest" : c === "open" ? "open weights" : c.replace(/^.*@/, "v"))}
                onPick={(c) => run(`${m.model}-${c}`, { model: m.model, choice: c }, c === "auto" ? `${m.model}: the newest usable base cartridge serves from the next start.` : c === "open" ? `${m.model}: its open weights serve from the next start.` : `${m.model}: ${c} serves from the next start.`)}
              />
            </div>
            <ul className="grid">
              {m.versions.map((v) => {
                const chosen = m.choice === v.key;
                return (
                  <li key={v.key} className="grid gap-1 py-1">
                    <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
                      <span aria-hidden className={v.serving ? "spectral-sw !h-4 !w-1" : "dither-slot !h-4 !w-2"} />
                      <span className="mono min-w-0 truncate text-[11.5px] text-text">{v.key}</span>
                      {v.serving ? <Pill tone="learned">serves</Pill> : !v.present ? <Pill tone="attention">not downloaded</Pill> : v.refused ? <Pill tone="error">refused</Pill> : <Pill>in the cartridge library</Pill>}
                      {chosen && <Pill tone="signal">chosen</Pill>}
                    </div>
                    <div className="mono flex flex-wrap items-center gap-x-3 pl-3.5 text-[10px] text-faint">
                      <span>
                        {bytes(v.bytes)}
                        {v.made ? ` · ${v.made.slice(0, 10)}` : ""}
                      </span>
                      <span className="ml-auto flex items-center gap-2">
                        {!chosen && (
                          <button type="button" className={`${small} border-line-strong text-muted hover:text-text`} disabled={busy !== null || !v.present} onClick={() => run(`${v.key}-use`, { model: m.model, choice: v.key }, `${m.model}: ${v.key} serves from the next start.`)}>
                            use this
                          </button>
                        )}
                        {!v.serving && !chosen && (
                          <DeleteFlow<{ audit_id?: string }>
                            path="/api/cartridges/bases/remove"
                            body={{ key: v.key }}
                            label="remove"
                            title="Asks what would go first, then asks you to type its name."
                            startClass={`${small} border-line text-faint hover:text-error`}
                            disabled={busy !== null}
                            done={() => `${v.key} removed from the cartridge library. Logged in Audit.`}
                            refresh={["/api/cartridges"]}
                          />
                        )}
                      </span>
                    </div>
                    {!v.present ? (
                      <p className="read pl-3.5 text-[10.5px] leading-relaxed text-faint">Not in the cartridge library yet; run the installer again. Until then it serves {m.serving === "open weights" ? "its open weights" : m.serving}.</p>
                    ) : (
                      v.refused && <p className="read pl-3.5 text-[10.5px] leading-relaxed text-faint">{cartridgeWords(v.refused.replace(`${v.key}: `, ""))}</p>
                    )}
                  </li>
                );
              })}
            </ul>
          </li>
        ))}
    </ul>
    </>
  );
}

/** The Base versions tab: the upgrade choice, then every MicroModel's versions. */
export function BasesTab({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<BasesPayload>("/api/cartridges/bases", 10000);
  const policy = q.data?.policy === "replace" ? "replace" : "keep";
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <UpgradePolicy onNote={onNote} />
        <Link href="/models/#downloads" className="link mono text-[10.5px]">
          get a version
        </Link>
      </div>
      <BaseVersions onNote={onNote} />
      <p className="mono text-[10px] text-quiet">{policy === "keep" ? "upgrades add versions beside the old ones" : "upgrades remove older versions once the new one is usable"} · choices apply at the next start · every change is in Audit</p>
    </div>
  );
}

/** One cartridge's spec sheet: its state, then its values set on the ground, one column. */
export function CartridgeSpec({ c, onNote }: { c: Cartridge; onNote: (n: Note) => void }) {
  const m = markOf(c);
  const family = familyOf(c);
  const [busy, setBusy] = useState(false);
  const loading = c.loaded && typeof c.bytes === "number" && typeof c.loaded_bytes === "number" && c.loaded_bytes < c.bytes;
  const flip = async (on: boolean) => {
    setBusy(true);
    try {
      await post<unknown>("/api/cartridges", { id: c.id, on });
      refreshAll("/api/cartridges");
      refreshAll("/api/audit");
      onNote({ text: `Cartridge ${c.id} switched ${on ? "on" : "off"}. Logged in Audit, where it can be undone.`, tone: "verified" });
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const total = c.bytes;
  const got = c.loaded_bytes ?? (c.loaded ? total : undefined);
  const verified = c.verified ?? /^(pass|passed|ok|verified|accept)$/i.test(c.check);
  const trial = m?.trial ?? c.check === "trial";
  const settingsOnly = c.model === "settings" || m?.target === "settings";
  const scopes = (c.scopes ?? []).map(scopeShort);
  return (
    <div className="grid gap-1" data-e2e="cartridge-spec" data-key={`${c.id}@${c.version}`}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
        <span className="mono min-w-0 truncate text-[12px] text-text" title={c.id}>
          {c.id} <span className="text-faint">v{c.version}</span>
        </span>
        {m ? <Pill tone={m.in_use === "served" ? "learned" : m.in_use === "trial" ? "attention" : "muted"}>{m.in_use === "served" ? "in use" : m.in_use === "trial" ? "in use on trial" : "in the cartridge library"}</Pill> : <Pill tone={loading ? "signal" : c.loaded ? "learned" : "muted"}>{loading ? "loading" : c.loaded ? "loaded" : "not loaded"}</Pill>}
        {trial && m?.in_use !== "trial" && <Pill tone="attention">on trial</Pill>}
        {family === "preference" && !m && (
          <span className="ml-auto">
            <Switch on={c.loaded} busy={busy} label={`Cartridge ${c.id}`} onChange={flip} />
          </span>
        )}
      </div>
      {c.description && <p className="read text-[11.5px] leading-relaxed text-muted">{cartridgeWords(c.description)}</p>}
      <KV k="Family">{family}</KV>
      <KV k="Scope" mono>{scopes.length ? scopes.join(", ") : m?.domain ? scopeShort(m.domain) : "general"}</KV>
      <KV k="Made for" mono>{settingsOnly ? "settings values" : (c.model ?? c.base)}</KV>
      <KV k="Socket" mono>{settingsOnly ? "settings, no layer" : c.layers ? `layers ${c.layers[0]} to ${c.layers[1]}` : "every layer"}</KV>
      <KV k="Kind">{c.kind === "affine" ? "gain and bias on a vector" : c.kind === "settings" ? "new settings values" : c.kind}</KV>
      <KV k="Trial">{trial ? <span className="text-attention">on trial, never changes what is served</span> : "no"}</KV>
      <KV k="Check">
        <span className={verified ? "text-verified" : "text-error"}>{verified ? "files match their hashes" : "a file fails its hash"}</span>
      </KV>
      {m && <KV k="Evidence" mono>{m.evidence_rows ?? 0} measurements</KV>}
      {m && <KV k="Stacks" mono>{m.stacks?.length ? m.stacks.map((s) => s.replace(/@\d+$/, "")).join(", ") : "none"}</KV>}
      {!m && <KV k="Precision" mono>{c.quantization}</KV>}
      {c.date && <KV k="Made" mono>{c.date}</KV>}
      <KV k="Size" mono>
        {total !== undefined ? (
          got !== undefined && got !== total && got > 0 ? (
            <>
              <Tick value={got} format={(n) => bytes(n)} /> of {bytes(total)}
            </>
          ) : (
            bytes(total)
          )
        ) : (
          `${c.components.length} components`
        )}
      </KV>
    </div>
  );
}

const SPEC_GROUPS: { family: Family; title: string; hint: string }[] = [
  { family: "fundamentals", title: "Fundamentals", hint: "Hardware operating numbers, code and ML fundamentals and the Nano's own system, fitted from stable reference material and its own measurements." },
  { family: "learning", title: "Learning", hint: "Learned from verified work with your coding agents only: a passed test, a fix that held, a place confirmed by use." },
  { family: "preference", title: "Preference", hint: "Your way of working, from edits kept or reverted and rules you stated. It plugs in on top, so your way wins." },
];

/** Every cartridge's spec sheet, by family; each opens its inspector. */
export function SpecSheets({ all, onNote, onOpen }: { all: Cartridge[]; onNote: (n: Note) => void; onOpen: (c: Cartridge) => void }) {
  return (
    <div className="grid gap-5" data-e2e="spec-sheets">
      {SPEC_GROUPS.map((g) => {
        const list = all.filter((c) => familyOf(c) === g.family);
        return (
          <section key={g.family} className="grid gap-2">
            <h3 className="eyebrow flex items-baseline justify-between gap-3 text-muted" title={g.hint}>
              <span>{g.title}</span>
              <span className="normal-case tracking-normal text-faint">
                {list.filter((c) => c.loaded).length} in use of {list.length}
              </span>
            </h3>
            {list.length === 0 ? (
              <div className="mono py-2 text-[11px] text-quiet">none yet</div>
            ) : (
              <ul className="grid">
                {list.map((c) => (
                  <li key={`${c.id}@${c.version}`} className="grid gap-1 border-b border-line/60 py-3">
                    <CartridgeSpec c={c} onNote={onNote} />
                    <div>
                      <button type="button" onClick={() => onOpen(c)} className="press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text">
                        {c.kind === "settings" ? "inspect / edit values" : "inspect"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </div>
  );
}

// 2026-11-25 (directions of Luis Jake Gabriel III: the interface explains the kinds of cartridges).
export const KINDS: { kind: KartKind; name: string; what: string }[] = [
  { kind: "base", name: "Base cartridge", what: "The MicroModel's own parameters, mapped and compressed for its job from the upstream open weights. One serves per MicroModel; other versions wait in the cartridge library." },
  { kind: "comp", name: "Compensation", what: "Small gains and biases that add back what one base cartridge version leaves out; it fits only that version." },
  { kind: "filter", name: "Job filter", what: "A filter over some of a MicroModel's layers for one job. Plugs in or out live, from the next call." },
  { kind: "adapter", name: "Adapter and lens", what: "Learned from one project's verified places; loads only for that project, on the base it was fitted on." },
  { kind: "preference", name: "Preference", what: "Your way of working, from edits you kept or reverted and rules you stated. Sits on top, so your way wins." },
  { kind: "settings", name: "Engine cartridge", what: "Operating numbers for a piece of hardware or corrections for the whole engine. Sets values, not parameters." },
];

/** The kinds, each with its colour, as a legend. */
export function KindsGuide() {
  return (
    <ul className="grid gap-2.5" aria-label="Kinds of cartridges" data-e2e="cartridge-kinds">
      {KINDS.map((k) => (
        <li key={k.kind} className="flex min-w-0 gap-2.5">
          <span aria-hidden className={`mt-0.5 inline-block h-7 w-1 shrink-0 ${KIND_SWATCH[k.kind]}`} />
          <div className="min-w-0">
            <div className="mono text-[11px] uppercase tracking-[0.12em] text-text">{k.name}</div>
            <p className="read mt-0.5 text-[11px] leading-relaxed text-faint">{k.what}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

const PART_WORD: Record<string, string> = { cartridges: "cartridges", maps: "parameter maps", units: "learned units" };

/** Where the user's own backup repositories touch this page: how many of them hold the named parts
 *  (cartridges, parameter maps, learned units) and when each last backed up. Read only; adding a
 *  repository, backing up, snapshots and restore are on Storage, Backups. */
export function BackupsLine({ parts }: { parts: string[] }) {
  const q = useApi<{ repos?: { id: string; url?: string; parts: string[]; last?: { at_s: number } }[]; error?: string }>("/api/backups", 30000);
  const now = useNow();
  const words = parts.map((p) => PART_WORD[p] ?? p).join(" and ");
  const repos = q.data?.repos ?? [];
  const holding = repos.filter((r) => parts.some((p) => r.parts.includes(p)));
  const last = holding.reduce<number | null>((a, r) => (r.last && (a === null || r.last.at_s > a) ? r.last.at_s : a), null);
  return (
    <div className="mono grid gap-1 text-[10.5px] text-faint" data-e2e="cart-backups">
      <span>
        {q.error && !q.data
          ? `backups not read: ${q.error}`
          : !q.data
            ? "reading the backup repositories"
            : repos.length === 0
              ? `no backup repository yet: the ${words} are on this machine only`
              : `${holding.length} of ${repos.length} backup ${repos.length === 1 ? "repository holds" : "repositories hold"} the ${words}${holding.length ? ` · last backed up ${last !== null ? ago(last * 1000, now) : "never"}` : ""}`}
      </span>
      <Link href="/storage/#backups" className="link w-fit" title="Your own repositories: add one, back up now, snapshots, preview and restore">
        back up and restore on Storage
      </Link>
    </div>
  );
}

/** The page's description, for the About tab. */
export function AboutCartridges() {
  return (
    <div className="read grid gap-3 text-[12px] leading-relaxed text-muted">
      <p>Each MicroModel (Reader, Writer, Embedder, Reranker) runs on a base cartridge, built from the upstream open weights. Cartridges dock on top of it in the order they apply: compensation, job filters, adapters and lenses, then preference. The base cartridges and the open weights never change; unplugging gives the base back exactly.</p>
      <p>Every plug, unplug and version choice goes through the engine and is logged in Audit, where it can be undone. A job filter or a learned unit applies from the next call; a base version from the next start of the engine.</p>
      <p className="mono text-[10.5px] text-faint">
        <Link href="/topology/#/learning/cartridges" className="link">
          see the cartridges in topology
        </Link>
        {" · "}
        <Link href="/models/#downloads" className="link">
          get base cartridges and libraries
        </Link>
        {" · "}
        <Link href="/parameters/" className="link">
          import a file or build a job filter
        </Link>
      </p>
    </div>
  );
}
