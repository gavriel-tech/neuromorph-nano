"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel."): the inspector now opens in
// the Parameters page's right rail (a map's detail), and the import panel is the rail's Import tab, so
// both are laid out for a narrow column and say what they do in a line, not a paragraph.
//
// 2026-11-26 (audit A2 and A4; Luis Jake Gabriel III: "Does the ui show the ability to map out and
// add and remove parameters from mapping through the interface for certain cartridges?"): open a
// parameter map and read what is in it, at any depth, without loading a 55 MB map whole; edit the
// maps that are meant to be edited (role.json, super-weights.json, tol-*, job filters, socket by
// socket and channel by channel) with the engine checking the shape before it saves and keeping the
// old file for Undo; and import your own maps and cartridges from this machine.

import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { roleTitle } from "@/components/MicroModel";
import { Empty, Placeholder } from "@/components/ui";
import { bytes, post, refreshAll, useApi } from "@/lib/api";
import { maskPath } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };
type KeyRow = { key: string; bytes: number; type: string };
type Read = {
  role: string;
  file: string;
  kind: string;
  about: string;
  bytes: number;
  editable: boolean;
  cap_bytes: number;
  mode?: "full" | "value" | "keys" | "items" | "rows";
  path?: string[];
  value_bytes?: number;
  type?: string;
  content?: unknown;
  keys?: KeyRow[];
  items?: unknown[];
  rows?: unknown[];
  offset?: number;
  limit?: number;
  total?: number;
  returned?: number;
  truncated?: boolean;
  note?: string;
  error?: string;
};
type Part = { target?: string; gain?: number[]; bias?: number[] };

const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const SHOW_CHARS = 60000;

const mean = (a: number[] | undefined) => (a && a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);

/** A job filter's parts as a table: the socket each part acts on and how far its gains and biases
 *  are from doing nothing; one part opens for channel-by-channel editing. */
function PartsTable({ parts, onEdit }: { parts: Part[]; onEdit: (i: number) => void }) {
  return (
    <div className="max-h-[40vh] overflow-auto">
      <table className="mono w-full text-left text-[10.5px]">
        <thead className="sticky top-0 bg-bg text-faint">
          <tr>
            <th className="py-1 pr-3 font-normal">socket</th>
            <th className="py-1 pr-3 font-normal">channels</th>
            <th className="py-1 pr-3 font-normal">gain mean</th>
            <th className="py-1 pr-3 font-normal">bias mean</th>
            <th className="py-1 font-normal" />
          </tr>
        </thead>
        <tbody>
          {parts.map((p, i) => (
            <tr key={`${p.target}-${i}`} className="border-t border-line/40">
              <td className="py-1 pr-3 text-muted">{p.target ?? `part ${i}`}</td>
              <td className="py-1 pr-3 text-faint">{p.gain?.length ?? 0}</td>
              <td className="py-1 pr-3 text-faint">{mean(p.gain)?.toFixed(6) ?? "-"}</td>
              <td className="py-1 pr-3 text-faint">{mean(p.bias)?.toFixed(6) ?? "-"}</td>
              <td className="py-1 text-right">
                <button type="button" className={link} onClick={() => onEdit(i)}>
                  edit channels
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One map, opened: its value at the path, browsable when large, editable when it is meant to be. */
export function MapInspector({ role, file, onClose }: { role: string; file: string; onClose: () => void }) {
  const [path, setPath] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);
  const [draft, setDraft] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const qs = new URLSearchParams({ role, file });
  if (path.length) qs.set("section", path.join("/"));
  if (offset) qs.set("offset", String(offset));
  const q = useApi<Read>(`/api/parameter-maps/read?${qs.toString()}`);
  const r = q.data;
  const go = (p: string[]) => {
    setPath(p);
    setOffset(0);
    setDraft(null);
  };
  const save = async () => {
    if (draft === null || !r) return;
    let content: unknown;
    try {
      content = JSON.parse(draft);
    } catch (e) {
      setNote({ text: `Not saved: the edit is not JSON (${e instanceof Error ? e.message : String(e)}).`, tone: "error" });
      return;
    }
    setBusy(true);
    try {
      const res = await post<{ audit_id?: string; backup?: string }>("/api/parameter-maps/save", { role, file, content, ...(path.length ? { section: path.join("/") } : {}) });
      setNote({ text: `Saved ${file}${path.length ? ` at /${path.join("/")}` : ""}. The old file is kept${res.backup ? ` (${maskPath(res.backup)})` : ""}; Audit ${res.audit_id ?? ""} undoes it.`, tone: "verified" });
      setDraft(null);
      for (const p of ["/api/parameter-maps", "/api/surgery", "/api/audit"]) refreshAll(p);
    } catch (e) {
      setNote({ text: `Not saved: ${e instanceof Error ? e.message : String(e)}`, tone: "error" });
    }
    setBusy(false);
  };
  const editable = !!r?.editable && (r.mode === "full" || r.mode === "value");
  const text = r?.content !== undefined ? JSON.stringify(r.content, null, 2) : "";
  const parts = r?.mode === "full" && file.startsWith("job-filter-") && r.content && typeof r.content === "object" ? ((r.content as { parts?: Part[] }).parts ?? null) : null;
  return (
    <div className="grid gap-3 border-l-2 border-sense-pale py-2 pl-3" data-e2e="map-inspector">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="mono flex flex-wrap items-center gap-1 text-[10.5px] text-faint">
          <button type="button" className="press hover:text-text" onClick={() => go([])}>
            {file}
          </button>
          {path.map((step, i) => (
            <span key={`${step}-${i}`} className="flex items-center gap-1">
              /
              <button type="button" className="press hover:text-text" onClick={() => go(path.slice(0, i + 1))}>
                {step}
              </button>
            </span>
          ))}
        </span>
        <span className="flex items-center gap-4">
          {editable && draft === null && (
            <button type="button" className={link} onClick={() => setDraft(text)}>
              edit
            </button>
          )}
          <button type="button" className={link} onClick={onClose}>
            close
          </button>
        </span>
      </div>
      <NoteLine note={note} />
      {!r && q.error ? (
        <Empty title="The map did not answer">{q.error}</Empty>
      ) : !r ? (
        <Placeholder className="h-16" what="map" />
      ) : r.error ? (
        <Empty title="Not readable">{r.error}</Empty>
      ) : (
        <>
          <p className="read text-[11px] leading-relaxed text-faint">
            {r.about} {bytes(r.bytes)} on disk{r.value_bytes !== undefined && path.length ? `; this value ${bytes(r.value_bytes)}` : ""}. {r.editable ? "Editable: the engine keeps its shape, saves the old file aside, and Audit undoes the save." : "A measurement a build reads: shown, not edited here."}
          </p>
          {r.note && <p className="mono text-[10.5px] text-quiet">{r.note}</p>}
          {r.mode === "keys" && (
            <ul className="grid max-h-[40vh] gap-0.5 overflow-auto">
              {(r.keys ?? []).map((k) => (
                <li key={k.key}>
                  <button type="button" onClick={() => go([...path, k.key])} className="press mono flex w-full justify-between gap-2 border-b border-line/40 py-1 text-left text-[10.5px] text-muted hover:text-text">
                    <span className="truncate">{k.key}</span>
                    <span className="shrink-0 text-faint">
                      {k.type} · {bytes(k.bytes)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {(r.mode === "items" || r.mode === "rows") && (
            <ol className="grid max-h-[40vh] gap-1 overflow-auto">
              {(r.mode === "items" ? r.items ?? [] : r.rows ?? []).map((it, i) => (
                <li key={i} className="grid grid-cols-[3rem_minmax(0,1fr)] gap-2 border-b border-line/40 py-1">
                  {r.mode === "items" ? (
                    <button type="button" onClick={() => go([...path, String((r.offset ?? 0) + i)])} className="press mono text-left text-[10.5px] text-faint hover:text-text">
                      {(r.offset ?? 0) + i}
                    </button>
                  ) : (
                    <span className="mono text-[10.5px] text-faint">{(r.offset ?? 0) + i}</span>
                  )}
                  <code className="mono min-w-0 truncate text-[10.5px] text-muted" title={JSON.stringify(it)}>
                    {JSON.stringify(it)}
                  </code>
                </li>
              ))}
            </ol>
          )}
          {(r.mode === "items" || r.mode === "rows" || r.mode === "keys") && (
            <div className="mono flex items-center gap-4 text-[10.5px] text-faint">
              <button type="button" className={link} disabled={(r.offset ?? 0) === 0} onClick={() => setOffset(Math.max(0, (r.offset ?? 0) - (r.limit ?? 100)))}>
                earlier
              </button>
              <span>
                from {r.offset ?? 0} of {r.total ?? "?"}
              </span>
              <button type="button" className={link} disabled={!r.truncated && !(r as { more_keys?: boolean }).more_keys} onClick={() => setOffset((r.offset ?? 0) + (r.mode === "items" ? r.returned ?? r.limit ?? 100 : r.limit ?? 100))}>
                later
              </button>
            </div>
          )}
          {parts && draft === null && (
            <PartsTable
              parts={parts}
              onEdit={(i) => {
                go(["parts", String(i)]);
              }}
            />
          )}
          {(r.mode === "full" || r.mode === "value") && draft === null && !parts && (
            <pre className="mono max-h-[50vh] overflow-auto border-l border-line pl-3 text-[10.5px] leading-relaxed text-muted">
              {text.length > SHOW_CHARS ? `${text.slice(0, SHOW_CHARS)}\n... ${text.length - SHOW_CHARS} more characters (open a key to read it all)` : text}
            </pre>
          )}
          {draft !== null && (
            <div className="grid gap-2">
              <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={16} spellCheck={false} className="mono w-full resize-y border border-line bg-transparent p-2 text-[10.5px] leading-relaxed text-text outline-none focus-visible:border-sense-pale" aria-label={`Edit ${file}`} />
              <div className="flex items-center gap-4">
                <button type="button" disabled={busy} onClick={() => void save()} className="press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40">
                  {busy ? "saving" : "save"}
                </button>
                <button type="button" disabled={busy} onClick={() => setDraft(null)} className={link}>
                  cancel
                </button>
                <span className="read text-[11px] text-quiet">The same keys and kinds; lists keep their length; every number finite. The engine refuses anything else.</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Your own maps and cartridges, from a path on this machine (nothing is fetched from a network). */
export function ImportPanel({ roles }: { roles: string[] }) {
  const [role, setRole] = useState(roles[0] ?? "reranker");
  const [mapPath, setMapPath] = useState("");
  const [replace, setReplace] = useState(false);
  const [cartPath, setCartPath] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const run = async (key: string, path: string, body: Record<string, unknown>, done: string) => {
    setBusy(key);
    try {
      const r = await post<{ audit_id?: string; file?: string; note?: string; added?: unknown }>(path, body);
      setNote({ text: `${done}${r.file ? ` (${maskPath(r.file)})` : ""}${r.note ? `. ${r.note}` : ""}. Logged in Audit${r.audit_id ? `, ${r.audit_id}, which can undo it` : ""}.`, tone: "verified" });
      for (const p of ["/api/parameter-maps", "/api/cartridges", "/api/surgery", "/api/audit"]) refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const field = "mono h-9 min-w-0 flex-1 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";
  const go = "press mono min-h-8 shrink-0 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
  return (
    <div className="grid gap-5" data-e2e="import-panel">
      <NoteLine note={note} />
      <section className="grid gap-2">
        <span className="eyebrow">Parameter map</span>
        <select value={role} onChange={(e) => setRole(e.target.value)} className="mono h-9 w-full border-b border-line-strong bg-transparent text-[11px] text-text outline-none focus-visible:border-sense-pale" aria-label="MicroModel">
          {roles.map((r) => (
            <option key={r} value={r} className="bg-bg-1">
              {roleTitle(r)}
            </option>
          ))}
        </select>
        <div className="flex min-w-0 items-center gap-2">
          <input value={mapPath} onChange={(e) => setMapPath(e.target.value)} placeholder="path to a .json map" className={field} aria-label="Path of the map" />
          <button type="button" disabled={busy !== null || !mapPath.trim()} onClick={() => void run("map", "/api/parameter-maps/import", { role, path: mapPath.trim(), replace }, `Map imported into the ${roleTitle(role)}'s maps`)} className={go}>
            {busy === "map" ? "importing" : "import map"}
          </button>
        </div>
        <label className="mono flex items-center gap-2 text-[10.5px] text-faint">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          replace a map of the same name (the old one is kept aside)
        </label>
        <span className="mono text-[10px] text-quiet">checked against its kind before it joins; never replaces a map unless you say so</span>
      </section>
      <section className="grid gap-2">
        <span className="eyebrow">Cartridge</span>
        <div className="flex min-w-0 items-center gap-2">
          <input value={cartPath} onChange={(e) => setCartPath(e.target.value)} placeholder="path to a .cartridge file" className={field} aria-label="Path of the cartridge" />
          <button type="button" disabled={busy !== null || !cartPath.trim()} onClick={() => void run("cart", "/api/cartridges/import", { path: cartPath.trim() }, "Cartridge added to the cartridge library")} className={go}>
            {busy === "cart" ? "loading" : "load cartridge"}
          </button>
        </div>
        <span className="mono text-[10px] text-quiet">checked against its own hashes; it joins the cartridge library on Cartridges, never written into a base cartridge</span>
      </section>
      <p className="mono border-t border-line/60 pt-3 text-[10px] leading-relaxed text-quiet" title="Provenance and licence, SHA-256 manifest, held-out evidence, build steps">
        From this machine only; nothing is fetched or sent. To share one with everyone, open a pull request on the public cartridge library on Hugging Face with the checklist in CONTRIBUTING.md.
      </p>
    </div>
  );
}
