"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... it
// should be centered on the interactive cartridge system"): the inspector opens in the Cartridges
// side panel beside the deck it came from; the stack editor sits in the Stacks view of the system,
// its explanation in the heading's tooltip.
// 2026-11-26 (audit A3): inside a cartridge and around it. Inspect any cartridge (its record, the
// stacks holding it, and its values when it holds settings or preferences) and edit those values;
// put a stack on trial or serve it; make a stack of your own, or a new version of one, in the
// stacking order the library keeps (base, hardware, adapter, learned, preference). A cartridge
// file is never written over: edited values become a new version, and Audit switches back. The
// cartridge's parameters (tensors, norms, strength, layers) open on the Parameters page by the
// "parameters" link.

import Link from "next/link";
import { useMemo, useState } from "react";
import { NoteLine } from "@/components/Controls";
import { Empty, Panel, Pill, Placeholder, Switch } from "@/components/ui";
import { post, refreshAll, useApi, type Library, type LibStack } from "@/lib/api";

type Note = { text: string; tone: "verified" | "error" };
type ValueRow = { key: string; value: string; code_value: string | null; kind: string };
type Inspect = {
  key?: string;
  source?: string;
  cartridge?: Record<string, unknown>;
  card?: Record<string, unknown>;
  files_present?: boolean;
  stacks?: { key: string; use: string; trial: boolean; active_for: string[]; cartridges: string[] }[];
  values?: ValueRow[] | null;
  points?: { target: string; gain: number; bias: number }[] | null;
  values_editable?: boolean;
  note?: string;
  error?: string;
};

const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const go = "press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const refreshCarts = () => {
  for (const p of ["/api/cartridges", "/api/audit", "/api/live"]) refreshAll(p);
};

/** "view parameters": opens an object on the Parameters page, focused on it (its tensors, its
 *  mapping, detailed editing). An object that holds no parameters says so in the link's place. */
export function ViewParams({ paramKey }: { paramKey: string }) {
  const q = useApi<{ holds?: boolean; error?: string }>(`/api/cartridges/parameters?key=${encodeURIComponent(paramKey)}`);
  if (!q.data) return null;
  if (q.data.error || !q.data.holds) {
    return (
      <span className="mono text-[10px] text-faint" data-e2e="view-params-none" title={q.data.error ?? undefined}>
        {q.data.error ? "parameters not read" : "holds no parameters"}
      </span>
    );
  }
  return (
    <Link href={`/parameters/?cartridge=${encodeURIComponent(paramKey)}`} className={link} data-e2e="view-params" title="Open its parameters on the Parameters page: every tensor on the layer strip, the maps beside it, detailed editing">
      view parameters
    </Link>
  );
}

/** One cartridge, opened: what it is, where it is stacked, and its values (editable for settings
 *  and preference cartridges). */
export function CartridgeInspector({ cartKey, onClose }: { cartKey: string; onClose?: () => void }) {
  const q = useApi<Inspect>(`/api/cartridges/inspect?key=${encodeURIComponent(cartKey)}`);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [newKey, setNewKey] = useState("");
  const [newVal, setNewVal] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const d = q.data;
  const save = async (values: Record<string, string | null>) => {
    setBusy(true);
    try {
      const r = await post<{ audit_id?: string; key?: string }>("/api/cartridges/values", { key: cartKey, values });
      setNote({ text: `Saved as ${r.key ?? "a new version"}; every use that ran the old one now runs it. Audit ${r.audit_id ?? ""} switches back.`, tone: "verified" });
      setDraft({});
      setNewKey("");
      setNewVal("");
      refreshCarts();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const changed = Object.entries(draft).filter(([k, v]) => d?.values?.find((x) => x.key === k)?.value !== v);
  const rec = d?.cartridge ?? d?.card ?? null;
  return (
    <div className="grid gap-3 border-l-2 border-learned py-2 pl-3" data-e2e="cartridge-inspector">
      <div className="flex items-center justify-between gap-3">
        <span className="mono text-[11px] text-text">{cartKey}</span>
        <span className="flex items-center gap-3">
          {/* Its parameters (tensors, norms, mapping, editing) live on the Parameters page. */}
          <ViewParams paramKey={cartKey} />
          {onClose && (
            <button type="button" className={link} onClick={onClose}>
              close
            </button>
          )}
        </span>
      </div>
      <NoteLine note={note} />
      {!d && q.error ? (
        <Empty title="The cartridge did not answer">{q.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-16" what="cartridge parameters" />
      ) : d.error ? (
        <Empty title="Not readable">{d.error}</Empty>
      ) : (
        <>
          {d.note && <p className="read text-[11px] leading-relaxed text-faint">{d.note}.</p>}
          {rec && (
            <dl className="mono grid grid-cols-1 gap-y-0.5 text-[10.5px]">
              {Object.entries(rec)
                .filter(([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean")
                .map(([k, v]) => (
                  <div key={k} className="flex min-w-0 justify-between gap-2 border-b border-line/40 py-0.5">
                    <dt className="text-faint">{k.replace(/_/g, " ")}</dt>
                    <dd className="truncate text-muted" title={String(v)}>{String(v)}</dd>
                  </div>
                ))}
            </dl>
          )}
          {(d.stacks ?? []).length > 0 && (
            <div className="mono text-[10.5px] text-faint">
              stacked in {(d.stacks ?? []).map((st) => `${st.key} (${st.use}${st.trial ? ", on trial" : ""}${st.active_for.length ? `, running for ${st.active_for.join(", ")}` : ""})`).join(" · ")}
            </div>
          )}
          {d.points && (
            <div className="mono grid max-h-40 gap-0.5 overflow-auto text-[10.5px] text-faint">
              {d.points.map((p) => (
                <span key={p.target}>
                  {p.target} · {p.gain} gains · {p.bias} biases
                </span>
              ))}
            </div>
          )}
          {d.values && (
            <div className="grid gap-2" data-e2e="cartridge-values">
              <table className="mono w-full text-left text-[10.5px]">
                <thead className="text-faint">
                  <tr>
                    <th className="py-1 pr-3 font-normal">key</th>
                    <th className="py-1 pr-3 font-normal">value</th>
                    <th className="py-1 font-normal">without it</th>
                  </tr>
                </thead>
                <tbody>
                  {d.values.map((v) => (
                    <tr key={v.key} className="border-t border-line/40">
                      <td className="py-1 pr-3 text-muted" title={v.kind}>{v.key}</td>
                      <td className="py-1 pr-3">
                        {d.values_editable ? (
                          <input value={draft[v.key] ?? v.value} onChange={(e) => setDraft((x) => ({ ...x, [v.key]: e.target.value }))} aria-label={`Value of ${v.key}`} className="mono h-7 w-full min-w-24 border-b border-line bg-transparent px-1 text-[10.5px] text-text outline-none focus-visible:border-sense-pale" />
                        ) : (
                          <span className="text-text">{v.value}</span>
                        )}
                      </td>
                      <td className="py-1 text-faint">{v.code_value ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {d.values_editable && (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="pref.key.<label>" aria-label="New preference key" className="mono h-8 w-48 border-b border-line bg-transparent px-1 text-[10.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
                    <input value={newVal} onChange={(e) => setNewVal(e.target.value)} placeholder="good/signals, e.g. 3/4" aria-label="New preference value" className="mono h-8 w-40 border-b border-line bg-transparent px-1 text-[10.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
                    <button type="button" disabled={busy || !newKey.trim() || !newVal.trim()} onClick={() => void save({ [newKey.trim()]: newVal.trim() })} className={link}>
                      add key
                    </button>
                  </div>
                  <div className="flex items-center gap-4">
                    <button type="button" disabled={busy || changed.length === 0} onClick={() => void save(Object.fromEntries(changed))} className={go}>
                      {busy ? "saving" : `save ${changed.length || ""} ${changed.length === 1 ? "change" : "changes"}`}
                    </button>
                    <span className="mono text-[10px] text-quiet" title="A preference value is good/signals (whole numbers, good at most signals); operating keys keep to the keys the engine reads.">good/signals</span>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Every stack: trial or serving, and a form to make one or a new version. */
export function StackEditor({ lib }: { lib: Library }) {
  const stacks = useMemo(() => lib.catalogue?.stacks ?? [], [lib]);
  const carts = useMemo(() => (lib.catalogue?.cartridges ?? []).map((c) => (c.version ? `${c.id}@${c.version}` : c.id)), [lib]);
  const active = lib.catalogue?.active ?? {};
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const [id, setId] = useState("");
  const [use, setUse] = useState("general");
  const [order, setOrder] = useState<string[]>([]);
  const [pick, setPick] = useState("");
  const [description, setDescription] = useState("");
  const key = (st: LibStack) => (st.version ? `${st.id}@${st.version}` : st.id);
  const trial = async (st: LibStack, on: boolean) => {
    setBusy(key(st));
    try {
      const r = await post<{ audit_id?: string | null }>("/api/cartridges/trial", { name: key(st), trial: on });
      setNote({ text: r.audit_id ? `${key(st)} ${on ? "put on trial (recorded, never applied)" : "now serves"}. Audit ${r.audit_id} undoes it.` : "Already so.", tone: "verified" });
      refreshCarts();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const saveStack = async () => {
    setBusy("save");
    try {
      const r = await post<{ audit_id?: string; stack?: { id?: string; version?: number } }>("/api/cartridges/stack", { id: id.trim(), use: use.trim(), cartridges: order, ...(description.trim() ? { description: description.trim() } : {}) });
      setNote({ text: `Stack ${r.stack?.id ?? id}${r.stack?.version ? `@${r.stack.version}` : ""} saved for ${use}. A new stack starts on trial until its evidence lets it serve. Audit ${r.audit_id ?? ""} undoes it.`, tone: "verified" });
      setOrder([]);
      refreshCarts();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const loadFrom = (st: LibStack) => {
    setId(st.id);
    setUse(st.use);
    setOrder([...st.cartridges]);
    setDescription(st.description ?? "");
  };
  const move = (i: number, d: -1 | 1) => setOrder((o) => {
    const n = [...o];
    const j = i + d;
    if (j < 0 || j >= n.length) return o;
    [n[i], n[j]] = [n[j], n[i]];
    return n;
  });
  return (
    <Panel title={<span title="A stack is the cartridges plugged in together for one use. On trial, the values it would change are recorded and never applied; serving needs its evidence. Make a stack of your own, or a new version of one, in the stacking order the cartridge library keeps: base, hardware, adapter, learned, then preference.">Stacks · trial, serve, and your own</span>} right={<span className="mono text-[11px] text-faint">{stacks.length} stacks</span>}>
      <NoteLine note={note} />
      {stacks.length === 0 ? (
        <Empty title="No stack in the cartridge library yet" />
      ) : (
        <ul className="grid" data-e2e="stacks">
          {stacks.map((st) => {
            const runs = Object.entries(active).filter(([, k]) => k === key(st)).map(([u]) => u);
            return (
              <li key={key(st)} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 border-b border-line/60 py-2.5">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="mono text-[12px] text-text">{key(st)}</span>
                    <span className="mono text-[10.5px] text-faint">for {st.use}</span>
                    {runs.length > 0 && <Pill tone="verified">running for {runs.join(", ")}</Pill>}
                    {st.trial && <Pill tone="attention">on trial</Pill>}
                  </div>
                  <div className="mono mt-0.5 truncate text-[10.5px] text-faint" title={st.cartridges.join(" > ")}>
                    {st.cartridges.join(" > ") || "no cartridge"}
                  </div>
                </div>
                <span className="flex items-center gap-3">
                  <button type="button" className={link} onClick={() => loadFrom(st)}>
                    new version
                  </button>
                  <span className="mono hidden text-[10px] uppercase tracking-[0.1em] text-faint sm:inline">trial</span>
                  <Switch on={!!st.trial} busy={busy === key(st)} label={`${st.trial ? "Serve" : "Put on trial"} ${key(st)}`} onChange={(on) => void trial(st, on)} />
                </span>
              </li>
            );
          })}
        </ul>
      )}
      <div className="mt-5 grid gap-3 border-t border-line pt-4" data-e2e="stack-form">
        <span className="eyebrow">Make a stack</span>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="grid gap-1">
            <span className="mono text-[10px] uppercase tracking-[0.1em] text-faint">stack id</span>
            <input value={id} onChange={(e) => setId(e.target.value)} placeholder="my-stack" className="mono h-8 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
          </label>
          <label className="grid gap-1">
            <span className="mono text-[10px] uppercase tracking-[0.1em] text-faint">use</span>
            <input value={use} onChange={(e) => setUse(e.target.value)} placeholder="general, project:<key>, job:<name>, preference" className="mono h-8 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
          </label>
          <label className="grid gap-1">
            <span className="mono text-[10px] uppercase tracking-[0.1em] text-faint">description</span>
            <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="what it is for" className="mono h-8 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Cartridge to add" className="mono h-8 max-w-full border-b border-line-strong bg-transparent text-[11px] text-text outline-none focus-visible:border-sense-pale">
            <option value="" className="bg-bg-1">
              add a cartridge
            </option>
            {carts.filter((c) => !order.includes(c)).map((c) => (
              <option key={c} value={c} className="bg-bg-1">
                {c}
              </option>
            ))}
          </select>
          <button type="button" disabled={!pick} onClick={() => { setOrder((o) => [...o, pick]); setPick(""); }} className={link}>
            add
          </button>
        </div>
        {order.length > 0 && (
          <ol className="grid gap-1">
            {order.map((c, i) => (
              <li key={c} className="mono flex items-center justify-between gap-3 border-b border-line/40 py-1 text-[11px] text-muted">
                <span>
                  {i + 1}. {c}
                </span>
                <span className="flex items-center gap-3">
                  <button type="button" className={link} disabled={i === 0} onClick={() => move(i, -1)}>
                    up
                  </button>
                  <button type="button" className={link} disabled={i === order.length - 1} onClick={() => move(i, 1)}>
                    down
                  </button>
                  <button type="button" className={link} onClick={() => setOrder((o) => o.filter((x) => x !== c))}>
                    take out
                  </button>
                </span>
              </li>
            ))}
          </ol>
        )}
        <div>
          <button type="button" disabled={busy !== null || !id.trim() || !use.trim() || order.length === 0} onClick={() => void saveStack()} className={go}>
            {busy === "save" ? "saving" : "save stack"}
          </button>
        </div>
      </div>
    </Panel>
  );
}
