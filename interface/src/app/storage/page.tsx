"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing."):
// Storage is now a working view of the disk. The middle: the whole drive as one bar (the Nano, the
// rest, free, the floor), the slab by store, one row per store with its size split by kind and its
// clean action, then the picked store's items with copy and delete. The rail: the storage settings
// (disk floor, folders, projects to forget), the kinds, and the detail of whatever is clicked.
// 2026-11-27 (his words: "The main center screen for every page should be easy to navigate and
// beautifully visual and interactive"): the middle has two tabs, kept in this browser: Disk (the
// drive, the slab by store, the stores and the picked store's items) and Backups (the user's own
// repositories), which no longer sits under the item table. The slab is drawn at the frame's own
// width, each store on it picks that store, and no label leaves the frame.

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import MainTabs from "@/components/MainTabs";
import BackupRepos, { BackupNotes } from "@/components/BackupRepos";
import Workspace, { RailSection, useRail } from "@/components/Workspace";
import { Empty, Failed, Masked, Pill, Placeholder, Segmented } from "@/components/ui";
import { ago, bytes, post, refresh, useApi, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { maskName, maskPath } from "@/lib/privacy";
import {
  canDelete,
  CATEGORIES,
  categoryOf,
  deleteBody,
  KIND_WORDS,
  parseStorage,
  parseStorageSettingsView,
  pruneBody,
  prunable,
  shares,
  STORAGE_API,
  STORAGE_DELETE,
  STORAGE_PRUNE,
  STORAGE_SETTINGS,
  settingsBody,
  type StorageCategory,
  type StorageFile,
  type StorageKind,
  type StorageReply,
} from "@/lib/storage";

type Note = { text: string; tone: "verified" | "error" };

const KIND_TONE: Record<StorageKind, string> = { "read-only": "text-faint", removable: "text-attention", junk: "text-attention", rebuildable: "text-sense-pale" };
const KIND_TIP: Record<StorageKind, string> = {
  rebuildable: "Caches and copies the engine makes again on demand. Pruned per store.",
  removable: "Learned filters, lenses, notes and adapters. Deleted one by one, after you confirm.",
  junk: "What tests, experiments and stopped runs left behind. Deleted with the junk store's clean action.",
  "read-only": "Base cartridges and the optional upstream open weights. Never deleted here.",
};
const pct = (n: number) => `${Math.round(n * 1000) / 10}%`;

/** The disk breakdown as one isometric slab on the night ground: one segment per store, its
 *  width by its share of what the Nano keeps on disk, a monospace leader up to its name and size.
 *  Drawn at the frame's own width, so the type keeps its size; a segment or its label picks the
 *  store. A label that fits to the right of its leader stands above the slab; one that does not
 *  (a store at the right end) stands under it, to the left of its leader, so no label leaves the
 *  frame and no leader runs across another label. */
function DiskSlab({ parts, total, picked, onPick }: { parts: ReturnType<typeof shares>; total: number; picked: string | null; onPick: (id: string) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const [measured, setMeasured] = useState(0);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setMeasured(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const W = Math.max(520, measured);
  const rowH = 17;
  const top = 20;
  const h = 26;
  const sk = 34;
  const dep = 18;
  const x0 = 24;
  const w = W - x0 * 2 - sk;
  const raw = parts.map((p) => (p.bytes > 0 ? Math.max(3, p.share * w) : 0));
  const scale = raw.reduce((a, b) => a + b, 0) > w ? w / raw.reduce((a, b) => a + b, 0) : 1;
  const widths = raw.map((v) => v * scale);
  const placed = parts.map((p, i) => {
    const x = x0 + widths.slice(0, i).reduce((a, b) => a + b, 0);
    const text = `${p.label} · ${bytes(p.bytes)} · ${pct(p.share)}`;
    const lx = x + widths[i] / 2 + sk / 2;
    // Above when the label fits to the right of its leader (6.4 px per character of the mono face).
    return { ...p, x, w: widths[i], text, lx, above: lx + 14 + text.length * 6.4 <= W - 8 };
  });
  const nAbove = Math.max(1, placed.filter((q) => q.above).length);
  const nBelow = placed.filter((q) => !q.above).length;
  const y = top + nAbove * rowH + 30;
  const axisY = y + h + 22;
  const H = axisY + 8 + (nBelow ? nBelow * rowH + 8 : 0);
  let ia = 0;
  let ib = 0;
  const segs = placed.map((q) => ({ ...q, row: q.above ? ia++ : ib++ }));
  return (
    <div ref={box} className="min-w-0 overflow-x-auto">
      {measured > 0 && (
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="group" aria-label={`Disk use by store: ${parts.map((p) => `${p.label} ${bytes(p.bytes)}`).join(", ")}; ${bytes(total)} in all`} data-e2e="storage-breakdown">
          <path d={`M${x0} ${y + h}h${w}l${sk} ${-dep}h${-w}z`} fill="none" stroke="#9fb0dd" strokeOpacity={0.18} strokeDasharray="1 3" />
          <path d={`M${x0} ${y}h${w}l${sk} ${-dep}h${-w}z`} fill="none" stroke="#9fb0dd" strokeOpacity={0.3} strokeWidth={0.6} />
          <rect x={x0} y={y} width={w} height={h} fill="none" stroke="#9fb0dd" strokeOpacity={0.3} strokeWidth={0.6} />
          {Array.from({ length: 21 }, (_, i) => (
            <line key={i} x1={x0 + (i * w) / 20} x2={x0 + (i * w) / 20} y1={y + h} y2={y + h + (i % 5 === 0 ? 6 : 3)} stroke="var(--line-3)" />
          ))}
          {segs.map((s) => {
            const on = s.id === picked;
            const left = !s.above;
            // Under the slab the leader starts below the scale, at the segment's own middle.
            const lx = left ? Math.min(s.x + s.w / 2, x0 + w - 2) : s.lx;
            const ly = left ? axisY + 14 + s.row * rowH : top + s.row * rowH;
            const say = `${s.label}: ${bytes(s.bytes)}, ${pct(s.share)} of what the Nano keeps. Show its items.`;
            return (
              <g key={s.id} className="fig-node" role="button" tabIndex={0} aria-label={say} aria-pressed={on} data-store={s.id} onClick={() => onPick(s.id)} onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onPick(s.id);
                  }
                }}>
                <title>{say}</title>
                <path d={`M${s.x} ${y}h${s.w}l${sk} ${-dep}h${-s.w}z`} fill={s.hue} opacity={on ? 0.6 : 0.36} />
                <rect x={s.x} y={y} width={s.w} height={h} fill={s.hue} opacity={on ? 0.95 : 0.7} />
                {on && s.w > 0 && <rect x={s.x} y={y + h + 8} width={Math.max(3, s.w)} height={2} fill="var(--sense-pale)" />}
                <line x1={lx} x2={lx} y1={left ? axisY + 4 : y - dep / 2} y2={ly} stroke={on ? "var(--sense-pale)" : "var(--line-3)"} strokeDasharray="1 2" />
                <line x1={lx} x2={lx + (left ? -10 : 10)} y1={ly} y2={ly} stroke={on ? "var(--sense-pale)" : "var(--line-3)"} />
                {/* The hit area of the label row, and its hover state. */}
                <rect className="fig-box" x={left ? lx - 14 - 330 : lx + 12} y={ly - 9} width={330} height={rowH - 1} fill="transparent" stroke="transparent" />
                <text x={lx + (left ? -14 : 14)} y={ly + 3.5} textAnchor={left ? "end" : "start"} className="mono" fontSize={10.5} fill={on ? "var(--text-1)" : "var(--text-2)"}>
                  {s.text}
                </text>
              </g>
            );
          })}
          <text x={x0} y={axisY} className="mono" fontSize={10} fill="var(--text-4)">
            0
          </text>
          <text x={x0 + w} y={axisY} textAnchor="end" className="mono" fontSize={10} fill="var(--text-4)">
            {bytes(total)}
          </text>
        </svg>
      )}
    </div>
  );
}

/** The whole drive as one bar: what the Nano keeps, what the rest of the drive holds, what is free,
 *  and the floor under which the engine stops writing. Only figures the engine reports. */
function DriveBar({ reply }: { reply: StorageReply }) {
  const total = reply.disk_total_bytes ?? null;
  const free = reply.free_bytes;
  if (total === null || total <= 0 || free === null) {
    return free !== null ? (
      <div className={`mono text-[10.5px] ${reply.below_floor ? "text-attention" : "text-faint"}`} data-e2e="storage-disk">
        {bytes(free)} free{reply.floor_bytes ? ` · floor ${bytes(reply.floor_bytes)}` : ""}
      </div>
    ) : null;
  }
  const nano = Math.min(reply.total_bytes, total);
  const other = Math.max(0, total - free - nano);
  const at = (n: number) => `${Math.max(0, Math.min(100, (n / total) * 100))}%`;
  const floor = reply.floor_bytes ?? null;
  return (
    <div className="grid gap-1.5" data-e2e="storage-disk">
      <div className="mono flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 text-[10.5px] uppercase tracking-[0.1em] text-faint">
        <span>drive · {bytes(total)}</span>
        <span className={reply.below_floor ? "text-attention" : undefined}>
          {bytes(free)} free{floor !== null ? ` · floor ${bytes(floor)}` : ""}
          {reply.below_floor ? " · below the floor" : ""}
        </span>
      </div>
      <div className="relative flex h-2.5 w-full overflow-hidden bg-line/40" role="img" aria-label={`The drive: the Nano keeps ${bytes(nano)}, other files ${bytes(other)}, ${bytes(free)} free of ${bytes(total)}`}>
        <i className="block h-full bg-sense-pale" style={{ width: at(nano), minWidth: nano > 0 ? 2 : 0 }} title={`the Nano: ${bytes(nano)}`} />
        <i className="block h-full bg-line-3" style={{ width: at(other) }} title={`other files on the drive: ${bytes(other)}`} />
        {floor !== null && <i className="absolute inset-y-0 w-px bg-attention" style={{ right: at(floor) }} title={`floor: the engine stops writing under ${bytes(floor)} free`} />}
      </div>
      <div className="mono flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] text-faint">
        <span className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-1.5 w-1.5 bg-sense-pale" />
          the Nano {bytes(nano)}
        </span>
        <span className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-1.5 w-1.5 bg-line-3" />
          other {bytes(other)}
        </span>
        <span className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-1.5 w-1.5 border border-line-3" />
          free {bytes(free)}
        </span>
        {floor !== null && (
          <span className="flex items-center gap-1.5">
            <i aria-hidden className="inline-block h-2 w-px bg-attention" />
            floor
          </span>
        )}
      </div>
    </div>
  );
}

/** A store's bytes by kind, as reported or summed from its items. */
function byKind(c: StorageCategory): { kind: StorageKind; bytes: number }[] {
  const summed: Record<StorageKind, number> = { rebuildable: 0, removable: 0, junk: 0, "read-only": 0 };
  for (const f of c.files) summed[f.kind] += f.bytes;
  const rb = c.rebuildable_bytes ?? null;
  const rm = c.removable_bytes ?? null;
  const ro = c.read_only_bytes ?? null;
  if (rb !== null || rm !== null || ro !== null) {
    const isJunk = c.id === "junk";
    return [
      { kind: isJunk ? ("junk" as const) : ("rebuildable" as const), bytes: rb ?? 0 },
      { kind: "removable" as const, bytes: rm ?? 0 },
      { kind: "read-only" as const, bytes: ro ?? 0 },
    ];
  }
  return (["rebuildable", "junk", "removable", "read-only"] as const).map((k) => ({ kind: k, bytes: summed[k] }));
}

/** What the rail shows for a store: what it holds, its note, and its bytes by kind. */
function StoreDetail({ c }: { c: StorageCategory }) {
  const def = CATEGORIES.find((k) => k.id === categoryOf(c.id, c.label));
  const n = c.file_count ?? c.files.length;
  return (
    <div className="grid gap-0">
      <RailSection title="Holds">
        <p className="read text-xs leading-relaxed text-muted">{def?.about ?? "Files the engine reports in this store."}</p>
        {c.note && <p className="read mt-2 text-[11px] leading-relaxed text-faint">{c.note}</p>}
      </RailSection>
      <RailSection title="By kind">
        <ul className="mono grid gap-1 text-[11px]">
          {byKind(c).map((k) => (
            <li key={k.kind} className="flex items-baseline justify-between gap-3" title={KIND_TIP[k.kind]}>
              <span className={KIND_TONE[k.kind]}>{KIND_WORDS[k.kind]}</span>
              <span className="text-text">{bytes(k.bytes)}</span>
            </li>
          ))}
          <li className="flex items-baseline justify-between gap-3 border-t border-line/60 pt-1">
            <span className="text-faint">
              {n.toLocaleString()} {n === 1 ? "file" : "files"}
              {c.files.length !== n ? ` in ${c.files.length} items` : ""}
            </span>
            <span className="text-text">{bytes(c.total_bytes)}</span>
          </li>
        </ul>
      </RailSection>
    </div>
  );
}

function copyText(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText)
    return navigator.clipboard.writeText(text).then(
      () => true,
      () => false,
    );
  return Promise.resolve(false);
}

/** What the rail shows for one item: its path, size, age and kind, and why it can or cannot go. */
function FileDetail({ f, root, now, onNote }: { f: StorageFile; root: string | null; now: number; onNote: (n: Note) => void }) {
  return (
    <div className="grid gap-0">
      <RailSection title="Path">
        <div className="mono break-all text-[11px] text-muted">
          <Masked id={`storage:${f.path}`} shown={maskPath(f.path, root)} full={f.path} />
        </div>
        <button
          type="button"
          onClick={async () => onNote((await copyText(f.path)) ? { text: `Copied the path of ${maskPath(f.path, root)}.`, tone: "verified" } : { text: "This browser did not allow copying; select the path after show instead.", tone: "error" })}
          className="press mono mt-2 min-h-8 border-b border-line-strong text-[10.5px] uppercase tracking-[0.1em] text-muted hover:text-text"
        >
          Copy path
        </button>
      </RailSection>
      <RailSection title="Item">
        <dl className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[11px]">
          <dt className="text-faint">size</dt>
          <dd className="text-right text-text">{bytes(f.bytes)}</dd>
          <dt className="text-faint">kind</dt>
          <dd className={`text-right ${KIND_TONE[f.kind]}`}>{KIND_WORDS[f.kind]}</dd>
          {f.label && (
            <>
              <dt className="text-faint">what</dt>
              <dd className="truncate text-right text-muted">{f.label}</dd>
            </>
          )}
          {f.entry === "folder" && (
            <>
              <dt className="text-faint">folder of</dt>
              <dd className="text-right text-muted">{typeof f.count === "number" ? `${f.count.toLocaleString()} ${f.count === 1 ? "file" : "files"}` : "files"}</dd>
            </>
          )}
          {f.project && (
            <>
              <dt className="text-faint">project</dt>
              <dd className="truncate text-right text-muted">{maskName(f.project)}</dd>
            </>
          )}
          <dt className="text-faint">changed</dt>
          <dd className="text-right text-muted">{f.modified_ms && now > 0 ? `${ago(f.modified_ms, now)} · ${new Date(f.modified_ms).toLocaleString()}` : "not reported"}</dd>
        </dl>
      </RailSection>
      <RailSection title="Can it go">
        <p className="read text-[11px] leading-relaxed text-faint">{canDelete(f) ? "Yes: Delete in its row shows what goes, then asks you to type its name." : KIND_TIP[f.kind]}</p>
      </RailSection>
    </div>
  );
}

/** One store as a row: its swatch and name, its size and share, a bar split by kind, and its clean
 *  action (prune what rebuilds, or delete all junk) through the delete flow. */
function StoreRow({ c, total, picked, onPick, root, onNote }: { c: StorageCategory; total: number; picked: boolean; onPick: () => void; root: string | null; onNote: (n: Note) => void }) {
  const def = CATEGORIES.find((k) => k.id === categoryOf(c.id, c.label));
  const hue = def?.hue ?? "var(--text-3)";
  const free = prunable(c);
  const isJunk = c.id === "junk";
  const n = c.file_count ?? c.files.length;
  const share = total > 0 ? c.total_bytes / total : 0;
  const parts = byKind(c).filter((k) => k.bytes > 0);
  return (
    <li className={`grid min-w-0 gap-1.5 border-b border-l-2 border-b-line/60 py-2.5 pr-1 pl-2.5 ${picked ? "border-l-sense-pale bg-bg-2/40" : "border-l-transparent hover:bg-bg-1/50"}`} data-store={c.id} data-picked={String(picked)}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
        <button type="button" onClick={onPick} aria-pressed={picked} title={def?.about ?? c.label} className="press flex min-w-0 flex-1 items-center gap-2.5 text-left">
          <i aria-hidden className="inline-block h-2 w-2 shrink-0" style={{ background: hue }} />
          <span className={`mono truncate text-[11.5px] uppercase tracking-[0.08em] ${picked ? "text-text" : "text-muted hover:text-text"}`}>{c.label}</span>
          <span className="mono shrink-0 text-[12px] text-text">{bytes(c.total_bytes)}</span>
          <span className="mono hidden shrink-0 text-[10.5px] text-faint sm:inline">
            {pct(share)} · {n.toLocaleString()} {n === 1 ? "file" : "files"}
          </span>
        </button>
        <span className={`mono shrink-0 text-[10.5px] ${free > 0 ? (isJunk ? "text-attention" : "text-sense-pale") : "text-quiet"}`} title={isJunk ? "Junk the clean action deletes" : "Rebuildable bytes a prune frees"}>
          {free > 0 ? `${bytes(free)} ${isJunk ? "junk" : "cleanable"}` : isJunk ? "no junk" : "nothing to prune"}
        </span>
        <div className="ml-auto min-w-0 max-w-full">
        <DeleteFlow<{ bytes?: number }>
          path={STORAGE_PRUNE}
          body={pruneBody(c.id)}
          label={isJunk ? "Delete junk" : "Prune"}
          disabled={free <= 0}
          title={isJunk ? "Delete all junk: the engine shows what goes, then asks for its word" : "Prune what rebuilds: the engine shows what goes, then asks for its word"}
          startClass="press mono min-h-8 border-b border-line-strong px-1.5 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text disabled:opacity-30"
          e2e="storage-prune"
          done={(r) => `${isJunk ? "Deleted the junk" : `Pruned ${c.label}`}: ${bytes(r?.bytes ?? free)}. Logged in Audit.`}
          refresh={[STORAGE_API]}
          onNote={onNote}
          root={root}
        />
        </div>
      </div>
      <div aria-hidden className="relative block h-2 w-full overflow-hidden bg-line/40">
        <span className="absolute inset-y-0 left-0 flex" style={{ width: `${Math.max(c.total_bytes > 0 ? 0.4 : 0, share * 100)}%`, minWidth: c.total_bytes > 0 ? 2 : 0 }}>
          {parts.map((k) => (
            <i
              key={k.kind}
              className="block h-full"
              title={`${KIND_WORDS[k.kind]}: ${bytes(k.bytes)}`}
              style={{
                width: `${(k.bytes / Math.max(1, c.total_bytes)) * 100}%`,
                background: k.kind === "read-only" ? hue : k.kind === "rebuildable" ? "var(--sense-pale)" : "var(--attention)",
                opacity: k.kind === "read-only" ? 0.4 : 0.9,
              }}
            />
          ))}
        </span>
      </div>
    </li>
  );
}

const KIND_FILTERS = ["all", "rebuildable", "removable", "junk", "read-only"] as const;
type KindFilter = (typeof KIND_FILTERS)[number];

/** The picked store's items, largest first: path, size, age and kind, with copy path and, for
 *  removable items, delete through the delete flow. A row's path opens it in the rail. */
function StoreFiles({ c, root, now, onNote }: { c: StorageCategory; root: string | null; now: number; onNote: (n: Note) => void }) {
  const rail = useRail();
  const [all, setAll] = useState(false);
  const [kind, setKind] = useState<KindFilter>("all");
  const [deleting, setDeleting] = useState<StorageFile | null>(null);
  const present = KIND_FILTERS.filter((k) => k === "all" || c.files.some((f) => f.kind === k));
  const k = present.includes(kind) ? kind : "all";
  const list = k === "all" ? c.files : c.files.filter((f) => f.kind === k);
  const files = all ? list : list.slice(0, 40);
  return (
    <section aria-label={`Items in ${c.label}`} className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-2">
        <span className="eyebrow text-muted">
          {c.label} · {list.length.toLocaleString()} {list.length === 1 ? "item" : "items"}
        </span>
        {present.length > 2 && (
          <div className="max-w-full overflow-x-auto">
            <Segmented values={[...present]} value={k} label="Kind" onPick={(v) => setKind(v as KindFilter)} render={(v) => (v === "all" ? "all" : KIND_WORDS[v as StorageKind])} />
          </div>
        )}
      </div>
      {deleting && (
        <DeleteFlow<{ bytes?: number }>
          key={deleting.path}
          path={STORAGE_DELETE}
          body={deleteBody(deleting.path)}
          label="Delete"
          autoStart
          done={(r) => `Deleted ${maskPath(deleting.path, root)} (${bytes(r?.bytes ?? deleting.bytes)}). Logged in Audit.`}
          refresh={[STORAGE_API]}
          onNote={onNote}
          onClose={() => setDeleting(null)}
          root={root}
        />
      )}
      {c.files.length === 0 ? (
        <Empty title="Nothing kept in this store" />
      ) : (
        <div className="overflow-x-auto">
          <table className="mono w-full min-w-[560px] text-[11px]" data-e2e="storage-files">
            <thead>
              <tr className="text-left text-faint">
                <th className="py-1 pr-3 font-normal">path</th>
                <th className="py-1 pr-3 text-right font-normal">size</th>
                <th className="py-1 pr-3 font-normal">changed</th>
                <th className="py-1 pr-3 font-normal">kind</th>
                <th className="py-1 font-normal" />
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.path} className="border-t border-line/60 text-muted hover:bg-bg-1/50">
                  <td className="max-w-[360px] py-1 pr-3">
                    <button type="button" onClick={() => rail.show({ title: f.label ?? maskPath(f.path, root), body: <FileDetail f={f} root={root} now={now} onNote={onNote} />, key: f.path })} className="press block w-full min-w-0 text-left hover:text-text" title={maskPath(f.path, root)}>
                      <span className="block truncate">{maskPath(f.path, root)}</span>
                      {(f.label || f.entry === "folder") && (
                        <span className="block truncate text-[10px] text-faint">
                          {f.label ?? ""}
                          {f.entry === "folder" ? `${f.label ? " · " : ""}folder${typeof f.count === "number" ? ` of ${f.count.toLocaleString()}` : ""}` : ""}
                          {f.project ? ` · ${maskName(f.project)}` : ""}
                        </span>
                      )}
                    </button>
                  </td>
                  <td className="py-1 pr-3 text-right text-text">{bytes(f.bytes)}</td>
                  <td className="py-1 pr-3 whitespace-nowrap text-faint">{f.modified_ms && now > 0 ? ago(f.modified_ms, now) : "-"}</td>
                  <td className={`py-1 pr-3 whitespace-nowrap ${KIND_TONE[f.kind]}`} title={KIND_TIP[f.kind]}>
                    {KIND_WORDS[f.kind]}
                  </td>
                  <td className="py-1 text-right whitespace-nowrap">
                    <button
                      type="button"
                      onClick={async () => onNote((await copyText(f.path)) ? { text: `Copied the path of ${maskPath(f.path, root)}.`, tone: "verified" } : { text: "This browser did not allow copying; select the path after show instead.", tone: "error" })}
                      className="press mr-3 text-[10.5px] uppercase tracking-[0.1em] text-faint hover:text-text"
                    >
                      Copy path
                    </button>
                    <button type="button" disabled={!canDelete(f) || deleting !== null} onClick={() => setDeleting(f)} title={canDelete(f) ? "Delete this item: the engine shows what goes, then asks you to type its name" : KIND_TIP[f.kind]} className="press text-[10.5px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-30">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {list.length > 40 && (
            <button type="button" onClick={() => setAll((x) => !x)} className="press mono mt-2 text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text">
              {all ? "Show the 40 largest" : `Show all ${list.length} items`}
            </button>
          )}
        </div>
      )}
    </section>
  );
}

/** The storage settings, for the rail: each key the engine lets the user change (the disk floor),
 *  saved at once and undone in Audit; the folders the Nano keeps its files in (fixed by its
 *  layout); and the projects it keeps memory for, each forgotten through the delete flow. */
function StorageSettingsRail({ onNote }: { onNote: (n: Note) => void }) {
  const [saving, setSaving] = useState<string | null>(null);
  const q = useApi<unknown>(STORAGE_SETTINGS, 30000);
  const view = useMemo(() => parseStorageSettingsView(q.data), [q.data]);
  const [draft, setDraft] = useState<Record<string, string>>({});
  if (q.error && !q.data) return <Empty title={q.error.startsWith("404") ? "Not served by this engine build" : "Storage settings did not answer"}>{q.error.startsWith("404") ? "This engine build does not serve /api/storage/settings yet." : q.error}</Empty>;
  if (!view) return <Placeholder className="h-20" what="storage settings" />;
  return (
    <div className="grid gap-0">
      <RailSection title="Change">
        {view.settable.length === 0 ? (
          <div className="mono text-[11px] text-faint">No storage setting reported.</div>
        ) : (
          <ul className="grid gap-3" data-e2e="storage-settings">
            {view.settable.map((st) => {
              const k = st.key;
              const v = st.value ?? st.default ?? "";
              const cur = draft[k] ?? String(v);
              const n = Number(cur);
              const valid = /^\d+$/.test(cur.trim()) && (st.min === null || n >= st.min) && (st.max === null || n <= st.max);
              const changed = cur !== String(v);
              return (
                <li key={k} className="grid gap-1">
                  <span className="mono truncate text-[10.5px] uppercase tracking-[0.1em] text-muted" title={`${k}${st.min !== null && st.max !== null ? ` · ${st.min} to ${st.max.toLocaleString()}` : ""}${st.default !== null ? ` · default ${st.default}` : ""}`}>
                    {k.replace(/^storage\./, "").replace(/[._]/g, " ")}
                  </span>
                  <span className="flex min-w-0 items-center gap-2">
                    <input value={cur} inputMode="numeric" onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))} aria-label={k} aria-invalid={changed && !valid} className="mono h-9 w-24 min-w-0 border-b border-line-strong bg-transparent px-1 text-xs text-text outline-none focus-visible:border-sense-pale" />
                    <span className="mono text-[10.5px] text-faint">
                      {st.unit ?? ""}
                      {st.default !== null ? ` · default ${st.default}` : ""}
                    </span>
                    <button
                      type="button"
                      disabled={!changed || !valid || saving === k}
                      onClick={async () => {
                        // 2026-11-26: a setting saves at once (no dialog); Audit undoes it.
                        setSaving(k);
                        try {
                          await post(STORAGE_SETTINGS, settingsBody(k, cur.trim()));
                          onNote({ text: `${k} set from ${String(v)} to ${cur.trim()}${st.unit ? ` ${st.unit}` : ""}. Logged in Audit, where it can be undone.`, tone: "verified" });
                          setDraft((d) => {
                            const next = { ...d };
                            delete next[k];
                            return next;
                          });
                        } catch (err) {
                          onNote({ text: err instanceof Error ? err.message : String(err), tone: "error" });
                        }
                        setSaving(null);
                        await refresh(STORAGE_SETTINGS);
                        await refresh(STORAGE_API);
                      }}
                      className="press mono ml-auto min-h-9 border-b border-line-strong px-2 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text disabled:opacity-40"
                    >
                      Save
                    </button>
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </RailSection>
      {view.paths.length > 0 && (
        <RailSection title="Folders">
          <dl className="mono grid gap-1.5 text-[10.5px]" data-e2e="storage-paths">
            {view.paths.map((p) => (
              <div key={p.id} className="grid min-w-0 gap-0.5">
                <dt className="text-faint">{p.label}</dt>
                <dd className="min-w-0 break-all text-muted">
                  <Masked id={`storage-setting:${p.id}`} shown={maskPath(p.path, null)} full={p.path} />
                </dd>
              </div>
            ))}
          </dl>
          {view.memory_dir_note && <p className="read mt-2 text-[11px] leading-relaxed text-quiet">{view.memory_dir_note}</p>}
        </RailSection>
      )}
      {view.memory_dirs.length > 0 && (
        // 2026-11-26 (Luis Jake Gabriel III): forget a project in one step, confirmed in more than
        // one: the engine shows what goes, then asks for the project's key.
        <RailSection title={`Projects with memory · ${view.memory_dirs.length}`}>
          <ul className="grid gap-1" data-e2e="storage-projects">
            {view.memory_dirs.map((m) => (
              <li key={m.path} className="mono grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/60 py-1 text-[10.5px]">
                <span className="min-w-0">
                  <span className="block truncate text-muted">{m.project === "nano" ? "the Nano itself" : maskName(m.project)}</span>
                  <span className="block truncate text-faint">
                    <Masked id={`storage-project:${m.path}`} shown={maskPath(m.path, null)} full={m.path} />
                  </span>
                </span>
                {m.project && m.project !== "nano" ? (
                  <DeleteFlow<{ bytes?: number; files?: number }>
                    path="/api/projects/forget"
                    body={{ project: m.project }}
                    label="Forget"
                    title="Removes everything the Nano keeps for this project; the project's own files are not touched. Shows what goes first, then asks you to type the project's key."
                    startClass="press text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error"
                    done={(r) => `Forgot ${maskName(m.project)}: ${bytes(r?.bytes ?? 0)} in ${r?.files ?? 0} files. Logged in Audit.`}
                    refresh={[STORAGE_SETTINGS, STORAGE_API, "/api/projects"]}
                    onNote={onNote}
                  />
                ) : (
                  <span className="text-quiet" title="The Nano's own memory holds the audit log and every setting's history; delete its items one by one">
                    kept
                  </span>
                )}
              </li>
            ))}
          </ul>
        </RailSection>
      )}
    </div>
  );
}

function KindsRail() {
  return (
    <ul className="grid gap-3 text-xs">
      {(["rebuildable", "removable", "junk", "read-only"] as const).map((k) => (
        <li key={k} className="grid gap-1">
          <Pill tone={k === "rebuildable" ? "signal" : k === "read-only" ? "muted" : "attention"}>{KIND_WORDS[k]}</Pill>
          <span className="read text-faint">{KIND_TIP[k]}</span>
        </li>
      ))}
    </ul>
  );
}

function StorageAbout({ root }: { root: string | null }) {
  return (
    <div className="grid gap-0">
      <RailSection title="What this is">
        <p className="read text-xs leading-relaxed text-muted">What the Nano keeps on this disk, by store, every item with its exact path and size. Prune what rebuilds, delete what is removable, clean the junk. Base cartridges and the upstream open weights stay read only. Every destructive step shows what goes and asks for a typed word first, and is logged in Audit.</p>
      </RailSection>
      <RailSection title="Stores">
        <dl className="grid gap-2 text-[11px]">
          {CATEGORIES.map((c) => (
            <div key={c.id} className="grid gap-0.5">
              <dt className="mono flex items-center gap-2 text-muted">
                <i aria-hidden className="inline-block h-1.5 w-1.5" style={{ background: c.hue }} />
                {c.label}
              </dt>
              <dd className="read text-faint">{c.about}</dd>
            </div>
          ))}
        </dl>
      </RailSection>
      <RailSection title="Root">
        <div className="mono break-all text-[11px] text-muted">{root ? <Masked id="storage-root" shown={maskPath(root, null)} full={root} /> : "not reported"}</div>
      </RailSection>
    </div>
  );
}

/** The working middle: the drive, the slab, the stores and the picked store's items. */
function StorageWork({ reply, root, onNote }: { reply: StorageReply; root: string | null; onNote: (n: Note) => void }) {
  const rail = useRail();
  const now = useNow();
  const parts = useMemo(() => shares(reply), [reply]);
  const largest = [...reply.categories].sort((a, b) => b.total_bytes - a.total_bytes)[0]?.id ?? null;
  const [pickedId, setPickedId] = useState<string | null>(null);
  const picked = reply.categories.find((c) => c.id === (pickedId ?? largest)) ?? null;
  const pick = (c: StorageCategory) => {
    setPickedId(c.id);
    rail.show({ title: c.label, body: <StoreDetail c={c} />, key: `store:${c.id}` });
  };
  if (!reply.categories.length) return <Empty title="Nothing on disk yet" slot />;
  return (
    <div className="grid min-w-0 gap-8">
      <DriveBar reply={reply} />
      <section aria-label="Disk use by store" className="min-w-0">
        <div className="topo night ticks border border-line px-2 py-3">
          <DiskSlab parts={parts} total={reply.total_bytes} picked={picked?.id ?? null} onPick={(id) => {
              const c = reply.categories.find((k) => k.id === id);
              if (c) pick(c);
            }} />
        </div>
        <div className="mono mt-1.5 text-[10px] uppercase tracking-[0.16em] text-faint">Fig. 1 · what the Nano keeps, by store · a store picks its items</div>
      </section>
      <ul aria-label="Stores" className="grid border-t border-line">
        {reply.categories.map((c) => (
          <StoreRow key={c.id} c={c} total={reply.total_bytes} picked={picked?.id === c.id} onPick={() => pick(c)} root={root} onNote={onNote} />
        ))}
      </ul>
      {picked && <StoreFiles key={picked.id} c={picked} root={root} now={now} onNote={onNote} />}
    </div>
  );
}

/** Storage: everything the Nano keeps on disk, by store, and the safe controls to clean it. */
export default function Storage() {
  const q = useApi<unknown>(STORAGE_API, 15000);
  const status = useApi<Status>("/api/status", 15000);
  const reply = useMemo(() => parseStorage(q.data), [q.data]);
  const [note, setNote] = useState<Note | null>(null);
  const root = reply?.root ?? status.data?.root ?? null;
  const rebuild = (reply?.categories ?? []).filter((c) => c.id !== "junk").reduce((a, c) => a + prunable(c), 0);
  const junk = prunable(reply?.categories.find((c) => c.id === "junk"));
  return (
    <Workspace
      title="Storage"
      status={
        reply ? (
          <>
            <span className="text-text">{bytes(reply.total_bytes)} kept</span>
            <span className={rebuild > 0 ? "text-sense-pale" : undefined} title="Rebuildable bytes a prune frees">
              {bytes(rebuild)} cleanable
            </span>
            <span className={junk > 0 ? "text-attention" : undefined}>{bytes(junk)} junk</span>
            {reply.free_bytes !== null && <span className={reply.below_floor ? "text-attention" : undefined}>{bytes(reply.free_bytes)} free</span>}
          </>
        ) : undefined
      }
      rail={[
        { id: "settings", label: "Settings", body: <StorageSettingsRail onNote={setNote} /> },
        { id: "kinds", label: "Kinds", body: <KindsRail /> },
        { id: "backup-notes", label: "Backups", body: <BackupNotes /> },
      ]}
      about={<StorageAbout root={root} />}
    >
      <NoteLine note={note} />
      <MainTabs
        storageKey="nm-main:storage"
        label="Storage views"
        tabs={[
          {
            id: "disk",
            label: "Disk",
            aliases: ["stores", "files"],
            count: reply ? bytes(reply.total_bytes) : undefined,
            title: "The drive, what the Nano keeps by store, prune and delete",
            body:
              q.error && !reply ? (
                q.error.startsWith("404") ? (
                  <Empty title="Not served by this engine build" slot>
                    This engine build does not serve /api/storage yet.
                  </Empty>
                ) : (
                  <Failed error={q.error} />
                )
              ) : !reply ? (
                <Placeholder className="h-64" what="storage" />
              ) : (
                <StorageWork reply={reply} root={root} onNote={setNote} />
              ),
          },
          {
            id: "backups",
            label: "Backups",
            aliases: ["backup", "repositories"],
            title: "Your own repositories: add, back up, snapshots, preview and restore",
            body: <BackupRepos />,
          },
        ]}
      />
    </Workspace>
  );
}
