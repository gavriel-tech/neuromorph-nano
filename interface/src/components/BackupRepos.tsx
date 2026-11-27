"use client";

// The user's own repositories for backing up and rolling back their Nano data (GET and POST
// /api/backups; MCP nm_settings backup_*; neuromorph-nano backup). A repository is any git remote
// (GitHub or another host, or a Hugging Face repository) or a local folder. Nothing is sent until
// "back up now"; a restore is previewed first, copies every file it changes aside, and Audit undoes it.
// 2026-11-27: the Backups tab of Storage. The path of a backup is drawn as four steps with the
// page's own counts (parts, snapshots, repositories, restore), then the repositories, then the
// form that adds one.

import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { Pill } from "@/components/ui";
import { RailSection } from "@/components/Workspace";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";

type Note = { text: string; tone: "verified" | "error" };
type Repo = { id: string; url: string; parts: string[]; token: boolean; added_ms: number; last?: { commit: string; at_s: number; message: string } };
type Reply = { repos: Repo[]; parts: { part: string; paths: string[] }[]; git: boolean; air_gap: boolean; max_file_bytes: number };
type Snap = { commit: string; at_s: number; message: string };
type Preview = { commit: string; new: number; changed: number; same: number; new_files: string[]; changed_files: string[] };

const PATH = "/api/backups";
const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const field = "mono h-9 min-w-0 border-b border-line-strong bg-transparent text-[12px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";
const PART_WORD: Record<string, string> = { memory: "memory", maps: "parameter maps", cartridges: "cartridges", units: "learned units", settings: "settings" };

function RepoRow({ r, airGap, onNote }: { r: Repo; airGap: boolean; onNote: (n: Note) => void }) {
  const now = useNow();
  const [busy, setBusy] = useState<string | null>(null);
  const [snaps, setSnaps] = useState<Snap[] | null>(null);
  const [pv, setPv] = useState<Preview | null>(null);
  const act = async <T,>(what: string, body: Record<string, unknown>, done: (x: T) => string | null) => {
    setBusy(what);
    try {
      const x = await post<T>(PATH, { id: r.id, ...body });
      const msg = done(x);
      if (msg) onNote({ text: msg, tone: "verified" });
      refreshAll(PATH);
      refreshAll("/api/audit");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  return (
    <li className="grid gap-2 border-b border-line/60 py-3" data-e2e="backup-repo">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="mono min-w-0 truncate text-[12px] text-text" title={r.url}>
          {r.url}
        </span>
        <span className="mono text-[10px] text-faint">{r.last ? `last ${ago(r.last.at_s * 1000, now)} · ${r.last.commit.slice(0, 7)}` : "no backup yet"}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {r.parts.map((p) => (
          <Pill key={p} tone="muted">
            {PART_WORD[p] ?? p}
          </Pill>
        ))}
        {r.token && <Pill tone="attention">token sent to this host</Pill>}
      </div>
      <div className="mono flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px]">
        <button
          type="button"
          className={link}
          data-e2e="backup-now"
          disabled={!!busy || airGap}
          title={airGap ? "The air gap is on: nothing is sent" : "Copies the chosen parts into a new snapshot and sends it to this repository"}
          onClick={() => void act<{ files: number; bytes: number; changed: boolean; left_out: unknown[] }>("backup", { action: "backup" }, (x) => `Backed up ${x.files} files (${bytes(x.bytes)})${x.changed ? "" : ", nothing changed since the last snapshot"}; ${x.left_out.length} left out (rebuilt by the Nano, or over 50 MiB).`)}
        >
          {busy === "backup" ? "backing up" : "back up now"}
        </button>
        <button type="button" className={link} data-e2e="backup-snapshots" disabled={!!busy} onClick={() => void act<{ snapshots: Snap[] }>("snaps", { action: "snapshots" }, (x) => { setSnaps(x.snapshots); setPv(null); return null; })}>
          {busy === "snaps" ? "reading" : "snapshots"}
        </button>
        <button type="button" className={link} disabled={!!busy} title="Takes the repository off this list; what it holds stays where it is" onClick={() => void act<{ audit_id: string }>("remove", { action: "remove" }, () => `${r.url} taken off the list; its snapshots stay in the repository. Undo in Audit.`)}>
          take off the list
        </button>
      </div>
      {snaps && (
        <ul className="grid gap-1 pl-3" data-e2e="backup-snapshot-list">
          {snaps.length === 0 && <li className="mono text-[10.5px] text-quiet">no snapshots yet</li>}
          {snaps.map((sn) => (
            <li key={sn.commit} className="mono flex flex-wrap items-baseline justify-between gap-x-3 text-[10.5px]">
              <span className="min-w-0 truncate text-muted" title={sn.message}>
                {sn.commit.slice(0, 7)} · {ago(sn.at_s * 1000, now)}
              </span>
              <button type="button" className={link} disabled={!!busy} data-e2e="backup-preview" onClick={() => void act<Preview>("preview", { action: "preview", commit: sn.commit }, (x) => { setPv(x); return null; })}>
                preview restore
              </button>
            </li>
          ))}
        </ul>
      )}
      {pv && (
        <div className="mono grid gap-1 border-l-2 border-attention/70 pl-3 text-[10.5px]" data-e2e="backup-restore-plan">
          <span className="text-text">
            restoring {pv.commit.slice(0, 7)} writes {pv.new} new and {pv.changed} changed files ({pv.same} already the same)
          </span>
          {[...pv.changed_files.slice(0, 8).map((f) => `changed ${f}`), ...pv.new_files.slice(0, 8).map((f) => `new ${f}`)].map((x) => (
            <span key={x} className="truncate text-faint">
              {x}
            </span>
          ))}
          <span className="text-quiet">every changed file is copied aside first; files only on this machine are kept; Audit undoes the restore</span>
          <div className="flex gap-4">
            <button
              type="button"
              className={`${link} text-attention`}
              data-e2e="backup-restore"
              disabled={!!busy || pv.new + pv.changed === 0}
              onClick={() => void act<{ new: number; changed: number }>("restore", { action: "restore", commit: pv.commit, confirm: true }, (x) => { setPv(null); return `Restored: ${x.new} new and ${x.changed} changed files written. Undo in Audit.`; })}
            >
              {busy === "restore" ? "restoring" : "restore this snapshot"}
            </button>
            <button type="button" className={link} onClick={() => setPv(null)}>
              cancel
            </button>
          </div>
        </div>
      )}
    </li>
  );
}

/** The side panel's Backups tab: what each part holds on disk (the engine's own paths), what a
 *  snapshot leaves out, and the rules of sending and restoring. Reference for the Backups tab. */
export function BackupNotes() {
  const q = useApi<Reply>(PATH, 15000);
  const d = q.data;
  return (
    <div className="grid gap-0" data-e2e="backup-notes">
      <RailSection title="Parts and what they hold">
        {d ? (
          <dl className="grid gap-2 text-[11px]">
            {d.parts.map((p) => (
              <div key={p.part} className="grid gap-0.5">
                <dt className="mono text-muted">{PART_WORD[p.part] ?? p.part}</dt>
                <dd className="mono break-all leading-relaxed text-faint">{p.paths.join(" · ")}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <span className="mono text-[11px] text-faint">{q.error ? `backups did not answer (${q.error})` : "reading"}</span>
        )}
      </RailSection>
      <RailSection title="Left out of a snapshot">
        <p className="text-[11.5px] leading-relaxed text-faint">
          Stores the Nano rebuilds (dense vectors, meaning caches, live snapshots, temporary files) and files over {d ? bytes(d.max_file_bytes) : "the size limit"}: git hosts refuse them without large-file support, and base cartridges come back from the cartridge library. Each snapshot is one commit with a MANIFEST.json naming what is in it and what was left out.
        </p>
      </RailSection>
      <RailSection title="Sending">
        <p className="text-[11.5px] leading-relaxed text-faint">
          Nothing is sent until back up now, and the air gap refuses it. It needs git on this machine{d ? (d.git ? " (found)" : " (not found)") : ""}. Send my token sends your token for that host only: HF_TOKEN or the hf login for Hugging Face, GITHUB_TOKEN or GH_TOKEN for other hosts; without it, git&apos;s own sign-in answers.
        </p>
      </RailSection>
      <RailSection title="Restoring">
        <p className="text-[11.5px] leading-relaxed text-faint">Preview restore writes nothing. Restore copies every file it changes aside first (backups/&lt;id&gt;-before-&lt;time&gt;/), keeps files only on this machine, and is undone in Audit. Restore while no serving process is running, or restart it after.</p>
      </RailSection>
      <RailSection title="The same elsewhere">
        <p className="mono text-[11px] leading-relaxed text-faint">MCP nm_settings: backups, backup_add, backup_remove, backup_now, backup_snapshots, backup_preview, backup_restore. Command line: neuromorph-nano backup add, now, snapshots, preview, restore.</p>
      </RailSection>
    </div>
  );
}

export default function BackupRepos() {
  const q = useApi<Reply>(PATH, 15000);
  const [note, setNote] = useState<Note | null>(null);
  const [url, setUrl] = useState("");
  const [parts, setParts] = useState<string[]>(["memory", "maps", "units", "settings"]);
  const [token, setToken] = useState(false);
  const [busy, setBusy] = useState(false);
  const d = q.data;
  const now = useNow();
  // The newest snapshot across the listed repositories (seconds), when any was made.
  const newest = (d?.repos ?? []).reduce((a, r) => Math.max(a, r.last?.at_s ?? 0), 0);
  const add = async () => {
    setBusy(true);
    try {
      await post(PATH, { action: "add", url: url.trim(), parts, token });
      setNote({ text: `${url.trim()} added. Nothing is sent until you back up. Undo in Audit.`, tone: "verified" });
      setUrl("");
      refreshAll(PATH);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  return (
    <section className="grid min-w-0 gap-6" data-e2e="backup-repos">
      <div className="flex min-h-7 flex-wrap items-center justify-between gap-3 border-b border-line pb-1.5">
        <h2 className="eyebrow flex items-center gap-2 text-muted">
          <span aria-hidden className="inline-block h-1 w-1 bg-sense-pale" />
          your repositories
        </h2>
        <span className="mono flex flex-wrap items-center gap-x-3 text-[10.5px] text-faint" title="Back up, store and roll back your own data in repositories you own">
          {d ? (
            <>
              <span>
                {d.repos.length} {d.repos.length === 1 ? "repository" : "repositories"}
              </span>
              <Pill tone={d.git ? "verified" : "attention"}>{d.git ? "git ready" : "git not found"}</Pill>
              {d.air_gap && <Pill tone="attention">air gap on: nothing is sent</Pill>}
            </>
          ) : q.error ? (
            `did not answer (${q.error})`
          ) : (
            "reading"
          )}
        </span>
      </div>
      <NoteLine note={note} />
      {/* The path of a backup, with this page's own counts. */}
      <ol className="grid grid-cols-2 border-t border-l border-line lg:grid-cols-4" aria-label="The path of a backup" data-e2e="backup-path">
        {(
          [
            ["Parts", d ? `${parts.length} of ${d.parts.length}` : null, "chosen below: what a snapshot of a new repository holds"],
            ["Snapshot", d ? (newest ? ago(newest * 1000, now) : "none yet") : null, "back up now copies the parts into one commit; nothing is sent before"],
            ["Repository", d ? String(d.repos.length) : null, "a git remote, a Hugging Face repository or a folder you own"],
            ["Restore", d ? "previewed" : null, "shows what changes first; changed files are copied aside; Audit undoes it"],
          ] as const
        ).map(([label, value, caption], i) => (
          <li key={label} className="grid min-w-0 content-start gap-1.5 border-r border-b border-line px-3 py-2.5">
            <span className="flex items-baseline justify-between gap-2">
              <span className="eyebrow truncate">
                <span className="text-quiet">{i + 1}</span> {label}
              </span>
              {i < 3 && (
                <span aria-hidden className="mono text-[11px] text-quiet">
                  →
                </span>
              )}
            </span>
            <span className="mono text-[13px] text-text">{value ?? "reading"}</span>
            <span className="mono text-[10px] leading-snug text-faint">{caption}</span>
          </li>
        ))}
      </ol>
      {d && d.repos.length === 0 ? (
        <div className="mono border-l border-line pl-3 text-[11px] leading-relaxed text-faint">No repository yet. Add one below; nothing is sent until you back up.</div>
      ) : (
        <ul className="grid border-t border-line/60">
          {(d?.repos ?? []).map((r) => (
            <RepoRow key={r.id} r={r} airGap={!!d?.air_gap} onNote={setNote} />
          ))}
        </ul>
      )}
      <form
        className="grid gap-3 border border-line bg-bg-1/50 px-3 pt-2.5 pb-3"
        data-e2e="backup-add"
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy && url.trim() && parts.length > 0) void add();
        }}
      >
        <span className="eyebrow">add a repository</span>
        <label className="grid gap-0.5">
          <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">git remote, Hugging Face repository or folder</span>
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/you/nano-backup.git" aria-label="Repository" className={`${field} w-full`} spellCheck={false} />
        </label>
        <div className="mono flex flex-wrap items-center gap-x-4 gap-y-2 text-[10.5px] text-muted">
          <span className="text-[10px] uppercase tracking-[0.12em] text-faint">parts</span>
          {(d?.parts ?? []).map((p) => (
            <label key={p.part} className="press flex min-h-8 cursor-pointer items-center gap-1.5 hover:text-text" title={p.paths.join(", ")}>
              <input type="checkbox" checked={parts.includes(p.part)} onChange={(e) => setParts((xs) => (e.target.checked ? [...xs, p.part] : xs.filter((x) => x !== p.part)))} />
              {PART_WORD[p.part] ?? p.part}
            </label>
          ))}
          <label className="press flex min-h-8 cursor-pointer items-center gap-1.5 hover:text-text" title="Sends your token for this host (HF_TOKEN for Hugging Face, GITHUB_TOKEN or GH_TOKEN for others) to this host only; without it, git's own sign-in answers">
            <input type="checkbox" checked={token} onChange={(e) => setToken(e.target.checked)} />
            send my token
          </label>
          <button type="submit" className="press mono ml-auto min-h-9 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40" disabled={busy || !url.trim() || parts.length === 0} data-e2e="backup-add-button">
            {busy ? "adding" : "add repository"}
          </button>
        </div>
      </form>
    </section>
  );
}
