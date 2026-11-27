"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... We need the interface to be
// amazing."): the six clients and their commands stay in the middle; how connecting works moves to
// ConnectGuide, which the Settings rail shows. Each client gets a copy of its dry run as well.

// 2026-11-26 (audit B4): which of your coding agents is connected to a project, read from each
// client's own settings file the way `neuromorph-nano connect` writes it (`GET /api/connect`),
// with the exact command to connect one and its dry-run preview. Nothing here writes a file: the
// command runs in your terminal.

import { useState } from "react";
import { Empty, Panel, Pill, Placeholder } from "@/components/ui";
import { useApi } from "@/lib/api";
import { maskPath } from "@/lib/privacy";

export type ClientRow = {
  client: string;
  name: string;
  command: string;
  dry_run_command: string;
  scope: string;
  file: string | null;
  connected: boolean | null;
  note?: string;
  preview?: string;
  entry?: boolean;
  serves_project?: boolean;
  launcher_matches?: boolean;
};
export type ConnectReply = { project?: string; launcher?: string; launcher_present?: boolean; clients?: ClientRow[]; error?: string };

/** The connect view of a project (its folder or key). */
export function useConnect(project: string | null) {
  return useApi<ConnectReply>(project ? `/api/connect?project=${encodeURIComponent(project)}` : null, 20000);
}

function Copy({ text, label = "copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={() => void navigator.clipboard?.writeText(text).then(() => setDone(true), () => setDone(false))}
      title={text}
      className="press mono min-h-8 shrink-0 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
    >
      {done ? "copied" : label}
    </button>
  );
}

/** One line per client: connected or not, where its setting lives, and the command. */
export function ConnectRows({ reply }: { reply: ConnectReply }) {
  return (
    <ul className="grid" data-e2e="connect-clients">
      {(reply.clients ?? []).map((c) => (
        <li key={c.client} className="grid min-w-0 gap-1 border-b border-line/60 py-2" data-client={c.client}>
          <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
            <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${c.connected === true ? "bg-verified" : "border border-line-3"}`} />
            <span className="mono w-28 shrink-0 truncate text-[12px] text-text">{c.name}</span>
            <Pill tone={c.connected === true ? "verified" : "muted"} title={c.note ?? undefined}>
              {c.connected === true ? "connected" : c.connected === false ? (c.entry ? "set up for another folder" : "not connected") : "add it by hand"}
            </Pill>
            <span className="mono min-w-0 flex-1 truncate text-[10.5px] text-faint" title={[c.note, c.file].filter(Boolean).join(" · ") || undefined}>
              {c.file ? `${c.scope} · ${maskPath(c.file)}` : c.scope}
            </span>
            <Copy text={c.command} />
            {c.dry_run_command && <Copy text={c.dry_run_command} label="dry run" />}
          </div>
          <code className="mono block min-w-0 truncate border-b border-line/60 px-1 py-0.5 text-[10.5px] text-muted" title={c.command}>
            {c.command}
          </code>
          {c.connected === true && c.launcher_matches === false && <span className="mono text-[10.5px] text-attention">its entry points at another copy of the launcher: run the command again</span>}
        </li>
      ))}
    </ul>
  );
}

/** The panel for Settings: the project picked, its six clients. */
export default function ConnectPanel({ project }: { project: string | null }) {
  const q = useConnect(project);
  const d = q.data;
  const n = (d?.clients ?? []).filter((c) => c.connected === true).length;
  return (
    <Panel title="Connect your coding agents" right={d && !d.error ? <span className="mono text-[11px] text-faint">{n} connected</span> : undefined}>
      {!project ? (
        <Empty title="No project known yet">Serve a project once, or run neuromorph-nano connect with --project and your folder.</Empty>
      ) : !d && q.error ? (
        <Empty title="Connection status did not answer">{q.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-24" what="connections" />
      ) : d.error ? (
        <Empty title="Not readable">{d.error}</Empty>
      ) : (
        <>
          <div className="mono mb-1 flex flex-wrap gap-x-4 text-[10.5px] text-faint">
            <span className="min-w-0 truncate">project {maskPath(d.project ?? project)}</span>
            <span className={d.launcher_present ? undefined : "text-attention"}>launcher {d.launcher_present ? "present" : "missing (run setup)"}</span>
          </div>
          <ConnectRows reply={d} />
        </>
      )}
    </Panel>
  );
}

/** How connecting works, for the rail. */
export function ConnectGuide() {
  return (
    <div className="grid gap-2 text-[11px]">
      <p className="read leading-relaxed text-muted">Neuromorph Nano works with every MCP client: Claude Code, Cursor, VS Code, Antigravity, Windsurf, and any other stdio client.</p>
      <p className="read leading-relaxed text-faint">
        Run a client&apos;s command in a terminal to connect the project to it; the dry run (<span className="mono">--dry-run</span>) shows the change first. Other servers&apos; entries are kept. Nothing here writes a file: the status is read from each client&apos;s own settings file.
      </p>
    </div>
  );
}
