"use client";

// 2026-11-26: the first-run strip on Live Activity. Shown only while something is missing for the
// Nano to work for you: a base cartridge not on the shelf, or no engine serving a project. It says
// what is missing, links to the downloads, and gives the exact command that connects a project to
// one of your coding agents. Calm and dismissible; it comes back only when what is missing changes.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel"): `variant="rail"` is the
// Set up tab of the side panel: always there, what is missing (or that nothing is), which coding
// agents are connected to the newest project (every MCP client, not one), and the connect command.

import Link from "next/link";
import { useState, useSyncExternalStore } from "react";
import { useApi, type Projects, type Status } from "@/lib/api";
import { useConnect } from "@/components/ConnectStatus";
import { Dot } from "@/components/ui";
import { maskName, maskPath } from "@/lib/privacy";
import { readPref, writePref } from "@/lib/prefs";

type Downloads = { cartridges?: { key: string; present: boolean }[]; air_gap?: boolean };

/** The MCP clients `neuromorph-nano connect` writes for, with the name each is known by. */
export const CLIENTS: { id: string; name: string }[] = [
  { id: "claude", name: "Claude Code" },
  { id: "cursor", name: "Cursor" },
  { id: "vscode", name: "VS Code" },
  { id: "antigravity", name: "Antigravity" },
  { id: "windsurf", name: "Windsurf" },
  { id: "generic", name: "Any other MCP client" },
];

/** The command that connects `project` to `client`, as typed in a terminal. */
export function connectCommand(client: string, project: string): string {
  const p = project.includes(" ") ? `"${project}"` : project;
  return `neuromorph-nano connect ${client} --project ${p}`;
}

const KEY = "nano.setup.dismissed";

function useStored(key: string): string | null {
  return useSyncExternalStore(
    (cb) => {
      window.addEventListener("storage", cb);
      return () => window.removeEventListener("storage", cb);
    },
    () => {
      try {
        return readPref(key);
      } catch {
        return null;
      }
    },
    () => null,
  );
}

/** One client picker and its command, with a copy button. */
export function ConnectCommand({ project, label = "Connect a project to a coding agent" }: { project: string; label?: string }) {
  const [client, setClient] = useState("claude");
  const [copied, setCopied] = useState(false);
  const cmd = connectCommand(client, project);
  return (
    <div className="grid min-w-0 gap-1.5" data-e2e="connect-command">
      <span className="eyebrow">{label}</span>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <label className="relative">
          <span className="sr-only">Coding agent</span>
          <select
            value={client}
            onChange={(e) => {
              setClient(e.target.value);
              setCopied(false);
            }}
            className="mono h-8 border-b border-line-strong bg-transparent pr-1 text-[11px] text-text outline-none focus-visible:border-sense-pale"
          >
            {CLIENTS.map((c) => (
              <option key={c.id} value={c.id} className="bg-bg-1">
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <code className="mono min-w-0 flex-1 break-all border-b border-line px-1 py-1 text-[11px] text-muted" title={cmd}>
          {connectCommand(client, maskPath(project) || project)}
        </code>
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard?.writeText(cmd).then(
              () => setCopied(true),
              () => setCopied(false),
            );
          }}
          className="press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
    </div>
  );
}

/** What is missing for the Nano to work for you, from the engine: the base cartridges not on the
 *  shelf, and whether any engine serves a project. */
function useSetup(status: Status | null) {
  const dl = useApi<Downloads>("/api/downloads", 15000);
  const projects = useApi<Projects>("/api/projects", 30000);
  const last = [...(projects.data?.projects ?? [])].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0))[0] ?? null;
  const connect = useConnect(last?.root ?? null);
  const missing = (dl.data?.cartridges ?? []).filter((c) => !c.present).map((c) => c.key);
  const notServing = status !== null && typeof status.serving === "number" && status.serving === 0;
  const reasons: string[] = [];
  if (missing.length) reasons.push(`${missing.length} base ${missing.length === 1 ? "cartridge is" : "cartridges are"} not in the cartridge library (${missing.join(", ")})`);
  if (notServing) reasons.push("no engine is serving a project yet");
  return { dl, last, connect, missing, notServing, reasons, sig: reasons.join("|") };
}

/** The Set up tab of the side panel. */
function SetupRail({ status }: { status: Status | null }) {
  const { dl, last, connect, missing, notServing, reasons } = useSetup(status);
  const clients = connect.data?.clients ?? [];
  const connected = clients.filter((c) => c.connected === true).map((c) => c.name);
  return (
    <section aria-label="Set up" className="grid gap-4" data-e2e="setup-strip" data-variant="rail">
      <div className="grid gap-1.5">
        <span className="eyebrow">What is missing</span>
        {!status || !dl.data ? (
          <span className="mono text-[11px] text-faint">{dl.error ? `the downloads did not answer (${dl.error})` : "reading"}</span>
        ) : reasons.length === 0 ? (
          <span className="mono flex items-center gap-2 text-[11px] text-muted">
            <Dot tone="verified" /> nothing: the base cartridges are in the cartridge library and an engine is serving
          </span>
        ) : (
          <ul className="grid gap-1">
            {reasons.map((r) => (
              <li key={r} className="mono flex items-start gap-2 text-[11px] text-muted">
                <Dot tone="attention" className="mt-1" />
                {r}
              </li>
            ))}
          </ul>
        )}
        <span className="mono flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
          {missing.length > 0 && (
            <Link href="/models/#downloads" className="link">
              {dl.data?.air_gap ? "downloads (air gap on)" : "get the base cartridges"}
            </Link>
          )}
          {notServing && <span className="text-faint">start one with neuromorph-nano serve or daemon</span>}
        </span>
      </div>

      <div className="grid gap-1.5">
        <span className="eyebrow flex items-baseline justify-between gap-3">
          <span>Coding agents</span>
          {last && <span className="mono normal-case tracking-normal text-quiet">newest project {maskName(last.key)}</span>}
        </span>
        {!last ? (
          <span className="mono text-[11px] text-faint">No project known yet. Connect one below and its folder shows here.</span>
        ) : !connect.data ? (
          <span className="mono text-[11px] text-faint">{connect.error ? `the connect view did not answer (${connect.error})` : "reading"}</span>
        ) : connect.data.error ? (
          <span className="mono text-[11px] text-faint">{connect.data.error}</span>
        ) : (
          <>
            <span className="mono text-[11px] text-faint" data-e2e="connect-summary">
              {connected.length ? `connected: ${connected.join(", ")}` : "no coding agent connected yet"}
            </span>
            <ul className="grid" data-e2e="setup-clients">
              {clients.map((c) => (
                <li key={c.client} className="mono flex min-w-0 items-center gap-2 border-b border-line/50 py-1 text-[11px]" title={[c.note, c.file ? maskPath(c.file) : null].filter(Boolean).join(" · ") || undefined}>
                  <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${c.connected === true ? "bg-verified" : "border border-line-3"}`} />
                  <span className="min-w-0 flex-1 truncate text-muted">{c.name}</span>
                  <span className={c.connected === true ? "text-verified" : "text-faint"}>{c.connected === true ? "connected" : c.connected === false ? (c.entry ? "another folder" : "not connected") : "by hand"}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      <ConnectCommand project={last?.root || "<your project folder>"} />
      <Link href="/settings/#tools" className="link mono text-[11px]">
        all six clients, with dry runs
      </Link>
    </section>
  );
}

export default function SetupStrip({ status, variant = "strip" }: { status: Status | null; variant?: "strip" | "rail" }) {
  if (variant === "rail") return <SetupRail status={status} />;
  return <Strip status={status} />;
}

function Strip({ status }: { status: Status | null }) {
  const { dl, last, connect, missing, notServing, reasons, sig } = useSetup(status);
  const dismissed = useStored(KEY);
  const [open, setOpen] = useState(false);
  const connected = (connect.data?.clients ?? []).filter((c) => c.connected === true).map((c) => c.name);
  // What is missing, as one line; the strip is shown only when there is something.
  if (!status || !dl.data || reasons.length === 0 || dismissed === sig) return null;
  const project = last?.root || "<your project folder>";
  return (
    // 2026-11-26: one line, so the organism keeps its room; the command opens on demand.
    <section aria-label="Set up" className="mt-2 grid gap-2 border-y border-line py-2" data-e2e="setup-strip">
      <div className="mono flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
        <span className="text-[10.5px] uppercase tracking-[0.14em] text-sense-pale">Set up</span>
        <span className="min-w-0 text-muted">{reasons.join("; ")}</span>
        {last?.root && connect.data && !connect.data.error && notServing && (
          <span className="text-faint" data-e2e="connect-summary">
            {connected.length ? `connected: ${connected.join(", ")}` : "no coding agent connected yet"}
          </span>
        )}
        <span className="ml-auto flex items-center gap-4">
          {missing.length > 0 && (
            <Link href="/models/#downloads" className="link">
              {dl.data.air_gap ? "downloads (air gap on)" : "get the base cartridges"}
            </Link>
          )}
          {notServing && (
            <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="press border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text">
              {open ? "hide the command" : "connect a coding agent"}
            </button>
          )}
          <Link href="/settings/#tools" className="link">
            all six clients
          </Link>
          <button
            type="button"
            onClick={() => {
              try {
                writePref(KEY, sig);
                window.dispatchEvent(new StorageEvent("storage", { key: KEY }));
              } catch {
                // storage unavailable: it stays for this visit
              }
            }}
            className="press text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text"
          >
            dismiss
          </button>
        </span>
      </div>
      {open && notServing && <ConnectCommand project={project} />}
    </section>
  );
}

/** How many things are missing, for a short mark on the page's status line. */
export function useSetupNeeded(status: Status | null): number {
  return useSetup(status).reasons.length;
}
