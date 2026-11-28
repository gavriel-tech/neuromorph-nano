"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing."):
// Settings' middle is the controls themselves, one group at a time under the section bar (machine,
// agents and tools, controls, privacy and network, stage times). The help text is in tooltips and
// in the rail's Guide tab, which follows the section shown; the design files the engine reads are
// a library in the rail. `/settings/#tools` (any section) opens that section; `/settings/#<design>`
// and `/settings/#designs` open the design files in the rail.

import Link from "next/link";
import { roleTitle } from "@/components/MicroModel";
import { useEffect, useMemo, useRef, useState } from "react";
import Controls, { NoteLine } from "@/components/Controls";
import LeanModePanel, { LeanModeGuide, mcpToolList } from "@/components/LeanModePanel";
import ProjectMenu from "@/components/ProjectMenu";
import ProjectsPanel, { ProjectsGuide } from "@/components/ProjectsPanel";
import { Empty, Masked, Panel, Pill, Placeholder } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi, useApiLevel, type Audit, type Control, type DevicesReply, type Projects, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { maskPath, maskPathsIn } from "@/lib/privacy";
import SectionBar, { useSection } from "@/components/SectionBar";
import ConnectPanel, { ConnectGuide } from "@/components/ConnectStatus";
import { setHash } from "@/lib/hash";
import { gpuCeiling, liveVramSegments, mb, vramSegments } from "@/lib/vram";
import { Capacity, LiveBadge, VramBar } from "@/components/LivePanels";
import StageTimesPanel from "@/components/StageTimes";
import HardwareLimitsPanel, { LimitsGuide } from "@/components/HardwareLimits";
import Workspace, { RailSection, useRail } from "@/components/Workspace";
import { limitNumber, machineCapacity, parseLimits, residency } from "@/lib/limits";
import { engineServing } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { useProjectKey } from "@/lib/evidence";
import { CONTROL_GROUP_ORDER, controlGroupName } from "@/lib/controls";

type Settings = Record<string, Record<string, unknown>>;

type AirGapReply = {
  air_gap: boolean;
  source: string;
  locked_by_env: boolean;
  network: string;
  loopback_only: boolean;
  lan_listeners: number;
  outbound_network_code: boolean;
  remote_imports_allowed: boolean;
  downloads_running?: boolean;
  models_local_dir: string;
};

type NanoIgnoreReply = {
  project?: string;
  root?: string;
  file?: string;
  exists?: boolean;
  content?: string;
  rules?: string[];
  nested?: string[];
  test?: { path: string; ignored: boolean };
  note?: string;
  error?: string;
};

function NanoIgnorePanel({ projects }: { projects: Projects | null | undefined }) {
  const { key: activeKey } = useProjectKey();
  const list = useMemo(() => [...(projects?.projects ?? [])].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0)), [projects]);
  const projectKey = activeKey || list[0]?.key || "nano";
  const [testPath, setTestPath] = useState("");
  const trimmedTest = testPath.trim();
  const url = `/api/nanoignore?project=${encodeURIComponent(projectKey)}${trimmedTest ? `&test=${encodeURIComponent(trimmedTest)}` : ""}`;
  const q = useApi<NanoIgnoreReply>(url, 15000);
  const d = q.data;

  const [draftMap, setDraftMap] = useState<Record<string, string>>({});
  const draft = projectKey in draftMap ? draftMap[projectKey] : (d?.content ?? "");
  const dirty = projectKey in draftMap && draftMap[projectKey] !== (d?.content ?? "");
  const [quickPat, setQuickPat] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);

  const saveContent = async (content: string) => {
    setBusy(true);
    try {
      await post("/api/nanoignore", { project: projectKey, action: "set", content });
      setDraftMap((m) => {
        const next = { ...m };
        delete next[projectKey];
        return next;
      });
      setNote({ text: "Saved .nanoignore. Substrate cache invalidated immediately; logged in Audit with Undo.", tone: "verified" });
      refreshAll("/api/nanoignore");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const addQuick = async () => {
    const pat = quickPat.trim();
    if (!pat) return;
    setBusy(true);
    try {
      await post("/api/nanoignore", { project: projectKey, action: "add", pattern: pat });
      setQuickPat("");
      setDraftMap((m) => {
        const next = { ...m };
        delete next[projectKey];
        return next;
      });
      setNote({ text: `Added "${pat}" to .nanoignore. Logged in Audit with Undo.`, tone: "verified" });
      refreshAll("/api/nanoignore");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const rules = d?.rules ?? [];
  const nested = d?.nested ?? [];

  return (
    <Panel
      title="File exclusions (.nanoignore)"
      right={
        d ? (
          <Pill tone={rules.length > 0 ? "verified" : "muted"}>
            {rules.length} {rules.length === 1 ? "rule" : "rules"}
          </Pill>
        ) : undefined
      }
    >
      <NoteLine note={note} />
      <div className="grid gap-3" data-e2e="nanoignore-panel">
        <div className="grid gap-3 sm:flex sm:flex-wrap sm:items-end sm:gap-4">
          <div className="min-w-0 sm:min-w-[12rem] sm:flex-1">
            <ProjectMenu placement="down" label="repository" />
          </div>
          <label className="grid min-w-0 gap-1 sm:min-w-[14rem] sm:flex-1">
            <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">test a relative path</span>
            <input
              value={testPath}
              onChange={(e) => setTestPath(e.target.value)}
              placeholder="e.g. secrets/keys.json or src/app.ts"
              className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
            />
          </label>
          {d?.test && (
            <Pill tone={d.test.ignored ? "verified" : "muted"}>
              {d.test.ignored ? "excluded by .nanoignore" : "readable by Nano"}
            </Pill>
          )}
        </div>

        {q.error && !d ? (
          <Empty title=".nanoignore did not answer">{q.error}</Empty>
        ) : d?.error ? (
          <Empty title=".nanoignore unavailable">{d.error}</Empty>
        ) : !d ? (
          <Placeholder className="h-28" what=".nanoignore" />
        ) : (
          <>
            <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
              <input
                value={quickPat}
                onChange={(e) => setQuickPat(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void addQuick();
                  }
                }}
                placeholder="Quick add pattern: *.pem, secrets/, internal/, !secrets/template.json"
                className="mono h-9 w-full min-w-0 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale sm:min-w-[16rem] sm:flex-1"
              />
              <button
                type="button"
                disabled={busy || !quickPat.trim()}
                onClick={() => void addQuick()}
                className="press mono min-h-9 w-fit border-b border-sense-pale px-2 text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
              >
                + add pattern
              </button>
            </div>

            <label className="grid gap-1">
              <span className="mono flex items-baseline justify-between text-[10px] uppercase tracking-[0.12em] text-faint">
                <span>{d.file ? maskPath(d.file) : ".nanoignore"}</span>
                <span className="normal-case tracking-normal text-quiet">{d.exists ? "on disk" : "not created yet"}</span>
              </span>
              <textarea
                rows={5}
                value={draft}
                onChange={(e) => setDraftMap((m) => ({ ...m, [projectKey]: e.target.value }))}
                placeholder={"# Gitignore-style patterns excluded from Neuromorph Nano\n*.pem\n.env*\nsecrets/\ninternal/"}
                className="mono w-full border border-line bg-bg-1/50 p-2.5 text-[11.5px] leading-relaxed text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
            </label>

            <div className="flex flex-wrap items-center gap-4">
              <button
                type="button"
                disabled={busy || !dirty}
                onClick={() => void saveContent(draft)}
                className="press mono min-h-10 border-b border-sense-pale px-1 text-[11px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
              >
                {busy ? "Saving" : "Save .nanoignore"}
              </button>
              {dirty && (
                <button
                  type="button"
                  onClick={() =>
                    setDraftMap((m) => {
                      const next = { ...m };
                      delete next[projectKey];
                      return next;
                    })
                  }
                  className="press mono min-h-10 text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text"
                >
                  discard edits
                </button>
              )}
              {nested.length > 0 && (
                <span className="mono text-[10.5px] text-faint">
                  nested: {nested.join(", ")}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}

function AirGapPanel() {
  const airgap = useApi<AirGapReply>("/api/airgap", 5000);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);
  const d = airgap.data;

  const toggle = async () => {
    if (!d) return;
    setBusy(true);
    const next = d.air_gap ? "off" : "on";
    try {
      await post("/api/controls", { key: "air_gap", value: next });
      setNote({
        text: next === "on"
          ? "Air-gap mode enabled: local network access and tablet pairing locked off. Logged in Audit."
          : "Air-gap mode disabled. Logged in Audit.",
        tone: "verified",
      });
      refreshAll("/api/airgap");
      refreshAll("/api/controls");
      refreshAll("/api/devices");
      refreshAll("/api/audit");
      // 2026-11-26: the downloads stop or open with the air gap, so their view follows at once.
      refreshAll("/api/downloads");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  return (
    <Panel
      title="Air-gap and local isolation"
      right={
        d ? (
          <Pill tone={d.air_gap ? "verified" : d.loopback_only ? "signal" : "attention"}>
            {d.air_gap ? (d.locked_by_env ? "air-gap locked (env)" : "air-gap on") : d.loopback_only ? "local only" : `lan ${d.network}`}
          </Pill>
        ) : undefined
      }
    >
      <NoteLine note={note} />
      {airgap.error && !d ? (
        <Empty title="Air-gap status did not answer">{airgap.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-24" what="air-gap status" />
      ) : (
        <div className="grid gap-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Pill tone={d.loopback_only ? "verified" : "attention"}>{d.loopback_only ? "127.0.0.1 only" : `${d.lan_listeners} lan listener(s)`}</Pill>
            {typeof d.outbound_network_code === "boolean" && (
              <Pill tone={d.outbound_network_code ? "signal" : "verified"}>{d.outbound_network_code ? "downloads only when you ask" : "no outbound network"}</Pill>
            )}
            {d.downloads_running === true && <Pill tone="signal">downloading now</Pill>}
            {typeof d.remote_imports_allowed === "boolean" && (
              <Pill tone={d.remote_imports_allowed ? "muted" : "verified"}>{d.remote_imports_allowed ? "remote imports allowed" : "remote imports blocked"}</Pill>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-4 border-t border-line pt-3">
            <button
              type="button"
              disabled={busy || d.locked_by_env}
              onClick={toggle}
              title="A strict local-only lock across the interface and MCP: local network access and tablet pairing forced off, remote URLs and imports refused, weights loaded from the local folder only."
              className="press mono min-h-11 border-b border-line-strong px-1 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-50"
            >
              {busy ? "Updating" : d.locked_by_env ? "Locked on by NANO_AIR_GAP" : d.air_gap ? "Disable air-gap mode" : "Enable air-gap mode"}
            </button>
            <span className="mono text-[11px] text-faint">source: {d.source}</span>
          </div>
        </div>
      )}
    </Panel>
  );
}

function summary(v: Record<string, unknown>): string {
  for (const k of ["note", "step", "purpose", "description"]) if (typeof v[k] === "string") return v[k] as string;
  return `${Object.keys(v).length} fields`;
}

function DevicesPanel() {
  const level = useApiLevel();
  const hasDevices = level === null || level >= 2;
  const devices = useApi<DevicesReply>(hasDevices ? "/api/devices" : null, 5000);
  const now = useNow();
  const [issued, setIssued] = useState<{ code: string; expires_ms: number } | null>(null);
  const [busyCode, setBusyCode] = useState(false);
  const [armRevoke, setArmRevoke] = useState<string | null>(null);
  const [busyRevoke, setBusyRevoke] = useState<string | null>(null);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);

  const d = devices.data;
  const activeIssued = issued && (!d || d.code_open) && (now === 0 || now <= issued.expires_ms) ? issued : null;
  const locked = d ? d.locked_until_ms > 0 && (now === 0 || d.locked_until_ms > now) : false;
  const lockSecs = d && locked && now > 0 ? Math.max(1, Math.ceil((d.locked_until_ms - now) / 1000)) : null;

  const createCode = async () => {
    setBusyCode(true);
    try {
      const res = await post<{ code: string; expires_ms: number }>("/api/pair/code", {});
      setIssued(res);
      const left = typeof res.expires_ms === "number" ? Math.round((res.expires_ms - Date.now()) / 1000) : null;
      const valid = left !== null && left > 0 ? (left >= 120 ? `valid about ${Math.round(left / 60)} minutes` : `valid ${left} seconds`) : "expiry not reported";
      setNote({ text: `Pairing code created (${valid}, once). Logged in Audit.`, tone: "verified" });
      refreshAll("/api/devices");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusyCode(false);
  };

  const revoke = async (id: string) => {
    setBusyRevoke(id);
    try {
      await post<DevicesReply>("/api/devices/revoke", { id });
      setNote({ text: `Device ${id} revoked. Logged in Audit.`, tone: "verified" });
      refreshAll("/api/devices");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusyRevoke(null);
    setArmRevoke(null);
  };

  const sorted = d ? [...d.devices].sort((a, b) => Number(a.revoked) - Number(b.revoked) || b.paired_ms - a.paired_ms) : [];

  return (
    <Panel
      title="Paired devices and local network"
      right={
        d ? (
          <Pill tone={d.network === "control" ? "attention" : d.network === "view" ? "signal" : "muted"}>
            network {d.network}
          </Pill>
        ) : undefined
      }
    >
      <NoteLine note={note} />
      {level !== null && level < 2 ? (
        <Empty title="Not reported by this engine build" slot>
          Device pairing comes with API 2.
        </Empty>
      ) : devices.error && !d ? (
        <Empty title="Devices did not answer">{devices.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-32" what="devices" />
      ) : (
        <div className="grid gap-4">
          {devices.error && <div className="mono text-xs text-error">Refresh failed ({devices.error}); showing last values.</div>}
          {d.network !== "off" && (
            <div className="mono border-l-2 border-attention pl-3 text-[11px] leading-relaxed text-attention">{d.encryption || "No encryption on the wire: anyone on your local network can read traffic and device tokens."}</div>
          )}
          {d.network === "off" ? (
            <div className="mono flex items-center gap-2 text-[11px] text-faint" title="Set the network control (Controls) to view or control, then restart the interface server, to allow paired tablets.">
              <span aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" />
              local network off
            </div>
          ) : d.addresses.length === 0 ? (
            <div className="mono text-[11px] text-attention" title="If you just turned network access on, restart the interface server (neuromorph-nano launch ui) to listen on your LAN addresses.">
              No local network address bound yet: restart the interface server.
            </div>
          ) : (
            <div className="grid gap-1">
              <div className="eyebrow">LAN addresses</div>
              <ul className="flex flex-wrap gap-x-5 gap-y-1">
                {d.addresses.map((addr) => (
                  <li key={addr} className="mono text-xs text-text">
                    {addr}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="border-t border-line pt-3">
            <div className="flex flex-wrap items-center gap-4">
              <button
                type="button"
                disabled={busyCode || locked}
                onClick={createCode}
                className="press mono min-h-11 border-b border-line-strong px-1 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-50"
              >
                {busyCode ? "Creating code" : "New pairing code"}
              </button>
              {locked && (
                <span className="mono text-xs text-error">
                  Pairing locked for {lockSecs ?? "..."}s after 5 wrong codes.
                </span>
              )}
              {activeIssued && (
                <span className="flex items-baseline gap-3">
                  <span className="mono text-xl tracking-[0.25em] text-text">{activeIssued.code}</span>
                  <span className="mono text-xs text-faint">
                    expires in {Math.max(0, Math.ceil((activeIssued.expires_ms - (now || activeIssued.expires_ms)) / 1000))}s
                  </span>
                </span>
              )}
              {!activeIssued && d.code_open && !locked && (
                <span className="mono text-xs text-muted">
                  A pairing code is open (valid until expiry or first use
                  {d.code_expires_ms > 0 && now > 0 ? `; ${Math.max(0, Math.ceil((d.code_expires_ms - now) / 1000))}s left` : ""}).
                </span>
              )}
            </div>
          </div>

          <div className="border-t border-line pt-3">
            <div className="eyebrow mb-2 flex items-center justify-between">
              <span>Paired devices</span>
              <span className="text-quiet">{sorted.length}</span>
            </div>
            {sorted.length === 0 ? (
              <div className="text-xs text-faint">No devices paired yet.</div>
            ) : (
              <ol>
                {sorted.map((dev) => {
                  const armed = armRevoke === dev.id;
                  const busy = busyRevoke === dev.id;
                  return (
                    <li key={dev.id} className="flex flex-wrap items-center justify-between gap-3 border-b border-line/60 py-2.5 last:border-b-0">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2.5">
                          <span className="mono text-[13px] text-text">{dev.name}</span>
                          <span className="mono text-[11px] text-faint">{dev.id}</span>
                          <Pill tone={dev.revoked ? "error" : "verified"}>{dev.revoked ? "revoked" : "active"}</Pill>
                        </div>
                        <div className="mono mt-0.5 text-[11px] text-faint">
                          paired {now > 0 && dev.paired_ms ? ago(dev.paired_ms, now) : "earlier"} · last seen {now > 0 && dev.last_seen_ms ? ago(dev.last_seen_ms, now) : "never"}
                          {dev.revoked && dev.revoked_ms > 0 && now > 0 ? ` · revoked ${ago(dev.revoked_ms, now)}` : ""}
                        </div>
                      </div>
                      {!dev.revoked && (
                        <div className="flex shrink-0 items-center gap-3">
                          {!armed ? (
                            <button
                              type="button"
                              disabled={busy}
                              onClick={() => setArmRevoke(dev.id)}
                              className="press mono min-h-11 border-b border-line-strong px-1 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text"
                            >
                              Revoke
                            </button>
                          ) : (
                            <>
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => revoke(dev.id)}
                                className="press mono min-h-11 border-b border-error px-1 text-[11px] uppercase tracking-[0.12em] text-error"
                              >
                                {busy ? "Revoking" : "Confirm revoke"}
                              </button>
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => setArmRevoke(null)}
                                className="press mono min-h-11 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text"
                              >
                                Keep
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

/** How many serving processes are active, in words; without a count, only that one is. */
function servingWords(n: number | null): string {
  return n !== null ? `${n} serving ${n === 1 ? "process" : "processes"} active` : "A serving process is active";
}

function DriveStorageSummary() {
  const storageQ = useApi<{ disk?: { path?: string; free_bytes?: number; total_bytes?: number } | null; disk_floor_gb?: number; below_floor?: boolean; total_bytes?: number }>("/api/storage", 15000);
  const s = storageQ.data;
  const disk = s?.disk;
  if (!disk || typeof disk.free_bytes !== "number" || typeof disk.total_bytes !== "number") return null;
  const drive = disk.path?.match(/^[A-Za-z]:\\/)?.[0] ?? (disk.path ? maskPath(disk.path) : "drive");
  const usedShare = disk.total_bytes > 0 ? Math.min(100, Math.round(((disk.total_bytes - disk.free_bytes) / disk.total_bytes) * 100)) : 0;
  return (
    <div className="mt-2 grid gap-1.5 border-t border-line/60 pt-2.5" data-e2e="settings-drive-storage">
      <div className="mono flex items-baseline justify-between gap-2 text-[10.5px]">
        <span className="text-faint" title={disk.path ? maskPath(disk.path) : undefined}>Drive {drive}</span>
        <span className={s?.below_floor ? "text-attention" : "text-text"}>
          {bytes(disk.free_bytes)} free / {bytes(disk.total_bytes)}
        </span>
      </div>
      <div className="h-1 w-full overflow-hidden bg-line/50">
        <div className={`h-full ${s?.below_floor ? "bg-attention" : "bg-sense-pale"}`} style={{ width: `${usedShare}%` }} />
      </div>
      <div className="mono flex flex-wrap items-baseline justify-between gap-2 text-[10px] text-quiet">
        <span>
          {typeof s?.total_bytes === "number" ? `Nano ${bytes(s.total_bytes)}` : ""}
          {typeof s?.disk_floor_gb === "number" ? ` · floor ${s.disk_floor_gb} GB` : ""}
        </span>
        <Link href="/storage/" className="link">storage</Link>
      </div>
    </div>
  );
}

function HomeostasisPanel({ status }: { status?: Status | null }) {
  const controls = useApi<Control[] | { controls?: Control[] }>("/api/controls", 5000);
  const audit = useApi<Audit>("/api/audit", 5000);

  const ctlMap = useMemo(() => {
    const map = new Map<string, Control>();
    const list = Array.isArray(controls.data) ? controls.data : controls.data?.controls ?? [];
    for (const c of list) map.set(c.key, c);
    return map;
  }, [controls.data]);
  const byKey = useMemo(() => new Map([...ctlMap].map(([k, c]) => [k, c.value] as const)), [ctlMap]);
  const controlsLoaded = controls.data !== null;

  const live = useLive();
  const serving = engineServing(live.reply);
  // This machine's detected hardware (`/api/limits` `profile`), shown where no engine is serving.
  const limitsQ = useApi<unknown>("/api/limits", 5000);
  const profile = useMemo(() => parseLimits(limitsQ.data)?.profile ?? null, [limitsQ.data]);
  const machine = machineCapacity(live.reply?.capacity, serving, profile);
  // What the serving engine holds on the card: its MicroModels' `vram_mb` (`/api/live` `models`),
  // else what `/api/status` `loaded[]` reports.
  const liveModels = serving !== false ? (live.reply?.models ?? []) : [];
  const liveHeldMb = liveModels.reduce((a, m) => a + Math.max(0, m.vram_mb ?? 0), 0);
  const gpuBytes = liveModels.length ? liveHeldMb * 1024 * 1024 : (status?.gpu_bytes ?? 0);
  const gpuUsedMb = Math.round(gpuBytes / (1024 * 1024));
  // `budget.gpu_mb_max` is `auto` by default (it follows this card) or a number up to 81,920 MiB.
  // Auto resolves to the engine's measured ceiling; until the engine reports one, the meters are
  // drawn against what is held, and the label says auto.
  const ceiling = gpuCeiling(byKey.get("budget.gpu_mb_max"), live.reply?.capacity);
  // With no ceiling reported, the per-model rows are drawn against what is held, and no
  // percentage of a ceiling is shown.
  const gpuCeilingMb = ceiling.mb ?? Math.max(1, gpuUsedMb);
  const gpuPct = ceiling.mb !== null ? Math.min(100, Math.round((gpuUsedMb / Math.max(1, ceiling.mb)) * 100)) : null;
  const vram = liveModels.length ? liveVramSegments(liveModels, gpuCeilingMb) : vramSegments(status, gpuCeilingMb);
  // The user's VRAM limit in force now (`capacity.limits.vram_mb`), marked on the live bar when
  // it is a number; and how the four MicroModels sit inside it (the engine's residency mode,
  // `capacity.limits.mode`, else `writer.kept_resident`).
  const vramCapMb = serving === false ? null : limitNumber(live.reply?.capacity?.limits?.vram_mb ?? null);
  const seat = serving === false ? null : residency(live.reply?.writer, live.reply?.capacity?.limits?.mode);
  // Every figure below is the control's real value from /api/controls; a key this engine does
  // not report reads "not reported", never a default typed in here.
  const NR = "not reported";
  const allTools = mcpToolList(ctlMap);
  const totalTools = allTools.length;
  const mcpProfile = byKey.get("mcp.tools");
  const rawToolCount = Number(byKey.get("mcp.tool_count"));
  const toolControlOn = byKey.get("mcp.tool_control") === "on";
  const enabledToolsRaw = (byKey.get("mcp.enabled_tools") ?? "").trim();
  const customToolList = toolControlOn && enabledToolsRaw && enabledToolsRaw.toLowerCase() !== "auto"
    ? enabledToolsRaw.split(",").map((s) => s.trim()).filter(Boolean)
    : [];
  const countKnown = Number.isFinite(rawToolCount) && rawToolCount > 0;
  const effToolCount: number | null = customToolList.length > 0
    ? customToolList.length
    : countKnown
      ? (mcpProfile === "lean" ? Math.min(5, rawToolCount) : totalTools > 0 ? Math.min(totalTools, rawToolCount) : rawToolCount)
      : totalTools > 0 ? totalTools : null;
  const dynamicTools = byKey.get("mcp.dynamic_tools");
  const fewerThanAll = effToolCount !== null && totalTools > 0 && effToolCount < totalTools;
  const effToolMode = toolControlOn
    ? (dynamicTools === "on" ? "custom+dyn" : "custom")
    : (dynamicTools === "on" && fewerThanAll ? "dynamic" : "default");
  const toolWords = effToolCount === null ? NR : totalTools > 0 ? `${effToolCount} of ${totalTools} tools` : `${effToolCount} tools`;
  const claudeLean = byKey.get("host.claude_lean");
  const adaptivePack = byKey.get("budget.adaptive_pack");
  const packTokens = byKey.get("budget.pack_tokens");
  const focusedTokens = byKey.get("budget.focused_tokens");
  const earlyDrop = byKey.get("rerank.early_drop");
  const writerIdle = byKey.get("writer.idle");
  // 0 is the engine's word for unlimited on the quota controls.
  const quota = (v: string | undefined, unit: string) => {
    if (v === undefined) return NR;
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? `${n.toLocaleString()} ${unit}` : n === 0 ? "unlimited" : v;
  };
  const quotaAction = byKey.get("quota.action");

  const packStats = useMemo(() => {
    const rows = (audit.data?.audit ?? []).filter(
      (r) => r.kind === "tool" && r.tool === "nm_context_pack" && r.state !== "running" && typeof r.ms === "number"
    );
    const total = rows.length;
    const withheld = rows.filter((r) => r.withheld === true).length;
    const meanMs = total > 0 ? Math.round(rows.reduce((acc, r) => acc + (r.ms ?? 0), 0) / total) : null;
    const latest = rows[0] ?? null;
    const timedSteps = (latest?.steps ?? []).filter((s) => typeof s.ms === "number" && (s.ms ?? 0) > 0).slice(0, 6);
    return { total, withheld, meanMs, latest, timedSteps };
  }, [audit.data]);

  return (
    <Panel
      title="Homeostasis, graphics card headroom and quota profile"
      right={
        <div className="flex flex-wrap items-center gap-2">
          {controlsLoaded && effToolCount !== null && (
            <Pill tone={effToolCount <= 5 ? "verified" : "signal"}>MCP {toolWords}</Pill>
          )}
          {controlsLoaded && (
            <Pill tone={ceiling.mb !== null && gpuUsedMb > ceiling.mb ? "attention" : "muted"}>
              VRAM {gpuUsedMb} / {ceiling.mb !== null ? `${Math.round(ceiling.mb)} MiB` : "ceiling not reported"}
              {ceiling.auto && ceiling.mb !== null ? " (auto)" : ""}
            </Pill>
          )}
          <LiveBadge />
        </div>
      }
    >
      {!controlsLoaded && controls.error ? (
        <div className="mono text-xs text-error">Controls did not answer: {controls.error}</div>
      ) : !controlsLoaded ? (
        <Placeholder className="h-48" what="settings" />
      ) : (
      // 2026-11-26 (Luis Jake Gabriel III: the composition looked disproportionate): the hardware in
      // one row of three (the budget, the card now, this machine), then the quota footprint and the
      // call telemetry side by side, so no column runs three times the height of the others.
      <div className="@container grid gap-8">
      <div className="grid gap-6 @4xl:grid-cols-3">
        <div className="grid content-start gap-2.5 border-b border-line/60 pb-4 @4xl:border-r @4xl:border-b-0 @4xl:pr-6 @4xl:pb-0">
          <div className="eyebrow flex items-center justify-between">
            <span>Graphics card budget</span>
            <span className={`mono ${gpuPct !== null ? "text-text" : "text-faint"}`}>{gpuPct !== null ? `${gpuPct}%` : "ceiling not reported"}</span>
          </div>
          <div className="h-2 w-full overflow-hidden bg-line/50">
            <div
              className={`h-full transition-all ${ceiling.mb !== null && gpuUsedMb > ceiling.mb ? "bg-attention" : "bg-sense-pale"}`}
              style={{ width: `${gpuPct ?? 0}%` }}
            />
          </div>
          <div className="mono text-xs text-muted">
            {bytes(gpuBytes)} resident on device · ceiling {ceiling.label}
          </div>
          {vram.length > 0 && (
            <ul className="grid gap-1.5" aria-label="Graphics card memory per MicroModel" data-e2e="vram-models">
              {vram.map((m) => (
                <li key={m.id} className="grid gap-0.5" data-model={m.id} data-held={m.held ? "yes" : "no"}>
                  <div className="mono flex items-baseline justify-between gap-3 text-[11px]">
                    <span className={`min-w-0 truncate ${m.held ? "text-text" : "text-faint"}`}>
                      {m.role && m.role !== m.id ? <span>{roleTitle(m.role)} </span> : null}
                      <span className="text-quiet" title={`${m.id}: the upstream open weights its base cartridge was built from`}>○ {m.id}</span>
                    </span>
                    <span className="shrink-0 text-faint">{m.held ? `${mb(m.gpuBytes)} MiB${m.device ? ` · ${m.device}` : ""}${m.precision ? ` · ${m.precision}` : ""}` : "not held"}</span>
                  </div>
                  <div className="h-1 w-full overflow-hidden bg-line/50">
                    <div className="h-full bg-sense-pale" style={{ width: `${Math.min(100, Math.round(m.share * 100))}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
          <div className="mono text-[11px] text-faint">
            {(status?.loaded ?? []).length > 0
              ? (status?.loaded ?? []).map((m) => `${m.id} (${m.device ?? "device not reported"})`).join(" · ")
              : serving && live.reply?.backend_note
                ? `${typeof live.reply.serving === "number" ? `${live.reply.serving} serving ${live.reply.serving === 1 ? "process" : "processes"} · ` : ""}${live.reply.backend_note}`
                : status?.serving || serving
                  ? servingWords(typeof live.reply?.serving === "number" ? live.reply.serving : typeof status?.serving === "number" ? status.serving : null)
                  : "No serving process holds device memory right now"}
          </div>
        </div>

        <div className="grid content-start gap-2 border-b border-line/60 pb-4 @4xl:border-r @4xl:border-b-0 @4xl:pr-6 @4xl:pb-0" data-e2e="vram-live">
            <div className="eyebrow">The card now, by MicroModel</div>
            <VramBar vram={live.reply?.vram ?? null} backend={live.reply?.backend ?? null} capMb={vramCapMb && vramCapMb > 0 ? vramCapMb : null} serving={serving} />
            <div className="mono flex items-center gap-2 text-[10.5px]" data-e2e="vram-residency" data-mode={seat?.mode ?? "unknown"}>
              <i aria-hidden className="inline-block h-1.5 w-1.5 shrink-0" style={{ background: seat?.mode === "seated" ? "var(--verified)" : seat?.mode === "swapping" ? "var(--learned)" : "var(--text-4)" }} />
              <span className={seat ? "text-muted" : "text-faint"}>{seat ? seat.words : serving === false ? "No engine serving: no MicroModel is seated" : live.reply ? "The live state does not say how the MicroModels sit" : "Reading the live state"}</span>
            </div>
          </div>

        <div className="grid content-start gap-2">
            <div className="eyebrow">This machine</div>
            <Capacity cap={machine.cap} gpuControl={byKey.get("budget.gpu_mb_max") ?? null} backend={live.reply?.backend ?? null} serving={serving} />
            <DriveStorageSummary />
          </div>
      </div>

      <div className="grid gap-6 border-t border-line pt-6 @3xl:grid-cols-2">
        <div className="grid content-start gap-2 border-b border-line/60 pb-4 @3xl:border-r @3xl:border-b-0 @3xl:pr-6 @3xl:pb-0">
          <div className="eyebrow">Quota and prompt footprint</div>
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-xs">
            <span className="text-faint">MCP tool list</span>
            <span className="mono text-right text-text">
              {effToolCount === null ? NR : `${toolWords} (${effToolMode})`}
            </span>
            <span className="text-faint">Claude Code Artifact</span>
            <span className="mono text-right text-text">{claudeLean === undefined ? NR : claudeLean === "on" ? "disabled (lean)" : "host default"}</span>
            <span className="text-faint">Pack token ceiling</span>
            <span className="mono text-right text-text">
              {adaptivePack === "on"
                ? `${focusedTokens ?? NR} focused / ${packTokens ?? NR} full`
                : adaptivePack === "off"
                  ? `${packTokens ?? NR} tokens fixed`
                  : packTokens !== undefined ? `${packTokens} full` : NR}
            </span>
            <span className="text-faint">Reranker early drop</span>
            <span className="mono text-right text-text">{earlyDrop ?? NR}</span>
            <span className="text-faint">Idle writer</span>
            <span className="mono text-right text-text">{writerIdle ?? NR}</span>
            <span className="text-faint">Quotas (session / 24h)</span>
            <span className="mono text-right text-text">
              {quota(byKey.get("quota.session_tokens"), "tok")} / {quota(byKey.get("quota.daily_tokens"), "tok")}
            </span>
            <span className="text-faint">Pack call quota · action</span>
            <span className="mono text-right text-text">
              {quota(byKey.get("quota.session_packs"), "packs")} · {quotaAction ?? NR}
            </span>
          </div>
        </div>

        <div className="grid content-start gap-2">
          <div className="eyebrow flex items-center justify-between">
            <span>Context pack telemetry</span>
            <span className="mono text-faint">{packStats.total} calls</span>
          </div>
          {packStats.total === 0 ? (
            <div className="text-xs text-faint">No context packs recorded in Audit yet.</div>
          ) : (
            <>
              <div className="mono text-xs text-muted">
                last {packStats.latest?.ms} ms · mean {packStats.meanMs} ms · {packStats.withheld} withheld
              </div>
              {packStats.timedSteps.length > 0 ? (
                <div className="grid gap-1 pt-1">
                  {packStats.timedSteps.map((st, idx) => (
                    <div key={`${st.part}-${idx}`} className="mono flex items-center justify-between text-[11px]">
                      <span className="truncate text-faint">{st.part}</span>
                      <span className="shrink-0 text-text">{st.ms} ms</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mono text-[11px] text-faint">
                  {packStats.latest?.mode ? `mode ${packStats.latest.mode} · ` : ""}
                  {Array.isArray(packStats.latest?.parts) ? `${packStats.latest.parts.length} parts ran` : "parts not reported"}
                </div>
              )}
            </>
          )}
        </div>
      </div>
      </div>
      )}
    </Panel>
  );
}

// 2026-11-25 (directions of Luis Jake Gabriel III: flow and visual hierarchy): one view per group
// under a section bar instead of one 11,000-pixel scroll; the view chosen last is kept in this
// browser. 2026-11-26: the design files moved to the rail, so the bar holds only the controls.
const VIEWS = [
  { id: "machine", label: "Machine", about: "The card, its budget and the hardware limits." },
  { id: "projects", label: "Projects", about: "Create custom projects, rename workspaces, inspect storage, and safely forget old projects." },
  { id: "tools", label: "Agents & tools", about: "Connect your coding agents, choose which MCP tools they see, and what a turn costs." },
  { id: "controls", label: "Controls", about: "Every switch and number the engine reports, by group. Each change is logged in Audit, where it can be undone." },
  { id: "privacy", label: "Privacy & network", about: "Air gap, file exclusions (.nanoignore), local isolation and paired devices." },
  { id: "speed", label: "Stage times", about: "Where the time of a question goes." },
] as const;
type View = (typeof VIEWS)[number]["id"];
const VIEW_KEY = "nano.settings.view";

/** The air gap and the devices, in words, for the rail. */
function PrivacyGuide() {
  const { data } = useApi<AirGapReply>("/api/airgap", 15000);
  return (
    <>
      <RailSection title="Air gap">
        <p className="read text-[11px] leading-relaxed text-muted">
          Air-gap mode is a strict local-only lock across both the interface and MCP (<span className="mono">nm_settings</span> <span className="mono">op=airgap</span>): local network access and tablet pairing are forced off, remote URLs and imports are refused, and all weights load from the local folder.
        </p>
        {data && (
          <dl className="mono mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[10.5px]">
            <dt className="text-faint">weights from</dt>
            <dd className="min-w-0 break-all text-right text-muted">
              <Masked id="settings:models-local" shown={maskPath(data.models_local_dir)} full={data.models_local_dir} />
            </dd>
            <dt className="text-faint">set by</dt>
            <dd className="text-right text-muted">{data.locked_by_env ? "NANO_AIR_GAP (locked)" : data.source}</dd>
            <dt className="text-faint">network</dt>
            <dd className="text-right text-muted">{data.network}</dd>
          </dl>
        )}
      </RailSection>
      <RailSection title="Paired devices">
        <ul className="read grid gap-1.5 text-[11px] text-faint">
          <li>Tablets on your local network pair with a code made here; a code works once, until it expires, and 5 wrong codes lock pairing for a while.</li>
          <li>Local network access is the network control (Controls): off, view or control. A change needs the interface server restarted.</li>
          <li>There is no encryption on the wire: anyone on your local network can read traffic and device tokens.</li>
        </ul>
      </RailSection>
      <RailSection title="File exclusions (.nanoignore)">
        <p className="read text-[11.5px] leading-relaxed text-muted">
          Excludes matching files and folders (root and nested <span className="mono">.nanoignore</span>, gitignore syntax) from the Global Context Substrate, substrate cache, dense vectors, idle Writer world model, fresh recall, build-help source snippets, hooks, and primes. Every edit here is logged in Audit with one-click Undo.
        </p>
        <p className="read mt-2 text-[11px] leading-relaxed text-faint">
          Scope boundary: <span className="mono">.nanoignore</span> governs what Neuromorph Nano reads and puts into Context Packets or primes; pair with your coding agent&apos;s own ignore and permission rules to block direct file reads by the agent.
        </p>
      </RailSection>
    </>
  );
}

/** One thing the centre shows, and where else it is read or set. */
type Surface = { what: string; api?: string[]; mcp?: string[]; cli?: string[]; keys?: string[]; note?: string };

/** The command line form of an `nm_settings` op: the same tool the agent calls, run once. */
const callSettings = (args: string) => `neuromorph-nano call nm_settings "${args}" --root <repo>`;

/** Where else each thing of a section is read and set: the API route, the MCP op, the command
 *  line and, for a switch that is a control, its key. The centre holds the controls themselves. */
const SURFACES: Record<View, Surface[]> = {
  machine: [
    { what: "Hardware limits (read)", api: ["GET /api/limits"], mcp: ["nm_settings op=limits", "nm_settings op=recommend (the recommendation alone)"], cli: [callSettings('{\\"op\\":\\"limits\\"}')] },
    {
      what: "Set one limit",
      api: ['POST /api/limits {"key": "vram_mb" | "ram_mb" | "cpu_threads" | "content_mb" | "caches_mb" | "weights_mb", "value": "auto" | number}'],
      mcp: ["nm_settings op=set_limit (key, value)"],
      cli: [callSettings('{\\"op\\":\\"set_limit\\",\\"key\\":\\"vram_mb\\",\\"value\\":\\"auto\\"}')],
      keys: ["limits.vram_mb", "limits.ram_mb", "limits.cpu_threads", "limits.memory.content_mb", "limits.memory.caches_mb", "limits.memory.weights_mb"],
    },
    { what: "Apply the recommended limits", api: ['POST /api/limits/apply-recommended {"confirm": true}'], mcp: ["nm_settings op=apply_recommended (confirm: true)"] },
    {
      what: "Card memory, residency and the budget",
      api: ["GET /api/status", "GET /api/live"],
      mcp: ["nm_settings op=status", "nm_live"],
      keys: ["budget.gpu_mb_max", "residency.idle_release"],
      note: "The meters are readings of the serving engine. The budget and the idle release are controls, set in Controls.",
    },
  ],
  projects: [
    { what: "List registered projects", api: ["GET /api/projects"], mcp: ["nm_settings op=projects"] },
    {
      what: "Create a custom or folder-backed project",
      api: ['POST /api/projects {"action": "create", "name", "root"?, "description"?}'],
      mcp: ["nm_settings op=project_create (name, root?, description?)"],
      note: "Omit root to create a custom workspace project without a folder. Audited with Undo.",
    },
    {
      what: "Rename a project",
      api: ['POST /api/projects {"action": "rename", "project", "name", "description"?}'],
      mcp: ["nm_settings op=project_rename (project, name, description?)"],
    },
    {
      what: "Forget a project's memory",
      api: ['POST /api/projects/forget {"project", "confirm_token"}'],
      mcp: ["nm_storage op=forget_project (project, confirm_token)"],
      note: "Two-step confirmation; never touches the repository's own files.",
    },
  ],
  tools: [
    {
      what: "Connection status of each coding agent",
      api: ["GET /api/connect?project=<folder>"],
      mcp: ["nm_settings op=connect (project)"],
      cli: ["neuromorph-nano connect <claude|antigravity|cursor|vscode|windsurf|generic> [--project <repo>] [--dry-run] [--remove]"],
      note: "Connecting writes the client's own settings file, so it is done from the command line; the interface and MCP read the status.",
    },
    {
      what: "The MCP tool list",
      api: ['POST /api/controls {"key", "value"}'],
      mcp: ["nm_settings op=set (key, value)"],
      cli: [callSettings('{\\"op\\":\\"set\\",\\"key\\":\\"mcp.tools\\",\\"value\\":\\"lean\\"}')],
      keys: ["mcp.tools", "mcp.tool_count", "mcp.tool_control", "mcp.enabled_tools", "mcp.dynamic_tools", "host.claude_lean"],
      note: "Applies when a client next reads the tool list.",
    },
  ],
  controls: [
    { what: "Read every control", api: ["GET /api/controls"], mcp: ["nm_settings op=view"], cli: [callSettings('{\\"op\\":\\"view\\"}')] },
    {
      what: "Set one control",
      api: ['POST /api/controls {"key", "value"}'],
      mcp: ["nm_settings op=set (key, value)"],
      cli: [callSettings('{\\"op\\":\\"set\\",\\"key\\":\\"<key>\\",\\"value\\":\\"<value>\\"}')],
      note: "The keys, values and defaults are listed under Control reference above.",
    },
    { what: "Undo a change", api: ['POST /api/undo {"id": "<audit id>"}'], mcp: ["nm_settings op=undo (id)"], note: "The audit id is in the reply of the change and on the Audit page." },
    { what: "Cartridge stack of a use", api: ['POST /api/cartridges {"library": "<use>", "stack": "<stack id>" | null}'], mcp: ["nm_cartridges op=use (use, stack)"] },
    {
      what: "The settings file, key by key",
      api: ["GET /api/controls/saved"],
      mcp: ["nm_settings op=saved"],
      cli: [callSettings('{\\"op\\":\\"saved\\"}')],
      keys: ["base_cartridges:<model id>", "base_compensation:<model id>", "role_models:<role>"],
      note: "An entry of these maps is set like a control, with the key forms above.",
    },
  ],
  privacy: [
    { what: "Air gap", api: ["GET /api/airgap"], mcp: ["nm_settings op=airgap"], keys: ["air_gap"], note: "The environment variable NANO_AIR_GAP=on locks it on; the switch is then read only." },
    {
      what: "File exclusions (.nanoignore)",
      api: ["GET /api/nanoignore?project=<key>[&test=<path>]", 'POST /api/nanoignore {"project", "action": "set" | "add" | "remove", "content" | "pattern"}'],
      mcp: ["nm_settings op=nanoignore | nanoignore_check | nanoignore_set | nanoignore_add | nanoignore_remove"],
      note: "Writes atomically, immediately invalidates substrate-cache-*.bin, and is undoable from Audit.",
    },
    { what: "Local network access", api: ['POST /api/controls {"key": "network", "value": "off" | "view" | "control"}'], mcp: ["nm_settings op=set"], keys: ["network"] },
    { what: "Paired devices (read)", api: ["GET /api/devices"], mcp: ["nm_settings op=devices"] },
    { what: "Make a pairing code", api: ["POST /api/pair/code (this computer's interface only)"], cli: ["neuromorph-nano pair"], note: "Not in MCP: a pairing code is made by the person at the computer." },
    { what: "Revoke a device", api: ['POST /api/devices/revoke {"id"}'], mcp: ["nm_settings op=device_revoke (id, confirm: true)"] },
  ],
  speed: [
    { what: "Stage times of a warm question", api: ["GET /api/stage-times[?file=STAGE-BENCH-<stamp>.txt]"], mcp: ["nm_settings op=stage_times (file)"], cli: ["neuromorph-nano lab stage-bench (writes the report the table reads)"] },
    { what: "The newest question, live", api: ["GET /api/live", "GET /api/live/stream (events)"], mcp: ["nm_live"] },
  ],
};

/** The surfaces of one section as a reference list. */
function SurfacesGuide({ view }: { view: View }) {
  const row = (label: string, lines?: string[]) =>
    lines && lines.length > 0 ? (
      <div className="grid grid-cols-[2.6rem_minmax(0,1fr)] gap-x-2">
        <dt className="mono text-[10px] uppercase tracking-[0.1em] text-quiet">{label}</dt>
        <dd className="mono grid min-w-0 gap-0.5 text-[10.5px] break-words text-muted">
          {lines.map((l) => (
            <span key={l}>{l}</span>
          ))}
        </dd>
      </div>
    ) : null;
  return (
    <RailSection title="Also read and set from">
      <ul className="grid gap-3" data-e2e="settings-surfaces" data-view={view}>
        {SURFACES[view].map((s) => (
          <li key={s.what} className="grid gap-1">
            <div className="read text-[11px] text-text">{s.what}</div>
            <dl className="grid gap-1">
              {row("API", s.api)}
              {row("MCP", s.mcp)}
              {row("CLI", s.cli)}
              {row("key", s.keys)}
            </dl>
            {s.note && <p className="read text-[11px] leading-relaxed text-faint">{s.note}</p>}
          </li>
        ))}
      </ul>
    </RailSection>
  );
}

/** What a control takes, in one line: its range and unit, or its values. */
function controlTakes(c: Control): string {
  const presets = (c.values ?? []).join(", ");
  if (c.kind === "integer" && typeof c.min === "number" && typeof c.max === "number") {
    const auto = (c.values ?? []).includes("auto") || c.default === "auto";
    return `${c.min} to ${c.max}${c.unit ? ` ${c.unit}` : ""}${typeof c.step === "number" && c.step > 1 ? `, step ${c.step}` : ""}${auto ? ", or auto" : ""}${c.key === "limits.vram_mb" ? ", or 0 (processor only)" : ""}`;
  }
  if (c.all_tools) return `auto, or a comma-separated list of tool names (${c.all_tools.length} tools)`;
  return presets;
}

/** Every control the engine reports, as a reference beside the control surface: its key, what it
 *  does, the values it takes, its default, when a change applies and where its value comes from.
 *  Read from the same reply as the centre, so the two never differ. */
function ControlReference() {
  const { data, error } = useApi<Control[]>("/api/controls", 15000);
  const [q, setQ] = useState("");
  // The groups opened by hand; a filter opens every group that holds a match.
  const [opened, setOpened] = useState<string[]>([]);
  const list = useMemo(() => (Array.isArray(data) ? data : []), [data]);
  const shown = useMemo(() => {
    const f = q.trim().toLowerCase();
    return f ? list.filter((c) => c.key.toLowerCase().includes(f) || c.label.toLowerCase().includes(f) || (c.help ?? "").toLowerCase().includes(f)) : list;
  }, [list, q]);
  const groups = useMemo(() => {
    const m = new Map<string, Control[]>();
    for (const c of shown) m.set(controlGroupName(c), [...(m.get(controlGroupName(c)) ?? []), c]);
    const order = [...(CONTROL_GROUP_ORDER as readonly string[]), ...[...m.keys()].filter((g) => !(CONTROL_GROUP_ORDER as readonly string[]).includes(g))];
    return order.filter((g) => m.has(g)).map((g) => [g, m.get(g) ?? []] as const);
  }, [shown]);
  return (
    <RailSection title={`Control reference${list.length ? ` · ${list.length}` : ""}`}>
      {error && !data && <p className="mono text-[11px] text-faint">Controls did not answer ({error}).</p>}
      {data && (
        <div className="grid gap-3" data-e2e="control-reference" data-count={list.length}>
          <label className="relative block">
            <span className="sr-only">Filter the control reference</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by key, name or text" className="mono h-9 w-full border-b border-line-strong bg-transparent text-xs outline-none placeholder:text-faint focus-visible:border-sense-pale" />
          </label>
          {groups.map(([g, cs]) => {
            const open = q.trim() !== "" || opened.includes(g);
            return (
            <div key={g} data-e2e="control-ref-group" data-group={g} data-open={open ? "true" : "false"}>
              <button
                type="button"
                aria-expanded={open}
                onClick={() => setOpened((o) => (o.includes(g) ? o.filter((x) => x !== g) : [...o, g]))}
                className="press mono flex min-h-8 w-full items-center justify-between gap-2 text-left text-[10.5px] uppercase tracking-[0.12em] text-muted hover:text-text"
              >
                <span>{g}</span>
                <span className="text-quiet">{cs.length}</span>
              </button>
              {open && (
              <ul className="grid gap-2.5 border-l border-line pt-1 pb-2 pl-3">
                {cs.map((c) => (
                  <li key={c.key} className="grid gap-0.5" data-e2e="control-ref" data-key={c.key}>
                    <div className="mono flex flex-wrap items-baseline justify-between gap-x-2 text-[11px]">
                      <span className="break-all text-text">{c.key}</span>
                      <span className="text-quiet">{c.source === "default" ? "default" : `set: ${c.value}`}</span>
                    </div>
                    <div className="read text-[11px] leading-relaxed text-muted">{c.label}</div>
                    {c.help && <div className="read text-[11px] leading-relaxed text-faint">{c.help}</div>}
                    <dl className="mono grid grid-cols-[3.4rem_minmax(0,1fr)] gap-x-2 text-[10.5px] text-faint">
                      <dt className="text-quiet">takes</dt>
                      <dd className="min-w-0 break-words">{controlTakes(c)}</dd>
                      <dt className="text-quiet">default</dt>
                      <dd>{c.default}</dd>
                      <dt className="text-quiet">applies</dt>
                      <dd>{c.applies}</dd>
                      {!c.wired && (
                        <>
                          <dt className="text-quiet">state</dt>
                          <dd className="text-attention">saved, not read by the engine yet</dd>
                        </>
                      )}
                    </dl>
                  </li>
                ))}
              </ul>
              )}
            </div>
            );
          })}
          {groups.length === 0 && <p className="mono text-[11px] text-faint">No control matches.</p>}
        </div>
      )}
    </RailSection>
  );
}

/** The rail's Guide: what the section shown is, and the reference for its controls. */
function SettingsGuide({ view }: { view: View }) {
  const v = VIEWS.find((x) => x.id === view);
  return (
    <div className="grid gap-0">
      <RailSection title={v?.label ?? "Settings"}>
        <p className="read text-[11px] leading-relaxed text-muted">{v?.about}</p>
      </RailSection>
      {view === "machine" && (
        <>
          <LimitsGuide />
          <RailSection title="Graphics card budget">
            <p className="read text-[11px] leading-relaxed text-faint">The budget (budget.gpu_mb_max) is auto by default, following this card, or a number up to 81,920 MiB. Auto resolves to the ceiling the engine measures; until it reports one, the meters are drawn against what is held.</p>
          </RailSection>
        </>
      )}
      {view === "projects" && <ProjectsGuide />}
      {view === "tools" && (
        <>
          <RailSection title="Connecting">
            <ConnectGuide />
          </RailSection>
          <LeanModeGuide />
        </>
      )}
      {view === "controls" && (
        <RailSection title="Undo">
          <p className="read text-[11px] leading-relaxed text-faint">
            Every change here is logged in{" "}
            <Link href="/audit/" className="link">
              Audit
            </Link>
            , where it can be undone. A control this engine build does not report is named as absent, never drawn.
          </p>
        </RailSection>
      )}
      {view === "privacy" && <PrivacyGuide />}
      {view === "speed" && (
        <RailSection title="Reading it">
          <ul className="read grid gap-1.5 text-[11px] text-faint">
            <li>The waterfall is the newest question, from the engine&apos;s own stage events, a running stage growing as it runs.</li>
            <li>The table is the median and p95 ms of every stage of a warm question, from the newest stage bench report (nano-lab stage-bench).</li>
            <li>Stages overlap in part, so they need not add up to the total.</li>
          </ul>
        </RailSection>
      )}
      {view === "controls" && <ControlReference />}
      <SurfacesGuide view={view} />
    </div>
  );
}

/** The design files the engine reads, as a library: search, the list, and the picked file. */
function DesignFiles({ data, error, root, sel, onPick }: { data: Settings | null; error: string | null; root?: string; sel: string | null; onPick: (n: string) => void }) {
  const [q, setQ] = useState("");
  const names = useMemo(() => Object.keys(data ?? {}).filter((n) => n.toLowerCase().includes(q.toLowerCase())), [data, q]);
  const shown = sel && data?.[sel] ? sel : (names[0] ?? null);
  const current = shown && data ? data[shown] : null;
  if (error && !data) return <Empty title="Design files did not answer">{error}</Empty>;
  if (!data) return <Placeholder className="h-40" what="design files" />;
  return (
    <div className="grid gap-3" data-e2e="design-files">
      <label className="relative block">
        <span className="sr-only">Search design files</span>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" className="mono h-9 w-full border-b border-line-strong bg-transparent text-xs outline-none placeholder:text-faint focus-visible:border-sense-pale" />
      </label>
      <ul className="max-h-[32vh] overflow-auto border-b border-line">
        {names.map((n) => (
          <li key={n}>
            <button type="button" onClick={() => onPick(n)} aria-current={shown === n ? "true" : undefined} title={summary(data[n])} className={`press relative flex w-full items-baseline justify-between gap-3 border-b border-line/50 py-1.5 pl-3 text-left ${shown === n ? "" : "hover:text-text"}`}>
              {shown === n && <span aria-hidden className="absolute top-1.5 bottom-1.5 left-0 w-[2px] bg-sense-pale" />}
              <span className={`mono truncate text-[12px] ${shown === n ? "text-text" : "text-muted"}`}>{n}</span>
              <span className="mono shrink-0 text-[10px] text-quiet">{Object.keys(data[n]).length}</span>
            </button>
          </li>
        ))}
        {names.length === 0 && <li className="mono py-2 text-[11px] text-faint">No design file matches.</li>}
      </ul>
      {current && shown && (
        <div className="grid gap-1.5">
          <div className="mono flex items-baseline justify-between gap-3 text-[11px]">
            <span className="text-text">designs/{shown}.json</span>
            <span className="text-quiet" title="Read by the engine; change them in the files themselves">read only</span>
          </div>
          <p className="read text-[11px] text-faint">{summary(current)}</p>
          <p className="mono text-[10px] leading-relaxed text-quiet">Also read from: API GET /api/settings · MCP nm_settings op=designs. The engine reads these files at start; they are changed in the files themselves.</p>
          <pre key={shown} className="mono fade-in max-h-[50vh] overflow-auto border-l border-line pl-3 text-[10.5px] leading-relaxed text-muted">
            {maskPathsIn(JSON.stringify(current, null, 2), root)}
          </pre>
        </div>
      )}
    </div>
  );
}

/** Opens the design files in the rail when the address names one (or `#designs`), once per name. */
function DesignDeepLink({ target }: { target: string | null }) {
  const rail = useRail();
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!target || done.current === target) return;
    done.current = target;
    rail.open("designs");
  }, [target, rail]);
  return null;
}

export default function SettingsPage() {
  const { data, error } = useApi<Settings>("/api/settings");
  const status = useApi<Status>("/api/status", 15000);
  const { key: activeProjectKey, projects: activeProjects } = useProjectKey();
  const projectsQ = useApi<Projects>("/api/projects", 30000);
  const airgap = useApi<AirGapReply>("/api/airgap", 15000);
  const controls = useApi<Control[]>("/api/controls", 15000);
  // Active global project's folder, else the project served last, else the Nano's own folder.
  const allProjectsList = activeProjects?.projects ?? projectsQ.data?.projects ?? [];
  const activeProjectRoot = allProjectsList.find((p) => p.key === activeProjectKey)?.root ?? null;
  const lastProject = activeProjectRoot ?? [...allProjectsList].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0))[0]?.root ?? status.data?.root ?? null;
  // 2026-11-26: a section name in the address opens that section (`/settings/#privacy`); any other
  // name in the address is a design file, which opens in the rail's design files.
  const section = useSection(VIEWS, VIEW_KEY);
  const inAddress = section.inAddress;
  const designInAddress = inAddress && !VIEWS.some((v) => v.id === inAddress) && inAddress !== "designs" ? inAddress : null;
  const [picked, setPicked] = useState<string | null>(null);
  const sel = designInAddress ?? picked;
  const view: View = section.id;
  const a = airgap.data;
  const nControls = Array.isArray(controls.data) ? controls.data.length : null;
  return (
    <Workspace
      title="Settings"
      railKey="Settings"
      defaultTab="guide"
      status={
        <>
          {a && <span className={a.air_gap ? "text-verified" : a.loopback_only ? undefined : "text-attention"}>{a.air_gap ? (a.locked_by_env ? "air gap locked" : "air gap on") : a.loopback_only ? "local only" : `lan ${a.network}`}</span>}
          {nControls !== null && <span>{nControls} controls</span>}
          <Link href="/audit/" className="link normal-case tracking-normal" title="Every change is logged in Audit, where it can be undone">
            undo in Audit
          </Link>
        </>
      }
      rail={[
        { id: "guide", label: "Guide", body: <SettingsGuide view={view} /> },
        {
          id: "designs",
          label: "Design files",
          count: data ? Object.keys(data).length : undefined,
          body: (
            <DesignFiles
              data={data}
              error={error}
              root={status.data?.root}
              sel={sel}
              onPick={(n) => {
                setPicked(n);
                setHash(`#${n}`, false);
              }}
            />
          ),
        },
      ]}
    >
      <DesignDeepLink target={designInAddress ?? (inAddress === "designs" ? "designs" : null)} />
      <SectionBar sections={VIEWS} value={view} onPick={(v) => section.pick(v)} label="Settings sections" />
      {view === "machine" && (
        <div className="grid gap-10">
          <HardwareLimitsPanel />
          <HomeostasisPanel status={status.data} />
        </div>
      )}
      {view === "projects" && <ProjectsPanel root={status.data?.root} />}
      {view === "tools" && (
        <div className="grid gap-10">
          <ConnectPanel project={lastProject} />
          <LeanModePanel />
        </div>
      )}
      {view === "controls" && <Controls />}
      {view === "privacy" && (
        <div className="grid gap-10">
          <AirGapPanel />
          <NanoIgnorePanel projects={projectsQ.data} />
          <DevicesPanel />
        </div>
      )}
      {view === "speed" && <StageTimesPanel />}
    </Workspace>
  );
}
