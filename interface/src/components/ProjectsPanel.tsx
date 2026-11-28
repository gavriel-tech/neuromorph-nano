"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import { Empty, Masked, Panel, Pill, Placeholder, Switch } from "@/components/ui";
import { RailSection } from "@/components/Workspace";
import { ago, bytes, post, projectLabel, refreshAll, useApi, type PrimesReply, type Project, type Projects } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { pickProject, useProjectKey } from "@/lib/evidence";
import { maskPath } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };

type NanoIgnoreMini = {
  exists?: boolean;
  rules?: string[];
  error?: string;
};

/** Live per-project controls shown inside an expanded project card's Settings menu. */
function ProjectLiveControls({
  project,
  onNote,
}: {
  project: Project;
  onNote: (n: Note) => void;
}) {
  const primesQ = useApi<PrimesReply>(`/api/primes?project=${encodeURIComponent(project.key)}`, 10000);
  const ignoreQ = useApi<NanoIgnoreMini>(
    project.root ? `/api/nanoignore?project=${encodeURIComponent(project.key)}` : null,
    15000,
  );
  const [quickPat, setQuickPat] = useState("");
  const [busy, setBusy] = useState(false);

  const builtins = primesQ.data?.builtins ?? [];
  const custom = primesQ.data?.custom ?? [];
  const activeBuiltins = builtins.filter((b) => b.enabled).length;
  const activeCustom = custom.filter((c) => c.enabled).length;
  const activeTotal = activeBuiltins + activeCustom;
  const ignoreRules = ignoreQ.data?.rules ?? [];

  const toggleBuiltin = async (id: string, enabled: boolean) => {
    setBusy(true);
    try {
      await post("/api/primes", { action: "toggle_builtin", project: project.key, id, enabled });
      onNote({ text: `Prime "${id}" switched ${enabled ? "ON" : "OFF"} for ${projectLabel(project)}.`, tone: "verified" });
      refreshAll("/api/primes");
      refreshAll("/api/audit");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const turnProjectPrimesOff = async () => {
    setBusy(true);
    try {
      await post("/api/primes", { action: "all_off", project: project.key });
      onNote({ text: `Switched all primes off for ${projectLabel(project)}.`, tone: "verified" });
      refreshAll("/api/primes");
      refreshAll("/api/audit");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const addIgnorePattern = async () => {
    const pat = quickPat.trim();
    if (!pat) return;
    setBusy(true);
    try {
      await post("/api/nanoignore", { project: project.key, action: "add", pattern: pat });
      setQuickPat("");
      onNote({ text: `Added "${pat}" to .nanoignore for ${projectLabel(project)}.`, tone: "verified" });
      refreshAll("/api/nanoignore");
      refreshAll("/api/audit");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {/* Per-project Primes control */}
      <div className="grid content-start gap-2.5 border border-line bg-bg-0/60 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/60 pb-2">
          <div className="flex items-center gap-2">
            <span className="mono text-[10.5px] uppercase tracking-[0.12em] text-text">Project primes</span>
            <Pill tone={activeTotal > 0 ? "verified" : "muted"}>{activeTotal} on</Pill>
          </div>
          <div className="flex items-center gap-3">
            {activeTotal > 0 && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void turnProjectPrimesOff()}
                className="press mono border-b border-line-strong text-[9.5px] uppercase tracking-[0.1em] text-faint hover:border-error hover:text-error disabled:opacity-40"
              >
                All off
              </button>
            )}
            <Link
              href="/audit/#primes"
              onClick={() => pickProject(project.key)}
              data-e2e="project-settings-primes"
              className="mono text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
            >
              Studio &rarr;
            </Link>
          </div>
        </div>
        {!primesQ.data ? (
          <span className="mono text-[10.5px] text-faint">Loading project primes...</span>
        ) : (
          <ul className="grid gap-1.5">
            {builtins.slice(0, 4).map((b) => (
              <li key={b.id} className="flex items-center justify-between gap-2 text-[11px]">
                <span className="mono min-w-0 truncate text-muted">{b.label}</span>
                <Switch
                  on={b.enabled}
                  disabled={busy}
                  label={`Toggle ${b.label} for ${projectLabel(project)}`}
                  onChange={(next) => void toggleBuiltin(b.id, next)}
                />
              </li>
            ))}
            {custom.length > 0 && (
              <li className="mono pt-0.5 text-[10px] text-faint">
                + {custom.length} custom {custom.length === 1 ? "prime" : "primes"} ({activeCustom} on)
              </li>
            )}
          </ul>
        )}
      </div>

      {/* Per-project .nanoignore & workspace navigation */}
      <div className="grid content-start gap-2.5 border border-line bg-bg-0/60 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/60 pb-2">
          <div className="flex items-center gap-2">
            <span className="mono text-[10.5px] uppercase tracking-[0.12em] text-text">.nanoignore</span>
            {project.root ? (
              <Pill tone={ignoreRules.length > 0 ? "verified" : "muted"}>
                {ignoreRules.length} {ignoreRules.length === 1 ? "rule" : "rules"}
              </Pill>
            ) : (
              <Pill tone="muted">no folder</Pill>
            )}
          </div>
          {project.root && (
            <Link
              href="/settings/#privacy"
              onClick={() => pickProject(project.key)}
              data-e2e="project-settings-nanoignore"
              className="mono text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
            >
              Full editor &rarr;
            </Link>
          )}
        </div>

        {project.root ? (
          <div className="grid gap-2">
            <div className="flex items-center gap-2">
              <input
                value={quickPat}
                onChange={(e) => setQuickPat(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void addIgnorePattern();
                  }
                }}
                placeholder="Quick exclude: *.pem, secrets/, .env*"
                className="mono h-8 min-w-0 flex-1 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <button
                type="button"
                disabled={busy || !quickPat.trim()}
                onClick={() => void addIgnorePattern()}
                className="press mono shrink-0 border-b border-sense-pale px-1.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text disabled:opacity-40"
              >
                + Add
              </button>
            </div>
            {ignoreRules.length > 0 && (
              <div className="mono truncate text-[10px] text-faint">
                Active: {ignoreRules.slice(0, 5).join(", ")}
                {ignoreRules.length > 5 ? ` (+${ignoreRules.length - 5} more)` : ""}
              </div>
            )}
          </div>
        ) : (
          <p className="read text-[11px] text-faint">Custom folderless workspace. File exclusions apply to folder-backed repositories.</p>
        )}
      </div>
    </div>
  );
}

export default function ProjectsPanel({ root }: { root?: string | null }) {
  const q = useApi<Projects>("/api/projects", 15000);
  const { key: activeKey } = useProjectKey();
  const now = useNow();
  const [note, setNote] = useState<Note | null>(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newRoot, setNewRoot] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [settingsOpen, setSettingsOpen] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [busy, setBusy] = useState(false);

  const list = useMemo(() => {
    const raw = q.data?.projects ?? [];
    return [...raw].sort((a, b) => {
      if (a.key === activeKey && b.key !== activeKey) return -1;
      if (b.key === activeKey && a.key !== activeKey) return 1;
      if (Boolean(a.serving) !== Boolean(b.serving)) return a.serving ? -1 : 1;
      return (b.last_ms ?? 0) - (a.last_ms ?? 0);
    });
  }, [q.data, activeKey]);

  const activeModalProject = list.find((p) => p.key === settingsOpen) ?? null;

  useEffect(() => {
    if (!settingsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSettingsOpen(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  const openSettingsModal = (p: Project) => {
    setSettingsOpen(p.key);
    setEditName(projectLabel(p));
    setEditDesc(p.description ?? "");
    setNote(null);
  };

  const submitCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const body: Record<string, unknown> = { action: "create", name };
      if (newRoot.trim()) body.root = newRoot.trim();
      if (newDesc.trim()) body.description = newDesc.trim();
      const res = await post<{ key?: string; project?: string }>("/api/projects", body);
      const createdKey = res.key ?? res.project ?? null;
      if (createdKey) pickProject(createdKey);
      setNewName("");
      setNewRoot("");
      setNewDesc("");
      setCreating(false);
      setNote({
        text: `Created project "${name}" (${createdKey ?? "saved"}) and set it as the active project. Logged in Audit with Undo.`,
        tone: "verified",
      });
      refreshAll("/api/projects");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const submitRename = async (key: string) => {
    const name = editName.trim();
    if (!name) return;
    setBusy(true);
    try {
      await post("/api/projects", {
        action: "rename",
        project: key,
        name,
        description: editDesc.trim(),
      });
      setNote({ text: `Saved settings for "${name}". Logged in Audit with Undo.`, tone: "verified" });
      refreshAll("/api/projects");
      refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const scratchCount = list.filter((p) => p.is_scratch || p.root_exists === false).length;

  return (
    <Panel
      title="Projects"
      right={
        <div className="flex flex-wrap items-center gap-2">
          {q.data && (
            <Pill tone={scratchCount > 0 ? "attention" : "muted"}>
              {list.length} {list.length === 1 ? "project" : "projects"}
            </Pill>
          )}
          <button
            type="button"
            onClick={() => {
              setCreating((c) => !c);
              setNote(null);
            }}
            data-e2e="projects-new-toggle"
            className="press mono min-h-8 border-b border-sense-pale px-1.5 text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
          >
            {creating ? "Close form" : "+ Add project"}
          </button>
        </div>
      }
    >
      <NoteLine note={note} />
      <div className="grid gap-5" data-e2e="projects-panel">
        {creating && (
          <form
            className="grid gap-3 border border-sense-pale/50 bg-bg-1/50 p-4"
            data-e2e="projects-create-form"
            onSubmit={(e) => {
              e.preventDefault();
              void submitCreate();
            }}
          >
            <div className="eyebrow">Add a new project</div>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">display name</span>
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Payments API or Research Notes"
                  maxLength={80}
                  required
                  className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                />
              </label>
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">folder path (optional)</span>
                <input
                  value={newRoot}
                  onChange={(e) => setNewRoot(e.target.value)}
                  placeholder="Leave blank for a custom folderless workspace"
                  className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                />
              </label>
            </div>
            <label className="grid gap-1">
              <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">description (optional)</span>
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="What this project tracks: codebase, domain, or scope"
                maxLength={240}
                className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
            </label>
            <div className="flex flex-wrap items-center gap-4 pt-1">
              <button
                type="submit"
                disabled={busy || !newName.trim()}
                className="press mono min-h-9 border-b border-sense-pale px-2 text-[11px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
              >
                {busy ? "Creating" : "Create & activate"}
              </button>
              <button
                type="button"
                onClick={() => setCreating(false)}
                className="press mono min-h-9 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text"
              >
                Cancel
              </button>
              <span className="mono ml-auto text-[10px] text-quiet">
                Transient scratch/ and fixture paths are ignored automatically
              </span>
            </div>
          </form>
        )}

        {q.error && !q.data ? (
          <Empty title="Projects did not answer">{q.error}</Empty>
        ) : !q.data ? (
          <Placeholder className="h-40" what="projects" />
        ) : list.length === 0 ? (
          <Empty title="No registered projects yet">
            Click &quot;+ Add project&quot; above to create a workspace, or serve any repository with neuromorph-nano.
          </Empty>
        ) : (
          <ul className="grid gap-3">
            {list.map((p) => {
              const display = projectLabel(p);
              const isSettingsOpen = settingsOpen === p.key;
              const isActive = activeKey === p.key;
              return (
                <li
                  key={p.key}
                  className={`grid gap-2.5 border px-4 py-4 ${
                    isActive
                      ? "border-line-strong border-l-2 border-l-sense-pale bg-bg-1/40"
                      : "border-line bg-bg-1/15"
                  }`}
                  data-e2e="project-row"
                  data-project={p.key}
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
                    <div className="flex min-w-0 flex-wrap items-baseline gap-2.5">
                      <span className="mono text-[13px] font-medium text-text">{display}</span>
                      <span className="mono text-[10.5px] text-quiet">{p.key}</span>
                      {isActive && <Pill tone="verified">active</Pill>}
                      {p.serving && <Pill tone="signal">serving</Pill>}
                      {p.custom && <Pill tone="muted">custom</Pill>}
                      {p.root && p.root_exists === false && <Pill tone="attention">folder missing</Pill>}
                      {p.is_scratch && <Pill tone="error">test/scratch</Pill>}
                    </div>
                    <div className="mono flex shrink-0 flex-wrap items-center gap-3 pr-1 text-[10.5px] text-faint">
                      <span>
                        {p.episodes} {p.episodes === 1 ? "episode" : "episodes"}
                      </span>
                      <span>{bytes(p.bytes)}</span>
                      {p.last_ms && now > 0 ? <span>{ago(p.last_ms, now)}</span> : null}
                    </div>
                  </div>

                  {p.description && (
                    <p className="read text-[11.5px] leading-relaxed text-muted">{p.description}</p>
                  )}

                  {p.root && (
                    <div className="mono truncate text-[10.5px] text-faint">
                      <Masked id={`project-root:${p.key}`} shown={maskPath(p.root, root)} full={p.root} />
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 pt-0.5">
                    {!isActive && (
                      <button
                        type="button"
                        onClick={() => {
                          pickProject(p.key);
                          setNote({ text: `Switched active project to "${display}" across all pages.`, tone: "verified" });
                        }}
                        className="press mono min-h-8 border-b border-sense-pale px-1 text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
                      >
                        Set active
                      </button>
                    )}
                    <button
                      type="button"
                      aria-haspopup="dialog"
                      aria-expanded={isSettingsOpen}
                      data-e2e="project-settings-toggle"
                      onClick={() => openSettingsModal(p)}
                      className="press mono inline-flex min-h-8 items-center gap-1.5 border border-line-strong px-2.5 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-sense-pale hover:text-text"
                    >
                      <span>Settings</span>
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        {/* Centered Project Settings Pop-Up Modal */}
        {activeModalProject && (
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`Project settings for ${projectLabel(activeModalProject)}`}
            data-e2e="project-settings-menu"
            className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6"
          >
            <button
              type="button"
              aria-label="Close project settings"
              onClick={() => setSettingsOpen(null)}
              className="fixed inset-0 bg-bg/75 backdrop-blur-xs"
            />
            <div className="relative z-10 flex max-h-[86dvh] w-full max-w-2xl flex-col border border-sense-pale/70 bg-bg-1 shadow-[0_18px_48px_rgb(0_0_0/0.75)]">
              {/* Modal Header */}
              <div className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-5">
                <div className="min-w-0">
                  <div className="mono truncate text-[12px] font-medium uppercase tracking-[0.14em] text-text">
                    {projectLabel(activeModalProject)} · Settings
                  </div>
                  <div className="mono truncate text-[10px] text-quiet">{activeModalProject.key}</div>
                </div>
                <button
                  type="button"
                  onClick={() => setSettingsOpen(null)}
                  className="press mono shrink-0 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text"
                >
                  × Close
                </button>
              </div>

              {/* Modal Body */}
              <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-4 sm:p-5">
                <NoteLine note={note} />

                {/* Identity: Display name & description */}
                <form
                  className="grid gap-3 border-b border-line/70 pb-4"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void submitRename(activeModalProject.key);
                  }}
                >
                  <div className="eyebrow">General</div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="grid gap-1">
                      <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Display name</span>
                      <input
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        maxLength={80}
                        required
                        className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none focus-visible:border-sense-pale"
                      />
                    </label>
                    <label className="grid gap-1">
                      <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Description</span>
                      <input
                        value={editDesc}
                        onChange={(e) => setEditDesc(e.target.value)}
                        maxLength={240}
                        placeholder="Optional short description"
                        className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                      />
                    </label>
                  </div>
                  <div>
                    <button
                      type="submit"
                      disabled={
                        busy ||
                        !editName.trim() ||
                        (editName.trim() === projectLabel(activeModalProject) &&
                          editDesc.trim() === (activeModalProject.description ?? ""))
                      }
                      className="press mono border border-sense-pale/70 bg-sense/15 px-3 py-1.5 text-[10px] uppercase tracking-[0.12em] text-text hover:bg-sense/30 disabled:opacity-40"
                    >
                      {busy ? "Saving..." : "Save changes"}
                    </button>
                  </div>
                </form>

                {/* Live interactive per-project controls (Primes + .nanoignore) */}
                <ProjectLiveControls project={activeModalProject} onNote={setNote} />

                {/* Danger Zone */}
                <div className="grid gap-2 border-t border-line/70 pt-3.5">
                  <span className="mono text-[10px] uppercase tracking-[0.14em] text-faint">Danger zone</span>
                  <DeleteFlow<{ project?: string }>
                    path="/api/projects/forget"
                    body={{ project: activeModalProject.key }}
                    label="Delete project"
                    title="Delete Neuromorph Nano's stored memory for this project (never touches your source code)."
                    done={() => `Deleted project "${projectLabel(activeModalProject)}" from Neuromorph Nano. Your repository files were not touched.`}
                    refresh={["/api/projects", "/api/storage", "/api/audit"]}
                    onNote={setNote}
                    onDone={() => setSettingsOpen(null)}
                    root={root}
                    startClass="press mono min-h-8 border-b border-line-strong px-1 text-[10.5px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error disabled:opacity-40"
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Panel>
  );
}

export function ProjectsGuide() {
  return (
    <>
      <RailSection title="Global active project">
        <p className="read text-[11px] leading-relaxed text-muted">
          The active project selected here or in the left sidebar connects every project-scoped view across the interface: Primes Studio, <span className="mono">.nanoignore</span>, Memory, Learning, and Audit.
        </p>
      </RailSection>
      <RailSection title="Where projects come from">
        <ul className="read grid gap-1.5 text-[11px] text-faint">
          <li>Each time Neuromorph Nano serves or runs hooks against a repository, it creates an isolated memory folder under <span className="mono">memory/projects/</span>.</li>
          <li>You can add custom workspaces (with or without a folder path), rename any project, or delete old/test projects safely with two-step confirmation.</li>
          <li>Deleting a project removes only Neuromorph Nano&apos;s stored memory folder; your source files on disk are never touched.</li>
          <li>Transient <span className="mono">scratch/</span> and internal test-fixture folders are excluded from automatic registration.</li>
        </ul>
      </RailSection>
    </>
  );
}
