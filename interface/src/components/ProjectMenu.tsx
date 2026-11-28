"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import { Pill } from "@/components/ui";
import { post, projectLabel, projectLabelByKey, refreshAll, type Project } from "@/lib/api";
import { pickProject, useProjectKey } from "@/lib/evidence";

type Note = { text: string; tone: "verified" | "error" };
type PanelMode = "list" | "add" | "edit" | "delete";

/** Unified global project dropdown menu with switch, + Add, Edit/Rename, and two-step double-confirmed Delete. */
export default function ProjectMenu({
  placement = "down",
  compact = false,
  label = "project",
  onSelect,
}: {
  /** "up" for the bottom-left sidebar, "down" for page toolbars. */
  placement?: "down" | "up";
  /** Compact full-width trigger for the left sidebar. */
  compact?: boolean;
  /** Optional eyebrow label beside or above the trigger. */
  label?: string | null;
  /** Optional callback invoked when the user picks a project from the menu. */
  onSelect?: (key: string) => void;
}) {
  const { key: activeKey, projects } = useProjectKey();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<PanelMode>("list");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);

  const [newName, setNewName] = useState("");
  const [newRoot, setNewRoot] = useState("");
  const [newDesc, setNewDesc] = useState("");

  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");

  const wrapRef = useRef<HTMLDivElement>(null);

  const list = useMemo(() => {
    const raw = projects?.projects ?? [];
    return [...raw].sort((a, b) => {
      if (a.key === activeKey && b.key !== activeKey) return -1;
      if (b.key === activeKey && a.key !== activeKey) return 1;
      if (Boolean(a.serving) !== Boolean(b.serving)) return a.serving ? -1 : 1;
      return (b.last_ms ?? 0) - (a.last_ms ?? 0);
    });
  }, [projects, activeKey]);

  const currentKey = activeKey ?? list[0]?.key ?? "nano";
  const currentProject: Project | null = list.find((p) => p.key === currentKey) ?? null;
  const currentLabel = projectLabelByKey(projects, currentKey);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
        setMode("list");
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        setMode("list");
      }
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const openEdit = () => {
    if (!currentProject) return;
    setEditName(projectLabel(currentProject));
    setEditDesc(currentProject.description ?? "");
    setNote(null);
    setMode("edit");
  };

  const submitAdd = async () => {
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
      setNote({ text: `Added "${name}".`, tone: "verified" });
      refreshAll("/api/projects");
      refreshAll("/api/audit");
      setMode("list");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const submitEdit = async () => {
    if (!currentProject || !editName.trim()) return;
    setBusy(true);
    try {
      await post("/api/projects", {
        action: "rename",
        project: currentProject.key,
        name: editName.trim(),
        description: editDesc.trim(),
      });
      setNote({ text: `Renamed to "${editName.trim()}".`, tone: "verified" });
      refreshAll("/api/projects");
      refreshAll("/api/audit");
      setMode("list");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const popPos =
    placement === "up"
      ? "fixed inset-x-3 top-16 max-h-[calc(100dvh-5rem)] overflow-y-auto sm:absolute sm:inset-x-auto sm:top-auto sm:bottom-full sm:left-0 sm:mb-2 sm:max-h-none sm:w-80 sm:max-w-[calc(100vw-2rem)] sm:overflow-visible"
      : "fixed inset-x-3 top-16 max-h-[calc(100dvh-5rem)] overflow-y-auto sm:absolute sm:inset-x-auto sm:top-full sm:left-0 sm:mt-1.5 sm:max-h-none sm:w-80 sm:max-w-[calc(100vw-2rem)] sm:overflow-visible";

  return (
    <div ref={wrapRef} className={`relative min-w-0 ${compact ? "grid gap-1" : "inline-flex max-w-full items-center gap-2"}`} data-e2e="project-menu">
      {label && !compact && (
        <span className="mono shrink-0 text-[10px] uppercase tracking-[0.12em] text-faint">{label}</span>
      )}
      {compact && label && (
        <div className="flex items-baseline justify-between gap-2">
          <span className="mono text-[9.5px] uppercase tracking-[0.14em] text-faint">{label}</span>
          <Link href="/settings/#projects" className="mono text-[9.5px] uppercase tracking-[0.1em] text-sense-pale hover:text-text">
            All
          </Link>
        </div>
      )}

      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        data-e2e="project-menu-trigger"
        onClick={() => {
          setOpen((v) => !v);
          setMode("list");
          setNote(null);
        }}
        className={`press mono flex min-w-0 items-center justify-between gap-2 border-b border-line-strong bg-transparent text-left text-text hover:border-sense-pale focus-visible:border-sense-pale ${
          compact ? "h-8 w-full pr-1 pl-0.5 text-[11px]" : "h-9 max-w-full px-1 text-[11.5px] sm:min-w-[10.5rem] sm:max-w-[16rem]"
        }`}
      >
        <span className="min-w-0 truncate">{currentLabel}</span>
        <svg aria-hidden viewBox="0 0 16 16" className={`h-3 w-3 shrink-0 text-faint transition-transform ${open ? "rotate-180" : ""}`}>
          <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
        </svg>
      </button>

      {open && (
        <>
        <div
          aria-hidden
          className="fixed inset-0 z-40 bg-bg/65 sm:hidden"
          onClick={() => {
            setOpen(false);
            setMode("list");
          }}
        />
        <div
          role="dialog"
          aria-label="Project menu"
          data-e2e="project-menu-popover"
          className={`z-50 grid gap-2.5 border border-line-strong bg-bg-1 p-3 shadow-[0_12px_36px_rgb(0_0_0/0.65)] ${popPos}`}
        >
          <NoteLine note={note} />

          {mode === "list" && (
            <>
              <div className="flex items-center justify-between border-b border-line pb-1.5">
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-faint">
                  Projects · {list.length}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setNote(null);
                    setMode("add");
                  }}
                  data-e2e="project-menu-add"
                  className="press mono border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
                >
                  + Add project
                </button>
              </div>

              <ul className="grid max-h-56 gap-0.5 overflow-y-auto">
                {list.map((p) => {
                  const active = p.key === currentKey;
                  return (
                    <li key={p.key}>
                      <button
                        type="button"
                        onClick={() => {
                          pickProject(p.key);
                          onSelect?.(p.key);
                          setOpen(false);
                        }}
                        className={`press flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left ${
                          active ? "border-l-2 border-sense-pale bg-bg-2/70 text-text" : "text-muted hover:bg-bg-2/40 hover:text-text"
                        }`}
                      >
                        <span className="min-w-0">
                          <span className="mono block truncate text-[11.5px]">{projectLabel(p)}</span>
                          <span className="mono block truncate text-[9.5px] text-quiet">{p.key}</span>
                        </span>
                        <span className="flex shrink-0 items-center gap-1.5">
                          {p.is_scratch && <Pill tone="error">test</Pill>}
                          {active && <Pill tone="verified">active</Pill>}
                        </span>
                      </button>
                    </li>
                  );
                })}
                <li>
                  <button
                    type="button"
                    onClick={() => {
                      pickProject("nano");
                      onSelect?.("nano");
                      setOpen(false);
                    }}
                    className={`press flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left ${
                      currentKey === "nano" ? "border-l-2 border-sense-pale bg-bg-2/70 text-text" : "text-muted hover:bg-bg-2/40 hover:text-text"
                    }`}
                  >
                    <span className="mono truncate text-[11.5px]">The Nano itself</span>
                    {currentKey === "nano" && <Pill tone="verified">active</Pill>}
                  </button>
                </li>
              </ul>

              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-2">
                {currentProject ? (
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={openEdit}
                      data-e2e="project-menu-edit"
                      className="press mono border-b border-line-strong text-[10px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text"
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNote(null);
                        setMode("delete");
                      }}
                      data-e2e="project-menu-delete"
                      className="press mono border-b border-line-strong text-[10px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error"
                    >
                      Delete
                    </button>
                  </div>
                ) : (
                  <span className="mono text-[9.5px] text-quiet">System workspace</span>
                )}
                <Link
                  href="/settings/#projects"
                  onClick={() => setOpen(false)}
                  className="mono text-[10px] uppercase tracking-[0.12em] text-faint hover:text-sense-pale"
                >
                  Manage all &rarr;
                </Link>
              </div>
            </>
          )}

          {mode === "add" && (
            <form
              className="grid gap-2.5"
              onSubmit={(e) => {
                e.preventDefault();
                void submitAdd();
              }}
            >
              <div className="flex items-center justify-between border-b border-line pb-1.5">
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-text">Add project</span>
                <button type="button" onClick={() => setMode("list")} className="press mono text-[10px] text-faint hover:text-text">
                  Back
                </button>
              </div>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Display name"
                maxLength={80}
                required
                className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <input
                value={newRoot}
                onChange={(e) => setNewRoot(e.target.value)}
                placeholder="Folder path (optional)"
                className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Description (optional)"
                maxLength={240}
                className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <div className="flex items-center gap-3 pt-1">
                <button
                  type="submit"
                  disabled={busy || !newName.trim()}
                  className="press mono border-b border-sense-pale text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
                >
                  {busy ? "Creating..." : "Create"}
                </button>
                <button type="button" onClick={() => setMode("list")} className="press mono text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text">
                  Cancel
                </button>
              </div>
            </form>
          )}

          {mode === "edit" && currentProject && (
            <form
              className="grid gap-2.5"
              onSubmit={(e) => {
                e.preventDefault();
                void submitEdit();
              }}
            >
              <div className="flex items-center justify-between border-b border-line pb-1.5">
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-text">Edit project</span>
                <button type="button" onClick={() => setMode("list")} className="press mono text-[10px] text-faint hover:text-text">
                  Back
                </button>
              </div>
              <input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Display name"
                maxLength={80}
                required
                className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none focus-visible:border-sense-pale"
              />
              <input
                value={editDesc}
                onChange={(e) => setEditDesc(e.target.value)}
                placeholder="Description (optional)"
                maxLength={240}
                className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <div className="flex items-center gap-3 pt-1">
                <button
                  type="submit"
                  disabled={busy || !editName.trim()}
                  className="press mono border-b border-sense-pale text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
                >
                  {busy ? "Saving..." : "Save"}
                </button>
                <button type="button" onClick={() => setMode("list")} className="press mono text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text">
                  Cancel
                </button>
              </div>
            </form>
          )}

          {mode === "delete" && currentProject && (
            <div className="grid gap-2">
              <div className="flex items-center justify-between border-b border-line pb-1.5">
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-error">Delete {currentLabel}</span>
                <button type="button" onClick={() => setMode("list")} className="press mono text-[10px] text-faint hover:text-text">
                  Back
                </button>
              </div>
              <DeleteFlow<{ project?: string }>
                autoStart
                path="/api/projects/forget"
                body={{ project: currentProject.key }}
                label="Delete project"
                done={() => `Deleted "${currentLabel}". Source files were not touched.`}
                refresh={["/api/projects", "/api/storage", "/api/audit"]}
                onNote={setNote}
                onDone={() => {
                  pickProject(null);
                  setMode("list");
                }}
                onClose={() => setMode("list")}
              />
            </div>
          )}
        </div>
        </>
      )}
    </div>
  );
}
