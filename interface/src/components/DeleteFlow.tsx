"use client";

// 2026-11-26 (directions of Luis Jake Gabriel III: deleting anything goes through verifications in
// several steps, to prevent accidental deleting). Every delete of the interface goes through this
// flow: (1) the button asks the engine what would be deleted, and nothing is deleted; (2) the
// preview is shown (what goes, its size, whether it can be undone) and the user types the word the
// engine names; (3) only the final button, enabled once the word matches, deletes. "Keep it"
// leaves at every step.

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { NoteLine } from "@/components/Controls";
import { Masked } from "@/components/ui";
import { bytes, cannotUndo, confirmDelete, previewDelete, refreshAll, type DeletePreview } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { maskPath } from "@/lib/privacy";

export type DeleteNote = { text: string; tone: "verified" | "error" };
type Phase = "idle" | "loading" | "confirm" | "deleting";

const START = "press mono min-h-11 border-b border-line-strong px-1 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-40";
const QUIET = "press mono min-h-11 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text";
/** Items listed in the preview before the rest is counted. */
const SHOW_ITEMS = 40;

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function DeleteFlow<T = unknown>({
  path,
  body,
  label,
  done,
  refresh = [],
  onNote,
  onDone,
  onClose,
  autoStart = false,
  disabled = false,
  title,
  startClass = START,
  e2e,
  root = null,
}: {
  /** The destructive endpoint, e.g. `/api/ledger/clear`. */
  path: string;
  /** The request, without any confirmation (the flow adds the token at step 3). */
  body: Record<string, unknown>;
  /** The start button's text. */
  label: ReactNode;
  /** The result in words, from the engine's reply. */
  done: (r: T) => string;
  /** Paths refreshed after the delete (by prefix); the audit log always is. */
  refresh?: string[];
  /** When given, results and errors go to the caller's note line instead of the flow's own. */
  onNote?: (n: DeleteNote) => void;
  onDone?: (r: T) => void;
  /** With `autoStart`: called when the flow ends (kept, deleted or refused). */
  onClose?: () => void;
  /** Starts at step 1 at once, with no start button (the caller drew it, e.g. in a table row). */
  autoStart?: boolean;
  disabled?: boolean;
  title?: string;
  startClass?: string;
  /** The start button's data-e2e. */
  e2e?: string;
  /** The Nano's folder, to shorten paths in the preview. */
  root?: string | null;
}) {
  const [phase, setPhase] = useState<Phase>(autoStart ? "loading" : "idle");
  const [preview, setPreview] = useState<DeletePreview | null>(null);
  const [word, setWord] = useState("");
  const [note, setNote] = useState<DeleteNote | null>(null);
  const field = useRef<HTMLInputElement>(null);
  const startBtn = useRef<HTMLButtonElement>(null);
  const started = useRef(false);
  // Bumped whenever the flow is left, so a reply that arrives after "Keep it" is dropped.
  const gen = useRef(0);
  const id = useId();
  const now = useNow();
  const bodyKey = JSON.stringify(body);

  const tell = (n: DeleteNote) => {
    if (onNote) onNote(n);
    else setNote(n);
  };
  const reset = (focusStart = true) => {
    gen.current += 1;
    setPhase("idle");
    setPreview(null);
    setWord("");
    if (autoStart) onClose?.();
    else if (focusStart) startBtn.current?.focus();
  };
  const showPreview = (p: DeletePreview) => {
    setPreview(p);
    setWord("");
    setPhase("confirm");
  };
  const failPreview = (e: unknown) => {
    tell({ text: errText(e), tone: "error" });
    reset(false);
  };

  // Step 1 at once when the caller drew the start button itself. The state is set only in the
  // promise's callbacks.
  useEffect(() => {
    if (!autoStart || started.current) return;
    started.current = true;
    const g = gen.current;
    previewDelete(path, JSON.parse(bodyKey) as Record<string, unknown>).then(
      (p) => g === gen.current && showPreview(p),
      (e) => g === gen.current && failPreview(e),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, path, bodyKey]);

  // Step 2: the field takes the focus as soon as the preview is shown.
  useEffect(() => {
    if (phase === "confirm") field.current?.focus();
  }, [phase]);

  /** Step 1: the engine says what would go; nothing is deleted. */
  const ask = async () => {
    const g = gen.current;
    setPhase("loading");
    try {
      const p = await previewDelete(path, body);
      if (g === gen.current) showPreview(p);
    } catch (e) {
      if (g === gen.current) failPreview(e);
    }
  };
  const start = () => {
    setNote(null);
    void ask();
  };

  const expired = preview !== null && preview.expires_ms > 0 && now > 0 && now > preview.expires_ms;
  const matches = preview !== null && word === preview.confirm_word;
  const permanent = preview ? cannotUndo(preview.undo) : true;

  const finish = async () => {
    if (!preview) return;
    if (expired) {
      await ask();
      return;
    }
    if (!matches) return;
    setPhase("deleting");
    try {
      const r = await confirmDelete<T>(path, body, preview.confirm_token);
      for (const p of [...refresh, "/api/audit"]) void refreshAll(p);
      tell({ text: done(r), tone: "verified" });
      onDone?.(r);
      reset();
    } catch (e) {
      tell({ text: errText(e), tone: "error" });
      reset();
    }
  };

  const open = phase !== "idle";
  const busy = phase === "loading" || phase === "deleting";
  const shown = preview?.items.slice(0, SHOW_ITEMS) ?? [];
  const more = (preview?.items.length ?? 0) - shown.length;

  return (
    <div className="min-w-0">
      {!onNote && <NoteLine note={note} />}
      {!autoStart && (
        <button ref={startBtn} type="button" disabled={disabled || open} onClick={start} title={title} className={startClass} data-e2e={e2e} aria-expanded={open}>
          {label}
        </button>
      )}
      {open && (
        <div role="group" aria-labelledby={`${id}-title`} className="mt-2 grid min-w-0 gap-2 border-l-2 border-error/70 py-1 pl-3" data-e2e="delete-flow" data-step={phase} data-confirm-word={preview?.confirm_word ?? ""}>
          {phase === "loading" || !preview ? (
            <div className="flex flex-wrap items-center gap-3">
              <span id={`${id}-title`} role="status" className="mono text-[11px] text-faint">
                Reading what would be deleted. Nothing is deleted yet.
              </span>
              <button type="button" onClick={() => reset()} className={QUIET}>
                Keep it
              </button>
            </div>
          ) : (
            <>
              <div className="mono text-[10px] uppercase tracking-[0.14em] text-faint">Check what goes · nothing is deleted until you type the word and press {permanent ? "Delete permanently" : "Delete"}</div>
              <h3 id={`${id}-title`} className="mono text-[12px] uppercase tracking-[0.12em] text-text">
                {preview.title}
              </h3>
              {shown.length > 0 && (
                <ul className="grid min-w-0 gap-1" data-e2e="delete-items">
                  {shown.map((it, i) => (
                    <li key={`${it.path ?? it.label}:${i}`} className="mono grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 text-[11px]">
                      <span className="min-w-0">
                        <span className="block break-words text-muted">{it.label}</span>
                        {it.path && it.path !== it.label && (
                          <span className="block break-all text-[10px] text-faint">
                            <Masked id={`delete:${it.path}`} shown={maskPath(it.path, root)} full={it.path} />
                          </span>
                        )}
                      </span>
                      <span className="text-right whitespace-nowrap text-text">
                        {it.bytes !== null ? bytes(it.bytes) : ""}
                        {it.files !== null ? `${it.bytes !== null ? " · " : ""}${it.files.toLocaleString()} ${it.files === 1 ? "file" : "files"}` : ""}
                      </span>
                    </li>
                  ))}
                  {more > 0 && <li className="mono text-[10.5px] text-faint">and {more.toLocaleString()} more</li>}
                </ul>
              )}
              {(preview.bytes !== null || preview.files !== null) && (
                <div className="mono text-[11px] text-text">
                  In all: {[preview.bytes !== null ? bytes(preview.bytes) : null, preview.files !== null ? `${preview.files.toLocaleString()} ${preview.files === 1 ? "file" : "files"}` : null].filter(Boolean).join(" · ")}
                </div>
              )}
              <p id={`${id}-undo`} className={`mono flex items-baseline gap-2 text-[11px] ${permanent ? "text-error" : "text-attention"}`}>
                <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${permanent ? "bg-error" : "bg-attention"}`} />
                {preview.undo.charAt(0).toUpperCase() + preview.undo.slice(1)}
                {/[.!]$/.test(preview.undo) ? "" : "."}
              </p>
              {preview.note && <p className="read text-[11px] leading-relaxed text-faint">{preview.note}</p>}
              <form
                className="grid min-w-0 gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void finish();
                }}
              >
                <label htmlFor={`${id}-word`} className="mono text-[10.5px] uppercase tracking-[0.12em] text-muted">
                  Type <span className="break-all normal-case tracking-normal text-text">{preview.confirm_word}</span> to confirm
                </label>
                <input
                  ref={field}
                  id={`${id}-word`}
                  type="text"
                  value={word}
                  disabled={busy}
                  onChange={(e) => setWord(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      e.preventDefault();
                      reset();
                    }
                  }}
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  aria-describedby={`${id}-undo`}
                  aria-invalid={word.length > 0 && !matches}
                  className="mono h-9 w-full max-w-xs min-w-0 border-b border-line-strong bg-transparent px-1 text-xs text-text outline-none focus-visible:border-sense-pale"
                  data-e2e="delete-word"
                />
                <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-line/60 pt-3">
                  <button
                    type="submit"
                    disabled={busy || (!expired && !matches)}
                    className="press mono min-h-11 border border-error/70 px-3.5 text-[11px] uppercase tracking-[0.12em] text-error hover:bg-error/10 disabled:opacity-40"
                    data-e2e="delete-final"
                  >
                    {phase === "deleting" ? "Deleting" : expired ? "Expired · preview again" : permanent ? "Delete permanently" : "Delete"}
                  </button>
                  <button type="button" disabled={phase === "deleting"} onClick={() => reset()} className={QUIET}>
                    Keep it
                  </button>
                  {expired && <span className="mono text-[10.5px] text-attention">The confirmation expired after 2 minutes; preview again to get a new one.</span>}
                </div>
              </form>
            </>
          )}
        </div>
      )}
    </div>
  );
}
