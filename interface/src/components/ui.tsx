"use client";

import { useState, useSyncExternalStore, type CSSProperties, type ReactNode } from "react";
import Link from "next/link";
import DotText from "@/components/DotText";
import { callLabel, captureMask, hide, reveal, useRevealed } from "@/lib/privacy";
import { useApi } from "@/lib/api";

/** Status tones. Each colour keeps one meaning: signal is "working now", learned is cartridges,
 *  verified is a checked outcome, attention needs a look, error failed. */
export type Tone = "muted" | "signal" | "learned" | "verified" | "attention" | "error";

const noSub = () => () => {};

/** A value shown masked until the viewer asks for it: per item, for this session only, never
 *  saved. In capture mode it cannot be revealed and no control is drawn. */
export function Masked({ id, shown, full, className = "" }: { id: string; shown: ReactNode; full?: string | null; className?: string }) {
  const on = useRevealed(id);
  const cap = useSyncExternalStore(noSub, captureMask, () => true);
  if (!full) return <span className={className}>{shown}</span>;
  return (
    <span className={className}>
      {on ? full : shown}
      {!cap && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            if (on) hide(id);
            else reveal(id);
          }}
          className="press mono ml-2 align-baseline text-[10px] uppercase tracking-[0.12em] text-quiet underline decoration-dotted underline-offset-2 hover:text-text"
        >
          {on ? "hide" : "show"}
        </button>
      )}
    </span>
  );
}

/** The label of a call: its kind in plain words, the question or prompt held back until asked. */
export function CallText({ id, label, tool, className = "" }: { id: string; label?: string | null; tool?: string | null; className?: string }) {
  const { short, hidden } = callLabel(label, tool);
  return <Masked id={`call:${id}`} className={className} shown={hidden ? `${short} · question hidden` : short} full={hidden ? `${short}: ${hidden}` : null} />;
}

/** A small reticle before a section heading, as on a spec sheet. */
export function Reticle({ className = "" }: { className?: string }) {
  return (
    <svg aria-hidden viewBox="0 0 12 12" className={`h-3 w-3 shrink-0 ${className}`} fill="none" stroke="currentColor" strokeWidth={1}>
      <circle cx={6} cy={6} r={3.5} />
      <path d="M6 0v3.2M6 8.8V12M0 6h3.2M8.8 6H12" />
    </svg>
  );
}

/** How this machine is reached, as GET /api/airgap reports it ("air gap on", "local only" or
 *  "shared on your LAN"), followed by a separator; nothing until it answers. */
function IsolationWord() {
  const { data } = useApi<{ air_gap?: boolean; loopback_only?: boolean; lan_listeners?: number }>("/api/airgap", 30000);
  if (!data) return null;
  const word = data.air_gap ? "air gap on" : (data.lan_listeners ?? 0) > 0 || data.loopback_only === false ? "shared on your LAN" : data.loopback_only ? "local only" : null;
  return word ? <>{word} · </> : null;
}

/** The page head: a monospace line with the page's place, the title, one sentence, then a hairline
 *  rule with numbered ticks. */
/** The pages about the Nano's weights carry a link to the one view of how they fit together. */
const FLOW_PAGES = new Set(["Models", "Cartridges", "Parameters", "Learning"]);

export function PageHeader({ title, sub, right, index }: { title: string; sub?: ReactNode; right?: ReactNode; index?: string }) {
  return (
    <header className="mb-8 md:mb-10">
      <div className="mono flex items-center justify-between gap-4 text-[10.5px] uppercase tracking-[0.18em] text-faint">
        <span className="flex flex-wrap items-center gap-x-4">
          <span>Neuromorph Nano{index ? ` / ${index}` : ""}</span>
          {index && FLOW_PAGES.has(index) && (
            <Link href="/cartridges/#flow" className="link normal-case tracking-normal" data-e2e="flow-link">
              components
            </Link>
          )}
        </span>
        <span className="hidden sm:inline">
          <IsolationWord />
          base cartridges and upstream weights read only
        </span>
      </div>
      <div className="mt-6 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="min-w-0">
          <h1 className="mono flex items-center gap-3 text-[15px] font-normal uppercase leading-none tracking-[0.32em] text-text md:text-[17px]">
            <Reticle className="h-3.5 w-3.5 text-sense-pale" />
            {title}
          </h1>
          {sub && <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-muted">{sub}</p>}
        </div>
        {right && <div className="mono flex flex-wrap items-center gap-4 text-[11px] uppercase tracking-[0.12em] text-muted">{right}</div>}
      </div>
      <div aria-hidden className="rule-ticks mt-6" />
    </header>
  );
}

/** A section set on the ground: a reticle and a monospace heading, a hairline rule, then the
 *  content. No box. */
export function Panel({ title, children, className = "", right }: { title?: ReactNode; children: ReactNode; className?: string; right?: ReactNode; pad?: boolean }) {
  return (
    <section className={`min-w-0 ${className}`}>
      {(title || right) && (
        <div className="flex min-h-7 flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-line pb-2">
          {title && (
            <h2 className="eyebrow flex items-center gap-2 text-muted">
              <Reticle />
              {title}
            </h2>
          )}
          {right}
        </div>
      )}
      <div className="pt-4">{children}</div>
    </section>
  );
}

/** A section that opens on demand (2026-11-25): the same reticle heading and hairline as a Panel,
 *  with a count or note on the right and a chevron; the detail stays folded until asked for. */
export function Fold({ title, right, children, open = false, id }: { title: ReactNode; right?: ReactNode; children: ReactNode; open?: boolean; id?: string }) {
  return (
    <details id={id} open={open} className="group min-w-0">
      <summary className="press flex min-h-9 cursor-pointer list-none items-center justify-between gap-3 border-b border-line pb-2 [&::-webkit-details-marker]:hidden">
        <h2 className="eyebrow flex items-center gap-2 text-muted group-open:text-text">
          <Reticle />
          {title}
        </h2>
        <span className="mono flex items-center gap-3 text-[11px] text-faint">
          {right}
          <svg aria-hidden viewBox="0 0 10 10" className="h-2.5 w-2.5 shrink-0 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" strokeWidth={1.2}>
            <path d="M1.5 3.5 5 7l3.5-3.5" />
          </svg>
        </span>
      </summary>
      <div className="pt-4">{children}</div>
    </details>
  );
}

const TONE_BG: Record<Tone, string> = { muted: "bg-quiet", signal: "bg-signal", learned: "bg-learned", verified: "bg-verified", attention: "bg-attention", error: "bg-error" };
const TONE_TX: Record<Tone, string> = { muted: "text-muted", signal: "text-signal", learned: "text-learned", verified: "text-verified", attention: "text-attention", error: "text-error" };

/** A state set in monospace with one small square of its colour. Not a badge. */
export function Pill({ children, tone = "muted", title }: { children: ReactNode; tone?: Tone; title?: string }) {
  return (
    <span title={title} className={`mono inline-flex shrink-0 items-center gap-1.5 text-[10.5px] uppercase tracking-[0.12em] ${TONE_TX[tone]}`}>
      <span aria-hidden className={`inline-block h-1.5 w-1.5 ${TONE_BG[tone]}`} />
      {children}
    </span>
  );
}

export function Dot({ tone = "muted", className = "" }: { tone?: Tone; className?: string }) {
  return <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${TONE_BG[tone]} ${className}`} />;
}

/** An empty state: a dotted slot in one-bit dither and one sentence. Still. */
export function Empty({ title, children, slot = false }: { title: string; children?: ReactNode; slot?: boolean }) {
  return (
    <div className="flex items-start gap-5 py-6">
      <span aria-hidden className={slot ? "dither-slot" : "dither-block"} />
      <div>
        <div className="mono text-[12px] uppercase tracking-[0.12em] text-muted">{title}</div>
        {children && <div className="read mt-2 max-w-md text-xs leading-relaxed text-faint">{children}</div>}
      </div>
    </div>
  );
}

export function Failed({ error }: { error: string }) {
  return (
    <div className="mb-8 flex items-start gap-3 border-t border-error/60 pt-3 text-[13px]">
      <span aria-hidden className="mt-1.5 inline-block h-1.5 w-1.5 bg-error" />
      <div>
        <div className="mono text-[12px] uppercase tracking-[0.12em] text-error">The engine is not answering</div>
        <div className="mt-1 text-muted">
          Start it with <span className="mono text-text">neuromorph-nano ui</span>, then reload. <span className="mono text-faint">({error})</span>
        </div>
      </div>
    </div>
  );
}

/** A large figure: a tiny monospace label, the number in the dot-matrix face, a caption. */
export function Figure({ label, value, caption, tone = "text-text", size = 44 }: { label: string; value: string; caption?: ReactNode; tone?: string; size?: number }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow">{label}</div>
      <div className={`mt-3 ${tone}`}>
        <DotText text={value} size={size} />
      </div>
      {caption && <div className="mono mt-2 truncate text-[11px] text-faint">{caption}</div>}
    </div>
  );
}

/** A loading state: a still dotted field at the size the content will take, with one quiet
 *  monospace line naming what it waits for. It appears only after 150 ms of waiting and never
 *  moves, so it cannot read as activity. */
export function Placeholder({ className = "h-32", what }: { className?: string; what?: string }) {
  return (
    <div role="status" aria-label={what ? `loading ${what}` : "loading"} data-loading={what ?? ""} className={`placeholder relative ${className}`}>
      {what && <span className="placeholder-label mono">loading {what}</span>}
    </div>
  );
}

/** The loading state of a figure: the figure's own frame at its final size (so nothing moves
 *  when it draws), a quiet "loading <what>" label and one faint scan line that crosses once. It is
 *  grey, never the signal colour, and stops under reduced motion. With `failed`, the frame says
 *  what did not answer instead. */
export function FigureLoading({ what, failed, className = "" }: { what: string; failed?: string | null; className?: string }) {
  if (failed)
    return (
      <div role="alert" data-e2e="figure-failed" className={`absolute inset-0 flex items-center justify-center ${className}`}>
        <div className="max-w-md px-6 text-center">
          <div className="mono text-[11px] uppercase tracking-[0.14em] text-error">{what} did not answer</div>
          <div className="mono mt-2 text-[10.5px] text-faint">{failed}</div>
        </div>
      </div>
    );
  return (
    <div role="status" aria-label={`loading ${what}`} data-loading={what} className={`figure-loading absolute inset-0 ${className}`}>
      <span className="placeholder-label mono">loading {what}</span>
    </div>
  );
}

/** A watched number: changed digits roll 4 px up and fade in; unchanged digits stay. */
export function Tick({ value, format = (n) => String(Math.round(n)), className = "" }: { value: number | null | undefined; format?: (n: number) => string; className?: string }) {
  const text = value === null || value === undefined ? "–" : format(value);
  const [shown, setShown] = useState<{ text: string; was: string }>({ text, was: text });
  if (shown.text !== text) setShown({ text, was: shown.text });
  const animate = shown.was !== shown.text && shown.was !== "–";
  return (
    <span className={`tnum inline-flex ${className}`} aria-label={shown.text}>
      {[...shown.text].map((ch, i) => (
        <span key={`${i}-${ch}-${animate && ch !== shown.was[i] ? shown.text : ""}`} aria-hidden className={animate && ch !== shown.was[i] ? "roll" : undefined}>
          {ch}
        </span>
      ))}
    </span>
  );
}

/** A row of choices with one sliding indicator under the chosen one. */
export function Segmented({
  values,
  value,
  onPick,
  busy = false,
  label,
  render = (v) => v,
}: {
  values: string[];
  value: string;
  onPick: (v: string) => void;
  busy?: boolean;
  label: string;
  render?: (v: string) => string;
}) {
  const i = Math.max(0, values.indexOf(value));
  const style: CSSProperties = { width: `calc(100% / ${values.length})`, transform: `translateX(${i * 100}%)` };
  return (
    <div role="radiogroup" aria-label={label} className={`relative inline-grid h-9 min-w-0 border-b border-line-strong p-0 ${busy ? "opacity-60" : ""}`} style={{ gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))` }}>
      <span aria-hidden className="absolute bottom-[-1px] left-0 h-[2px] bg-sense-pale" style={{ ...style, transition: "transform var(--d-2) var(--ease)" }} />
      {values.map((v) => {
        const on = v === value;
        return (
          <button
            key={v}
            role="radio"
            aria-checked={on}
            disabled={busy}
            onClick={() => !on && onPick(v)}
            className={`press mono relative z-10 min-h-11 min-w-[48px] px-3 text-[11px] uppercase tracking-[0.08em] ${on ? "text-text" : "text-faint hover:text-text"}`}
          >
            {render(v)}
          </button>
        );
      })}
    </div>
  );
}

/** An on/off switch. On is the bright thumb on a raised track; colour is kept for state. */
export function Switch({ on, onChange, label, busy = false, disabled = false }: { on: boolean; onChange?: (v: boolean) => void; label: string; busy?: boolean; disabled?: boolean }) {
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={busy || disabled}
      onClick={() => onChange?.(!on)}
      className={`press relative inline-flex h-9 w-11 shrink-0 items-center justify-center rounded-full disabled:cursor-not-allowed ${busy ? "opacity-60" : ""} ${disabled ? "opacity-50" : ""}`}
    >
      <span aria-hidden className={`relative block h-4 w-7 rounded-full border border-line-3 ${on ? "bg-bg-3" : "bg-bg-2"}`}>
        <span
          aria-hidden
          className={`absolute top-[2px] left-[2px] h-2.5 w-2.5 rounded-full ${on ? "bg-text" : "bg-faint"}`}
          style={{ transform: `translateX(${on ? 12 : 0}px)`, transition: "transform 160ms var(--ease), background-color 160ms var(--ease)" }}
        />
      </span>
    </button>
  );
}

export function KV({ k, children, mono = false }: { k: ReactNode; children: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-xs">
      <span className="shrink-0 text-faint">{k}</span>
      <span className={`min-w-0 truncate text-right text-muted ${mono ? "mono" : ""}`}>{children}</span>
    </div>
  );
}

/** The Neuropod's own shape as the mark: eight arms leaving a ring. One arm carries the signal
 *  colour while the engine is working. */
export function Mark({ className = "h-7 w-7", working = false }: { className?: string; working?: boolean }) {
  // The organism in plan, in dots: a ring and eight arms of three dots each, between the ring's
  // eight joins.
  const dots: { x: number; y: number; r: number; arm: number }[] = [];
  for (let i = 0; i < 12; i++) dots.push({ x: 16 + Math.cos((i / 12) * Math.PI * 2) * 4.5, y: 16 + Math.sin((i / 12) * Math.PI * 2) * 4.5, r: 0.9, arm: -1 });
  for (let i = 0; i < 8; i++) {
    const a = -Math.PI / 2 + Math.PI / 8 + (i / 8) * Math.PI * 2;
    for (let k = 0; k < 3; k++) dots.push({ x: 16 + Math.cos(a) * (8 + k * 2.7), y: 16 + Math.sin(a) * (8 + k * 2.7), r: 1.1 - k * 0.25, arm: i });
  }
  return (
    <svg aria-hidden viewBox="0 0 32 32" className={className}>
      {dots.map((d, i) => (
        <circle key={i} cx={d.x} cy={d.y} r={d.r} fill={working && d.arm === 4 ? "var(--signal)" : "currentColor"} style={{ transition: "fill var(--d-5) var(--ease)" }} />
      ))}
    </svg>
  );
}
