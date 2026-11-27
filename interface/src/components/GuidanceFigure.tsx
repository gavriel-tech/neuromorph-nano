"use client";

// 2026-11-27 (Luis Jake Gabriel III, his words: "We should be able to visualize the system in a
// high level professional and technical way" ... "The main center screen for every page should be
// easy to navigate and beautifully visual and interactive"): the agent guidance as one circuit.
// Left, the hook events of the coding agent; middle, the three kinds of guidance they reach
// (reminders, loops, the step graph); right, what was sent to the agent and the evidence ledger's
// verdicts on it. A line is bright only where the project's own guidance uses it: an event with an
// enabled reminder, an active loop at the stop, a ready step to nudge. Every count is the
// `/api/guidance` reply's; nothing moves unless the data changes.

import { useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { TRIGGERS, verdictTally, type GuidanceReply } from "@/lib/guidance";

/** The hook events in the figure's words (short: they sit in a narrow column). */
export const TRIGGER_SHORT: Record<string, string> = {
  prompt: "prompt",
  "session-start": "session start",
  compact: "compaction",
  stop: "stop",
  edit: "edit",
  failure: "failed check",
  tools: "tool call",
  "error-repeats": "repeated error",
};

export type GuidanceTarget = "reminders" | "loops" | "steps" | "deliveries";

const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const ROW = 23;
const TOP = 48;
const H = TOP + ROW * TRIGGERS.length + 4;

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

const keyAct = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fn();
  }
};

export default function GuidanceFigure({ g, onOpen, onTrigger }: { g: GuidanceReply; onOpen: (t: GuidanceTarget) => void; onTrigger: (trigger: string) => void }) {
  const [box, measured] = useWidth();
  const W = Math.max(600, measured);
  const reminders = g.reminders ?? [];
  const loops = g.loops ?? [];
  const steps = g.steps ?? [];
  const remOn = reminders.filter((r) => r.enabled);
  const remindersOff = g.settings?.reminders === "off";
  const loopsOff = g.settings?.loops === "off";
  const memoryOff = g.settings?.memory === "off";
  const active = loops.filter((l) => l.state === "active" && l.enabled);
  const lead = active[0] ?? null;
  const done = g.counts?.done ?? steps.filter((s) => s.status === "done").length;
  const ready = g.counts?.ready ?? steps.filter((s) => s.status === "ready").length;
  const blocked = g.counts?.blocked ?? steps.filter((s) => s.status === "blocked").length;
  const tally = verdictTally(g.verdicts);
  const judged = tally.held + tally.missed + tally.open + tally.unjudged;
  const delivered = g.delivered_total ?? 0;
  const perTrigger = (t: string) => remOn.filter((r) => r.trigger === t).length;
  const sentBy = (kind: string) => (g.deliveries ?? []).some((d) => d.kind.startsWith(kind));

  // Columns: the events, the three kinds of guidance, what was sent.
  const aX = 18;
  const aR = aX + 172;
  const bW = clamp(W * 0.25, 176, 280);
  const cW = clamp(W * 0.23, 168, 250);
  const cX = W - 18 - cW;
  const bX = Math.round(aR + (cX - aR - bW) / 2);
  const nodeH = 54;
  const gapY = (H - TOP + 12 - nodeH * 3) / 2;
  const nodeY = (i: number) => TOP - 12 + i * (nodeH + gapY);
  const rowY = (i: number) => TOP + i * ROW;
  const mid = (i: number) => nodeY(i) + nodeH / 2;
  const cTop = nodeY(0);
  const cH = nodeY(2) + nodeH - cTop;

  const wire = (x1: number, y1: number, x2: number, y2: number) => {
    const m = (x1 + x2) / 2;
    return `M${x1} ${y1} C${m} ${y1} ${m} ${y2} ${x2} ${y2}`;
  };
  const lit = { stroke: "var(--sense-pale)", strokeOpacity: 0.85, strokeWidth: 1 };
  const dim = { stroke: "var(--line-3)", strokeOpacity: 0.75, strokeWidth: 0.8, strokeDasharray: "1 3" };

  const stopI = TRIGGERS.indexOf("stop");
  const promptI = TRIGGERS.indexOf("prompt");
  const loopLit = active.length > 0 && !loopsOff && !memoryOff;
  const stepLit = ready > 0 && !remindersOff && !memoryOff;

  const node = (i: number, target: GuidanceTarget, title: string, off: boolean, label: string, body: ReactNode) => (
    <g className="fig-node" role="button" tabIndex={0} aria-label={label} onClick={() => onOpen(target)} onKeyDown={keyAct(() => onOpen(target))} data-e2e="guidance-fig-node" data-target={target}>
      <title>{label}</title>
      <rect className="fig-box" x={bX} y={nodeY(i)} width={bW} height={nodeH} fill="var(--bg-1)" fillOpacity={0.7} stroke="var(--line-3)" strokeDasharray={off ? "3 3" : undefined} />
      <text x={bX + 10} y={nodeY(i) + 16} className="mono" fontSize={9.5} letterSpacing="0.14em" fill="var(--text-3)">
        {title}
      </text>
      {off && (
        <text x={bX + bW - 10} y={nodeY(i) + 16} textAnchor="end" className="mono" fontSize={9.5} fill="var(--attention)">
          setting off
        </text>
      )}
      {body}
    </g>
  );

  return (
    <div ref={box} className="night ticks min-w-0 overflow-x-auto border border-line" data-e2e="guidance-figure">
      {measured > 0 && (
        <svg width={W} height={H + 14} viewBox={`0 0 ${W} ${H + 14}`} className="block" role="group" aria-label={`Guidance circuit: ${remOn.length} of ${reminders.length} reminders on, ${active.length} loops active, ${done} of ${steps.length} steps done, ${delivered} sent to the agent`}>
          {/* Column heads */}
          <text x={aX} y={22} className="mono" fontSize={9.5} letterSpacing="0.18em" fill="var(--text-4)">
            HOOK EVENTS
          </text>
          <text x={bX} y={22} className="mono" fontSize={9.5} letterSpacing="0.18em" fill="var(--text-4)">
            GUIDANCE
          </text>
          <text x={cX} y={22} className="mono" fontSize={9.5} letterSpacing="0.18em" fill="var(--text-4)">
            TO THE AGENT
          </text>

          {/* Wires: every event can carry a reminder; the stop carries loops; prompts and stops carry the step nudge. */}
          {TRIGGERS.map((t, i) => (
            <path key={`w-${t}`} d={wire(aR, rowY(i) - 3, bX, mid(0))} fill="none" {...(perTrigger(t) > 0 && !remindersOff && !memoryOff ? lit : dim)} />
          ))}
          <path d={wire(aR, rowY(stopI) - 3, bX, mid(1))} fill="none" {...(loopLit ? lit : dim)} />
          <path d={wire(aR, rowY(promptI) - 3, bX, mid(2))} fill="none" {...(stepLit ? lit : dim)} />
          <path d={wire(aR, rowY(stopI) - 3, bX, mid(2))} fill="none" {...(stepLit ? lit : dim)} />
          {(["reminder", "loop", "step"] as const).map((k, i) => (
            <path key={`s-${k}`} d={wire(bX + bW, mid(i), cX, cTop + cH * (0.3 + i * 0.2))} fill="none" {...(sentBy(k) ? lit : dim)} />
          ))}

          {/* The hook events: one that carries an enabled reminder is bright, with its count. */}
          {TRIGGERS.map((t, i) => {
            const n = perTrigger(t);
            const y = rowY(i);
            const say = `${TRIGGER_SHORT[t] ?? t}: ${n} ${n === 1 ? "reminder" : "reminders"} on. Add a reminder at this event.`;
            return (
              <g key={t} className="fig-node" role="button" tabIndex={0} aria-label={say} onClick={() => onTrigger(t)} onKeyDown={keyAct(() => onTrigger(t))} data-e2e="guidance-fig-trigger" data-trigger={t} data-count={n}>
                <title>{say}</title>
                <rect className="fig-box" x={aX - 6} y={y - 14} width={aR - aX + 6} height={ROW - 3} fill="transparent" stroke="transparent" />
                <rect x={aX} y={y - 8} width={6} height={6} fill={n > 0 ? "var(--sense-pale)" : "none"} stroke={n > 0 ? "var(--sense-pale)" : "var(--line-3)"} />
                <text x={aX + 14} y={y} className="mono" fontSize={11} fill={n > 0 ? "var(--text-1)" : "var(--text-3)"}>
                  {TRIGGER_SHORT[t] ?? t}
                </text>
                <text x={aR - 8} y={y} textAnchor="end" className="mono" fontSize={10.5} fill={n > 0 ? "var(--text-1)" : "var(--text-4)"}>
                  {n > 0 ? n : "·"}
                </text>
                <circle cx={aR} cy={y - 3} r={1.6} fill={n > 0 ? "var(--sense-pale)" : "var(--line-3)"} />
              </g>
            );
          })}

          {node(
            0,
            "reminders",
            "REMINDERS",
            remindersOff,
            `Reminders: ${remOn.length} on, ${reminders.length} set. Open the reminders.`,
            <>
              <text x={bX + 10} y={nodeY(0) + 41} className="mono" fontSize={19} fill={remOn.length ? "var(--text-1)" : "var(--text-4)"}>
                {remOn.length}
              </text>
              <text x={bX + 10 + String(remOn.length).length * 12 + 8} y={nodeY(0) + 41} className="mono" fontSize={10.5} fill="var(--text-3)">
                on · {reminders.length} set
              </text>
            </>,
          )}
          {node(
            1,
            "loops",
            "LOOPS",
            loopsOff,
            `Loops: ${active.length} active, ${g.outcomes_total ?? 0} ended. Open the loops.`,
            <>
              <text x={bX + 10} y={nodeY(1) + 41} className="mono" fontSize={19} fill={active.length ? "var(--text-1)" : "var(--text-4)"}>
                {active.length}
              </text>
              <text x={bX + 10 + String(active.length).length * 12 + 8} y={nodeY(1) + 41} className="mono" fontSize={10.5} fill="var(--text-3)">
                active · {g.outcomes_total ?? 0} ended
              </text>
              {/* The first active loop's iterations, one pip per allowed continuation. */}
              {lead &&
                Array.from({ length: Math.min(20, lead.max) }, (_, k) => (
                  <rect key={k} x={bX + bW - 10 - Math.min(20, lead.max) * 5 + k * 5} y={nodeY(1) + 9} width={3.5} height={8} fill={k < lead.iterations ? "var(--sense-pale)" : "none"} stroke={k < lead.iterations ? "var(--sense-pale)" : "var(--line-3)"} strokeWidth={0.6} />
                ))}
            </>,
          )}
          {node(
            2,
            "steps",
            "STEP GRAPH",
            false,
            `Step graph: ${done} done, ${ready} ready, ${blocked} blocked${g.next_step ? `, next ${g.next_step}` : ""}. Open the step graph.`,
            <>
              <text x={bX + 10} y={nodeY(2) + 41} className="mono" fontSize={19} fill={steps.length ? "var(--text-1)" : "var(--text-4)"}>
                {done}/{steps.length}
              </text>
              <text x={bX + 10 + `${done}/${steps.length}`.length * 12 + 8} y={nodeY(2) + 41} className="mono" fontSize={10.5} fill="var(--text-3)">
                done · {ready} ready{bW > 220 ? ` · ${blocked} blocked` : ""}
              </text>
              {g.next_step && (
                <text x={bX + bW - 10} y={nodeY(2) + 16} textAnchor="end" className="mono" fontSize={9.5} fill="var(--sense-pale)">
                  next {g.next_step}
                </text>
              )}
              {steps.length > 0 && (
                <g>
                  <rect x={bX} y={nodeY(2) + nodeH - 2} width={(bW * done) / steps.length} height={2} fill="var(--verified)" />
                  <rect x={bX + (bW * done) / steps.length} y={nodeY(2) + nodeH - 2} width={(bW * ready) / steps.length} height={2} fill="var(--sense-pale)" />
                </g>
              )}
            </>,
          )}

          {/* What was sent, and the ledger's verdicts on it. */}
          <g className="fig-node" role="button" tabIndex={0} aria-label={`Sent to the agent: ${delivered}. ${tally.held} held, ${tally.missed} missed, ${tally.open} open, ${tally.unjudged} not judged. Open the deliveries.`} onClick={() => onOpen("deliveries")} onKeyDown={keyAct(() => onOpen("deliveries"))} data-e2e="guidance-fig-node" data-target="deliveries">
            <title>{`Sent to the agent: ${delivered}. Open the deliveries.`}</title>
            <rect className="fig-box" x={cX} y={cTop} width={cW} height={cH} fill="var(--bg-1)" fillOpacity={0.7} stroke="var(--line-3)" strokeDasharray={memoryOff ? "3 3" : undefined} />
            <text x={cX + 10} y={cTop + 16} className="mono" fontSize={9.5} letterSpacing="0.14em" fill="var(--text-3)">
              SENT
            </text>
            {memoryOff && (
              <text x={cX + cW - 10} y={cTop + 16} textAnchor="end" className="mono" fontSize={9.5} fill="var(--attention)">
                memory off
              </text>
            )}
            <text x={cX + 10} y={cTop + 46} className="mono" fontSize={24} fill={delivered ? "var(--text-1)" : "var(--text-4)"}>
              {delivered}
            </text>
            <text x={cX + 10 + String(delivered).length * 15 + 8} y={cTop + 46} className="mono" fontSize={10.5} fill="var(--text-3)">
              delivered
            </text>
            <text x={cX + 10} y={cTop + 72} className="mono" fontSize={9.5} letterSpacing="0.14em" fill="var(--text-4)">
              VERDICTS
            </text>
            <rect x={cX + 10} y={cTop + 79} width={cW - 20} height={4} fill="var(--line-1)" />
            {judged > 0 &&
              (() => {
                const parts: [number, string][] = [
                  [tally.held, "var(--verified)"],
                  [tally.missed, "var(--attention)"],
                  [tally.open, "var(--sense-pale)"],
                  [tally.unjudged, "var(--text-4)"],
                ];
                let x = cX + 10;
                return parts.map(([n, c], k) => {
                  const w = ((cW - 20) * n) / judged;
                  const r = <rect key={k} x={x} y={cTop + 79} width={w} height={4} fill={c} />;
                  x += w;
                  return r;
                });
              })()}
            {(
              [
                ["held", tally.held, "var(--verified)"],
                ["missed", tally.missed, "var(--attention)"],
                ["open", tally.open, "var(--sense-pale)"],
                ["not judged", tally.unjudged, "var(--text-4)"],
              ] as const
            ).map(([word, n, c], k) => (
              <g key={word}>
                <rect x={cX + 10} y={cTop + 96 + k * 16} width={5} height={5} fill={c} fillOpacity={n > 0 ? 1 : 0.35} />
                <text x={cX + 22} y={cTop + 102 + k * 16} className="mono" fontSize={10.5} fill={n > 0 ? "var(--text-2)" : "var(--text-4)"}>
                  {word}
                </text>
                <text x={cX + cW - 10} y={cTop + 102 + k * 16} textAnchor="end" className="mono" fontSize={10.5} fill={n > 0 ? "var(--text-1)" : "var(--text-4)"}>
                  {n}
                </text>
              </g>
            ))}
          </g>
        </svg>
      )}
    </div>
  );
}
