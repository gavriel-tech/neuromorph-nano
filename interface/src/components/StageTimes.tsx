"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... The main
// core of the pages should be on actual work flow and core function visuals ... We need the
// interface to be amazing."): the waterfall and the stage table stay; the reading notes are in the
// tooltip of the table, and the panel fits the narrow side panel (it sits in the Audit rail).

import { useState } from "react";
import { LiveWaterfall } from "@/components/LivePanels";
import { Empty, Panel, Pill, Placeholder, Segmented } from "@/components/ui";
import { useApi } from "@/lib/api";
import { lastLiveCall, msShort, slowest, STAGE_NAMES, stageBars, stampText, type StageTimes } from "@/lib/stagetimes";

/** The per-stage breakdown of a warm question (`NANO_STAGE_TIMES`): first the live waterfall of
 *  the newest question from the engine's own stage events (`/api/live` and its stream, a running
 *  stage growing as it runs), then median and p95 ms per stage for each question set of the
 *  newest stage-bench report (`/api/stage-times`). */
export default function StageTimesPanel() {
  const q = useApi<StageTimes>("/api/stage-times", 30000);
  const d = q.data;
  const sets = d?.sets ?? [];
  const [pick, setPick] = useState<string | null>(null);
  const set = sets.find((x) => x.set === pick) ?? sets[0] ?? null;
  const { rows, total } = stageBars(set);
  const top = slowest(set);
  const live = lastLiveCall(d);
  return (
    <Panel
      title="Stage times"
      right={d ? <Pill tone={d.live ? "verified" : "muted"}>{d.live ? "live" : "from the newest stage bench"}</Pill> : undefined}
    >
      <div data-e2e="stage-times" className="@container">
        <div className="mb-4 border-b border-line/60 pb-3">
          <LiveWaterfall />
        </div>
        {live && live.length > 0 && (
          <div className="mb-4 border-b border-line/60 pb-3" data-e2e="stage-times-live">
            <div className="eyebrow mb-1.5">Newest call, live ({d?.live_calls?.length ?? 0} kept)</div>
            <div className="mono flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-faint">
              {live.map((r) => (
                <span key={r.stage}>
                  {STAGE_NAMES[r.stage] ?? r.stage} <span className="text-text">{msShort(r.ms)}</span>
                </span>
              ))}
            </div>
          </div>
        )}
        {q.error && !d ? (
          <Empty title="Stage times did not answer">{q.error.startsWith("404") ? "This engine build does not serve /api/stage-times yet." : q.error}</Empty>
        ) : !d ? (
          <Placeholder className="h-24" what="stage times" />
        ) : !set ? (
          <Empty title="No stage table yet" slot>
            Run nano-lab stage-bench to fill it.
          </Empty>
        ) : (
          <div className="grid gap-3">
            <div className="mono flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-[11px] text-faint">
              <span className="truncate">{d.header || d.file}</span>
              <span>{stampText(d.stamp)}</span>
            </div>
            {sets.length > 1 && <Segmented values={sets.map((x) => x.set)} value={set.set} onPick={setPick} label="Question set" />}
            {set.warm && <div className="mono text-[11px] text-muted">warm question ms: {set.warm}</div>}
            <table className="mono w-full text-[11px]" data-e2e="stage-times-table" title={`Median and p95 ms of every stage of a warm question. Stages overlap in part (some run beside others), so they need not add up to the total. ${d.live ? "The table is from the newest stage bench; the newest call above is live." : "Not live: the engine keeps per-call stage times on its standard error only."}`}>
              <thead>
                <tr className="text-left text-faint">
                  <th className="py-1 pr-3 font-normal">stage</th>
                  <th className="hidden w-[40%] py-1 pr-3 font-normal @xs:table-cell">share of median</th>
                  <th className="py-1 pr-3 text-right font-normal">median</th>
                  <th className="py-1 pr-3 text-right font-normal">p95</th>
                  <th className="py-1 text-right font-normal">n</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.stage} className="border-t border-line/60 text-muted">
                    <td className={`max-w-[10rem] truncate py-1 pr-3 ${top?.stage === r.stage ? "text-text" : ""}`}>{STAGE_NAMES[r.stage] ?? r.stage}</td>
                    <td className="hidden py-1 pr-3 @xs:table-cell">
                      <div className="h-1.5 w-full bg-line/50">
                        <div className="h-full bg-sense-pale" style={{ width: `${Math.min(100, Math.round(r.share * 100))}%` }} />
                      </div>
                    </td>
                    <td className="py-1 pr-3 text-right">{msShort(r.median_ms)}</td>
                    <td className="py-1 pr-3 text-right">{msShort(r.p95_ms)}</td>
                    <td className="py-1 text-right">{r.n}</td>
                  </tr>
                ))}
                {total && (
                  <tr className="border-t border-line-strong text-text">
                    <td className="py-1 pr-3">total</td>
                    <td className="hidden @xs:table-cell" />
                    <td className="py-1 pr-3 text-right">{msShort(total.median_ms)}</td>
                    <td className="py-1 pr-3 text-right">{msShort(total.p95_ms)}</td>
                    <td className="py-1 text-right">{total.n}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  );
}
