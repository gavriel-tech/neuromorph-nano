"use client";

// Distillation: curated teacher pairs trained into a learned adapter cartridge for a MicroModel,
// through the held-out gate. One shared builder (`DistilBuilder`, also used by the Cartridges Build
// tab), the teacher data (`DistilData`: import, sets, outcome labels) and the cartridges it made
// (`DistilCartridges`: gate figures, plug in or out). The same routes as MCP: `/api/training/distil`
// (nm_learning distil, distil_import, distil_label), `/api/lab/build` kind `distil` (nm_cartridges
// distil_build) and `/api/learned-units` (nm_learning unit_on, unit_off).

import { maskPathsIn } from "@/lib/privacy";
import Link from "next/link";
import TeachPanel from "@/components/TeachPanel";
import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { roleTitle } from "@/components/MicroModel";
import { Panel, Pill, Placeholder, Segmented, Switch } from "@/components/ui";
import { post, refreshAll, useApi } from "@/lib/api";

type Note = { text: string; tone: "verified" | "error" };
type GateCheck = { pass: boolean; figures: string };
export type DistilSet = { set: string; rows: number; healthy: number; splits: Record<string, number>; outcomes: Record<string, number>; teachers: string[]; licences: string[]; with_topk: number; with_score?: number; cartridge: string };
export type DistilRowView = { id: string; set: string; split: string; outcome: string; teacher: string; licence: string; topk: boolean; prompt: string; completion: string };
export type DistilCartridge = {
  id: string;
  key: string;
  role: string;
  model: string;
  version: number;
  versions: number;
  technique: string;
  set: string;
  base: string;
  scale?: string | number;
  gate?: Record<string, GateCheck> | null;
  measured: string;
  teachers?: string[];
  made: string;
  plugged: boolean;
  path: string;
};
export type DistilReply = {
  file: string;
  rows_total: number;
  sets: DistilSet[];
  rows: DistilRowView[];
  outcomes: string[];
  techniques: string[];
  targets?: string[];
  fills?: { target: string; technique: string }[];
  built: { role: string; target?: string; technique: string }[];
  unavailable?: { role: string; target: string; technique: string; why: string }[];
  roles: string[];
  cartridges: DistilCartridge[];
  gate: string;
  gate_scores?: string;
  gate_similarity?: string;
};
type Build = { status: string; kind?: string; role: string; job: string; message: string };
type Surgery = { lab_build?: Build; job_filter_build?: Build };

const field = "mono h-9 min-w-0 border-b border-line-strong bg-transparent text-[12px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";
const btn = "press mono min-h-9 border px-3 text-[10.5px] uppercase tracking-[0.12em] disabled:opacity-40";
const linkBtn = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";

const TARGET_WORD: Record<string, string> = { adapter: "adapter cartridge", filter: "filter cartridge (job filter)", preference: "preference cartridge", readout: "read-out cartridge" };
const TECHNIQUE_DOES: Record<string, string> = {
  lens: "A low-rank map at the MicroModel's final norm, fitted in closed form so its next-token choices (for the Reranker, its yes or no score) follow the teacher's.",
  "teacher-loop": "The lens refitted over rounds: each round weighs up the pairs the student still follows worst, until the tune split stops gaining.",
  channel: "Per-channel gain and bias toward the same targets (the job filter's form).",
  unit: "A learning unit at the Reader's learning target, its up matrix fitted forward-only (MeZO) against the distil loss within a time budget per fit; it loads as every learning unit does (serve --unit).",
  readout: "A ridge read-out of the Reranker's score toward the teacher's scores, served along its yes-minus-no direction (for a preference cartridge: verdict counts per circumstance).",
  steer: "A steering vector at the last layer's residual, the mean shift toward the teacher in closed form, its scale measured with the vector in place; served by the MicroModel's steer seat.",
};

function useNote() {
  const [note, setNote] = useState<Note | null>(null);
  const [busy, setBusy] = useState("");
  const run = async (what: string, fn: () => Promise<string>) => {
    setBusy(what);
    try {
      setNote({ text: await fn(), tone: "verified" });
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy("");
  };
  return { note, busy, run };
}

/** The live state of the lab build when it is a distillation (else null). */
export function useDistilBuild(): Build | null {
  const surgery = useApi<Surgery>("/api/surgery", 3000);
  const b = surgery.data?.lab_build ?? surgery.data?.job_filter_build;
  return b && b.kind === "distil" && b.status !== "idle" ? b : null;
}

/** The distillation builder: MicroModel, technique, teacher data set and rounds, then build. One
 *  build at a time; a cartridge is written only when the held-out gate passes. */
export function DistilBuilder({ e2e = "distil-builder", wide = false }: { e2e?: string; wide?: boolean }) {
  const data = useApi<DistilReply>("/api/training/distil?limit=0", 10000);
  const build = useDistilBuild();
  const { note, busy, run } = useNote();
  const [role, setRole] = useState("writer");
  const [target, setTarget] = useState("adapter");
  const [technique, setTechnique] = useState("lens");
  const [set, setSet] = useState("all");
  const [rounds, setRounds] = useState("4");
  const [budget, setBudget] = useState("20");
  const d = data.data;
  const targets = d?.targets ?? ["adapter", "filter", "preference", "readout"];
  const fills = d?.fills ?? [];
  const techniques = (d?.techniques ?? ["lens", "teacher-loop", "channel", "unit", "readout", "steer"]).filter((t) => !fills.length || fills.some((f) => f.target === target && f.technique === t));
  const built = (r: string, t: string) => (d?.built ?? []).some((b) => b.role === r && (b.target ?? "adapter") === target && b.technique === t);
  const pickTarget = (k: string) => {
    setTarget(k);
    const first = fills.find((f) => f.target === k && (d?.built ?? []).some((b) => b.role === role && b.target === k && b.technique === f.technique)) ?? fills.find((f) => f.target === k);
    if (first) setTechnique(first.technique);
  };
  const why = (d?.unavailable ?? []).find((u) => u.role === role && u.target === target && u.technique === technique)?.why;
  const sets = d?.sets ?? [];
  const chosen = set === "all" ? sets.reduce((a, s) => a + s.healthy, 0) : (sets.find((s) => s.set === set)?.healthy ?? 0);
  const running = build?.status === "running";
  const start = () =>
    run("build", async () => {
      const body: Record<string, unknown> = { kind: "distil", role, target, technique, set };
      if (technique === "teacher-loop") body.rounds = Number(rounds) || 4;
      if (technique === "unit") body.budget_s = Math.min(120, Math.max(1, Number(budget) || 20));
      await post("/api/lab/build", body);
      for (const p of ["/api/surgery", "/api/live", "/api/audit"]) refreshAll(p);
      return `Distillation into a ${TARGET_WORD[target] ?? target} for the ${roleTitle(role)} started (${technique}, set ${set}). It fits on the fit split, chooses on the tune split and writes a cartridge only if the held-out gate passes.`;
    });
  return (
    <div className={`grid gap-4 ${wide ? "lg:grid-cols-2 lg:items-start lg:gap-x-8 lg:[&>*:nth-child(n+5)]:col-span-2" : ""}`} data-e2e={e2e}>
      <label className="grid gap-1">
        <span className="eyebrow">MicroModel</span>
        <select value={role} onChange={(e) => setRole(e.target.value)} className={`${field} w-full`} data-e2e="distil-role">
          {(d?.roles ?? ["reader", "writer", "embedder", "reranker"]).map((r) => (
            <option key={r} value={r} className="bg-bg-1">
              {roleTitle(r)}
            </option>
          ))}
        </select>
      </label>
      <label className="grid gap-1">
        <span className="eyebrow">learned cartridge</span>
        <select value={target} onChange={(e) => pickTarget(e.target.value)} className={`${field} w-full`} data-e2e="distil-target">
          {targets.map((k) => (
            <option key={k} value={k} className="bg-bg-1">
              {TARGET_WORD[k] ?? k}
              {(d?.built ?? []).some((b) => b.role === role && b.target === k) ? "" : " (not available yet)"}
            </option>
          ))}
        </select>
      </label>
      <div className="grid gap-1.5" data-e2e="distil-technique">
        <span className="eyebrow">technique</span>
        <select value={technique} onChange={(e) => setTechnique(e.target.value)} className={`${field} w-full`} aria-label="Technique">
          {techniques.map((t) => (
            <option key={t} value={t} className="bg-bg-1">
              {t}
              {built(role, t) ? "" : " (not available yet)"}
            </option>
          ))}
        </select>
        <span className="mono text-[10px] leading-relaxed text-quiet">{TECHNIQUE_DOES[technique] ?? ""}</span>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-3">
        <label className="grid gap-1">
          <span className="eyebrow">teacher data</span>
          <select value={set} onChange={(e) => setSet(e.target.value)} className={`${field} w-full`} data-e2e="distil-set">
            <option value="all" className="bg-bg-1">
              every set
            </option>
            {sets.map((s) => (
              <option key={s.set} value={s.set} className="bg-bg-1">
                {s.set} · {s.healthy} healthy pairs
              </option>
            ))}
          </select>
        </label>
        {technique === "unit" && (
          <label className="grid gap-1">
            <span className="eyebrow">seconds per fit</span>
            <input value={budget} onChange={(e) => setBudget(e.target.value)} aria-label="Seconds per fit" className={`${field} w-16`} />
          </label>
        )}
        {technique === "teacher-loop" && (
          <label className="grid gap-1">
            <span className="eyebrow">rounds</span>
            <input value={rounds} onChange={(e) => setRounds(e.target.value)} aria-label="Rounds" className={`${field} w-16`} />
          </label>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" disabled={!!busy || running || !built(role, technique) || chosen === 0} onClick={() => void start()} data-e2e="distil-build" className={`${btn} border-sense-pale/70 text-sense-pale hover:bg-sense-pale/10 hover:text-text`}>
          {running ? "training" : "train"}
        </button>
        <span className="mono text-[10px] text-quiet">
          {!built(role, technique) ? `${technique} into a ${TARGET_WORD[target] ?? target} for the ${roleTitle(role)} is not available yet${why ? `: ${why}` : ""}` : chosen === 0 ? "no healthy teacher pairs yet; import some first" : `${chosen} healthy pairs · kept only if the held-out gate passes`}
        </span>
      </div>
      {build && (
        <div className="mono grid gap-1 border-l-2 border-line-strong pl-3 text-[10.5px]" data-e2e="distil-status">
          <span className="flex items-center gap-2">
            <Pill tone={build.status === "done" ? (build.message.includes("REFUSED") ? "attention" : "verified") : build.status === "error" ? "error" : build.status === "stopped" ? "attention" : "signal"}>{build.status === "done" && build.message.includes("REFUSED") ? "refused by its gate" : build.status}</Pill>
            <span className="text-text">
              {roleTitle(build.role)} · {build.job}
            </span>
          </span>
          {build.message && <span className="whitespace-pre-wrap break-words text-faint">{maskPathsIn(build.message).split(" | ").join("\n")}</span>}
        </div>
      )}
      <p className="mono text-[10px] leading-relaxed text-faint" title={role === "reranker" ? d?.gate_scores : role === "embedder" ? d?.gate_similarity : d?.gate}>
        {role === "embedder"
          ? "Teacher data: pairs with a similarity score from 0 to 1 (prompt the query, completion the document). Expected effect: a request's similarity to the pieces follows the teacher's; only request vectors are mapped, so the stored vectors stay valid. A new cartridge starts unplugged."
          : role === "reranker"
          ? "Teacher data: pairs with a score from 0 to 1 (prompt the query, completion the document). Expected effect: the judge's yes or no score follows the teacher's. A new cartridge starts unplugged on the MicroModel's deck."
          : "Expected effect: the teacher's style, format and wording, not new knowledge. A new cartridge starts unplugged on the MicroModel's deck."}
      </p>
      <NoteLine note={note} />
    </div>
  );
}

/** The teacher data: import a JSON lines file of teacher pairs, the sets with their splits and
 *  outcomes, an outcome label per set, and the newest pairs. */
export function DistilData() {
  const data = useApi<DistilReply>("/api/training/distil?limit=12", 10000);
  const { note, busy, run } = useNote();
  const [file, setFile] = useState("");
  const [set, setSet] = useState("");
  const [licence, setLicence] = useState("");
  const d = data.data;
  const importIt = () =>
    run("import", async () => {
      const r = await post<{ set: string; rows: number; passed_over: number; reference: string }>("/api/training/distil/import", { file: file.trim(), set: set.trim() || undefined, licence: licence.trim() || undefined });
      for (const p of ["/api/training", "/api/audit"]) refreshAll(p);
      setFile("");
      return `Imported ${r.rows} teacher pairs into set ${r.set} (${r.passed_over} already here); the file is listed as ${r.reference} with its licence and hash. Undo in Audit.`;
    });
  const labelSet = (s: string, outcome: string) =>
    run(`label:${s}`, async () => {
      await post("/api/training/distil/label", { set: s, outcome, why: "labelled on the Learning page" });
      for (const p of ["/api/training", "/api/audit"]) refreshAll(p);
      return `Set ${s} labelled ${outcome}. ${outcome === "used" || outcome === "superseded" ? "Its pairs are trained on." : "Its pairs are kept apart and never trained on."} Undo in Audit.`;
    });
  return (
    <div className="grid gap-4" data-e2e="distil-data">
      <div className="grid gap-2">
        <input value={file} onChange={(e) => setFile(e.target.value)} placeholder="a JSON lines file of teacher pairs on this machine" aria-label="Teacher pairs file" className={`${field} w-full`} spellCheck={false} />
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <input value={set} onChange={(e) => setSet(e.target.value)} placeholder="set name (the file's by default)" aria-label="Set" className={`${field} min-w-0 flex-1`} />
          <input value={licence} onChange={(e) => setLicence(e.target.value)} placeholder="licence (Apache-2.0, own)" aria-label="Licence" className={`${field} w-44`} />
          <button type="button" disabled={!!busy || !file.trim()} onClick={() => void importIt()} className={`${btn} border-sense-pale text-sense-pale hover:text-text`} data-e2e="distil-import">
            {busy === "import" ? "importing" : "import"}
          </button>
        </div>
        <span className="mono text-[10px] text-quiet" title="Each line: prompt, completion, teacher, licence, and optionally topk (the teacher's top log-probabilities per token). The split is fixed by the prompt's hash; nothing is added twice; nothing is sent anywhere.">
          prompt, completion, teacher, licence per line; split fixed by the prompt&apos;s hash
        </span>
      </div>
      {data.error && !d ? (
        <span className="mono text-[11px] text-error">{data.error}</span>
      ) : !d ? (
        <Placeholder className="h-16" what="distillation sources" />
      ) : d.sets.length === 0 ? (
        <span className="mono text-[11px] text-faint">No teacher data yet.</span>
      ) : (
        <ul className="grid gap-1.5" data-e2e="distil-sets">
          {d.sets.map((s) => (
            <li key={s.set} className="grid gap-1 border-b border-line/60 py-1.5">
              <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-2">
                <span className="mono truncate text-[11.5px] text-text">{s.set}</span>
                <span className="mono text-[10px] text-faint">
                  {s.rows} pairs · fit {s.splits.fit ?? 0} · tune {s.splits.tune ?? 0} · held out {s.splits["held-out"] ?? 0}
                </span>
              </div>
              <div className="mono flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1 text-[10px] text-faint">
                <span>{Object.entries(s.outcomes).map(([o, n]) => `${o} ${n}`).join(" · ")}</span>
                <span>{s.teachers.join(", ")}</span>
                <span>{s.licences.join("; ")}</span>
                {s.with_topk > 0 && <span>{s.with_topk} with teacher log-probabilities</span>}
              </div>
              <div className="flex flex-wrap gap-3">
                {s.healthy < s.rows && (
                  <button type="button" disabled={!!busy} onClick={() => void labelSet(s.set, "used")} className={linkBtn}>
                    label used
                  </button>
                )}
                {s.healthy > 0 && (
                  <button type="button" disabled={!!busy} onClick={() => void labelSet(s.set, "unused")} className={linkBtn} title="Kept, never trained on">
                    label unused
                  </button>
                )}
                {s.healthy > 0 && (
                  <button type="button" disabled={!!busy} onClick={() => void labelSet(s.set, "failed")} className={linkBtn} title="Kept apart as failure data">
                    label failed
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {d && d.rows.length > 0 && (
        <ul className="grid gap-1" data-e2e="distil-rows">
          {d.rows.map((r) => (
            <li key={r.id} className="grid min-w-0 gap-0.5 border-b border-line/40 py-1" title={`${r.id} · ${r.teacher} · ${r.licence}`}>
              <span className="mono truncate text-[10.5px] text-muted">{r.prompt}</span>
              <span className="mono truncate text-[10.5px] text-text">{r.completion}</span>
              <span className="mono text-[9.5px] text-quiet">
                {r.set} · {r.split} · {r.outcome || "not stated"}
                {r.topk ? " · log-probabilities" : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
      <NoteLine note={note} />
    </div>
  );
}

/** The distilled cartridges with their held-out gate figures, each plugged in or out. */
export function DistilCartridges() {
  const data = useApi<DistilReply>("/api/training/distil?limit=0", 10000);
  const { note, busy, run } = useNote();
  const carts = data.data?.cartridges ?? [];
  const plug = (c: DistilCartridge, on: boolean) =>
    run(c.id, async () => {
      await post("/api/learned-units", { unit: c.key, on });
      for (const p of ["/api/training", "/api/learned-units", "/api/audit"]) refreshAll(p);
      return `${c.id} ${on ? "plugged in; the MicroModel uses it from its next load" : "unplugged; the MicroModel's own outputs come back"}. Undo in Audit.`;
    });
  if (!data.data) return <Placeholder className="h-12" what="distillation" />;
  return (
    <div className="grid gap-2" data-e2e="distil-cartridges">
      {carts.length === 0 ? (
        <span className="mono text-[11px] text-faint">No distilled cartridge yet: a build writes one only when it passes its held-out gate.</span>
      ) : (
        <ul className="grid gap-2">
          {carts.map((c) => (
            <li key={c.id} className="grid gap-1 border-b border-line/60 py-2">
              <div className="flex min-w-0 items-center justify-between gap-3">
                <span className="mono truncate text-[11.5px] text-text" title={c.path}>
                  {c.id}@{c.version}
                </span>
                <Switch on={c.plugged} label={`Plug ${c.id} in`} busy={busy === c.id} onChange={(on) => void plug(c, on)} />
              </div>
              <span className="mono text-[10px] text-faint">
                {roleTitle(c.role)} · {c.technique} · set {c.set} · on {c.base} · scale {String(c.scale ?? "1")} · {c.versions} {c.versions === 1 ? "version" : "versions"}
              </span>
              {c.gate && (
                <ul className="grid gap-0.5">
                  {Object.entries(c.gate).map(([k, g]) => (
                    <li key={k} className="mono flex min-w-0 gap-2 text-[10px]">
                      <span className={g.pass ? "text-verified" : "text-error"}>{g.pass ? "pass" : "fail"}</span>
                      <span className="text-muted">{k}</span>
                      <span className="truncate text-faint" title={g.figures}>
                        {g.figures}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
      <span className="mono text-[10px] text-quiet">
        Plugged in or out, inspected and edited on{" "}
        <Link href="/cartridges/" className="link">
          Cartridges
        </Link>{" "}
        (the MicroModel&apos;s deck and the compatibility table).
      </span>
      <NoteLine note={note} />
    </div>
  );
}

/** The Learning page's Distillation section: the teacher data, the builder and what it made. */
export function DistillationSection() {
  const [view, setView] = useState("train");
  // The same reply the builder reads (one request), for the path's own counts.
  const data = useApi<DistilReply>("/api/training/distil?limit=0", 10000);
  const build = useDistilBuild();
  const d = data.data;
  const healthy = (d?.sets ?? []).reduce((a, x) => a + x.healthy, 0);
  const plugged = (d?.cartridges ?? []).filter((c) => c.plugged).length;
  const kinds = d?.targets?.length ?? null;
  // 2026-11-27: distillation's path as five steps with the engine's own counts: teacher data, what
  // to fill, the build, the held-out gate, the cartridges written. A step opens its view.
  const steps: { label: string; value: string | null; caption: string; view: string; tone?: string }[] = [
    { label: "Teacher data", value: d ? String(d.rows_total) : null, caption: d ? `${healthy} healthy pairs in ${d.sets.length} ${d.sets.length === 1 ? "set" : "sets"}` : data.error ? "did not answer" : "reading", view: "teacher data" },
    { label: "Fill", value: d ? (kinds !== null ? String(kinds) : "not reported") : null, caption: "kinds of learned cartridge, for every MicroModel", view: "train" },
    { label: "Train", value: build ? build.status : d ? "idle" : null, caption: build ? `${build.role} · ${build.job}` : "one build at a time; shown in Live Activity", view: "train", tone: build?.status === "running" ? "text-signal" : undefined },
    { label: "Gate", value: "held out", caption: "a cartridge is written only when the held-out split gains", view: "train", tone: "text-learned" },
    { label: "Cartridges", value: d ? String(d.cartridges.length) : null, caption: d ? `${plugged} plugged in` : "reading", view: "cartridges", tone: "text-verified" },
  ];
  return (
    <Panel title="Distillation" right={<Pill tone="learned">held-out gate</Pill>}>
      <div className="grid gap-5" data-e2e="distil-section">
        <ol className="grid grid-cols-2 border-t border-l border-line sm:grid-cols-5" aria-label="The path of a distillation" data-e2e="distil-path">
          {steps.map((st, i) => (
            <li key={st.label} className="min-w-0 border-r border-b border-line">
              <button type="button" onClick={() => setView(st.view)} title={`Open ${st.view}`} className={`press grid h-full w-full content-start gap-1.5 px-3 py-2.5 text-left hover:bg-bg-2/40 ${view === st.view ? "bg-bg-2/25" : ""}`}>
                <span className="flex items-baseline justify-between gap-2">
                  <span className="eyebrow truncate">
                    <span className="text-quiet">{i + 1}</span> {st.label}
                  </span>
                  {i < steps.length - 1 && (
                    <span aria-hidden className="mono hidden text-[11px] text-quiet 2xl:inline">
                      →
                    </span>
                  )}
                </span>
                <span className={`mono text-[15px] ${st.value === null ? "text-faint" : (st.tone ?? "text-text")}`}>{st.value ?? "reading"}</span>
                <span className="mono line-clamp-3 text-[10px] leading-snug text-faint" title={st.caption}>
                  {st.caption}
                </span>
              </button>
            </li>
          ))}
        </ol>
        <Segmented label="Distillation view" values={["train", "live teacher", "teacher data", "cartridges"]} value={view} onPick={setView} />
        {view === "train" && <DistilBuilder wide />}
        {view === "live teacher" && <TeachPanel />}
        {view === "teacher data" && <DistilData />}
        {view === "cartridges" && <DistilCartridges />}
      </div>
    </Panel>
  );
}
