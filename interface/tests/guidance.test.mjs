import "./register-ts.mjs";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const g = await import("../src/lib/guidance.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const src = (p) => readFileSync(resolve(HERE, "../src", p), "utf8");

/* ---------- Words ---------- */

test("triggers, repetition and loop conditions read as the engine means them", () => {
  assert.equal(g.triggerWords({ trigger: "prompt", every: 1, glob: null }), "at every prompt");
  assert.equal(g.triggerWords({ trigger: "prompt", every: 3, glob: null }), "at every 3rd prompt");
  assert.equal(g.triggerWords({ trigger: "tools", every: 12, glob: null }), "after every 12th tool call");
  assert.equal(g.triggerWords({ trigger: "edit", every: 1, glob: "src/**" }), "before an edit of src/**");
  assert.equal(g.triggerWords({ trigger: "edit", every: 1, glob: null }), "before every edit");
  assert.equal(g.repeatWords({ repeat: "max", max: 1 }), "at most 1 time a session");
  assert.equal(g.repeatWords({ repeat: "once", max: 1 }), "once a session");
  assert.equal(g.untilWords({ until: "goal-done", goal: 7 }), "until goal 7 is closed");
  assert.equal(g.untilWords({ until: "steps-done", goal: null }), "until every step is done");
  assert.equal(g.nth(1), "1st");
  assert.equal(g.nth(11), "11th");
  assert.equal(g.nth(22), "22nd");
  assert.deepEqual(g.listOf(" a, b ,,a, c"), ["a", "b", "c"]);
  assert.equal(g.loopShare({ iterations: 5, max: 3 }), 1);
  assert.equal(g.fit("a very long step title indeed", 12), "a very lo...");
  assert.equal(g.channelWords({ channel: "none", host: "antigravity" }), "not acted (Antigravity)");
});

test("stall, budgets, combined conditions, error repeats and loop ends read as the engine means them", () => {
  assert.equal(g.triggerWords({ trigger: "error-repeats", every: 1, glob: null, error: "E0308", count: 3 }), "when E0308 comes back in 3 tool results");
  assert.equal(g.triggerWords({ trigger: "error-repeats", every: 1, glob: null, error: "E0308", count: 9 }), "when E0308 comes back in 5 tool results");
  assert.equal(g.untilWords({ until: "all-of", goal: 2, parts: ["check-passes", "goal-done"] }), "until all of: a check passes and the goal is closed (goal 2)");
  assert.equal(g.untilWords({ until: "any-of", goal: null, parts: ["check-passes", "steps-done"] }), "until any of: a check passes or the steps are done");
  assert.equal(g.budgetWords({ budget_ms: 90000, budget_tools: 40 }), "budget 90 s, 40 tool calls");
  assert.equal(g.budgetWords({ budget_ms: null, budget_tools: null }), "");
  assert.equal(g.msWords(3600000), "60 min");
  assert.equal(g.msWords(7200000), "2 h");
  assert.equal(g.loopTone("stalled"), "attention");
  assert.equal(g.loopTone("budget-spent"), "attention");
  assert.equal(g.loopTone("done"), "verified");
  assert.equal(g.endWords("cap"), "maximum reached");
  assert.equal(g.endWords("removed"), "removed while active");
  assert.equal(g.endTone("met"), "verified");
  assert.equal(g.endTone("ended"), "muted");
  assert.ok(g.CONDITIONS.includes("all-of") && g.CONDITIONS.includes("any-of") && g.TRIGGERS.includes("error-repeats"));
});

/* ---------- The step graph ---------- */

test("the step graph lays dependencies left to right and survives a cycle or a missing step", () => {
  const steps = [
    { id: "a", title: "A", status: "done", depends_on: [] },
    { id: "b", title: "B", status: "ready", depends_on: ["a"] },
    { id: "c", title: "C", status: "ready", depends_on: ["a"] },
    { id: "d", title: "D", status: "blocked", depends_on: ["b", "c", "gone"] },
  ];
  const l = g.layout(steps);
  const at = Object.fromEntries(l.nodes.map((n) => [n.id, n]));
  assert.deepEqual([at.a.layer, at.b.layer, at.c.layer, at.d.layer], [0, 1, 1, 2]);
  assert.deepEqual([at.b.row, at.c.row], [0, 1]);
  assert.ok(at.b.x > at.a.x && at.d.x > at.b.x);
  assert.equal(l.edges.length, 4, "the missing dependency draws no edge");
  assert.ok(l.width > 0 && l.height > 0);
  assert.match(g.edgePath(l.edges[0]), /^M\d+ \d+ C/);
  // A cycle (the engine refuses one, but an old file could hold it) still lays out.
  const cyc = g.layout([
    { id: "x", title: "X", status: "blocked", depends_on: ["y"] },
    { id: "y", title: "Y", status: "blocked", depends_on: ["x"] },
  ]);
  assert.equal(cyc.nodes.length, 2);
  assert.deepEqual(g.layout([]), { nodes: [], edges: [], width: 0, height: 0 });
});

test("OR-join edges and probe steps are laid out and marked apart from AND edges", () => {
  const l = g.layout([
    { id: "a", title: "A", status: "done", depends_on: [] },
    { id: "b", title: "B", status: "ready", depends_on: [] },
    { id: "c", title: "C", status: "ready", depends_on: [], depends_any: ["a", "b"] },
    { id: "p", title: "P", status: "blocked", depends_on: ["c"], probe: true },
  ]);
  const at = Object.fromEntries(l.nodes.map((n) => [n.id, n]));
  assert.deepEqual([at.a.layer, at.b.layer, at.c.layer, at.p.layer], [0, 0, 1, 2], "an OR dependency counts for the column");
  assert.deepEqual(l.edges.map((e) => `${e.from}>${e.to}:${e.any ? "any" : "all"}`), ["a>c:any", "b>c:any", "c>p:all"]);
  assert.deepEqual([at.p.probe, at.c.probe], [true, false]);
});

/* ---------- Verdicts ---------- */

test("delivery verdicts read as the evidence ledger's rules mean them", () => {
  const v = (kind, state, strength = null, why = undefined) => ({ delivery: `guide:${kind}:a:s:1`, kind, item: "a", window: "closed", verdict: { rule: "r", state, strength, why } });
  assert.equal(g.verdictWords(v("step", "used", "direct")), "followed (direct)");
  assert.equal(g.verdictWords(v("step", "passed-over", "weak")), "passed over (weak)");
  assert.equal(g.verdictWords(v("loop", "verified", "direct")), "condition met before the cap (direct)");
  assert.equal(g.verdictWords(v("loop", "refuted", "direct")), "ended without its condition met (cap, stalled or budget spent) (direct)");
  assert.equal(g.verdictWords(v("reminder", "verified", "indirect")), "check passed after edits under its glob (indirect)");
  assert.equal(g.verdictWords(v("reminder", "unknown", null, "the reminder names no edit glob")), "not judged: the reminder names no edit glob");
  assert.equal(g.verdictWords(null), null);
  assert.equal(g.verdictTone(v("step", "used")), "verified");
  assert.equal(g.verdictTone(v("loop", "refuted")), "attention");
  assert.equal(g.verdictTone(v("reminder", "none")), "muted");
  assert.deepEqual(g.verdictTally({ by_kind: { step: { used: 2, none: 1 }, loop: { refuted: 1 }, reminder: { unknown: 3, verified: 1 } } }), { held: 3, missed: 1, open: 1, unjudged: 3 });
  assert.deepEqual(g.verdictTally(undefined), { held: 0, missed: 0, open: 0, unjudged: 0 });
});

/* ---------- The section ---------- */

test("the Guidance section keeps its end-to-end hooks and no em dash", () => {
  const panel = src("components/GuidancePanel.tsx");
  for (const id of ["guidance", "guidance-reminders", "guidance-reminder-add", "guidance-loops", "guidance-loop-add", "guidance-loop-end", "guidance-graph", "guidance-step-add", "guidance-step-done", "guidance-deliveries", "guidance-verdicts", "guidance-delivery-verdict", "guidance-outcomes", "guidance-loop-parts", "guidance-loop-stall", "guidance-loop-budget-ms", "guidance-loop-budget-tools", "guidance-reminder-error", "guidance-outcome", "guidance-step-edge", "guidance-step-after-any", "guidance-step-probe"]) {
    assert.ok(panel.includes(`data-e2e="${id}"`) || panel.includes(` e2e="${id}"`), `data-e2e="${id}" missing`);
  }
  assert.ok(panel.includes('path="/api/guidance"'), "removal goes through the delete flow");
  const page = src("app/memory/page.tsx");
  assert.ok(page.includes("GuidancePanel") && page.includes('"guidance"'), "the Memory page holds the section and its rail tab");
  for (const f of ["components/GuidancePanel.tsx", "lib/guidance.ts"]) assert.ok(!src(f).includes(String.fromCharCode(0x2014)), `${f} holds an em dash`);
});
