import "./register-ts.mjs";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const p = await import("../src/lib/preferences.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const src = (f) => readFileSync(resolve(HERE, "../src", f), "utf8");

test("the stored kind ban reads avoid; want and the others keep their names", () => {
  assert.equal(p.kindLabel("ban"), "avoid");
  assert.equal(p.kindLabel("want"), "want");
  assert.equal(p.kindLabel("replace"), "replace");
  assert.equal(p.kindLabel("ban", "avoid"), "avoid", "the engine's label is taken as given");
});

test("strengths read and print in the engine's range", () => {
  assert.equal(p.strengthText(0.5), "0.5");
  assert.equal(p.strengthText(1), "1");
  assert.equal(p.strengthText(0.25), "0.25");
  assert.equal(p.parseStrength("0.5"), 0.5);
  assert.equal(p.parseStrength(" 1.25 "), 1.25);
  assert.equal(p.parseStrength("2"), 2);
  assert.equal(p.parseStrength(".1"), 0.1);
  assert.equal(p.parseStrength("2.5"), null);
  assert.equal(p.parseStrength("0.05"), null);
  assert.equal(p.parseStrength("much"), null);
  assert.equal(p.parseStrength("-1"), null);
  assert.equal(p.ruleStrength({ kind: "want", strength: "1" }), 1);
  assert.equal(p.ruleStrength({ kind: "want", strength: 0.25 }), 0.25);
  assert.equal(p.ruleStrength({ kind: "ban" }), null);
});

test("the builder labels wants and avoided words, never ban, and edits a want's strength as a new version", () => {
  const b = src("components/PreferenceBuilder.tsx");
  assert.ok(b.includes("kindLabel(r.kind, r.label)"), "rule rows show the kind's label");
  assert.ok(b.includes('data-e2e="pref-strength"'), "a want shows its strength control");
  assert.ok(b.includes("{ rule: r.id, strength"), "the strength edit names the rule and its strength");
  assert.ok(/\/api\/preferences\/edit/.test(b), "edits go through the engine's edit route (a new version, audited)");
  const visible = b.replace(/\/\/.*$/gm, "").replace(/data-[a-z0-9-]+="[^"]*"/g, "");
  assert.ok(!/\bban\b/i.test(visible.replace(/kind === "ban"/g, "")), "no user-visible ban wording");
});
