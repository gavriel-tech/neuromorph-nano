import "./register-ts.mjs";
import assert from "node:assert/strict";
import test from "node:test";

const g = await import("../src/lib/paramgeom.ts");

const q = { name: "model.layers.3.self_attn.q_proj.weight", shape: [2048, 1024], rows: 2048, cols: 1024 };
const norm = { name: "model.norm.weight", shape: [1024], rows: 1, cols: 1024 };
const conv = { name: "conv.weight", shape: [8, 4, 3], rows: 8, cols: 12 };
const dims = [q, norm, conv];

test("editor paths read back to the rectangles they cover, as pedit::parse_path reads them", () => {
  assert.deepEqual(g.rectsOf(`${q.name}[10:11, 20:21]`, dims)?.rects, [{ r0: 10, r1: 11, c0: 20, c1: 21 }]);
  assert.deepEqual(g.rectsOf(`${q.name}[4:6, :]`, dims)?.rects, [{ r0: 4, r1: 6, c0: 0, c1: 1024 }]);
  assert.deepEqual(g.rectsOf(`${q.name}[:, 7:8]`, dims)?.rects, [{ r0: 0, r1: 2048, c0: 7, c1: 8 }]);
  assert.deepEqual(g.rectsOf(q.name, dims)?.rects, [{ r0: 0, r1: 2048, c0: 0, c1: 1024 }]);
  assert.deepEqual(g.rectsOf(`${norm.name}[4:10]`, dims)?.rects, [{ r0: 0, r1: 1, c0: 4, c1: 10 }]);
  // A flat range over three dimensions: the tail of a row, whole rows, the head of the last.
  assert.deepEqual(g.rectsOf(`${conv.name}[5:40]`, dims)?.rects, [
    { r0: 0, r1: 1, c0: 5, c1: 12 },
    { r0: 1, r1: 3, c0: 0, c1: 12 },
    { r0: 3, r1: 4, c0: 0, c1: 4 },
  ]);
  assert.deepEqual(g.rectsOf(`${conv.name}[12:36]`, dims)?.rects, [{ r0: 1, r1: 3, c0: 0, c1: 12 }]);
  // A layer, the whole object, an unknown tensor and a bad slice name no tensor rectangle.
  for (const p of ["L3", "*", "nope[0:1]", `${q.name}[5:2, 0:1]`, `${q.name}[0:1`]) assert.equal(g.rectsOf(p, dims), null, p);
});

test("rectangles overlap only when they share a weight", () => {
  const a = { r0: 0, r1: 4, c0: 0, c1: 4 };
  assert.ok(g.overlaps(a, { r0: 3, r1: 5, c0: 3, c1: 5 }));
  assert.ok(!g.overlaps(a, { r0: 4, r1: 5, c0: 0, c1: 4 }));
});

test("ruler steps are 1, 2 or 5 times a power of ten, whole for indices", () => {
  assert.equal(g.niceStep(1024, 10, 1), 100);
  assert.equal(g.niceStep(37, 5, 1), 5);
  assert.equal(g.niceStep(3, 10, 1), 1);
  assert.equal(g.niceStep(0.2, 4), 0.05);
  assert.deepEqual(g.ticks(0, 1023, 200), [0, 200, 400, 600, 800, 1000]);
  assert.deepEqual(g.ticks(-0.1, 0.1, 0.05), [-0.1, -0.05, 0, 0.05, 0.1]);
});

test("a value's percentile comes from the whole-percentile values", () => {
  const qs = Array.from({ length: 101 }, (_, p) => p / 100);
  assert.equal(g.percentileOf(0.5, qs), 50);
  assert.ok(Math.abs(g.percentileOf(0.255, qs) - 25.5) < 1e-9);
  assert.equal(g.percentileOf(-1, qs), 0);
  assert.equal(g.percentileOf(2, qs), 100);
  // A run of equal values takes its middle.
  const ties = Array.from({ length: 101 }, (_, p) => (p < 40 ? -1 : p <= 60 ? 0 : 1));
  assert.equal(g.percentileOf(0, ties), 50);
});

test("the colour scales rise in lightness from 0 on both arms", () => {
  const hex = (s) => g.parseHex(s);
  const pal = { zero: hex("#131a44"), neg: hex("#0f67f5"), negHi: hex("#7fb0ff"), pos: hex("#e05cff"), posHi: hex("#c58bff"), top: hex("#eaf2ff") };
  const d = g.diverging(pal);
  assert.equal(d.length, 257);
  assert.deepEqual(d[128], pal.zero);
  const L = (c) => g.toLab(c)[0];
  for (let i = 129; i < 257; i++) assert.ok(L(d[i]) >= L(d[i - 1]) - 1e-3, `positive arm at ${i}`);
  for (let i = 127; i >= 0; i--) assert.ok(L(d[i]) >= L(d[i + 1]) - 1e-3, `negative arm at ${i}`);
  const s = g.sequential(pal);
  for (let i = 1; i < s.length; i++) assert.ok(L(s[i]) >= L(s[i - 1]) - 1e-3, `sequential at ${i}`);
  assert.equal(g.indexOf(0, 1, 257, "diverging"), 128);
  assert.equal(g.indexOf(-5, 1, 257, "diverging"), 0);
  assert.equal(g.indexOf(-1, 1, 256, "sequential"), 255);
  // OKLab round trip.
  assert.deepEqual(g.fromLab(g.toLab([127, 176, 255])), [127, 176, 255]);
});

test("cell values are printed short, or not at all", () => {
  assert.equal(g.shortFmt(-0.0441894, 6), "-.0442");
  assert.equal(g.shortFmt(-0.0441894, 4), "-.04");
  assert.equal(g.shortFmt(0.5, 2), ".5");
  assert.equal(g.shortFmt(0, 1), "0");
  assert.equal(g.shortFmt(-0.0441894, 1), null);
  assert.equal(g.shortFmt(123456, 5), "1.2e5");
  assert.equal(g.shortFmt(123456, 3), "1e5");
});
