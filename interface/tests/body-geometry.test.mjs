import "./register-ts.mjs";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const { buildTree } = await import("../src/lib/topology.ts");
const { bodyScene, LONGEST_JOB } = await import("../src/lib/body.ts");
const { polyLen } = await import("../src/lib/scene.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const ANATOMY_PATH = resolve(HERE, "../../designs/anatomy.json");
const CRATES_DIR = resolve(HERE, "../../crates");
const baseAnatomy = JSON.parse(readFileSync(ANATOMY_PATH, "utf8"));
const baseParts = Object.fromEntries(
  readdirSync(CRATES_DIR)
    .filter((d) => existsSync(resolve(CRATES_DIR, d, "part.json")))
    .map((d) => [d, JSON.parse(readFileSync(resolve(CRATES_DIR, d, "part.json"), "utf8"))]),
);

const VIEWPORTS = [
  [390, 844],
  [480, 360],
  [640, 480],
  [768, 1024],
  [800, 600],
  [1024, 768],
  [1280, 720],
  [1340, 620],
  [1920, 1080],
  [900, 1200],
  [650, 1400],
];

const TENTACLE_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24];

function makeAnatomy(nT) {
  const staticNodes = baseAnatomy.nodes.filter(
    (n) =>
      !n.id.startsWith("nmpod.unit#t") &&
      !n.id.startsWith("nmpod.unit#d") &&
      !n.id.startsWith("nmapi.tentacle_core#t") &&
      !n.id.startsWith("nmbrain.g0#t"),
  );
  const staticEdges = baseAnatomy.edges.filter(
    (e) =>
      !e.from.includes("#t") &&
      !e.to.includes("#t") &&
      !e.from.includes("#d") &&
      !e.to.includes("#d"),
  );
  const dims = ["global", "local", "structural", "semantic", "temporal", "operational", "epistemic", "objective"];
  const nodes = [...staticNodes];
  const edges = [...staticEdges];
  for (let i = 0; i < nT; i++) {
    nodes.push(
      {
        id: `nmpod.unit#t${i}`,
        kind: "tentacle",
        parent: "nmpod.pod#p0",
        ports_in: [{ name: "plan", kind: "json", required: true }],
        ports_out: [{ name: "assignment", kind: "json", required: true }],
        config: { default_dimension: dims[i % dims.length], job: "any" },
      },
      {
        id: `nmpod.unit#d${i}`,
        kind: "drone",
        parent: `nmpod.unit#t${i}`,
        ports_in: [{ name: "assignment", kind: "json", required: true }],
        ports_out: [{ name: "findings", kind: "json", required: true }],
        config: { role: "tip", readers: ["lexical", "semantic", "symbols"], job: "any" },
      },
      {
        id: `nmapi.tentacle_core#t${i}`,
        kind: "other",
        parent: `nmpod.unit#t${i}`,
        ports_in: [{ name: "findings", kind: "json", required: true }],
        ports_out: [{ name: "report", kind: "json", required: true }],
        config: { role: "core" },
      },
      {
        id: `nmbrain.g0#t${i}`,
        kind: "memory",
        parent: `nmpod.unit#t${i}`,
        ports_in: [{ name: "report", kind: "json", required: true }],
        ports_out: [{ name: "packets", kind: "packets", required: true }],
        config: { role: "g0_tx", id: `g0_t${i + 1}` },
      },
    );
    edges.push(
      { from: "nmbrain.stack#plan", from_port: "plan", to: `nmpod.unit#t${i}`, to_port: "plan", mode: { kind: "direct" } },
      { from: `nmpod.unit#t${i}`, from_port: "assignment", to: `nmpod.unit#d${i}`, to_port: "assignment", mode: { kind: "conditional", predicate: "assigned" } },
      { from: `nmpod.unit#d${i}`, from_port: "findings", to: `nmapi.tentacle_core#t${i}`, to_port: "findings", mode: { kind: "direct" } },
      { from: `nmapi.tentacle_core#t${i}`, from_port: "report", to: `nmbrain.g0#t${i}`, to_port: "report", mode: { kind: "direct" } },
      { from: `nmbrain.g0#t${i}`, from_port: "packets", to: "nmbrain.g1#g1", to_port: "packets", mode: { kind: "buffered", bound: 256 } },
      { from: "nmbrain.g2#g2", from_port: "behaviour", to: `nmpod.unit#t${i}`, to_port: "plan", mode: { kind: "feedback", max_iter: 1 } },
    );
  }
  return { ...baseAnatomy, nodes, edges };
}

function annotBox(a, detail) {
  if (a.kind === "chip") {
    const w = Math.max(a.label.length * 6.75 + 26, detail === 2 && a.sub ? a.sub.length * 6.4 : 0);
    const x = a.side === "l" ? a.x : a.side === "r" ? a.x - w : a.x - w / 2;
    const bottom = detail === 2 && a.sub ? a.y + 28 : a.y + 10;
    return { x, y: a.y - 10, w, h: bottom - (a.y - 10) };
  }
  const tw = (detail === 2 && a.key.startsWith("neuropod/t")
    ? Math.max(a.label.length, `T0 · ${LONGEST_JOB}`.length)
    : a.label.length) * 6.4;
  const x = a.side === "l" ? a.x : a.side === "r" ? a.x - tw : a.x - tw / 2;
  return { x, y: a.y - 11, w: tw, h: 15 };
}

function overlaps(a, b, pad = 0) {
  return (
    a.x < b.x + b.w + pad &&
    a.x + a.w + pad > b.x &&
    a.y < b.y + b.h + pad &&
    a.y + a.h + pad > b.y
  );
}

function insideFrame(b, W, H) {
  return b.x >= 0 && b.y >= 0 && b.x + b.w <= W && b.y + b.h <= H;
}

test("bodyScene renders exact anatomy wire count parity for default 8-tentacle design", () => {
  const tree = buildTree({ anatomy: baseAnatomy, parts: baseParts, status: null, cartridges: [] });
  const scene = bodyScene(tree, tree.root, 1600, 1000, [], null, null);
  assert.equal(scene.wires.length, baseAnatomy.edges.length);
  assert.equal(tree.tentacles.length, 8);
});

test("bodyScene handles nT = 0 safely without crashing", () => {
  const anatomy = makeAnatomy(0);
  const tree = buildTree({ anatomy, parts: baseParts, status: null, cartridges: [] });
  const scene = bodyScene(tree, tree.root, 1024, 768, [], null, null);
  assert.equal(tree.tentacles.length, 0);
  assert.ok(scene.pts.length > 0);
});

for (const [W, H] of VIEWPORTS) {
  for (const nT of TENTACLE_COUNTS) {
    test(`organism geometry invariants hold at (${W}x${H}) with nT=${nT}`, () => {
      const anatomy = makeAnatomy(nT);
      const tree = buildTree({ anatomy, parts: baseParts, status: null, cartridges: [] });
      const scene = bodyScene(tree, tree.root, W, H, [], null, [
        { key: "p1", root: null, first_ms: 1, last_ms: 2, episodes: 1, bytes: 1, dir: "p1" },
      ]);

      // 1. Tentacle proportionality: L > 0 and stddev(L_i) / mean(L_i) < 0.02
      const tipAnnots = scene.annots.filter((a) => a.kind === "minor" && /^neuropod\/t\d+$/.test(a.key));
      assert.equal(tipAnnots.length, nT, "expected one tipAnnot per tentacle");

      const lengths = [];
      const basePts = [];
      const baseWidths = [];
      for (let i = 0; i < nT; i++) {
        const tKey = `neuropod/t${i}`;
        const route = scene.routes.find((r) => r.from === tKey && r.to === `${tKey}/tip`);
        assert.ok(route && route.pts.length >= 2, `missing route for ${tKey}`);
        const tipAnnot = tipAnnots.find((a) => a.key === tKey);
        assert.ok(tipAnnot, `missing tipAnnot for ${tKey}`);
        const centerline = [...route.pts, { x: tipAnnot.mx, y: tipAnnot.my }];
        const L = polyLen(centerline);
        assert.ok(L > 0, `tentacle ${i} length must be > 0, got ${L}`);
        lengths.push(L);
        basePts.push(route.pts[0]);

        // Measure actual transverse base width from rendered tentacle body points near s = 0
        const dx = route.pts[1].x - route.pts[0].x;
        const dy = route.pts[1].y - route.pts[0].y;
        const dl = Math.hypot(dx, dy) || 1;
        const nx = -dy / dl;
        const ny = dx / dl;
        const g0Pts = scene.pts.filter((p) => p.k === `${tKey}/g0` && p.t === "b" && Math.hypot(p.x - route.pts[0].x, p.y - route.pts[0].y) < L * 0.12);
        let minN = Infinity;
        let maxN = -Infinity;
        for (const p of g0Pts) {
          const proj = (p.x - route.pts[0].x) * nx + (p.y - route.pts[0].y) * ny;
          if (proj < minN) minN = proj;
          if (proj > maxN) maxN = proj;
        }
        baseWidths.push(Number.isFinite(minN) ? maxN - minN : 0);
      }

      const meanL = lengths.reduce((a, b) => a + b, 0) / lengths.length;
      const varianceL = lengths.reduce((acc, x) => acc + (x - meanL) ** 2, 0) / lengths.length;
      const cvL = Math.sqrt(varianceL) / meanL;
      assert.ok(cvL < 0.02, `tentacle length CV ${cvL} >= 0.02`);

      // 2. Non-overlapping tentacle bases when nT >= 2
      if (nT >= 2) {
        for (let i = 0; i < nT; i++) {
          const j = (i + 1) % nT;
          const chord = Math.hypot(basePts[j].x - basePts[i].x, basePts[j].y - basePts[i].y);
          assert.ok(
            baseWidths[i] < chord,
            `tentacle base ${i} width ${baseWidths[i].toFixed(2)} >= chord ${chord.toFixed(2)}`,
          );
        }
      }

      // 3. All tentacle body/tip/reader points strictly inside [0, W] x [0, H]
      const memRegion = scene.regions.find((r) => r.key === "memory")?.box;
      const portRegion = scene.regions.find((r) => r.key === "runtime")?.box;
      const rackRegion = scene.regions.find((r) => r.key === "learning")?.box;
      assert.ok(memRegion, "memory region missing");
      assert.ok(portRegion, "runtime/port region missing");
      assert.ok(rackRegion, "learning/rack region missing");

      const tentaclePts = scene.pts.filter((p) => p.k.startsWith("neuropod/t"));
      assert.ok(tentaclePts.length > 0);
      for (const p of tentaclePts) {
        assert.ok(
          p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= H,
          `tentacle point (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) clipped outside ${W}x${H}`,
        );
        assert.ok(
          p.x < memRegion.x || p.x > memRegion.x + memRegion.w || p.y < memRegion.y || p.y > memRegion.y + memRegion.h,
          `tentacle point (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) overlaps Memory region`,
        );
      }

      // 4. All annotations inside [0, W] x [0, H] and clear of Memory, Port, Rack, and each other
      const chips = scene.annots.filter((a) => a.kind === "chip");
      for (const a of scene.annots) {
        const b = annotBox(a, scene.detail);
        assert.ok(
          insideFrame(b, W, H),
          `annot "${a.label}" box ${JSON.stringify(b)} clipped outside ${W}x${H}`,
        );
        if (a.key !== "memory") {
          assert.ok(!overlaps(b, memRegion), `annot "${a.label}" overlaps Memory region`);
        }
      }

      // Tip annotations must not overlap chips, port, or rack
      for (const ta of tipAnnots) {
        const tb = annotBox(ta, scene.detail);
        for (const c of chips) {
          const cb = annotBox(c, scene.detail);
          assert.ok(!overlaps(tb, cb), `tipAnnot "${ta.label}" overlaps chip "${c.label}"`);
        }
        assert.ok(!overlaps(tb, portRegion), `tipAnnot "${ta.label}" overlaps port`);
        assert.ok(!overlaps(tb, rackRegion), `tipAnnot "${ta.label}" overlaps rack`);
      }

      // Regions (memory, port, rack) must fit inside [0, W] x [0, H] and not overlap each other
      assert.ok(insideFrame(memRegion, W, H), "memory region outside frame");
      assert.ok(insideFrame(portRegion, W, H), "port region outside frame");
      assert.ok(insideFrame(rackRegion, W, H), "rack region outside frame");
      assert.ok(!overlaps(memRegion, portRegion), "memory overlaps port");
      assert.ok(!overlaps(memRegion, rackRegion), "memory overlaps rack");
      assert.ok(!overlaps(portRegion, rackRegion), "port overlaps rack");
    });
  }
}
