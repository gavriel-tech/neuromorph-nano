import "./register-ts.mjs";
import assert from "node:assert/strict";
import test from "node:test";

const api = await import("../src/lib/api.ts");
const controls = await import("../src/lib/controls.ts");

function makeControl(overrides = {}) {
  return {
    key: "budget.pack_tokens",
    group: "Budgets and allowances",
    label: "Pack tokens",
    help: "Maximum context pack token budget.",
    kind: "integer",
    min: 1000,
    max: 16000,
    step: 250,
    unit: "tokens",
    values: ["1000", "2000", "4000", "8000", "16000"],
    default: "4000",
    value: "4000",
    source: "default",
    wired: true,
    applies: "next call",
    ...overrides,
  };
}

async function withMockEnv(fn) {
  const origFetch = globalThis.fetch;
  const origWindow = globalThis.window;
  const origDocument = globalThis.document;
  const origEnv = process.env.NEXT_PUBLIC_API_BASE;
  api._resetApiForTest();
  try {
    await fn();
  } finally {
    api._resetApiForTest();
    globalThis.fetch = origFetch;
    if (origWindow === undefined) delete globalThis.window;
    else globalThis.window = origWindow;
    if (origDocument === undefined) delete globalThis.document;
    else globalThis.document = origDocument;
    if (origEnv === undefined) delete process.env.NEXT_PUBLIC_API_BASE;
    else process.env.NEXT_PUBLIC_API_BASE = origEnv;
  }
}

function setMetaToken(tok) {
  let current = tok;
  globalThis.document = {
    querySelector(sel) {
      if (sel === 'meta[name="nano-token"]') {
        return {
          getAttribute: (name) => (name === "content" ? current : null),
          setAttribute: (name, val) => {
            if (name === "content") current = val;
          },
        };
      }
      return null;
    },
  };
  return () => current;
}

test("controls.ts groups all budget, quota-shield, and memory cache controls into their expected groups", () => {
  const numericBudgetKeys = [
    "budget.pack_tokens",
    "budget.tentacle_tokens",
    "budget.judge_window",
    "budget.fast_k",
    "budget.max_run_s",
    "budget.gpu_mb_max",
    "budget.focused_tokens",
    "quota.session_tokens",
    "quota.daily_tokens",
    "quota.session_packs",
  ];
  for (const key of numericBudgetKeys) {
    const c = makeControl({ key, group: undefined });
    assert.equal(controls.controlGroupName(c), "Budgets and allowances");
    assert.equal(controls.isNumericControl(c), true);
  }
  for (const key of ["budget.adaptive_pack", "rerank.early_drop", "writer.idle", "quota.action"]) {
    const c = makeControl({ key, group: undefined, kind: "enum", min: undefined, max: undefined });
    assert.equal(controls.controlGroupName(c), "Budgets and allowances");
    assert.equal(controls.isNumericControl(c), false);
  }
  for (const key of ["mcp.tools", "mcp.tool_control", "mcp.dynamic_tools", "host.claude_lean"]) {
    const c = makeControl({ key, group: undefined, kind: "enum", min: undefined, max: undefined });
    assert.equal(controls.controlGroupName(c), "Coding agents");
  }
  for (const key of ["muscle", "learning", "learning.cycle", "examples"]) {
    const c = makeControl({ key, group: undefined, kind: "enum", min: undefined, max: undefined });
    assert.equal(controls.controlGroupName(c), "Learning");
  }
  const toolCount = makeControl({ key: "mcp.tool_count", group: undefined, kind: "integer", min: 1, max: 19, step: 1, unit: "tools" });
  assert.equal(controls.controlGroupName(toolCount), "Coding agents");
  assert.equal(controls.isNumericControl(toolCount), true);
  assert.equal(
    controls.controlGroupName(makeControl({ key: "recall_cache", group: undefined, kind: "enum", min: undefined, max: undefined })),
    "Memory and privacy",
  );
  assert.equal(
    controls.controlGroupName(makeControl({ key: "prompts.days", group: undefined, kind: "integer", min: 1, max: 90, unit: "days" })),
    "Memory and privacy",
  );
  assert.equal(
    controls.controlGroupName(makeControl({ key: "network", group: undefined, kind: "enum", min: undefined, max: undefined })),
    "Network",
  );
  assert.equal(
    controls.controlGroupName(makeControl({ key: "precision.base", group: undefined, kind: "enum", min: undefined, max: undefined })),
    "Precision",
  );
});

test("validateIntegerControl validates bounds, rejects non-integers, and normalizes leading zeros", () => {
  const c = makeControl({ min: 1000, max: 16000, unit: "tokens" });
  assert.deepEqual(controls.validateIntegerControl(c, ""), { ok: false, error: "Enter a whole number" });
  assert.deepEqual(controls.validateIntegerControl(c, "   "), { ok: false, error: "Enter a whole number" });
  assert.deepEqual(controls.validateIntegerControl(c, "4000.5"), { ok: false, error: "Must be a whole number" });
  assert.deepEqual(controls.validateIntegerControl(c, "1e3"), { ok: false, error: "Must be a whole number" });
  assert.deepEqual(controls.validateIntegerControl(c, "abc"), { ok: false, error: "Must be a whole number" });
  assert.deepEqual(controls.validateIntegerControl(c, "999"), { ok: false, error: "Must be at least 1000 tokens" });
  assert.deepEqual(controls.validateIntegerControl(c, "16001"), { ok: false, error: "Must be at most 16000 tokens" });
  assert.deepEqual(controls.validateIntegerControl(c, "1000"), { ok: true, value: 1000, normalized: "1000" });
  assert.deepEqual(controls.validateIntegerControl(c, "16000"), { ok: true, value: 16000, normalized: "16000" });
  assert.deepEqual(controls.validateIntegerControl(c, " 04000 "), { ok: true, value: 4000, normalized: "4000" });
});

test("stepIntegerControl steps by step and clamps to [min, max]", () => {
  const c = makeControl({ min: 1000, max: 2000, step: 250, value: "1250", default: "1500" });
  assert.equal(controls.stepIntegerControl(c, "1250", 1), "1500");
  assert.equal(controls.stepIntegerControl(c, "1250", -1), "1000");
  assert.equal(controls.stepIntegerControl(c, "1000", -1), "1000");
  assert.equal(controls.stepIntegerControl(c, "2000", 1), "2000");
  assert.equal(controls.stepIntegerControl(c, "bad", 1), "1500");
});

test("formatControlValue formats precision, days, default, and numeric units", () => {
  assert.equal(controls.formatControlValue("default", "precision.base"), "auto");
  assert.equal(controls.formatControlValue("8", "precision.base"), "8-bit");
  assert.equal(controls.formatControlValue("1", "prompts.days"), "1 day");
  assert.equal(controls.formatControlValue("7", "prompts.days"), "7 days");
  assert.equal(controls.formatControlValue("4000", "budget.pack_tokens", "tokens"), "4000 tokens");
  assert.equal(controls.formatControlValue("60", "budget.max_run_s", "s"), "60 s");
  assert.equal(controls.formatControlValue("3500", "budget.gpu_mb_max", "MiB"), "3500 MiB");
});

test("api.ts refresh strips all time_ms fields globally and avoids redundant object replacement", async () => {
  await withMockEnv(async () => {
    let call = 0;
    globalThis.fetch = async () => {
      call++;
      const t1 = 1000 + call * 100;
      const t2 = 2000 + call * 100;
      const body =
        call <= 2
          ? JSON.stringify({ audit: [{ id: "a1", time_ms: t1 }, { id: "a2", time_ms: t2 }], episodes: [] })
          : JSON.stringify({ audit: [{ id: "a1", time_ms: t1 }, { id: "a3", time_ms: t2 }], episodes: [] });
      return new Response(body, { status: 200, headers: { "Content-Type": "application/json" } });
    };

    await api.refresh("/api/audit");
    const first = api.readCache("/api/audit").data;
    await api.refresh("/api/audit");
    const second = api.readCache("/api/audit").data;
    assert.equal(second, first, "identical payload with changed time_ms across multiple rows must keep object identity");

    await api.refresh("/api/audit");
    const third = api.readCache("/api/audit").data;
    assert.notEqual(third, first, "changed payload must replace cached object");
  });
});

test("api.ts resolves window.__NANO_API_BASE__, NEXT_PUBLIC_API_BASE, port 3000 fallback, and relative prod", async () => {
  await withMockEnv(async () => {
    const urls = [];
    globalThis.fetch = async (url) => {
      urls.push(String(url));
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };

    // 1. window.__NANO_API_BASE__ wins and trims trailing slashes
    globalThis.window = { __NANO_API_BASE__: "http://127.0.0.1:4999///", location: { port: "3000", pathname: "/" } };
    process.env.NEXT_PUBLIC_API_BASE = "http://127.0.0.1:4888";
    await api.refresh("/api/devices");

    // 2. NEXT_PUBLIC_API_BASE used when window.__NANO_API_BASE__ unset
    globalThis.window = { location: { port: "3000", pathname: "/" } };
    await api.refresh("/api/devices");

    // 3. port 3000 fallback to 4870 when env unset
    delete process.env.NEXT_PUBLIC_API_BASE;
    await api.refresh("/api/devices");

    // 4. static production mode uses relative path ""
    globalThis.window = { location: { port: "4870", pathname: "/" } };
    await api.refresh("/api/devices");

    assert.deepEqual(urls, [
      "http://127.0.0.1:4999/api/devices",
      "http://127.0.0.1:4888/api/devices",
      "http://127.0.0.1:4870/api/devices",
      "/api/devices",
    ]);
  });
});

test("api.ts post() auto-refreshes stale token once on 403 in dev mode and in static HTML meta mode", async () => {
  // Dev mode via /api/token
  await withMockEnv(async () => {
    process.env.NEXT_PUBLIC_API_BASE = "http://127.0.0.1:4870";
    const calls = [];
    globalThis.fetch = async (url, init) => {
      const u = String(url);
      calls.push({ url: u, token: init?.headers?.["X-Nano-Token"] ?? null });
      if (u.endsWith("/api/token")) {
        const tok = calls.filter((c) => c.url.endsWith("/api/token")).length === 1 ? "tok-1" : "tok-2";
        return new Response(JSON.stringify({ token: tok }), { status: 200 });
      }
      if (init?.headers?.["X-Nano-Token"] === "tok-1") {
        return new Response("stale token", { status: 403 });
      }
      return new Response(JSON.stringify({ code: "654321", expires_ms: 999999 }), { status: 200 });
    };

    const res = await api.post("/api/pair/code", {});
    assert.deepEqual(res, { code: "654321", expires_ms: 999999 });
    assert.equal(calls.filter((c) => c.url.endsWith("/api/pair/code")).length, 2);
  });

  // Static engine-served HTML meta fallback when /api/token is 404
  await withMockEnv(async () => {
    globalThis.window = { location: { port: "4870", pathname: "/settings/" } };
    const getMeta = setMetaToken("tok-old");
    globalThis.fetch = async (url, init) => {
      const u = String(url);
      if (u === "/settings/") {
        return new Response('<html><head><meta name="nano-token" content="tok-fresh"></head></html>', { status: 200 });
      }
      if (init?.headers?.["X-Nano-Token"] === "tok-old") {
        return new Response("writes are accepted only from this machine", { status: 403 });
      }
      return new Response(JSON.stringify({ rows: 3, project: "nano" }), { status: 200 });
    };

    const res = await api.post("/api/ledger/clear", { project: "nano" });
    assert.deepEqual(res, { rows: 3, project: "nano" });
    assert.equal(getMeta(), "tok-fresh");
  });
});

test("api.ts post() does not loop on persistent 403 and parses both JSON and text/plain errors", async () => {
  await withMockEnv(async () => {
    globalThis.window = { location: { port: "4870", pathname: "/learning/" } };
    setMetaToken("tok-same");
    let postCount = 0;
    globalThis.fetch = async (url) => {
      if (String(url) === "/learning/") {
        return new Response('<meta name="nano-token" content="tok-same">', { status: 200 });
      }
      postCount++;
      return new Response("remote device is view-only", { status: 403 });
    };

    await assert.rejects(() => api.post("/api/examples/clear", { project: "nano" }), /remote device is view-only/);
    assert.equal(postCount, 1, "unchanged token must not trigger a second POST");
  });

  await withMockEnv(async () => {
    globalThis.window = { location: { port: "4870", pathname: "/settings/" } };
    setMetaToken("tok-ok");
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ error: "budget.pack_tokens must be between 1000 and 16000" }), { status: 400 });

    await assert.rejects(
      () => api.post("/api/controls", { key: "budget.pack_tokens", value: "99999" }),
      /budget\.pack_tokens must be between 1000 and 16000/,
    );
  });
});

test("api.ts refreshAll invalidates matching prefixes and coalesces concurrent inflight refreshes", async () => {
  await withMockEnv(async () => {
    let controlsFetches = 0;
    let releaseFirst;
    const firstGate = new Promise((r) => {
      releaseFirst = r;
    });
    let cartridgesFetches = 0;
    let devicesFetches = 0;

    globalThis.fetch = async (url) => {
      const u = String(url);
      if (u === "/api/controls") {
        controlsFetches++;
        if (controlsFetches === 1) await firstGate;
        return new Response(JSON.stringify([{ key: "budget.pack_tokens", value: String(controlsFetches * 1000) }]), { status: 200 });
      }
      if (u === "/api/cartridges") {
        cartridgesFetches++;
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (u === "/api/cartridges/library") {
        cartridgesFetches++;
        return new Response(JSON.stringify({ exists: true }), { status: 200 });
      }
      if (u === "/api/devices") {
        devicesFetches++;
        return new Response(JSON.stringify({ network: "off", addresses: [], devices: [] }), { status: 200 });
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };

    // Prime caches
    await api.refresh("/api/cartridges");
    await api.refresh("/api/cartridges/library");
    await api.refresh("/api/devices");

    // Trigger inflight /api/controls then request a post-write refresh while inflight
    const p1 = api.refresh("/api/controls");
    const p2 = api.refresh("/api/controls");
    releaseFirst();
    await Promise.all([p1, p2]);
    assert.equal(controlsFetches, 2);
    assert.equal(api.readCache("/api/controls").data[0].value, "2000");

    // Undo cache invalidation prefix check
    await Promise.all([api.refreshAll("/api/cartridges"), api.refreshAll("/api/devices")]);
    assert.equal(cartridgesFetches, 4);
    assert.equal(devicesFetches, 2);
  });
});
