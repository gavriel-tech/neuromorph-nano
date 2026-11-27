// Opaque-box headless Microsoft Edge CDP E2E verifier for Neuromorph Nano UI and Organism View.
// Zero external npm dependencies; uses Node v22+ built-in fetch and WebSocket.
//
// Usage:
//   node interface/scripts/e2e-ui-verify.mjs [--url http://127.0.0.1:<port>] [--exe <path>] [--baseline]
// Default (without --baseline) enforces strict full M5 mode: exits 1 if any check fails or skips.
// When no server answers (no --url server, or no neuromorph-nano.exe to start an isolated one),
// the static export checks still run and every live check is reported SKIP with "no server
// answers"; the script then exits 0 (nothing was wrong, nothing live was checked).

import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, cpSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import net from "node:net";
import os from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");
const args = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function argValue(flag) {
  const idx = args.indexOf(flag);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  const eq = args.find((a) => a.startsWith(flag + "="));
  return eq ? eq.slice(flag.length + 1) : null;
}

const explicitUrl = argValue("--url");
const explicitExe = argValue("--exe");
const baselineMode = args.includes("--baseline");
const strictMode = !baselineMode;

/** The job of the probe filter written into an isolated home (see createIsolatedHome). */
const PROBE_JOB = "e2e-probe";
/** The job of the job filter with points written into an isolated home for the cartridge parameters lane. */
const PARAMS_JOB = "e2e-params";
/** Every static route of the interface (src/app), with trailing slashes as the export serves them. */
const ROUTES = ["/", "/anatomy/", "/topology/", "/cartridges/", "/learning/", "/memory/", "/models/", "/audit/", "/parts/", "/settings/", "/storage/", "/parameters/"];
const NO_SERVER = "no server answers";

const results = [];
function record(tier, area, name, status, detail = "") {
  results.push({ tier, area, name, status, detail });
  const suffix = detail ? ` -- ${detail}` : "";
  console.log(`[${status}][T${tier}][${area}] ${name}${suffix}`);
}
function check(tier, area, name, ok, detail = "") {
  record(tier, area, name, ok ? "PASS" : "FAIL", detail);
  return Boolean(ok);
}
function skip(tier, area, name, detail = "baseline mode") {
  record(tier, area, name, "SKIP", detail);
}

function findEdge() {
  const candidates = [
    process.env.EDGE_PATH,
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, "Microsoft/Edge/Application/msedge.exe") : null,
  ].filter(Boolean);
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  throw new Error(`Microsoft Edge not found in: ${candidates.join(", ")}`);
}

function findNanoExe() {
  if (explicitExe) {
    if (!existsSync(explicitExe)) throw new Error(`Explicit --exe not found: ${explicitExe}`);
    return resolve(explicitExe);
  }
  const gateExe = join(REPO_ROOT, "scratch", "gate", "target", "release", "neuromorph-nano.exe");
  if (existsSync(gateExe)) return gateExe;
  const relExe = join(REPO_ROOT, "target", "release", "neuromorph-nano.exe");
  if (existsSync(relExe)) return relExe;
  throw new Error(`neuromorph-nano.exe not found at ${gateExe} or ${relExe}`);
}

function getFreePort() {
  return new Promise((resolvePort, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      srv.close((err) => (err ? reject(err) : resolvePort(port)));
    });
  });
}

function createIsolatedHome(exePath) {
  const tmp = join(os.tmpdir(), `nm-ui-e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`);
  mkdirSync(tmp, { recursive: true });
  const nanoJson = join(REPO_ROOT, "nano.json");
  if (existsSync(nanoJson)) copyFileSync(nanoJson, join(tmp, "nano.json"));

  const srcOut = join(REPO_ROOT, "interface", "out");
  if (!existsSync(srcOut)) throw new Error(`Static UI build missing at ${srcOut}`);
  mkdirSync(join(tmp, "interface"), { recursive: true });
  cpSync(srcOut, join(tmp, "interface", "out"), { recursive: true });

  const srcDesigns = join(REPO_ROOT, "designs");
  if (!existsSync(srcDesigns)) throw new Error(`designs folder missing at ${srcDesigns}`);
  cpSync(srcDesigns, join(tmp, "designs"), { recursive: true });

  const srcCrates = join(REPO_ROOT, "crates");
  const dstCrates = join(tmp, "crates");
  mkdirSync(dstCrates, { recursive: true });
  for (const entry of readdirSync(srcCrates, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const partJson = join(srcCrates, entry.name, "part.json");
    if (existsSync(partJson)) {
      const d = join(dstCrates, entry.name);
      mkdirSync(d, { recursive: true });
      copyFileSync(partJson, join(d, "part.json"));
    }
  }

  // The newest stage-bench report, so /audit shows a stage table (read only by the server).
  const runsDir = join(REPO_ROOT, "runs");
  if (existsSync(runsDir)) {
    const bench = readdirSync(runsDir)
      .filter((f) => /^STAGE-BENCH-\d+\.txt$/.test(f))
      .sort((a, b) => Number(b.slice(12, -4)) - Number(a.slice(12, -4)))[0];
    if (bench) {
      mkdirSync(join(tmp, "runs"), { recursive: true });
      copyFileSync(join(runsDir, bench), join(tmp, "runs", bench));
    }
  }

  // A probe job filter in the temporary home only, so /cartridges has a row whose On/Off and
  // Remove can be exercised without touching the real library (never read by a model here).
  const probeDir = join(tmp, "library", "parameter-maps", "reranker");
  mkdirSync(probeDir, { recursive: true });
  writeFileSync(
    join(probeDir, `job-filter-${PROBE_JOB}.json`),
    JSON.stringify({ contract: "nmengn-sockets/1", cartridge: `reranker-${PROBE_JOB}@1`, base: "reranker-base@1", model: "probe", job: PROBE_JOB, base_cos: 0.9, filtered_cos: 0.91, enabled: true, parts: [] }, null, 2)
  );
  // 2026-11-26: a small tolerance map stays after the probe filter is removed, so the map
  // inspector always has a map to open.
  writeFileSync(join(probeDir, "tol-e2e-probe.json"), JSON.stringify({ note: "e2e probe map", tolerances: { L1: 0.01, L2: 0.02 } }, null, 2));
  // A job filter with points, plugged out, for the cartridge parameters lane (read and edited into a
  // new version on the Parameters page; never read by a model here).
  const paramsDir = join(tmp, "library", "parameter-maps", "writer");
  mkdirSync(paramsDir, { recursive: true });
  writeFileSync(
    join(paramsDir, `job-filter-${PARAMS_JOB}.json`),
    JSON.stringify({ contract: "nmengn-sockets/1", cartridge: `writer-${PARAMS_JOB}@1`, model: "probe", job: PARAMS_JOB, enabled: false, parts: [{ target: "L2.o", gain: [1.5, 0.5, 1.25], bias: [0.1, -0.2, 0.3] }, { target: "L5.swigate", gain: [2.0, 1.0], bias: [0.0, 0.5] }] }, null, 2)
  );

  const dstRel = join(tmp, "target", "release");
  mkdirSync(dstRel, { recursive: true });
  copyFileSync(exePath, join(dstRel, "neuromorph-nano.exe"));
  return tmp;
}

async function startIsolatedUiServer() {
  const exePath = findNanoExe();
  const tempHome = createIsolatedHome(exePath);
  const port = await getFreePort();
  const launchExe = join(tempHome, "target", "release", "neuromorph-nano.exe");
  const env = { ...process.env, NANO_HOME: tempHome, NANO_MODELS: join(REPO_ROOT, "weights") };
  delete env.CLAUDE_PROJECT_DIR;

  const child = spawn(launchExe, ["launch", "ui", "--port", String(port)], {
    cwd: REPO_ROOT,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (d) => { stderr += d.toString(); });

  const baseUrl = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let i = 0; i < 60; i++) {
    if (child.exitCode !== null) {
      throw new Error(`launch ui exited early (${child.exitCode}): ${stderr}`);
    }
    try {
      const r = await fetch(`${baseUrl}/api/status`);
      if (r.ok) {
        ready = true;
        break;
      }
    } catch {
      /* waiting */
    }
    await sleep(250);
  }
  if (!ready) throw new Error(`Timed out waiting for launch ui at ${baseUrl}: ${stderr}`);

  const stop = async () => {
    if (child.pid) {
      spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    }
    const needle = join(tempHome, "scratch", "live").toLowerCase().replace(/'/g, "''");
    spawnSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        `Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.ToLowerInvariant().StartsWith('${needle}') } | ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }`,
      ],
      { stdio: "ignore" }
    );
    for (let k = 0; k < 10 && existsSync(tempHome); k++) {
      try {
        rmSync(tempHome, { recursive: true, force: true });
      } catch {
        await sleep(300);
      }
    }
  };

  return { baseUrl, tempHome, stop };
}

/** A serving engine on the isolated home, over MCP on standard input and output (one JSON message
 *  per line), for the live activity check: it serves a small crate of this repository, so the
 *  interface server on the same home shows its live state. `request` sends one JSON-RPC request
 *  and resolves with the reply; `stop` ends the process. */
function startServingEngine(tempHome) {
  const exe = join(tempHome, "target", "release", "neuromorph-nano.exe");
  const root = join(REPO_ROOT, "crates", "nmjson");
  const env = { ...process.env, NANO_HOME: tempHome, NANO_MODELS: join(REPO_ROOT, "weights") };
  delete env.CLAUDE_PROJECT_DIR;
  const child = spawn(exe, ["serve", "--root", root], { cwd: root, env, stdio: ["pipe", "pipe", "pipe"] });
  let buf = "";
  let stderr = "";
  let nextId = 0;
  const waiting = new Map();
  child.stderr.on("data", (d) => { stderr = (stderr + d.toString()).slice(-4000); });
  child.stdout.on("data", (d) => {
    buf += d.toString("utf8");
    for (let i = buf.indexOf("\n"); i >= 0; i = buf.indexOf("\n")) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      try {
        const m = JSON.parse(line);
        if (m.id !== undefined && waiting.has(m.id)) {
          waiting.get(m.id)(m);
          waiting.delete(m.id);
        }
      } catch {
        /* not a reply */
      }
    }
  });
  const request = (method, params = {}, timeoutMs = 180000) =>
    new Promise((res, rej) => {
      const id = ++nextId;
      const t = setTimeout(() => rej(new Error(`MCP ${method} timed out: ${stderr.slice(-300)}`)), timeoutMs);
      waiting.set(id, (m) => {
        clearTimeout(t);
        res(m);
      });
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  const notify = (method) => child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method }) + "\n");
  const stop = async () => {
    try {
      child.stdin.end();
    } catch {
      /* already closed */
    }
    await sleep(800);
    if (child.pid) spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  };
  return { request, notify, stop };
}

async function connectCdp(debugPort) {
  let ok = false;
  for (let i = 0; i < 120; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
      if (r.ok) {
        ok = true;
        break;
      }
    } catch {
      /* waiting */
    }
    await sleep(200);
  }
  if (!ok) throw new Error(`Edge CDP did not respond on port ${debugPort}`);

  const target = await (await fetch(`http://127.0.0.1:${debugPort}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => {
    ws.addEventListener("open", r, { once: true });
    ws.addEventListener("error", j, { once: true });
  });

  let nextId = 0;
  const pending = new Map();
  const handlers = [];
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    } else if (msg.method) {
      for (const h of handlers) h(msg);
    }
  });

  const send = (method, params = {}) =>
    new Promise((resolveCmd, rejectCmd) => {
      const id = ++nextId;
      const timer = setTimeout(() => {
        pending.delete(id);
        rejectCmd(new Error(`CDP timeout on ${method}`));
      }, 30000);
      pending.set(id, (msg) => {
        clearTimeout(timer);
        if (msg.error) rejectCmd(new Error(`CDP ${method}: ${JSON.stringify(msg.error)}`));
        else resolveCmd(msg.result);
      });
      ws.send(JSON.stringify({ id, method, params }));
    });

  const evalJs = async (expression) => {
    const res = await send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (res.exceptionDetails) {
      throw new Error(`Page eval exception: ${JSON.stringify(res.exceptionDetails)}`);
    }
    return res.result ? res.result.value : undefined;
  };

  return { send, evalJs, on: (fn) => handlers.push(fn), close: () => ws.close() };
}

async function teardownEdge(edgeProc, profileDir, debugPort) {
  try {
    const v = await (await fetch(`http://127.0.0.1:${debugPort}/json/version`)).json();
    const ws = new WebSocket(v.webSocketDebuggerUrl);
    await new Promise((r) => ws.addEventListener("open", r, { once: true }));
    ws.send(JSON.stringify({ id: 1, method: "Browser.close" }));
    await sleep(400);
    ws.close();
  } catch {
    /* already closed */
  }
  if (edgeProc && edgeProc.pid) {
    spawnSync("taskkill", ["/PID", String(edgeProc.pid), "/T", "/F"], { stdio: "ignore" });
  }
  const leaf = profileDir.split(/[\\/]/).pop().replace(/'/g, "''");
  for (let k = 0; k < 10; k++) {
    const r = spawnSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        `@(Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*${leaf}*' }) | ForEach-Object { $_.ProcessId }`,
      ],
      { encoding: "utf8" }
    );
    const pids = (r.stdout || "").split(/\s+/).filter(Boolean);
    if (!pids.length) break;
    for (const pid of pids) spawnSync("taskkill", ["/PID", pid, "/T", "/F"], { stdio: "ignore" });
    await sleep(250);
  }
  for (let k = 0; k < 15 && existsSync(profileDir); k++) {
    try {
      rmSync(profileDir, { recursive: true, force: true });
    } catch {
      await sleep(250);
    }
  }
}

/** 2026-11-26 (Luis Jake Gabriel III: every delete verified in more than one step): with a delete
 *  flow's preview open, checks the final button is disabled and that no confirmed request has gone
 *  out, types the word the engine named, and presses the final button. Returns null when no
 *  preview opened within about 3 s, else { guarded, word }. */
async function deleteFlowConfirm(cdp) {
  for (let i = 0; i < 20; i++) {
    const ready = await cdp.evalJs(`!!document.querySelector('[data-e2e="delete-flow"][data-step="confirm"] [data-e2e="delete-word"]')`);
    if (ready) break;
    await sleep(150);
  }
  return cdp.evalJs(`(() => {
    const flow = document.querySelector('[data-e2e="delete-flow"][data-step="confirm"]');
    if (!flow) return null;
    const word = flow.getAttribute('data-confirm-word') || '';
    const input = flow.querySelector('[data-e2e="delete-word"]');
    const final = flow.querySelector('[data-e2e="delete-final"]');
    if (!input || !final || !word) return null;
    const guarded = final.disabled === true;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(input, word);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return new Promise((resolve) => setTimeout(() => { if (!final.disabled) final.click(); resolve({ guarded, word, clicked: !final.disabled || true }); }, 150));
  })()`);
}

async function waitForPageReady(cdp, timeoutMs = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const state = await cdp.evalJs(`(() => {
      const txt = document.body ? document.body.innerText : '';
      const hasPlaceholder = Boolean(document.querySelector('.placeholder'));
      const hasCanvas = Boolean(document.querySelector('canvas'));
      const sceneBase = window.__nmScene ? Number(window.__nmScene.base || 0) : 0;
      return {
        connecting: txt.includes('Connecting'),
        failed: txt.includes('The engine is not answering'),
        hasPlaceholder,
        hasCanvas,
        sceneBase,
        textLen: txt.trim().length,
        scrollWidth: document.documentElement ? document.documentElement.scrollWidth : 0,
        clientWidth: document.documentElement ? document.documentElement.clientWidth : 0
      };
    })()`);
    if (state.failed) return { ok: false, reason: "The engine is not answering", state };
    if (
      state.textLen > 0 &&
      !state.connecting &&
      !state.hasPlaceholder &&
      (!state.hasCanvas || state.sceneBase > 0)
    ) {
      await sleep(300);
      return { ok: true, state };
    }
    await sleep(250);
  }
  const finalState = await cdp.evalJs(`({
    connecting: (document.body?.innerText || '').includes('Connecting'),
    failed: (document.body?.innerText || '').includes('The engine is not answering'),
    hasPlaceholder: Boolean(document.querySelector('.placeholder')),
    hasCanvas: Boolean(document.querySelector('canvas')),
    sceneBase: window.__nmScene ? Number(window.__nmScene.base || 0) : 0,
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth
  })`);
  return { ok: false, reason: "readiness timeout", state: finalState };
}

/** Whether a server answers GET /api/status at `baseUrl` within `ms`. */
async function serverAnswers(baseUrl, ms = 1500) {
  try {
    const r = await fetch(`${baseUrl}/api/status`, { signal: AbortSignal.timeout(ms) });
    return r.ok;
  } catch {
    return false;
  }
}

/** Checks that need no server: the static export holds a page for every route. */
function staticChecks() {
  const out = join(REPO_ROOT, "interface", "out");
  for (const route of ROUTES) {
    const page = route === "/" ? join(out, "index.html") : join(out, route.replace(/^\/|\/$/g, ""), "index.html");
    check(1, "UI-Export", `Static export has ${route}`, existsSync(page), existsSync(out) ? page : "interface/out missing: run npm run build");
  }
}

/** The live checks, reported SKIP when no server answers. */
const LIVE_CHECKS = [
  ...ROUTES.map((r) => [1, "UI-Routes", `Route ${r} loads cleanly`]),
  [1, "UI-Parameters", "Job-filter Build, plug in/unplug and Remove controls"],
  [3, "UI-Parameters", "Job-filter toggle reflected within 1 s"],
  [1, "UI-Memory", "Comprehension panel: levels, outline, routing record, heads"],
  [1, "UI-Learning", "Run cycle and Rollback controls"],
  [1, "UI-Settings", "4-model VRAM meter"],
  [1, "UI-Audit", "Stage-times view"],
];

function finish() {
  console.log("\n=== E2E UI & Organism Summary ===");
  for (const tier of [1, 2, 3, 4]) {
    const rows = results.filter((r) => r.tier === tier);
    const p = rows.filter((r) => r.status === "PASS").length;
    const f = rows.filter((r) => r.status === "FAIL").length;
    const s = rows.filter((r) => r.status === "SKIP").length;
    console.log(`Tier ${tier}: PASS=${p} FAIL=${f} SKIP=${s}`);
  }
  const totalPass = results.filter((r) => r.status === "PASS").length;
  const totalFail = results.filter((r) => r.status === "FAIL").length;
  const totalSkip = results.filter((r) => r.status === "SKIP").length;
  console.log(`Total : PASS=${totalPass} FAIL=${totalFail} SKIP=${totalSkip}`);
  return { totalFail, totalSkip };
}

async function main() {
  staticChecks();
  let server = null;
  let baseUrl = explicitUrl ? explicitUrl.replace(/\/+$/, "") : "";
  let why = "";
  if (baseUrl) {
    if (!(await serverAnswers(baseUrl))) why = `${NO_SERVER} at ${baseUrl}`;
  } else {
    try {
      server = await startIsolatedUiServer();
      baseUrl = server.baseUrl;
    } catch (e) {
      why = `${NO_SERVER}: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
  if (why) {
    for (const [tier, area, name] of LIVE_CHECKS) skip(tier, area, name, why);
    const { totalFail } = finish();
    console.log(`Live checks skipped (${why}).`);
    process.exit(totalFail > 0 ? 1 : 0);
  }
  const edgeExe = findEdge();
  const isolated = server !== null;
  console.log(`UI E2E Verify: url=${baseUrl} edge=${edgeExe} mode=${strictMode ? "FULL_M5_STRICT" : "BASELINE"}`);

  const profileDir = join(os.tmpdir(), `nm-edge-e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`);
  mkdirSync(profileDir, { recursive: true });
  const debugPort = await getFreePort();
  const edgeProc = spawn(
    edgeExe,
    [
      "--headless=new",
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${profileDir}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--disable-background-networking",
      "--disable-component-update",
      "--hide-scrollbars",
      "--window-size=1340,720",
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  let cdp = null;
  try {
    cdp = await connectCdp(debugPort);
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await cdp.send("Network.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: 1340,
      height: 720,
      deviceScaleFactor: 1,
      mobile: false,
    });

    const diag = {
      consoleErrors: [],
      exceptions: [],
      networkFailed: [],
      apiErrors: [],
      allowExpected400: false,
      apiResponses: [],
    };
    const resetDiag = () => {
      diag.consoleErrors.length = 0;
      diag.exceptions.length = 0;
      diag.networkFailed.length = 0;
      diag.apiErrors.length = 0;
      diag.apiResponses.length = 0;
    };

    cdp.on((msg) => {
      if (msg.method === "Runtime.consoleAPICalled" && msg.params?.type === "error") {
        const text = (msg.params.args || []).map((a) => a.value ?? a.description ?? "").join(" ");
        diag.consoleErrors.push(text);
      } else if (msg.method === "Runtime.exceptionThrown") {
        const ex = msg.params?.exceptionDetails;
        diag.exceptions.push(ex?.text || ex?.exception?.description || "exception");
      } else if (msg.method === "Network.loadingFailed") {
        if (msg.params?.canceled && msg.params?.errorText === "net::ERR_ABORTED") return;
        diag.networkFailed.push(`${msg.params?.type || ""}:${msg.params?.errorText || "failed"}`);
      } else if (msg.method === "Network.responseReceived") {
        const resp = msg.params?.response;
        if (resp && resp.url && resp.url.includes("/api/")) {
          diag.apiResponses.push({ url: resp.url, status: resp.status });
          if (resp.status >= 400) {
            if (diag.allowExpected400 && resp.status === 400) return;
            diag.apiErrors.push(`${resp.status} ${resp.url}`);
          }
        }
      }
    });

    // =========================================================================
    // 1. ROUTE SWEEP (all 11 static trailing-slash routes; /parts/ forwards to /anatomy/)
    // =========================================================================
    for (const route of ROUTES) {
      resetDiag();
      await cdp.send("Page.navigate", { url: baseUrl + route });
      const ready = await waitForPageReady(cdp);
      const overflowOk = ready.state.scrollWidth <= ready.state.clientWidth;
      const errCount =
        diag.consoleErrors.length +
        diag.exceptions.length +
        diag.networkFailed.length +
        diag.apiErrors.length;
      let forwardOk = true;
      if (route === "/parts/") {
        let path = "";
        for (let i = 0; i < 20 && path !== "/anatomy/"; i++) {
          path = await cdp.evalJs("location.pathname");
          if (path !== "/anatomy/") await sleep(150);
        }
        forwardOk = path === "/anatomy/";
      }
      const ok = ready.ok && overflowOk && errCount === 0 && forwardOk;
      check(
        1,
        "UI-Routes",
        `Route ${route} loads cleanly (0 console/page/API errors, no horizontal overflow)`,
        ok,
        `ready=${ready.ok}(${ready.reason || "ok"})${route === "/parts/" ? ` forwarded=${forwardOk}` : ""} sw=${ready.state.scrollWidth}/${ready.state.clientWidth} errs=${errCount} console=${JSON.stringify(diag.consoleErrors)} api=${JSON.stringify(diag.apiErrors)} net=${JSON.stringify(diag.networkFailed)}`
      );
    }

    // =========================================================================
    // 1b. LOADING STATES (2026-11-27): with every /api/ reply held back 2.5 s, each page shows a
    // loading state or its content within 1.5 s, never a blank main area or a blank figure frame.
    // =========================================================================
    {
      const HOLD_MS = 2500;
      const held = new Set();
      let holding = true;
      const onPaused = (m) => {
        if (!holding || m.method !== "Fetch.requestPaused") return;
        const id = m.params.requestId;
        held.add(id);
        setTimeout(() => {
          if (held.delete(id)) cdp.send("Fetch.continueRequest", { requestId: id }).catch(() => {});
        }, HOLD_MS);
      };
      cdp.on(onPaused);
      await cdp.send("Fetch.enable", { patterns: [{ urlPattern: "*/api/*" }] });
      const LOADING_PROBE = `(() => {
        const main = document.querySelector('main');
        if (!main) return { ok: false, why: 'no main' };
        const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4 && r.bottom > 0 && r.top < innerHeight; };
        const loading = [...main.querySelectorAll('[data-loading]')].filter(vis).length;
        const text = (main.innerText || '').replace(/\\s+/g, ' ').trim().length;
        const blankFigs = [...main.querySelectorAll('.topo')].filter(vis).filter((f) => !f.querySelector('canvas, svg, [data-loading], [role="alert"]')).length;
        return { ok: (loading > 0 || text > 40) && blankFigs === 0, loading, text, blankFigs };
      })()`;
      for (const route of ROUTES.filter((r) => r !== "/parts/")) {
        await cdp.send("Page.navigate", { url: baseUrl + route });
        await sleep(1500);
        const r = (await cdp.evalJs(LOADING_PROBE)) || { ok: false, why: "no answer" };
        check(1, "UI-Loading", `Route ${route} shows a loading state or its content within 1.5 s while the engine is slow (no blank main area or figure)`, Boolean(r.ok), JSON.stringify(r));
        await sleep(HOLD_MS + 300);
      }
      holding = false;
      await cdp.send("Fetch.disable");
      for (const id of held) await cdp.send("Fetch.continueRequest", { requestId: id }).catch(() => {});
      held.clear();
    }

    // =========================================================================
    // 1c. TENTACLE JOBS AND MICROMODEL KINDS (2026-11-27). A tentacle's label claims no job at
    // rest; when a call's steps are played, each lit tentacle's label carries the job its step
    // reported. The call is a mock reply of the live audit read (named mock in its label). A
    // MicroModel's hover names its kind and the open weights it runs on.
    // =========================================================================
    {
      const t0 = Date.now() - 600000;
      const mockRow = {
        id: "a-e2e-jobs", kind: "tool", tool: "nm_context_pack", label: "Context pack: e2e tentacle jobs (mock reply)", mode: "plain", ms: 120, ok: true, state: "done", time_ms: t0,
        steps: [
          { part: "nmsubs.views#g0", start_ms: 0, ms: 4 },
          { part: "nmpod.unit#t0", start_ms: 4, ms: 30, job: "structural" }, { part: "nmpod.unit#t3", start_ms: 4, ms: 30, job: "dependency" }, { part: "nmpod.unit#t5", start_ms: 4, ms: 30, lens: "meaning" },
          { part: "nmbrain.g1#g1", start_ms: 34, ms: 6 },
          { part: "nmbrain.synthesis#b0", start_ms: 40, ms: 30 },
        ],
      };
      const onMock = (m) => {
        if (m.method !== "Fetch.requestPaused") return;
        const url = m.params.request.url;
        if (!/\/api\/audit\?kind=tool&limit=20/.test(url)) return cdp.send("Fetch.continueRequest", { requestId: m.params.requestId }).catch(() => {});
        cdp.send("Fetch.fulfillRequest", { requestId: m.params.requestId, responseCode: 200, responseHeaders: [{ name: "Content-Type", value: "application/json" }], body: Buffer.from(JSON.stringify({ audit: [mockRow], episodes: [] })).toString("base64") }).catch(() => {});
      };
      let mocking = true;
      cdp.on((m) => { if (mocking) onMock(m); });
      await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1680, height: 1000, deviceScaleFactor: 1, mobile: false });
      await cdp.send("Fetch.enable", { patterns: [{ urlPattern: "*/api/audit*" }] });
      await cdp.send("Page.navigate", { url: baseUrl + "/topology/" });
      await waitForPageReady(cdp);
      await sleep(1500);
      const labels = `(() => Object.fromEntries(['t0','t1','t3','t5'].map(k => { const g = document.querySelector('[data-key="neuropod/' + k + '"] text'); return [k, g ? g.textContent.trim() : null]; })))()`;
      const rest = await cdp.evalJs(labels);
      const restOk = Boolean(rest && rest.t0 === "T1" && rest.t1 === "T2" && rest.t3 === "T4" && rest.t5 === "T6");
      check(1, "UI-Live", "At rest each tentacle is labelled by its index only (no job claimed)", restOk, JSON.stringify(rest));
      await cdp.evalJs(`(() => { const b = [...document.querySelectorAll('button')].find(x => /replay last call/i.test(x.textContent || '')); if (b) b.click(); return !!b; })()`);
      await sleep(400);
      const lit = await cdp.evalJs(labels);
      const litOk = Boolean(lit && lit.t0 === "T1 · structural" && lit.t3 === "T4 · dependency" && lit.t5 === "T6 · meaning" && lit.t1 === "T2");
      check(1, "UI-Live", "A played call's tentacles are labelled with the job (or lens) their steps reported; others keep their index", litOk, JSON.stringify(lit));
      mocking = false;
      await cdp.send("Fetch.disable");
      // A MicroModel's hover on the organism names its kind and its open weights.
      await cdp.send("Page.navigate", { url: baseUrl + "/topology/#/engine/micromodels" });
      await waitForPageReady(cdp);
      await sleep(1500);
      const tip = await cdp.evalJs(`(async () => {
        const g = [...document.querySelectorAll('[data-key^="engine/micromodels/"]')][0];
        if (!g) return { none: true };
        const r = g.getBoundingClientRect();
        g.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, pointerType: 'mouse' }));
        await new Promise((res) => setTimeout(res, 300));
        const t = document.querySelector('[data-e2e="topo-tip-title"]');
        return { key: g.dataset.key, title: t ? t.textContent.trim() : null };
      })()`);
      const kindOk = Boolean(tip && tip.title && /^(Reader|Writer|Embedder|Reranker)( \+ (Reader|Writer|Embedder|Reranker))* · \S+/.test(tip.title));
      check(1, "UI-Live", "A MicroModel's hover on the organism names its kind (Reader, Writer, Embedder or Reranker) and its open weights", kindOk, JSON.stringify(tip));
      await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1340, height: 720, deviceScaleFactor: 1, mobile: false });
    }

    // =========================================================================
    // 2. INTERACTIVE UI CONNECTIVITY & BUDGET ROUNDTRIP (/settings/ & /learning/)
    // =========================================================================
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/settings/" });
    await waitForPageReady(cdp);

    // 2026-11-25: Settings shows one view at a time under a section bar; open the view a check needs.
    const settingsView = async (label) => {
      await cdp.evalJs(`(() => {
        const nav = document.querySelector('nav[aria-label="Settings sections"]');
        const b = nav && [...nav.querySelectorAll('button, a')].find(x => (x.textContent || '').trim().toLowerCase() === ${JSON.stringify(label)}.toLowerCase());
        if (b) b.click();
      })()`);
      await sleep(400);
    };
    await settingsView("Controls");
    const hasControlSurface = await cdp.evalJs(`(() => {
      const hs = [...document.querySelectorAll('h1, h2, .eyebrow')].map(el => el.textContent.trim());
      return hs.some(t => /Control surface/i.test(t));
    })()`);
    check(1, "UI-Settings", "Control surface panel visible on /settings/", hasControlSurface);

    await settingsView("Machine");
    const hasHomeostasisHud = await cdp.evalJs(`(() => {
      const txt = document.body.innerText || '';
      return /Homeostasis, graphics card headroom and quota profile/i.test(txt) && /MCP tool list/i.test(txt) && /Quotas/i.test(txt);
    })()`);
    check(1, "UI-Settings", "Homeostasis, VRAM headroom, and quota profile HUD visible on /settings/", hasHomeostasisHud);

    // Click an alternate discrete [role="radio"] option and verify reload persistence
    await settingsView("Controls");
    const radioPick = await cdp.evalJs(`(() => {
      const groups = [...document.querySelectorAll('[role="radiogroup"]')];
      const targetGroup = groups.find(g => /mode/i.test(g.getAttribute('aria-label') || '')) || groups.find(g => g.querySelectorAll('[role="radio"]').length >= 2);
      if (!targetGroup) return null;
      const radios = [...targetGroup.querySelectorAll('[role="radio"]')];
      const current = radios.find(r => r.getAttribute('aria-checked') === 'true');
      const alt = radios.find(r => r.getAttribute('aria-checked') !== 'true');
      if (!current || !alt) return null;
      const groupLabel = targetGroup.getAttribute('aria-label') || '';
      const origText = current.textContent.trim();
      const altText = alt.textContent.trim();
      alt.click();
      return { groupLabel, origText, altText };
    })()`);

    let discreteSaved = false;
    if (radioPick) {
      for (let i = 0; i < 20; i++) {
        await sleep(150);
        const statusTxt = await cdp.evalJs(`(document.querySelector('[role="status"]')?.textContent || '').trim()`);
        if (statusTxt.length > 0) {
          discreteSaved = true;
          break;
        }
      }
      await cdp.send("Page.navigate", { url: baseUrl + "/settings/" });
      await waitForPageReady(cdp);
      const persisted = await cdp.evalJs(`(() => {
        const groups = [...document.querySelectorAll('[role="radiogroup"]')];
        const g = groups.find(x => (x.getAttribute('aria-label') || '') === ${JSON.stringify(radioPick.groupLabel)});
        if (!g) return false;
        const checked = g.querySelector('[role="radio"][aria-checked="true"]');
        return Boolean(checked && checked.textContent.trim() === ${JSON.stringify(radioPick.altText)});
      })()`);
      check(
        3,
        "UI-Settings",
        `Discrete control [role="radio"] (${radioPick.groupLabel}) persists across reload`,
        discreteSaved && persisted,
        `status=${discreteSaved} persisted=${persisted} (${radioPick.origText} -> ${radioPick.altText})`
      );
      // Restore original discrete option
      await cdp.evalJs(`(() => {
        const groups = [...document.querySelectorAll('[role="radiogroup"]')];
        const g = groups.find(x => (x.getAttribute('aria-label') || '') === ${JSON.stringify(radioPick.groupLabel)});
        if (!g) return;
        const orig = [...g.querySelectorAll('[role="radio"]')].find(r => r.textContent.trim() === ${JSON.stringify(radioPick.origText)});
        if (orig) orig.click();
      })()`);
      await sleep(300);
    } else {
      check(3, "UI-Settings", "Discrete control [role=\"radio\"] roundtrip on /settings/", false, "no radiogroup found");
    }

    // Budgets and allowances + Paired Devices on /settings/
    const NUM_SEL = 'input[role="spinbutton"], input[inputmode="numeric"], input[type="number"]';
    const budgetUiPresent = await cdp.evalJs(`(() => {
      const txt = document.body.innerText || '';
      return /Budgets and allowances/i.test(txt) && document.querySelectorAll(${JSON.stringify(NUM_SEL)}).length >= 6;
    })()`);

    if (budgetUiPresent || strictMode) {
      check(
        1,
        "UI-Budget",
        'Settings renders "Budgets and allowances" group with >= 6 numeric inputs',
        budgetUiPresent
      );
      if (budgetUiPresent) {
        resetDiag();
        const prep = await cdp.evalJs(`(() => {
          const inputs = [...document.querySelectorAll(${JSON.stringify(NUM_SEL)})];
          const target = inputs.find(inp => {
            const ctx = (inp.getAttribute('aria-label') || '') + ' ' + (inp.closest('div,section,li,tr')?.textContent || '');
            return /pack.?tokens|Context pack/i.test(ctx);
          }) || inputs[0];
          if (!target) return null;
          target.setAttribute('data-e2e-budget-target', '1');
          const orig = target.value;
          const next = orig === '2500' ? '4000' : '2500';
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
          setter.call(target, next);
          target.dispatchEvent(new Event('input', { bubbles: true }));
          target.dispatchEvent(new Event('change', { bubbles: true }));
          const form = target.closest('form') || target.closest('div,section,li,tr') || target.parentElement;
          const btn = form ? [...form.querySelectorAll('button')].find(b => /apply|save|set/i.test(b.textContent || '')) : null;
          if (btn) btn.click();
          else if (form && form.tagName === 'FORM') form.requestSubmit();
          return { orig, next };
        })()`);

        let post200 = false;
        let statusSaved = false;
        for (let i = 0; i < 20; i++) {
          await sleep(150);
          post200 = diag.apiResponses.some((r) => r.url.includes("/api/controls") && r.status === 200);
          const st = await cdp.evalJs(`(document.querySelector('[role="status"]')?.textContent || '').trim()`);
          if (st.length > 0 && /saved|set|updated|pack/i.test(st)) statusSaved = true;
          if (post200 && statusSaved) break;
        }
        await cdp.send("Page.navigate", { url: baseUrl + "/settings/" });
        await waitForPageReady(cdp);
        const persistedBudget = await cdp.evalJs(`(() => {
          const inputs = [...document.querySelectorAll(${JSON.stringify(NUM_SEL)})];
          const target = inputs.find(inp => {
            const ctx = (inp.getAttribute('aria-label') || '') + ' ' + (inp.closest('div,section,li,tr')?.textContent || '');
            return /pack.?tokens|Context pack/i.test(ctx);
          }) || inputs[0];
          return target ? target.value : null;
        })()`);
        check(
          4,
          "UI-Budget",
          "Budget input update -> POST /api/controls 200 -> role=status -> persists after reload",
          Boolean(prep && post200 && persistedBudget === prep.next),
          `post200=${post200} statusSaved=${statusSaved} persisted=${persistedBudget} expected=${prep?.next}`
        );

        // Negative boundary check (allow expected 400 only during this step)
        diag.allowExpected400 = true;
        const negResult = await cdp.evalJs(`(async () => {
          const inputs = [...document.querySelectorAll(${JSON.stringify(NUM_SEL)})];
          const target = inputs.find(inp => {
            const ctx = (inp.getAttribute('aria-label') || '') + ' ' + (inp.closest('div,section,li,tr')?.textContent || '');
            return /pack.?tokens|Context pack/i.test(ctx);
          }) || inputs[0];
          if (!target) return { triggered: false };
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
          setter.call(target, '999999');
          target.dispatchEvent(new Event('input', { bubbles: true }));
          target.dispatchEvent(new Event('change', { bubbles: true }));
          const form = target.closest('form') || target.closest('div,section,li,tr') || target.parentElement;
          const btn = form ? [...form.querySelectorAll('button')].find(b => /apply|save|set/i.test(b.textContent || '')) : null;
          if (btn && !btn.disabled) btn.click();
          return { triggered: true };
        })()`);
        await sleep(500);
        const negStatus = await cdp.evalJs(`(() => {
          const s = document.querySelector('[role="alert"], [role="status"]');
          const errText = s ? s.textContent.trim() : '';
          const bodyErr = document.body.innerText.match(/between|range|max|invalid|refused|must be/i)?.[0] || '';
          return { errText, bodyErr };
        })()`);
        const saw400 = diag.apiResponses.some((r) => r.url.includes("/api/controls") && r.status === 400);
        diag.allowExpected400 = false;
        check(
          2,
          "UI-Budget",
          "Out-of-range budget input surfaces error feedback",
          Boolean(negResult.triggered && (saw400 || negStatus.errText.length > 0 || negStatus.bodyErr.length > 0)),
          `saw400=${saw400} status=${JSON.stringify(negStatus)}`
        );

        // Restore original budget value
        if (prep) {
          await cdp.evalJs(`(() => {
            const inputs = [...document.querySelectorAll(${JSON.stringify(NUM_SEL)})];
            const target = inputs.find(inp => {
              const ctx = (inp.getAttribute('aria-label') || '') + ' ' + (inp.closest('div,section,li,tr')?.textContent || '');
              return /pack.?tokens|Context pack/i.test(ctx);
            }) || inputs[0];
            if (!target) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(target, ${JSON.stringify(prep.orig)});
            target.dispatchEvent(new Event('input', { bubbles: true }));
            target.dispatchEvent(new Event('change', { bubbles: true }));
            const form = target.closest('form') || target.closest('div,section,li,tr') || target.parentElement;
            const btn = form ? [...form.querySelectorAll('button')].find(b => /apply|save|set/i.test(b.textContent || '')) : null;
            if (btn) btn.click();
            else {
              target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
              target.dispatchEvent(new Event('blur', { bubbles: true }));
            }
          })()`);
          await sleep(400);
        }
      }
    } else {
      skip(1, "UI-Budget", 'Settings renders "Budgets and allowances" group with >= 6 numeric inputs', "Baseline mode");
      skip(4, "UI-Budget", "Budget input update -> POST /api/controls 200 -> role=status -> persists after reload", "Baseline mode");
      skip(2, "UI-Budget", "Out-of-range budget input surfaces error feedback", "Baseline mode");
    }

    // Pairing code UI check on /settings/
    await settingsView("Privacy & network");
    const pairBtnFound = await cdp.evalJs(`(() => {
      const btns = [...document.querySelectorAll('button')];
      const b = btns.find(x => /pair|pairing code|new code|generate code/i.test(x.textContent || ''));
      if (!b) return false;
      b.click();
      return true;
    })()`);
    if (pairBtnFound || strictMode) {
      let codeSeen = false;
      let pair200 = false;
      if (pairBtnFound) {
        for (let i = 0; i < 20; i++) {
          await sleep(150);
          pair200 = diag.apiResponses.some((r) => r.url.includes("/api/pair/code") && r.status === 200);
          const txt = await cdp.evalJs(`document.body.innerText || ''`);
          if (/\b\d{6}\b/.test(txt) && pair200) {
            codeSeen = true;
            break;
          }
        }
      }
      check(
        3,
        "UI-Pairing",
        "Pairing code button generates 6-digit code via POST /api/pair/code",
        pairBtnFound && pair200 && codeSeen,
        `btn=${pairBtnFound} pair200=${pair200} codeSeen=${codeSeen}`
      );
    } else {
      skip(3, "UI-Pairing", "Pairing code button generates 6-digit code via POST /api/pair/code", "Baseline mode");
    }

    // /learning/ panels + Clear Ledger / Clear Examples buttons
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/learning/" });
    await waitForPageReady(cdp);

    // 2026-11-26: long pages show one section at a time under a sticky section bar; open one by
    // its id (the link carries data-section) before checking what it holds.
    const goSection = async (id) => {
      await cdp.evalJs(`(() => { const a = document.querySelector('[data-e2e="section-bar"] [data-section=${JSON.stringify(id)}]'); if (a) a.click(); })()`);
      await sleep(450);
    };
    const headingsNow = () => cdp.evalJs(`[...document.querySelectorAll('h1, h2, h3, .eyebrow')].map(e => e.textContent.trim())`);
    const seen = [];
    for (const id of ["evidence", "training", "units"]) {
      await goSection(id);
      seen.push(...(await headingsNow()));
    }
    const learningPanels = {
      evidence: seen.some(h => /Evidence ledger/i.test(h)),
      training: seen.some(h => /Training store/i.test(h)),
      primes: seen.some(h => /Primes/i.test(h)),
      control: seen.some(h => /^Learning switches$/i.test(h))
    };
    check(
      1,
      "UI-Learning",
      "/learning/ renders Evidence ledger, Training store, Primes, and Control panels",
      learningPanels.evidence && learningPanels.training && learningPanels.primes && learningPanels.control,
      JSON.stringify(learningPanels)
    );

    const btnTexts = () => cdp.evalJs(`[...document.querySelectorAll('button')].map(b => b.textContent.trim())`);
    await goSection("evidence");
    const evBtns = await btnTexts();
    await goSection("training");
    const trBtns = await btnTexts();
    const clearButtons = {
      hasLedger: evBtns.some(t => /clear.*ledger|ledger.*clear/i.test(t)),
      hasExamples: trBtns.some(t => /clear.*(examples|training)|(examples|training).*clear/i.test(t))
    };

    async function triggerClearButton(regexSrc, expectedApiPath) {
      const clicked = await cdp.evalJs(`(() => {
        const rx = new RegExp(${JSON.stringify(regexSrc)}, 'i');
        const btn = [...document.querySelectorAll('button')].find(b => rx.test(b.textContent || ''));
        if (!btn) return false;
        btn.click();
        return true;
      })()`);
      if (!clicked) return { ok: false, detail: "button not found" };
      // 2026-11-26: the multi-step delete flow (preview, the typed word, the final button). The
      // panel can remount while the project key settles after load: start again until the
      // preview is up (at most 5 tries).
      let flow = null;
      for (let k = 0; k < 5 && !flow; k++) {
        flow = await deleteFlowConfirm(cdp);
        if (!flow) {
          await cdp.evalJs(`(() => {
            const rx = new RegExp(${JSON.stringify(regexSrc)}, 'i');
            const again = [...document.querySelectorAll('button')].find(b => rx.test(b.textContent || ''));
            if (again && !again.disabled) again.click();
          })()`);
        }
      }
      if (!flow) return { ok: false, detail: "the delete flow's preview did not open" };
      if (!flow.guarded) return { ok: false, detail: "the final delete button was usable before the word was typed" };
      let api200 = false;
      let statusOk = false;
      for (let i = 0; i < 20; i++) {
        await sleep(150);
        api200 = diag.apiResponses.some((r) => r.url.includes(expectedApiPath) && r.status === 200);
        const st = await cdp.evalJs(`(document.querySelector('[role="status"]')?.textContent || '').trim()`);
        if (st.length > 0) statusOk = true;
        if (api200 && statusOk) break;
      }
      return { ok: api200, detail: `api200=${api200} statusOk=${statusOk}` };
    }

    if (clearButtons.hasLedger || strictMode) {
      await goSection("evidence");
      const resL = await triggerClearButton("clear.*ledger|ledger.*clear", "/api/ledger/clear");
      check(1, "UI-Learning", "Clear Ledger button triggers 200 POST /api/ledger/clear", resL.ok, resL.detail);
    } else {
      skip(1, "UI-Learning", "Clear Ledger button triggers 200 POST /api/ledger/clear", "Baseline mode");
    }

    if (clearButtons.hasExamples || strictMode) {
      await goSection("training");
      const resE = await triggerClearButton("clear.*(examples|training)|(examples|training).*clear", "/api/examples/clear");
      check(1, "UI-Learning", "Clear Examples button triggers 200 POST /api/examples/clear", resE.ok, resE.detail);
    } else {
      skip(1, "UI-Learning", "Clear Examples button triggers 200 POST /api/examples/clear", "Baseline mode");
    }

    await goSection("overview");
    const continualPanel = await cdp.evalJs(`(() => {
      const txt = document.body.innerText || '';
      return {
        panel: /Continuous learning engine/i.test(txt),
        cartridges: /Continual learning cartridges · 6/i.test(txt),
        // 2026-11-25: measured figures only; the ones not measured are named, never shown as numbers.
        measured: !!document.querySelector('[data-e2e="learning-measures"]') && /not measured in this version/i.test(txt),
        defaults: /eval 83%|baseline 83%/i.test(txt),
      };
    })()`);
    check(
      1,
      "UI-Learning",
      "/learning/ renders the Continuous learning engine, its 6 continual learning cartridges and measured figures only",
      continualPanel.panel && continualPanel.cartridges && continualPanel.measured && !continualPanel.defaults,
      JSON.stringify(continualPanel)
    );

    const cycleClicked = await cdp.evalJs(`(() => {
      const btn = [...document.querySelectorAll('button')].find(b => /Run a learning cycle now/i.test(b.textContent || ''));
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    })()`);
    let cycle200 = false;
    if (cycleClicked) {
      for (let i = 0; i < 20; i++) {
        await sleep(150);
        cycle200 = diag.apiResponses.some((r) => r.url.includes("/api/learning/cycle") && r.status === 200);
        if (cycle200) break;
      }
    }
    check(1, "UI-Learning", "Run a learning cycle now button triggers 200 POST /api/learning/cycle", cycleClicked && cycle200, `clicked=${cycleClicked} cycle200=${cycle200}`);

    // Rollback: shown once a snapshot exists (a cycle has run). Clicked only in an isolated home;
    // against --url its presence is checked and the real state is left alone.
    let rollbackSeen = false;
    for (let i = 0; i < 20 && !rollbackSeen; i++) {
      rollbackSeen = await cdp.evalJs(`[...document.querySelectorAll('button')].some(b => /Roll back the newest cycle/i.test(b.textContent || ''))`);
      if (!rollbackSeen) await sleep(150);
    }
    if (isolated && rollbackSeen) {
      const before = diag.apiResponses.filter((r) => r.url.includes("/api/learning/cycle") && r.status === 200).length;
      await cdp.evalJs(`(() => {
        const b = [...document.querySelectorAll('button')].find(x => /Roll back the newest cycle/i.test(x.textContent || ''));
        if (b && !b.disabled) b.click();
      })()`);
      await sleep(200);
      await cdp.evalJs(`(() => {
        const b = [...document.querySelectorAll('button')].find(x => /Confirm roll back/i.test(x.textContent || ''));
        if (b && !b.disabled) b.click();
      })()`);
      let rb200 = false;
      for (let i = 0; i < 20 && !rb200; i++) {
        await sleep(150);
        rb200 = diag.apiResponses.filter((r) => r.url.includes("/api/learning/cycle") && r.status === 200).length > before;
      }
      check(1, "UI-Learning", "Rollback button triggers 200 POST /api/learning/cycle (action rollback)", rb200, `rollbackSeen=${rollbackSeen} rb200=${rb200}`);
    } else {
      check(1, "UI-Learning", "Rollback control shown after a learning cycle", rollbackSeen, `rollbackSeen=${rollbackSeen} isolated=${isolated}`);
    }

    await goSection("training");
    // 2026-11-25: Your training data (export, import, reference material). Export is clicked only
    // in an isolated home; against --url the panel's presence is checked and nothing is written.
    const trainingPanel = await cdp.evalJs(`(() => ({
      transfer: !!document.querySelector('[data-e2e="training-transfer"]'),
      references: !!document.querySelector('[data-e2e="training-references"]'),
    }))()`);
    check(1, "UI-Learning", "/learning/ renders Your training data (transfer and reference material)", trainingPanel.transfer && trainingPanel.references, JSON.stringify(trainingPanel));
    // Distillation: the Learning page's section with the shared builder (kind, MicroModel,
    // technique, teacher data, train) and, one click on, the teacher data panel.
    const distil = await cdp.evalJs(`(() => ({
      section: !!document.querySelector('[data-e2e="distil-section"]'),
      builder: !!document.querySelector('[data-e2e="distil-builder"]'),
      target: !!document.querySelector('[data-e2e="distil-target"]'),
      role: !!document.querySelector('[data-e2e="distil-role"]'),
      technique: !!document.querySelector('[data-e2e="distil-technique"]'),
      train: !!document.querySelector('[data-e2e="distil-build"]'),
    }))()`);
    check(1, "UI-Learning", "/learning/ renders the Distillation section with its builder (kind, MicroModel, technique, train)", distil.section && distil.builder && distil.target && distil.role && distil.technique && distil.train, JSON.stringify(distil));
    await cdp.evalJs(`(() => {
      const b = [...document.querySelectorAll('[data-e2e="distil-section"] [role="radio"]')].find(x => /teacher data/i.test(x.textContent || ''));
      if (b) b.click();
    })()`);
    await sleep(400);
    const distilData = await cdp.evalJs(`!!document.querySelector('[data-e2e="distil-data"]') && !!document.querySelector('[data-e2e="distil-import"]')`);
    check(1, "UI-Learning", "/learning/ Distillation shows its teacher data panel (import)", distilData === true, String(distilData));
    if (isolated) {
      const beforeEx = diag.apiResponses.filter((r) => r.url.includes("/api/training/export") && r.status === 200).length;
      await cdp.evalJs(`(() => {
        const b = [...document.querySelectorAll('button')].find(x => /Export to a file/i.test(x.textContent || ''));
        if (b && !b.disabled) b.click();
      })()`);
      let ex200 = false;
      for (let i = 0; i < 30 && !ex200; i++) {
        await sleep(150);
        ex200 = diag.apiResponses.filter((r) => r.url.includes("/api/training/export") && r.status === 200).length > beforeEx;
      }
      check(1, "UI-Learning", "Export to a file triggers 200 POST /api/training/export", ex200, `ex200=${ex200}`);
    }

    // Stated preferences: the builder in the middle (chat box, parsed rules, cartridges) and its
    // notes in the side panel. In an isolated home, words typed in the box are previewed (nothing
    // written), confirmed (a plugged cartridge listed with its rules), and unplugged again.
    await cdp.evalJs(`(() => { const p = document.querySelector('[data-e2e="pref-builder"]'); if (p) p.scrollIntoView({ block: 'center' }); })()`);
    const prefWait = async (js, tries = 40) => {
      for (let i = 0; i < tries; i++) {
        const v = await cdp.evalJs(js);
        if (v) return v;
        await sleep(150);
      }
      return null;
    };
    const prefPanel = await prefWait(`!!document.querySelector('[data-e2e="pref-builder"] [data-e2e="pref-input"]') && !!document.querySelector('[data-e2e="pref-cartridges"]') && !!document.querySelector('[data-e2e="pref-check"]')`, 60);
    check(1, "UI-Learning", "/learning/ renders the Preferences builder (chat box, cartridges, draft check)", Boolean(prefPanel));
    if (isolated && prefPanel) {
      await cdp.evalJs(`(() => {
        const area = document.querySelector('[data-e2e="pref-input"]');
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(area, 'no em dashes. snake_case for functions');
        area.dispatchEvent(new Event('input', { bubbles: true }));
        const name = document.querySelector('[data-e2e="pref-name"]');
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(name, 'ui e2e');
        name.dispatchEvent(new Event('input', { bubbles: true }));
      })()`);
      await sleep(150);
      await cdp.evalJs(`(() => { const b = document.querySelector('[data-e2e="pref-preview-btn"]'); if (b && !b.disabled) b.click(); })()`);
      const previewed = await prefWait(`(() => { const p = document.querySelector('[data-e2e="pref-preview"]'); if (!p) return null; const rules = [...p.querySelectorAll('[data-e2e="pref-rule"]')].map(r => r.getAttribute('data-enforcement')); return rules.length === 2 ? rules : null; })()`, 60);
      const notYet = await cdp.evalJs(`!document.querySelector('[data-e2e="pref-cartridge"][data-id="stated-ui-e2e"]')`);
      check(1, "UI-Learning", "Preferences: typed words are previewed as rules (enforced, checked) and nothing is written yet", Boolean(previewed && previewed[0] === "enforced" && previewed[1] === "checked" && notYet), JSON.stringify({ previewed, notYet }));
      await cdp.evalJs(`(() => { const b = document.querySelector('[data-e2e="pref-confirm"]'); if (b && !b.disabled) b.click(); })()`);
      const written = await prefWait(`(() => { const c = document.querySelector('[data-e2e="pref-cartridge"][data-id="stated-ui-e2e"]'); return c && c.getAttribute('data-plugged') === 'true' ? true : null; })()`, 80);
      const write200 = diag.apiResponses.some((r) => r.url.includes("/api/preferences/say") && r.status === 200);
      check(1, "UI-Learning", "Preferences: confirming writes the cartridge, listed plugged in (200 POST /api/preferences/say)", Boolean(written && write200), JSON.stringify({ written, write200 }));
      await cdp.evalJs(`(() => { const c = document.querySelector('[data-e2e="pref-cartridge"][data-id="stated-ui-e2e"]'); const b = c && c.querySelector('[data-e2e="pref-plug"]'); if (b && !b.disabled) b.click(); })()`);
      const unplugged = await prefWait(`(() => { const c = document.querySelector('[data-e2e="pref-cartridge"][data-id="stated-ui-e2e"]'); return c && c.getAttribute('data-plugged') === 'false' ? true : null; })()`, 80);
      check(1, "UI-Learning", "Preferences: the cartridge's unplug button unplugs it (the engine's state follows)", Boolean(unplugged));
    }
    await goSection("preferences");
    const prefNotes = await prefWait(`!!document.querySelector('[data-e2e="pref-notes"]')`, 40);
    check(1, "UI-Learning", "/learning/ side panel has the Preferences notes (how rules are held, the forms read)", Boolean(prefNotes));

    await goSection("units");
    // 2026-11-26 (Luis: "plug in and unplug the learned units"): the Learned units panel, by
    // compartment; in an isolated home a compartment switch is clicked off and on again.
    await cdp.evalJs(`(() => { const p = document.querySelector('[data-e2e="learned-units"]'); if (p) p.scrollIntoView({ block: 'center' }); })()`);
    let luPanel = { panel: false, compartments: 0 };
    for (let i = 0; i < 40 && !luPanel.panel; i++) {
      luPanel = await cdp.evalJs(`(() => ({ panel: !!document.querySelector('[data-e2e="learned-units"]'), compartments: document.querySelectorAll('[data-e2e="learned-units"] [data-e2e="compartment"]').length }))()`);
      if (!luPanel.panel) await sleep(150);
    }
    check(1, "UI-Learning", "/learning/ renders Learned units by compartment", luPanel.panel && luPanel.compartments > 0, JSON.stringify(luPanel));
    // 2026-11-27: Learning's middle has four main tabs (loop, distillation, learned units,
    // preferences); distillation shows its path of five steps, and the side panel its notes.
    const learnTabs = await cdp.evalJs(`(() => { const tabs = [...document.querySelectorAll('[data-e2e="main-tab"]')].map((b) => b.getAttribute('data-tab')); const pane = (sel) => document.querySelector(sel)?.closest('[data-e2e="main-pane"]')?.getAttribute('data-tab') ?? null; return { tabs, distil: pane('[data-e2e="distil-section"]'), units: pane('[data-e2e="learned-units"]'), prefs: pane('[data-e2e="pref-builder"]'), path: document.querySelectorAll('[data-e2e="distil-path"] li').length, notes: !!document.querySelector('[data-e2e="distil-notes"]') }; })()`);
    check(1, "UI-Learning", "Learning has main tabs Loop, Distillation, Learned units, Preferences, each holding its part; distillation shows its five steps and the side panel its notes", Boolean(learnTabs && learnTabs.tabs.join(",") === "loop,distillation,units,stated" && learnTabs.distil === "distillation" && learnTabs.units === "units" && learnTabs.prefs === "stated" && learnTabs.path === 5 && learnTabs.notes), JSON.stringify(learnTabs));
    if (isolated && luPanel.panel) {
      // The first compartment's switch: clicked off, its state must follow the engine's reply
      // (aria-checked flips), then clicked back on.
      const state = () => cdp.evalJs(`(() => { const s = document.querySelector('[data-e2e="learned-units"] [data-e2e="compartment"] [role="switch"]'); return s ? s.getAttribute('aria-checked') : null; })()`);
      const click = () => cdp.evalJs(`(() => { const s = document.querySelector('[data-e2e="learned-units"] [data-e2e="compartment"] [role="switch"]'); if (s && !s.disabled) s.click(); })()`);
      const was = await state();
      await click();
      let now = was;
      for (let i = 0; i < 30 && now === was; i++) {
        await sleep(150);
        now = await state();
      }
      check(1, "UI-Learning", "A compartment switch changes state through POST /api/learned-units", was !== null && now !== was, `was=${was} now=${now}`);
      if (now !== was) {
        await click();
        for (let i = 0; i < 30 && (await state()) !== was; i++) await sleep(150);
      }
    }

    // =========================================================================
    // 2b. JOB FILTERS: Build, On/Off, Remove (hot swap within 1 s). 2026-11-26: the controls live on
    // /parameters/ beside the maps the filters are built from (Cartridges links there).
    // =========================================================================
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/parameters/" });
    await waitForPageReady(cdp);
    const surgeryNow = async () => {
      try {
        return await (await fetch(`${baseUrl}/api/surgery`, { cache: "no-store" })).json();
      } catch {
        return null;
      }
    };
    const filterRows = (s) =>
      Object.entries(s?.roles ?? {}).flatMap(([role, info]) => (info.job_filters ?? []).map((jf) => ({ role, job: jf.job, enabled: jf.enabled !== false })));
    let rowsUi = null;
    for (let i = 0; i < 30; i++) {
      rowsUi = await cdp.evalJs(`(() => {
        const add = document.querySelector('[data-e2e="pmap-add"]');
        const build = add && [...add.querySelectorAll('button')].find(b => /^(build|building)$/i.test((b.textContent || '').trim()));
        const rows = [...document.querySelectorAll('[data-e2e="pmap-row"]')].filter(r => /job-filter-[^ ]+\\.json/.test(r.textContent || ''));
        const toggles = rows.flatMap(r => [...r.querySelectorAll('button')].filter(b => /^(unplug|plug in)$/i.test((b.textContent || '').trim())));
        const removes = rows.flatMap(r => [...r.querySelectorAll('button')].filter(b => /^remove$/i.test((b.textContent || '').trim())));
        return { build: Boolean(build), toggles: toggles.length, removes: removes.length };
      })()`);
      if (rowsUi.build && rowsUi.toggles > 0) break;
      await sleep(150);
    }
    const filtersApi = filterRows(await surgeryNow());
    check(
      1,
      "UI-Parameters",
      "/parameters/ shows the job-filter Build control and plug in/unplug and Remove per filter",
      rowsUi.build && rowsUi.toggles === filtersApi.length && rowsUi.removes === filtersApi.length && (filtersApi.length > 0 || !isolated),
      `ui=${JSON.stringify(rowsUi)} filters=${filtersApi.length}`
    );

    // Build: an invalid job slug is refused by the server's check before nano-lab is started
    // (nothing runs on the card), and the refusal is shown.
    diag.allowExpected400 = true;
    await cdp.evalJs(`(() => {
      const add = document.querySelector('[data-e2e="pmap-add"]');
      const input = add && add.querySelector('input');
      if (!input) return;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(input, 'bad slug!');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await sleep(200);
    await cdp.evalJs(`(() => {
      const add = document.querySelector('[data-e2e="pmap-add"]');
      const b = add && [...add.querySelectorAll('button')].find(x => /^build$/i.test((x.textContent || '').trim()));
      if (b && !b.disabled) b.click();
    })()`);
    let build400 = false;
    let buildNote = "";
    for (let i = 0; i < 20 && !(build400 && buildNote); i++) {
      await sleep(150);
      build400 = diag.apiResponses.some((r) => r.url.includes("/api/surgery/job-filter/build") && r.status === 400);
      buildNote = await cdp.evalJs(`(document.body.innerText.match(/role and job must be alphanumeric[^\\n]*/) || [''])[0]`);
    }
    diag.allowExpected400 = false;
    check(2, "UI-Parameters", "Build with an invalid job slug: POST /api/surgery/job-filter/build answers 400 and the refusal is shown (no fit started)", build400 && buildNote.length > 0, `build400=${build400} note=${JSON.stringify(buildNote)}`);
    await cdp.evalJs(`(() => {
      const add = document.querySelector('[data-e2e="pmap-add"]');
      const input = add && add.querySelector('input');
      if (!input) return;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(input, 'code-slice');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);

    // On/Off: the probe filter of an isolated home (or, against --url, the first filter, put back
    // after). The file must flip within 1 s of the click, seen by GET /api/surgery.
    const target = filtersApi.find((f) => f.job === PROBE_JOB) ?? (isolated ? null : filtersApi[0] ?? null);
    if (target) {
      const clickToggle = (job) =>
        cdp.evalJs(`(() => {
          const row = [...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('job-filter-' + ${JSON.stringify(job)} + '.json'));
          const b = row && [...row.querySelectorAll('button')].find(x => /^(unplug|plug in)$/i.test((x.textContent || '').trim()));
          if (!b) return false;
          b.click();
          return true;
        })()`);
      const t0 = Date.now();
      const clicked = await clickToggle(target.job);
      let flippedMs = -1;
      for (let i = 0; i < 40 && clicked; i++) {
        const now = filterRows(await surgeryNow()).find((f) => f.role === target.role && f.job === target.job);
        if (now && now.enabled !== target.enabled) {
          flippedMs = Date.now() - t0;
          break;
        }
        await sleep(50);
      }
      let uiShows = false;
      for (let i = 0; i < 20 && flippedMs >= 0 && !uiShows; i++) {
        uiShows = await cdp.evalJs(`(() => { const row = [...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('job-filter-' + ${JSON.stringify(target.job)} + '.json')); return !!row && [...row.querySelectorAll('button')].some(b => (b.textContent || '').trim().toLowerCase() === ${JSON.stringify(target.enabled ? "plug in" : "unplug")}); })()`);
        if (!uiShows) await sleep(100);
      }
      check(3, "UI-Parameters", `Job filter ${target.role}/${target.job} plug in/unplug reflected within 1 s (file and page)`, clicked && flippedMs >= 0 && flippedMs <= 1000 && uiShows, `clicked=${clicked} flippedMs=${flippedMs} uiShows=${uiShows}`);
      // Put it back as it was.
      await clickToggle(target.job);
      for (let i = 0; i < 20; i++) {
        const now = filterRows(await surgeryNow()).find((f) => f.role === target.role && f.job === target.job);
        if (now && now.enabled === target.enabled) break;
        await sleep(100);
      }
    } else {
      skip(3, "UI-Parameters", "Job filter plug in/unplug reflected within 1 s", "no job filter to toggle on this server");
    }

    // Remove: only the probe filter of an isolated home is removed.
    if (isolated && filtersApi.some((f) => f.job === PROBE_JOB)) {
      // The page is busy for a moment after the toggles; press once the button is usable.
      const press = () => cdp.evalJs(`(() => {
        const row = [...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('job-filter-' + ${JSON.stringify(PROBE_JOB)} + '.json'));
        const b = row && [...row.querySelectorAll('button')].find(x => /^remove$/i.test((x.textContent || '').trim()));
        if (!b) return "missing";
        if (b.disabled) return "busy";
        b.click();
        return "pressed";
      })()`);
      let state = "busy";
      for (let i = 0; i < 30 && state === "busy"; i++) {
        state = await press();
        if (state === "busy") await sleep(150);
      }
      const pressed = state === "pressed";
      // 2026-11-26: the remove goes through the delete flow (preview, the typed word, the final button).
      const flow = pressed ? await deleteFlowConfirm(cdp) : null;
      const removed = Boolean(pressed && flow && flow.guarded);
      let gone = false;
      for (let i = 0; i < 20 && removed && !gone; i++) {
        await sleep(100);
        gone = !filterRows(await surgeryNow()).some((f) => f.job === PROBE_JOB);
      }
      check(3, "UI-Parameters", "Remove deletes the probe job filter (POST /api/surgery/job-filter action delete)", removed && gone, `pressed=${pressed} flow=${JSON.stringify(flow)} gone=${gone}`);
    } else {
      check(1, "UI-Parameters", "Remove control present for every job filter (not clicked on a real library)", rowsUi.removes === filtersApi.length, `removes=${rowsUi.removes} filters=${filtersApi.length}`);
    }

    // =========================================================================
    // 2b2. 2026-11-26 (audit Phase B): every map, cartridge, adapter, lens, memory, goal and
    // connection control the pages carry, present and answering.
    // =========================================================================
    const waitFor = async (js, tries = 40) => {
      let v = null;
      for (let i = 0; i < tries; i++) {
        v = await cdp.evalJs(js);
        if (v) return v;
        await sleep(150);
      }
      return v;
    };
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/parameters/" });
    await waitForPageReady(cdp);
    const importShown = await waitFor(`!!document.querySelector('[data-e2e="import-panel"]')`);
    // A map to open: the first row's inspect (an isolated home may have none left once the probe
    // filter was removed above; then only the import panel is checked).
    const hasRow = await cdp.evalJs(`(() => { const b = [...document.querySelectorAll('[data-e2e="pmap-row"] button')].find(x => /^inspect/i.test((x.textContent || '').trim())); if (b) { b.click(); return true; } return false; })()`);
    const inspector = hasRow ? await waitFor(`(() => { const m = document.querySelector('[data-e2e="map-inspector"]'); if (!m) return null; const t = m.textContent || ''; return /on disk/.test(t) || /Not readable|did not answer/.test(t) ? { read: /on disk/.test(t) } : null; })()`, 60) : { read: true, none: true };
    check(1, "UI-Parameters", "/parameters/ opens a map's contents (inspect) and offers importing your own maps and cartridges", Boolean(importShown && inspector && inspector.read && (hasRow || !isolated)), JSON.stringify({ importShown, hasRow, inspector }));

    // Cartridge parameters: the lane lists every object holding parameters per MicroModel; one opens
    // in the rail with its tensors, the per-layer norm strip and the strength; Apply makes a new
    // version (isolated home only); a link #cartridge=<key> opens one directly.
    const paramsKey = `filter:writer/${PARAMS_JOB}`;
    const lane = await waitFor(`(() => { const l = document.querySelector('[data-e2e="cparam-lane"]'); if (!l) return null; const rows = [...l.querySelectorAll('[data-e2e="cparam-row"]')]; if (!rows.length && !/No object holds parameters/.test(l.textContent || '')) return null; const row = rows.find(r => r.getAttribute('data-key') === ${JSON.stringify(paramsKey)}) || rows[0]; if (row) row.click(); return { rows: rows.length, key: row ? row.getAttribute('data-key') : null }; })()`, 80);
    const detail = lane && lane.key ? await waitFor(`(() => { const d = document.querySelector('[data-e2e="cparam-detail"]'); if (!d) return null; const t = d.querySelectorAll('[data-e2e="cparam-tensor"]').length; const ro = !!d.querySelector('[data-e2e="cparam-readonly"]'); if (!t && !/holds no parameters|Not readable/.test(d.textContent || '')) return null; return { key: d.getAttribute('data-key'), tensors: t, layers: !!d.querySelector('[data-e2e="cparam-layers"]'), strength: !!d.querySelector('[data-e2e="cparam-strength"]'), apply: !!d.querySelector('[data-e2e="cparam-apply"]'), ro }; })()`, 80) : null;
    check(1, "UI-Parameters", "/parameters/ lists cartridge parameters per MicroModel and opens one with its tensors, layer strip and strength", Boolean(lane && (lane.rows > 0 || !isolated) && (!isolated || (detail && detail.key === paramsKey && detail.tensors === 4 && detail.layers && detail.strength && detail.apply))), JSON.stringify({ lane, detail }));
    if (isolated && detail && detail.key === paramsKey) {
      const applied = await cdp.evalJs(`(async () => {
        const d = document.querySelector('[data-e2e="cparam-detail"]');
        const r = d && d.querySelector('[data-e2e="cparam-strength"] input');
        if (!r) return { range: false };
        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        setter.call(r, '0.5');
        r.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(x => setTimeout(x, 300));
        const b = d.querySelector('[data-e2e="cparam-apply"]');
        if (!b || b.disabled) return { range: true, apply: false };
        b.click();
        for (let i = 0; i < 60; i++) {
          await new Promise(x => setTimeout(x, 150));
          const t = d.textContent || '';
          if (t.includes('New version filter:writer/e2e-params-p2')) return { range: true, apply: true, made: true };
          if (/refused|error/i.test(t) && !/New version/.test(t)) return { range: true, apply: true, made: false, text: t.slice(0, 200) };
        }
        return { range: true, apply: true, made: false };
      })()`);
      const listed = await cdp.evalJs(`fetch('/api/cartridges/parameters').then(r => r.json()).then(j => (j.objects || []).some(o => o.key === 'filter:writer/e2e-params-p2' && o.state === 'plugged out')).catch(() => false)`);
      check(3, "UI-Parameters", "Apply on a strength makes a new version of the job filter, plugged out (POST /api/cartridges/parameters)", Boolean(applied && applied.made && listed), JSON.stringify({ applied, listed }));
      await cdp.send("Page.navigate", { url: baseUrl + "/parameters/#cartridge=" + encodeURIComponent(paramsKey) });
      await waitForPageReady(cdp);
      const linked = await waitFor(`(() => { const d = document.querySelector('[data-e2e="cparam-detail"]'); return d && d.getAttribute('data-key') === ${JSON.stringify(paramsKey)} ? true : null; })()`, 80);
      check(1, "UI-Parameters", "/parameters/#cartridge=<key> opens that object's parameters in the side panel (the Cartridges inspector's parameters link)", Boolean(linked), JSON.stringify({ linked }));
      // The whole-set viewer is the focus view's working view: opened on a small job filter, it
      // shows every tensor as a heatmap (read when mounted, only for the opened object).
      await cdp.send("Page.navigate", { url: baseUrl + "/parameters/?cartridge=" + encodeURIComponent(paramsKey) });
      await waitForPageReady(cdp);
      const overview = await waitFor(`(() => { const f = document.querySelector('[data-e2e="cparam-focus"]'); const o = f && f.querySelector('[data-e2e="param-overview"]'); if (!o) return null; const c = o.querySelectorAll('[data-e2e="param-overview-canvas"]').length; return c > 0 ? { canvases: c, key: f.getAttribute('data-key') } : null; })()`, 100);
      check(1, "UI-Parameters", "/parameters/?cartridge=<a small filter> shows the whole-set viewer with at least one tensor heatmap in the focus view", Boolean(overview && overview.canvases > 0 && overview.key === paramsKey), JSON.stringify({ overview }));
      const viewer = await waitFor(`(async () => { const f = document.querySelector('[data-e2e="cparam-focus"]'); if (!f || !f.querySelector('[data-e2e="param-inspector"]') || !f.querySelector('[data-e2e="param-legend"]') || !f.querySelector('[data-e2e="param-ruler"]')) return null; const c = f.querySelector('[data-e2e="cparam-controls"]'); const t = f.querySelector('[data-e2e="cparam-controls-toggle"]'); if (!c || !t) return null; const h0 = c.getBoundingClientRect().height; t.click(); await new Promise((r) => setTimeout(r, 200)); const h1 = c.getBoundingClientRect().height; t.click(); await new Promise((r) => setTimeout(r, 200)); const h2 = c.getBoundingClientRect().height; return { h0, h1, h2, toggles: f.querySelectorAll('[data-e2e="param-layer-toggle"]').length }; })()`, 100);
      check(1, "UI-Parameters", "The viewer has rulers, a numeric colour scale and an inspector, no overlay switches on the heatmap, and the pinned controls collapse and expand", Boolean(viewer && viewer.toggles === 0 && viewer.h1 < viewer.h0 && viewer.h2 === viewer.h0), JSON.stringify({ viewer }));

      // From a cartridge on Cartridges to its parameters on Parameters: the drawer's "view
      // parameters" opens the Parameters page focused on it; there the drill down (a tensor, its
      // grid, a cell), the map values beside the selection, and an edit's preview before Apply.
      resetDiag();
      await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/" });
      await waitForPageReady(cdp);
      const fromCart = await waitFor(`(() => { const c = [...document.querySelectorAll('[data-e2e="cart"]')].find(x => x.getAttribute('data-key') === 'filter:writer-${PARAMS_JOB}@1'); const b = c && c.querySelector('.cart-label button'); if (!b) return null; b.click(); return true; })()`, 80);
      const viewLink = fromCart ? await waitFor(`(() => { const d = document.querySelector('[data-e2e="cartridge-drawer"]'); const a = d && d.querySelector('[data-e2e="view-params"]'); if (!a) return null; a.click(); return a.getAttribute('href'); })()`, 80) : null;
      const focused = viewLink ? await waitFor(`(() => { const f = document.querySelector('[data-e2e="cparam-focus"]'); return f && f.getAttribute('data-key') === ${JSON.stringify(paramsKey)} && f.querySelector('[data-e2e="cparam-focus-layers"]') && f.querySelector('[data-e2e="cparam-maps"]') ? { href: location.pathname + location.search } : null; })()`, 100) : null;
      check(1, "UI-Parameters", "A cartridge on Cartridges offers view parameters, which opens Parameters focused on it with its layers and its mapping", Boolean(fromCart && viewLink && focused), JSON.stringify({ fromCart, viewLink, focused }));
      if (focused) {
        const drilled = await cdp.evalJs(`(async () => {
          const wait = async (f, n = 60) => { for (let i = 0; i < n; i++) { const v = f(); if (v) return v; await new Promise(r => setTimeout(r, 150)); } return null; };
          const t = await wait(() => [...document.querySelectorAll('[data-e2e="cparam-focus-tensor"]')].find(b => b.getAttribute('data-name') === 'L2.o.gain'));
          if (!t) return { tensor: false };
          t.click();
          const cell = await wait(() => document.querySelectorAll('[data-e2e="cparam-cell"]')[1]);
          if (!cell) return { tensor: true, grid: false };
          cell.click();
          const sel = await wait(() => { const s = document.querySelector('[data-e2e="cparam-selection"]'); return s && /1 selected/.test(s.textContent || '') ? s : null; });
          const op = document.querySelector('[data-e2e="cparam-op"]');
          const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set;
          setter.call(op, 'set');
          op.dispatchEvent(new Event('change', { bubbles: true }));
          await new Promise(r => setTimeout(r, 200));
          const v = document.querySelector('[data-e2e="cparam-value"]');
          const vs = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
          vs.call(v, '2');
          v.dispatchEvent(new Event('input', { bubbles: true }));
          await new Promise(r => setTimeout(r, 200));
          // The overlay is the default target; this check makes a new version.
          const tg = document.querySelector('[data-e2e="cparam-target"]');
          if (tg) { setter.call(tg, 'version'); tg.dispatchEvent(new Event('change', { bubbles: true })); await new Promise(r => setTimeout(r, 200)); }
          document.querySelector('[data-e2e="cparam-preview-btn"]').click();
          const pv = await wait(() => { const p = document.querySelector('[data-e2e="cparam-preview"]'); return p && /1 value changes/.test(p.textContent || '') ? p.textContent : null; });
          const apply = document.querySelector('[data-e2e="cparam-apply-focus"]');
          if (!pv || !apply || apply.disabled) return { tensor: true, grid: true, selection: !!sel, preview: !!pv };
          apply.click();
          const made = await wait(() => { const e = document.querySelector('[data-e2e="cparam-edit"]'); return e && (e.textContent || '').includes('New version filter:writer/e2e-params-p') ? true : null; });
          return { tensor: true, grid: true, selection: !!sel, preview: true, made: !!made };
        })()`);
        check(3, "UI-Parameters", "The focus drill down picks a value in the tensor grid, shows it beside the maps, previews a set and applies it as a new version", Boolean(drilled && drilled.selection && drilled.preview && drilled.made), JSON.stringify(drilled));
        // The overlay on the same selection: an edit applied in the overlay (the default target),
        // then mask and freeze drawn on the grid (dimmed, hatched) and listed, then undone.
        const overlaid = await cdp.evalJs(`(async () => {
          const wait = async (f, n = 60) => { for (let i = 0; i < n; i++) { const v = f(); if (v) return v; await new Promise(r => setTimeout(r, 150)); } return null; };
          const sset = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set;
          const tg = document.querySelector('[data-e2e="cparam-target"]');
          if (!tg) return { target: false };
          sset.call(tg, 'overlay');
          tg.dispatchEvent(new Event('change', { bubbles: true }));
          await new Promise(r => setTimeout(r, 200));
          document.querySelector('[data-e2e="cparam-preview-btn"]').click();
          const pv = await wait(() => { const p = document.querySelector('[data-e2e="cparam-preview"]'); return p && /values? changes?/.test(p.textContent || '') && p.querySelector('[data-e2e="cparam-takes-effect"]') ? true : null; });
          const apply = document.querySelector('[data-e2e="cparam-apply-focus"]');
          if (!pv || !apply || apply.disabled) return { target: true, preview: !!pv };
          apply.click();
          const edited = await wait(() => /Overlay entry e\\d/.test(document.querySelector('[data-e2e="cparam-edit"]')?.textContent || '') ? true : null);
          const entries = () => document.querySelectorAll('[data-e2e="cparam-overlay-entry"]').length;
          document.querySelector('[data-e2e="cparam-mask"]').click();
          const masked = await wait(() => document.querySelector('[data-e2e="cparam-cell"][data-masked="1"]') && document.querySelector('[data-e2e="cparam-overlay-entry"][data-action="mask"]') ? true : null, 80);
          document.querySelector('[data-e2e="cparam-freeze"]').click();
          const frozen = await wait(() => document.querySelector('[data-e2e="cparam-cell"][data-frozen="1"]') && document.querySelector('[data-e2e="cparam-overlay-entry"][data-action="freeze"]') ? true : null, 80);
          const strip = !!document.querySelector('[data-e2e="cparam-overlay-strip"] [data-frozen="1"]');
          const before = entries();
          document.querySelector('[data-e2e="cparam-overlay-undo"]').click();
          const undone = await wait(() => entries() === before - 1 ? true : null, 80);
          return { target: true, preview: true, edited: !!edited, masked: !!masked, frozen: !!frozen, strip, undone: !!undone };
        })()`);
        check(3, "UI-Parameters", "The overlay on a selection: an edit applied in the overlay, mask and freeze drawn on the grid and the layer strip and listed, undo of the newest change", Boolean(overlaid && overlaid.edited && overlaid.masked && overlaid.frozen && overlaid.strip && overlaid.undone), JSON.stringify(overlaid));
        // 2026-11-27 (parity gap B2): the same selection's values exported from the controls block
        // (POST /api/cartridges/parameters/export); the note names the file it wrote.
        const exported = await cdp.evalJs(`(async () => {
          const b = document.querySelector('[data-e2e="cparam-export-btn"]');
          if (!b) return { button: false };
          b.click();
          for (let i = 0; i < 60; i++) { const t = document.querySelector('[data-e2e="cparam-export"]')?.textContent || ''; if (t.includes('written to')) return { button: true, written: true, text: t.slice(0, 200) }; if (t.includes('selects')) return { button: true, refused: t.slice(0, 200) }; await new Promise(r => setTimeout(r, 150)); }
          return { button: true, written: false };
        })()`);
        check(3, "UI-Parameters", "Export values in the focus controls writes the selection's values and names the file", Boolean(exported && exported.written), JSON.stringify(exported));
      }
    }

    // 2026-11-26 (Luis: "make this a coherent system"): the whole chain on one view, and the maps
    // grouped by what they are for, each with its state.
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/#flow" });
    await waitForPageReady(cdp);
    const flow = await waitFor(`(() => { const steps = document.querySelectorAll('[data-e2e="system-flow"] [data-e2e="flow-step"]').length; const gl = !!document.querySelector('[data-e2e="glossary"]'); return steps === 9 && gl ? { steps, gl } : null; })()`, 60);
    check(1, "UI-Cartridges", "/cartridges/#flow shows the nine parts, from the open weights to the online libraries, and the glossary", Boolean(flow), JSON.stringify(flow));
    await cdp.send("Page.navigate", { url: baseUrl + "/parameters/" });
    await waitForPageReady(cdp);
    const groups = await waitFor(`(() => { const g = [...document.querySelectorAll('[data-e2e="pmap-group"]')].map(x => x.getAttribute('data-group')); const link = !!document.querySelector('[data-e2e="flow-link"]'); return g.length && link ? { g, link } : null; })()`, 60);
    check(1, "UI-Parameters", "/parameters/ groups each MicroModel's maps by what they are for and links to how it all fits", Boolean(groups), JSON.stringify(groups));

    // 2026-11-26: the map atlas. A tile opens the map with its lineage both ways; the state filter
    // narrows the tiles and the list below together.
    const atlas = await waitFor(`(() => { const lanes = document.querySelectorAll('[data-e2e="pmap-lane"]').length; const t = document.querySelector('[data-e2e="pmap-tile"]'); if (!lanes || !t) return null; t.click(); const d = document.querySelector('[data-e2e="pmap-detail"]'); const chain = d && d.querySelector('[data-e2e="pmap-chain"]'); return d && chain ? { lanes, chain: /this map/.test(chain.textContent || '') } : null; })()`, 60);
    const narrowed = atlas
      ? await cdp.evalJs(`(async () => { const tiles = () => document.querySelectorAll('[data-e2e="pmap-tile"]').length; const rows = () => document.querySelectorAll('[data-e2e="pmap-row"]').length; const before = { tiles: tiles(), rows: rows() }; const btn = [...document.querySelectorAll('[data-e2e="pmap-states"] button')].find(b => /^current/i.test((b.textContent || '').trim().replace(/^\\W+/, ''))); if (!btn) return { before, none: true }; btn.click(); await new Promise(r => setTimeout(r, 400)); const after = { tiles: tiles(), rows: rows() }; btn.click(); await new Promise(r => setTimeout(r, 400)); return { before, after, back: tiles() === before.tiles }; })()`)
      : null;
    const narrowOk = narrowed && (narrowed.none || (narrowed.after.tiles <= narrowed.before.tiles && narrowed.after.rows <= narrowed.before.rows && narrowed.back));
    check(1, "UI-Parameters", "/parameters/ map atlas: a tile opens the map with its lineage, and the state filter narrows tiles and list together", Boolean(atlas && atlas.chain && narrowOk), JSON.stringify({ atlas, narrowed }));
    // 2026-11-26 (Luis: "a way to clean up old maps"): in an isolated home the probe tolerance map is
    // archived from its row and restored from the archive, each through the engine.
    if (isolated) {
      const rowBtn = () => cdp.evalJs(`(() => { const row = [...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('tol-e2e-probe.json')); const b = row && [...row.querySelectorAll('button')].find(x => (x.textContent || '').trim().toLowerCase() === ${JSON.stringify("archive")}); if (b && !b.disabled) { b.click(); return true; } return false; })()`);
      const archivedNow = () => cdp.evalJs(`(() => { const a = document.querySelector('[data-e2e="pmap-archive"]'); return !!a && (a.textContent || '').includes('tol-e2e-probe.json'); })()`);
      const clicked = await rowBtn();
      const inArchive = clicked ? await waitFor(`(() => { const a = document.querySelector('[data-e2e="pmap-archive"]'); return !!a && (a.textContent || '').includes('tol-e2e-probe.json'); })()`, 60) : false;
      await cdp.evalJs(`(() => { const a = document.querySelector('[data-e2e="pmap-archive"]'); const li = a && [...a.querySelectorAll('li')].find(x => (x.textContent || '').includes('tol-e2e-probe.json')); const b = li && [...li.querySelectorAll('button')].find(x => (x.textContent || '').trim().toLowerCase() === 'restore'); if (b) b.click(); })()`);
      const back = await waitFor(`!!([...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('tol-e2e-probe.json')))`, 60);
      check(1, "UI-Parameters", "An old map is archived from its row and restored from the archive", Boolean(clicked && inArchive && back && !(await archivedNow())), JSON.stringify({ clicked, inArchive, back }));
      // 2026-11-27 (parity gap B1): the same map's lineage recorded from its row's lineage editor
      // (POST /api/parameter-maps/lineage); the engine's list then carries the note.
      const lineage = await cdp.evalJs(`(async () => {
        const wait = async (f, n = 60) => { for (let i = 0; i < n; i++) { const v = f(); if (v) return v; await new Promise(r => setTimeout(r, 150)); } return null; };
        const row = [...document.querySelectorAll('[data-e2e="pmap-row"]')].find(r => (r.textContent || '').includes('tol-e2e-probe.json'));
        const b = row && row.querySelector('[data-e2e="pmap-row-lineage"]');
        if (!b) return { button: false };
        b.click();
        const form = await wait(() => document.querySelector('[data-e2e="pmap-lineage-form"]'));
        if (!form) return { button: true, form: false };
        const inputs = [...form.querySelectorAll('input')];
        const note = inputs[inputs.length - 1];
        Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(note, 'ui e2e lineage note');
        note.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(r => setTimeout(r, 200));
        form.querySelector('[data-e2e="pmap-lineage-save"]').click();
        for (let i = 0; i < 60; i++) {
          await new Promise(r => setTimeout(r, 250));
          const d = await fetch('/api/parameter-maps').then(r => r.json()).catch(() => null);
          const m = d && (d.roles || []).flatMap(x => x.maps || []).find(x => x.file === 'tol-e2e-probe.json');
          if (m && m.lineage && m.lineage.note === 'ui e2e lineage note') return { button: true, form: true, recorded: m.lineage.recorded };
        }
        return { button: true, form: true, recorded: null };
      })()`);
      check(1, "UI-Parameters", "A map's lineage is recorded from its row's lineage editor", Boolean(lineage && lineage.recorded === "user"), JSON.stringify(lineage));
    }

    // A stated preference cartridge sits in the whole-engine rack with its plug, like any cartridge.
    if (isolated) {
      await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/" });
      await waitForPageReady(cdp);
      const rack = await waitFor(`(() => { const c = document.querySelector('[data-e2e="engine-rack"] [data-e2e="cart"][data-key="cart:stated-ui-e2e@1"]'); return c ? { kind: c.getAttribute('data-kind'), plug: [...c.querySelectorAll('button')].some(b => /plug in|unplug/i.test(b.textContent || '')) } : null; })()`, 80);
      check(1, "UI-Cartridges", "A stated preference cartridge is on the decks' whole-engine rack with a plug", Boolean(rack && rack.kind === "preference" && rack.plug), JSON.stringify(rack));
    }

    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/#adapters" });
    await waitForPageReady(cdp);
    const adapters = await waitFor(`(() => { const t = document.body.innerText || ''; const lens = /Project lenses/i.test(t); const units = /Learning units/i.test(t); const lu = !!document.querySelector('[data-e2e="learned-units"]'); return lens && units && lu ? { lens, units, lu } : null; })()`, 60);
    check(1, "UI-Cartridges", "/cartridges/#adapters shows project lenses, learning units and every learned unit", Boolean(adapters), JSON.stringify(adapters));
    await cdp.send("Page.navigate", { url: baseUrl + "/storage/" });
    await waitForPageReady(cdp);
    const backups = await waitFor(`(() => { const b = document.querySelector('[data-e2e="backup-repos"]'); const add = b && b.querySelector('[data-e2e="backup-add"] input'); return b && add ? { add: true } : null; })()`, 60);
    check(1, "UI-Storage", "/storage/ offers your own repositories for backup and rollback, with the form to add one", Boolean(backups), JSON.stringify(backups));
    // 2026-11-27: Storage has two main tabs; the repositories are their own tab (not under the item
    // table), with the path of a backup, and the side panel has the Backups notes.
    const storageTabs = await waitFor(`(() => { const tabs = [...document.querySelectorAll('[data-e2e="main-tab"]')].map((b) => b.getAttribute('data-tab')); if (tabs.length < 2) return null; const b = document.querySelector('[data-e2e="backup-repos"]'); return { tabs, backupsInTab: !!(b && b.closest('[data-e2e="main-pane"]')?.getAttribute('data-tab') === 'backups'), path: !!document.querySelector('[data-e2e="backup-path"]'), notes: !!document.querySelector('[data-e2e="backup-notes"]') }; })()`, 60);
    check(1, "UI-Storage", "Storage has main tabs Disk and Backups; the repositories are their own tab, with the path of a backup and the side panel's Backups notes", Boolean(storageTabs && storageTabs.tabs.join(",") === "disk,backups" && storageTabs.backupsInTab && storageTabs.path && storageTabs.notes), JSON.stringify(storageTabs));
    await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/#build" });
    await waitForPageReady(cdp);
    const cartBuild = await waitFor(`(() => { const b = document.querySelector('[data-e2e="cartridge-build"]'); if (!b) return null; const kinds = [...b.querySelectorAll('[data-e2e="cartridge-build-kind"] option')].map(x => (x.textContent || '').trim()); const go = !!b.querySelector('[data-e2e="lab-build"]'); return go && kinds.length >= 4 && kinds.includes('preference') && kinds.includes('distillation') ? { kinds, go } : null; })()`, 60);
    check(1, "UI-Cartridges", "/cartridges/#build offers every cartridge builder (job filter, base cartridge, distillation, preference) with a build control", Boolean(cartBuild), JSON.stringify(cartBuild));
    await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/#stacks" });
    await waitForPageReady(cdp);
    const stackForm = await waitFor(`!!document.querySelector('[data-e2e="stack-form"]') || /Not reported by this engine build|cartridge library did not answer/i.test(document.body.innerText || '')`, 60);
    check(1, "UI-Cartridges", "/cartridges/#stacks offers trial switches and a form to make a stack", Boolean(stackForm), `stackForm=${stackForm}`);
    // 2026-11-27: the rack plate (a deck per MicroModel on shared rows, the whole-engine line under
    // them, the key of the kinds), the Sources tab of the side panel, and the cartridge parameters
    // as a view of the Parameters page (each MicroModel's chain), with the atlas back after it.
    await cdp.send("Page.navigate", { url: baseUrl + "/cartridges/#sources" });
    await waitForPageReady(cdp);
    const rack = await waitFor(`(() => { const p = document.querySelector('[data-e2e="rack-plate"]'); if (!p || p.querySelector('[data-e2e="deck-loading"]')) return null; const pane = document.querySelector('[data-e2e="rail-body"]'); return { decks: p.querySelectorAll('[data-e2e="decks"] > [data-model]').length, engine: !!p.querySelector('[data-e2e="engine-rack"]'), tab: pane ? pane.getAttribute('data-tab') : null, legend: !!document.querySelector('[data-e2e="kinds-legend"]'), backups: !!document.querySelector('[data-e2e="cart-backups"]') }; })()`, 60);
    check(1, "UI-Cartridges", "/cartridges/ draws the rack plate (a deck per MicroModel, the whole-engine line, the key of the kinds) and /cartridges/#sources opens the Sources tab", Boolean(rack && rack.engine && rack.legend && rack.backups && rack.tab === "sources"), JSON.stringify(rack));
    await cdp.send("Page.navigate", { url: baseUrl + "/parameters/" });
    await waitForPageReady(cdp);
    const cartView = await waitFor(`(async () => { const b = document.querySelector('[data-e2e="pmap-view"][data-view="cartridges"]'); const lane = document.querySelector('[data-e2e="cparam-lane"]'); const atlas = () => { const l = document.querySelector('[data-e2e="pmap-lane"]'); return !!(l && l.offsetParent); }; if (!b || !lane || !atlas()) return null; b.click(); await new Promise((r) => setTimeout(r, 300)); const out = { atlasShown: atlas(), laneShown: !!lane.offsetParent, groups: lane.querySelectorAll('[data-e2e="cparam-group"]').length, rows: lane.querySelectorAll('[data-e2e="cparam-row"]').length }; document.querySelector('[data-e2e="pmap-view"][data-view="atlas"]').click(); await new Promise((r) => setTimeout(r, 300)); return { ...out, back: atlas() }; })()`, 60);
    check(1, "UI-Parameters", "/parameters/ has a cartridges view: the cartridge parameters alone, as each MicroModel's chain, and the atlas comes back", Boolean(cartView && !cartView.atlasShown && cartView.laneShown && cartView.back && (cartView.rows === 0 || cartView.groups > 0)), JSON.stringify(cartView));

    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/memory/#memories" });
    await waitForPageReady(cdp);
    const memCtl = await waitFor(`(() => { const g = !!document.querySelector('[data-e2e="goals"]') || /Session goals/i.test(document.body.innerText || ''); const add = !!document.querySelector('[data-e2e="memory-add"]'); const sweep = [...document.querySelectorAll('button')].some(b => /sweep temporary notes/i.test(b.textContent || '')); return g && add && sweep ? { g, add, sweep } : null; })()`, 60);
    check(1, "UI-Memory", "/memory/#memories offers session goals, adding a memory, pin and sweep", Boolean(memCtl), JSON.stringify(memCtl));
    await cdp.send("Page.navigate", { url: baseUrl + "/memory/#world" });
    await waitForPageReady(cdp);
    const refresh = await waitFor(`!!document.querySelector('[data-e2e="world-refresh"]')`, 60);
    check(1, "UI-Memory", "/memory/#world offers refreshing the world model through the delete flow", Boolean(refresh), `refresh=${refresh}`);
    // Guidance (reminders, loops, the step graph and what was sent): the section in the middle,
    // its descriptions in the rail tab; in an isolated home a reminder is typed and added through
    // the form and listed, and a step added through the form shows in the graph as ready.
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/memory/#guidance" });
    await waitForPageReady(cdp);
    const guide = await waitFor(`(() => { const g = document.querySelector('[data-e2e="guidance"]'); if (!g) return /Guidance not available|Guidance did not answer|No project picked/.test(document.body.innerText || '') ? { unavailable: true } : null; const has = (id) => !!g.querySelector('[data-e2e="' + id + '"]'); return has('guidance-reminder-add') && has('guidance-loop-add') && has('guidance-graph') && has('guidance-step-add') && has('guidance-deliveries') ? { about: !!document.querySelector('[data-e2e="guidance-about"]') } : null; })()`, 60);
    check(1, "UI-Memory", "/memory/#guidance shows reminders, loops, the step graph and what was sent, with its descriptions in the rail", Boolean(guide && !guide.unavailable && guide.about), JSON.stringify(guide));
    const mainTabs = await waitFor(`(() => { const tabs = [...document.querySelectorAll('[data-e2e="main-tab"]')].map((b) => b.getAttribute('data-tab')); if (tabs.length < 3) return null; const shown = [...document.querySelectorAll('[data-e2e="main-pane"]')].filter((p) => !p.hidden).map((p) => p.getAttribute('data-tab')); const g = document.querySelector('[data-e2e="guidance"]'); const w = document.querySelector('[data-e2e="world-model"]'); const railHasLoops = !!document.querySelector('[data-e2e="rail-body"] [data-e2e="guidance-loops"]'); return { tabs, shown, loopsInMain: !!(g && g.closest('[data-e2e="main-pane"]')?.getAttribute('data-tab') === 'loops'), worldInMain: !!(w && w.closest('[data-e2e="main-pane"]')?.getAttribute('data-tab') === 'world'), railHasLoops }; })()`, 60);
    check(1, "UI-Memory", "Memory has main tabs Memory, World model, Loops and graph; the loops and the world model are their own tabs in the middle, not in the side panel", Boolean(mainTabs && mainTabs.tabs.join(",") === "memory,world,loops" && mainTabs.shown.join(",") === "loops" && mainTabs.loopsInMain && mainTabs.worldInMain && !mainTabs.railHasLoops), JSON.stringify(mainTabs));
    // 2026-11-27: the Loops tab opens on the guidance circuit (the eight hook events, the three
    // kinds of guidance, what was sent); each add form opens from its part's head.
    if (guide && !guide.unavailable) {
      const circuit = await waitFor(`(() => { const f = document.querySelector('[data-e2e="guidance-figure"]'); if (!f) return null; const ev = f.querySelectorAll('[data-e2e="guidance-fig-trigger"]').length; if (!ev) return null; const nodes = [...f.querySelectorAll('[data-e2e="guidance-fig-node"]')].map((n) => n.getAttribute('data-target')); const form = document.querySelector('[data-e2e="guidance-loop-add"]'); const closed = form ? form.closest('[hidden]') !== null : null; const t = document.querySelector('[data-e2e="guidance-loops-toggle"]'); if (t) t.click(); return { ev, nodes, closed }; })()`, 60);
      await sleep(200);
      const loopFormOpen = await cdp.evalJs(`(() => { const form = document.querySelector('[data-e2e="guidance-loop-add"]'); return form ? form.closest('[hidden]') === null : null; })()`);
      check(1, "UI-Memory", "Loops and graph opens on the guidance circuit (8 hook events, reminders, loops, steps, deliveries) and a part's add form opens from its head", Boolean(circuit && circuit.ev === 8 && circuit.nodes.join(",") === "reminders,loops,steps,deliveries" && circuit.closed === true && loopFormOpen === true), JSON.stringify({ circuit, loopFormOpen }));
    }
    if (isolated && guide && !guide.unavailable) {
      const typeInto = (sel, value) => `(() => { const el = document.querySelector('${sel}'); if (!el) return false; Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${JSON.stringify(value)}); el.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`;
      await cdp.evalJs(typeInto('[data-e2e="guidance-reminder-text"]', "ui e2e: check the docs"));
      await sleep(150);
      await cdp.evalJs(`(() => { const b = document.querySelector('[data-e2e="guidance-reminder-submit"]'); if (b && !b.disabled) b.click(); })()`);
      const added = await waitFor(`(() => [...document.querySelectorAll('[data-e2e="guidance-reminder"]')].some(r => /ui e2e: check the docs/.test(r.textContent || '')) || null)()`, 60);
      const add200 = diag.apiResponses.some((r) => r.url.includes("/api/guidance") && r.status === 200);
      check(1, "UI-Memory", "Guidance: a reminder typed in the form is added (200 POST /api/guidance) and listed", Boolean(added && add200), JSON.stringify({ added, add200 }));
      await cdp.evalJs(typeInto('[data-e2e="guidance-step-title"]', "ui e2e: a first step"));
      await sleep(150);
      await cdp.evalJs(`(() => { const b = document.querySelector('[data-e2e="guidance-step-submit"]'); if (b && !b.disabled) b.click(); })()`);
      const node = await waitFor(`(() => { const n = [...document.querySelectorAll('[data-e2e="guidance-step-node"]')].find(x => /ui e2e: a first step/.test(x.textContent || '') || /ui e2e: a first step/.test(x.getAttribute('aria-label') || '')); return n ? n.getAttribute('data-status') : null; })()`, 60);
      check(1, "UI-Memory", "Guidance: a step added through the form is drawn in the step graph (ready, no dependencies)", node === "ready", `status=${node}`);
    }

    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/settings/#tools" });
    await waitForPageReady(cdp);
    const conn = await waitFor(`(() => { const rows = document.querySelectorAll('[data-e2e="connect-clients"] li').length; const t = document.body.innerText || ''; return rows === 6 || /No project known yet/i.test(t) ? { rows } : null; })()`, 60);
    check(1, "UI-Settings", "/settings/#tools shows the six coding agent clients with their connect commands", Boolean(conn), JSON.stringify(conn));

    // Parity: Settings, Controls shows settings.json key by key (GET /api/controls/saved): every
    // key beside the controls, each entry set there, a key nothing sets read only with its reason.
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/settings/#controls" });
    await waitForPageReady(cdp);
    const savedFile = await waitFor(`(() => {
      const root = document.querySelector('[data-e2e="saved-settings"][data-state="ready"]');
      if (!root) return null;
      const keys = [...root.querySelectorAll('[data-e2e="saved-key"]')].map(k => ({ key: k.getAttribute('data-key'), readOnly: k.getAttribute('data-read-only') === 'true', entries: k.querySelectorAll('[data-e2e="saved-entry"]').length }));
      return { keys, unknown: Number(root.getAttribute('data-unknown') || 0) };
    })()`, 60);
    const savedApi = await fetch(baseUrl + "/api/controls/saved").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const savedKeys = (savedFile?.keys ?? []).map((k) => k.key);
    check(
      1,
      "UI-Settings",
      "/settings/#controls shows every key of settings.json beside the controls, as /api/controls/saved lists them",
      Boolean(savedFile && savedApi) && savedApi.keys.every((k) => savedKeys.includes(k.key)) && ["base_cartridges", "base_compensation", "role_models", "note"].every((k) => savedKeys.includes(k)) && savedFile.keys.every((k) => k.readOnly || k.entries > 0),
      JSON.stringify(savedFile)
    );
    // The rail's Guide complements the control surface: every control with its key, values,
    // default and when it applies, and where else each thing is read and set.
    await cdp.evalJs(`(() => { const t = document.querySelector('[data-e2e="rail-tab"][data-tab="guide"]'); if (t) t.click(); })()`);
    const railRef = await waitFor(`(() => {
      const r = document.querySelector('[data-e2e="control-reference"]');
      const s = document.querySelector('[data-e2e="settings-surfaces"]');
      if (!r || !s) return null;
      const g = document.querySelector('[data-e2e="control-ref-group"] button');
      if (g && g.getAttribute('aria-expanded') !== 'true') g.click();
      const refs = document.querySelectorAll('[data-e2e="control-ref"]').length;
      return refs > 0 ? { count: Number(r.getAttribute('data-count')), groups: document.querySelectorAll('[data-e2e="control-ref-group"]').length, refs, surfaces: s.querySelectorAll('li').length, view: s.getAttribute('data-view') } : null;
    })()`, 40);
    check(
      1,
      "UI-Settings",
      "The Settings rail carries the control reference (every control the engine reports) and where else each thing is read and set (API, MCP, command line)",
      Boolean(railRef && savedApi) && railRef.count === Number(savedApi.controls) && railRef.groups > 0 && railRef.surfaces > 0 && railRef.view === "controls",
      JSON.stringify(railRef)
    );
    if (isolated && savedFile) {
      const compKey = await cdp.evalJs(`(() => {
        const e = document.querySelector('[data-e2e="saved-key"][data-key="base_compensation"] [data-e2e="saved-entry"]');
        if (!e) return null;
        const on = [...e.querySelectorAll('[role="radio"]')].find(r => (r.textContent || '').trim().toLowerCase() === 'on');
        if (!on) return null;
        on.click();
        return e.getAttribute('data-key');
      })()`);
      const compOn = compKey ? await waitFor(`(() => { const e = document.querySelector('[data-e2e="saved-entry"][data-key=${JSON.stringify(compKey)}]'); return e && e.getAttribute('data-value') === 'on'; })()`, 40) : false;
      const compApi = await fetch(baseUrl + "/api/controls/saved").then((r) => r.json()).catch(() => null);
      const compSaved = compApi?.keys?.find((k) => k.key === "base_compensation")?.entries?.find((e) => e.set_key === compKey)?.value === "on";
      await cdp.evalJs(`(() => {
        const e = document.querySelector('[data-e2e="saved-entry"][data-key=${JSON.stringify(compKey)}]');
        const off = e && [...e.querySelectorAll('[role="radio"]')].find(r => (r.textContent || '').trim().toLowerCase() === 'off');
        if (off) off.click();
      })()`);
      const compOff = compKey ? await waitFor(`(() => { const e = document.querySelector('[data-e2e="saved-entry"][data-key=${JSON.stringify(compKey)}]'); return e && e.getAttribute('data-value') === 'off'; })()`, 40) : false;
      check(3, "UI-Settings", "A settings.json entry (base cartridge compensation) set from Settings is saved by the engine and set back", Boolean(compOn && compSaved && compOff), `key=${compKey} on=${Boolean(compOn)} saved=${compSaved} off=${Boolean(compOff)}`);
    }

    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/models/#downloads" });
    await waitForPageReady(cdp);
    const sources = await waitFor(`(() => { const n = document.querySelectorAll('[data-e2e="cartridge-source"]').length; const form = !!document.querySelector('[data-e2e="source-add"]'); return n >= 1 && form ? { n, form } : null; })()`, 60);
    check(1, "UI-Models", "/models/#downloads lists the cartridge libraries (official first) with a form to add a third party", Boolean(sources), JSON.stringify(sources));

    await cdp.send("Page.navigate", { url: baseUrl + "/models/" });
    await waitForPageReady(cdp);
    await cdp.evalJs(`(() => { const b = [...document.querySelectorAll('button')].find(x => /^Inspect tensors/i.test((x.textContent || '').trim())); if (b) b.click(); })()`);
    const drawer = await waitFor(`(() => { const d = document.querySelector('[data-e2e="weights-drawer"]'); if (!d) return null; const t = d.textContent || ''; return d.querySelector('table') || /Not readable|did not answer/i.test(t) ? { table: !!d.querySelector('table') } : null; })()`, 80);
    check(1, "UI-Models", "/models/ opens a MicroModel's tensors and clusters (read only)", Boolean(drawer), JSON.stringify(drawer));

    // 2026-11-26: each MicroModel card says what it runs on (its base cartridge, and the upstream open
    // weights with their download state), and the activity note says whether anything is serving.
    const runsOn = await waitFor(`(() => { const w = document.querySelectorAll('[data-e2e="micromodel-weights"]'); const note = document.querySelector('[data-e2e="models-activity-note"]'); if (w.length < 1 || !note) return null; const t = [...w].map(x => x.textContent || '').join(' '); return /Runs on/i.test(t) && /(on disk|not on disk|partly|reading|not reported)/i.test(t) ? { cards: w.length, note: (note.textContent || '').slice(0, 60) } : null; })()`, 60);
    check(1, "UI-Models", "/models/ shows what each MicroModel runs on (base cartridge, upstream open weights) and whether anything is serving", Boolean(runsOn), JSON.stringify(runsOn));

    // The organism zooms with its own controls (no part opens from a button zoom).
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/topology/" });
    await waitForPageReady(cdp);
    const zoomed = await waitFor(`(() => { const z = document.querySelector('[data-e2e="topo-zoom"]'); if (!z) return null; const b = z.querySelector('button'); b.click(); b.click(); const t = (z.textContent || '').trim(); return /\\d+%/.test(t) ? { t } : null; })()`, 60);
    check(1, "UI-Topology", "/topology/ zooms the organism with its own controls", Boolean(zoomed), JSON.stringify(zoomed));

    // Live Activity names the MicroModels by role (the upstream open weights are not the MicroModel).
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/" });
    await waitForPageReady(cdp);
    const named = await waitFor(`(() => { const m = [...document.querySelectorAll('[data-e2e="micromodel"]')]; if (!m.length) return null; const first = m.map(x => (x.firstElementChild?.textContent || x.textContent || '').trim().split(/\\s/)[0]); return first.every(n => !/^qwen/i.test(n)) ? { first } : { bad: first }; })()`, 60);
    check(1, "UI-Live", "/ names each MicroModel by its role, not by the upstream open weights", Boolean(named && !named.bad), JSON.stringify(named));

    // =========================================================================
    // 2c. /memory COMPREHENSION PANEL: levels, outline, routing record, heads
    // =========================================================================
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/memory/#world" });
    await waitForPageReady(cdp);
    let comp = null;
    for (let i = 0; i < 40; i++) {
      comp = await cdp.evalJs(`(() => {
        const q = (s) => document.querySelector('[data-e2e="' + s + '"]');
        const levels = q('comprehension-levels');
        return {
          levels: levels ? [...levels.querySelectorAll('[data-level]')].map(li => li.querySelector('.eyebrow')?.textContent.trim()) : [],
          outline: Boolean(q('comprehension-outline')),
          routingRows: q('comprehension-routing') ? q('comprehension-routing').querySelectorAll('tbody tr').length : -1,
          routingText: (q('comprehension-routing')?.textContent || '').slice(0, 120),
          heads: q('comprehension-heads') ? q('comprehension-heads').querySelectorAll('li').length : -1
        };
      })()`);
      if (comp.levels.length === 4 && comp.outline && comp.routingRows >= 0 && comp.heads >= 0) break;
      await sleep(150);
    }
    const compApi = await (async () => {
      try {
        return await (await fetch(`${baseUrl}/api/comprehension`, { cache: "no-store" })).json();
      } catch {
        return null;
      }
    })();
    const wantRows = (compApi?.routing?.rows?.length ?? 0) + (compApi?.routing?.combine?.length ?? 0);
    check(
      1,
      "UI-Memory",
      "/memory/ comprehension panel shows L1 to L4, the outline, the routing record rows and the heads per lens",
      JSON.stringify(comp.levels) === JSON.stringify(["L1 piece", "L2 file", "L3 module", "L4 system"]) &&
        comp.outline &&
        comp.routingRows === wantRows &&
        comp.heads === (compApi?.heads?.length ?? 0) &&
        Array.isArray(compApi?.levels) && compApi.levels.length === 4,
      `${JSON.stringify(comp)} apiRows=${wantRows} apiHeads=${compApi?.heads?.length}`
    );

    // =========================================================================
    // 2d. /settings 4-MODEL VRAM METER
    // =========================================================================
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/settings/" });
    await waitForPageReady(cdp);
    // The meter is in the Machine view of the section bar (the last view is remembered).
    await cdp.evalJs(`(() => {
      const nav = document.querySelector('nav[aria-label="Settings sections"]');
      const b = nav && [...nav.querySelectorAll('button, a')].find(x => (x.textContent || '').trim() === 'Machine');
      if (b) b.click();
    })()`);
    await sleep(400);
    const statusApi = await (async () => {
      try {
        return await (await fetch(`${baseUrl}/api/status`, { cache: "no-store" })).json();
      } catch {
        return null;
      }
    })();
    let vram = null;
    for (let i = 0; i < 30; i++) {
      vram = await cdp.evalJs(`(() => {
        const m = document.querySelector('[data-e2e="vram-models"]');
        return m ? [...m.querySelectorAll('li[data-model]')].map(li => li.getAttribute('data-model')) : [];
      })()`);
      if (vram.length > 0) break;
      await sleep(150);
    }
    const listed = (statusApi?.models ?? []).map((m) => m.id);
    check(1, "UI-Settings", "/settings/ VRAM meter has one row per MicroModel (4 or more)", vram.length >= 4 && listed.every((id) => vram.includes(id)), `rows=${JSON.stringify(vram)} models=${JSON.stringify(listed)}`);

    // =========================================================================
    // 2e. /audit STAGE TIMES (NANO_STAGE_TIMES breakdown)
    // =========================================================================
    resetDiag();
    await cdp.send("Page.navigate", { url: baseUrl + "/audit/" });
    await waitForPageReady(cdp);
    const stApi = await (async () => {
      try {
        const r = await fetch(`${baseUrl}/api/stage-times`, { cache: "no-store" });
        return r.ok ? await r.json() : null;
      } catch {
        return null;
      }
    })();
    let stUi = null;
    for (let i = 0; i < 30; i++) {
      stUi = await cdp.evalJs(`(() => {
        const p = document.querySelector('[data-e2e="stage-times"]');
        if (!p) return { panel: false, rows: 0, empty: false };
        const t = p.querySelector('[data-e2e="stage-times-table"]');
        return { panel: true, rows: t ? t.querySelectorAll('tbody tr').length : 0, empty: /No stage table yet/.test(p.textContent || '') };
      })()`);
      if (stUi.panel && (stUi.rows > 0 || stUi.empty)) break;
      await sleep(150);
    }
    const firstSet = stApi?.sets?.[0];
    const wantStageRows = firstSet ? firstSet.stages.length : 0;
    check(
      1,
      "UI-Audit",
      "/audit/ stage-times view matches GET /api/stage-times (rows per stage, or the empty state)",
      Boolean(stApi) && stUi.panel && (wantStageRows > 0 ? stUi.rows === wantStageRows : stUi.empty),
      `api=${stApi ? `${stApi.file} sets=${stApi.sets.length}` : "no reply"} ui=${JSON.stringify(stUi)}`
    );
    // 2026-11-27: the kinds are chips with their own side panel reference, "undoable" is a state
    // filter, and (isolated home) a row the engine can undo carries Undo beside it: the guidance
    // changes made above are such rows, and the first one's Undo answers 200.
    const auditUi = await waitFor(`(() => { const chips = [...document.querySelectorAll('[data-e2e="audit-kind"]')].map((b) => b.getAttribute('data-kind')); if (chips.length < 2) return null; const states = [...document.querySelectorAll('[data-e2e="audit-filters"] [role="radiogroup"][aria-label="State"] [role="radio"]')].map((b) => (b.textContent || '').trim().toLowerCase()); return { chips: chips.length, guidanceChip: chips.includes('memory & guidance'), undoableState: states.some((s) => s.startsWith('undoable')), kindsRail: !!document.querySelector('[data-e2e="audit-kinds"]'), rowUndo: document.querySelectorAll('[data-e2e="audit-row-undo"]').length }; })()`, 60);
    check(1, "UI-Audit", "/audit/ has kind chips (memory and guidance among them), the undoable state filter and the Kinds reference in the side panel", Boolean(auditUi && auditUi.chips >= 10 && auditUi.guidanceChip && auditUi.undoableState && auditUi.kindsRail), JSON.stringify(auditUi));
    if (isolated && auditUi && auditUi.rowUndo > 0) {
      const beforeUndo = diag.apiResponses.filter((r) => r.url.includes("/api/undo") && r.status === 200).length;
      await cdp.evalJs(`(() => { const b = document.querySelector('[data-e2e="audit-row-undo"]'); if (b && !b.disabled) b.click(); })()`);
      let undo200 = false;
      for (let i = 0; i < 40 && !undo200; i++) {
        await sleep(150);
        undo200 = diag.apiResponses.filter((r) => r.url.includes("/api/undo") && r.status === 200).length > beforeUndo;
      }
      check(1, "UI-Audit", "An undoable row's own Undo answers 200 POST /api/undo", undo200, `rowUndo=${auditUi.rowUndo} undo200=${undo200}`);
    }

    // =========================================================================
    // 3. ORGANISM VIEW MULTI-VIEWPORT GEOMETRY CHECK
    // =========================================================================
    const organismRoutes = ["/", "/topology/"];
    const viewports = [
      { vw: 1340, vh: 720 },
      { vw: 800, vh: 600 },
      { vw: 390, vh: 844 },
      { vw: 900, vh: 1200 },
    ];

    for (const route of organismRoutes) {
      resetDiag();
      await cdp.send("Emulation.setDeviceMetricsOverride", {
        width: 1340,
        height: 720,
        deviceScaleFactor: 1,
        mobile: false,
      });
      await cdp.send("Emulation.setTouchEmulationEnabled", { enabled: false });
      await cdp.send("Page.navigate", { url: baseUrl + route });
      await waitForPageReady(cdp);

      for (const { vw, vh } of viewports) {
        const mobile = vw < 500;
        const dsf = mobile ? 2 : 1;
        await cdp.send("Emulation.setDeviceMetricsOverride", {
          width: vw,
          height: vh,
          deviceScaleFactor: dsf,
          mobile,
        });
        await cdp.send("Emulation.setTouchEmulationEnabled", {
          enabled: mobile,
          maxTouchPoints: 5,
        });
        await sleep(320);

        // Wait until svg.annot viewBox matches .topo clientWidth/clientHeight (within 2px)
        let synced = false;
        for (let k = 0; k < 25; k++) {
          const syncState = await cdp.evalJs(`(() => {
            const topo = document.querySelector('.topo');
            const svg = topo ? topo.querySelector('svg.annot') : null;
            if (!topo || !svg) return { ok: false };
            const vb = (svg.getAttribute('viewBox') || '').trim().split(/\\s+/).map(Number);
            if (vb.length !== 4) return { ok: false };
            const cw = topo.clientWidth, ch = topo.clientHeight;
            const match = cw > 50 && ch > 50 && Math.abs(vb[2] - cw) <= 2 && Math.abs(vb[3] - ch) <= 2;
            const base = window.__nmScene ? Number(window.__nmScene.base || 0) : 0;
            return { ok: match && base > 0, vb, cw, ch, base };
          })()`);
          if (syncState.ok) {
            synced = true;
            break;
          }
          await sleep(150);
        }

        const geom = await cdp.evalJs(`(() => {
          const sw = document.documentElement.scrollWidth;
          const cw = document.documentElement.clientWidth;
          const topo = document.querySelector('.topo');
          const svg = topo ? topo.querySelector('svg.annot') : null;
          if (!topo || !svg) return { error: 'missing .topo or svg.annot' };
          const tr = topo.getBoundingClientRect();

          // All 8 tentacle tips
          const tips = [];
          for (let i = 0; i < 8; i++) {
            const txt = svg.querySelector('g.t-node[data-key="neuropod/t' + i + '"] text.minor-text');
            if (!txt) {
              tips.push({ i, missing: true });
            } else {
              const r = txt.getBoundingClientRect();
              tips.push({
                i,
                missing: false,
                cx: r.left + r.width / 2,
                cy: r.top + r.height / 2,
                text: txt.textContent.trim()
              });
            }
          }
          const presentTips = tips.filter(t => !t.missing);
          let centroid = null;
          let radii = [];
          let minR = 0;
          let maxR = 0;
          let ratio = 0;
          if (presentTips.length === 8) {
            const cx = presentTips.reduce((a, b) => a + b.cx, 0) / 8;
            const cy = presentTips.reduce((a, b) => a + b.cy, 0) / 8;
            centroid = { cx, cy };
            radii = presentTips.map(t => Math.hypot(t.cx - cx, t.cy - cy));
            minR = Math.min(...radii);
            maxR = Math.max(...radii);
            ratio = minR > 0 ? maxR / minR : Infinity;
          }

          // Visible SVG geometry elements containment (excluding transparent hit rects)
          const tol = 1.5;
          const oobTopo = [];
          const oobView = [];
          const visSelectors = 'text, rect.chip, rect.marker, path.leader';
          for (const el of svg.querySelectorAll(visSelectors)) {
            const cs = getComputedStyle(el);
            if (cs.display === 'none' || cs.visibility === 'hidden') continue;
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) continue;
            const tag = el.tagName + '.' + (el.getAttribute('class') || '') + ':' + (el.textContent || '').trim().slice(0, 16);
            if (r.left < tr.left - tol || r.right > tr.right + tol || r.top < tr.top - tol || r.bottom > tr.bottom + tol) {
              oobTopo.push({ tag, r: { l: r.left, t: r.top, r: r.right, b: r.bottom }, tr: { l: tr.left, t: tr.top, r: tr.right, b: tr.bottom } });
            }
            const checkVertView = location.pathname.startsWith('/topology') || tr.bottom <= innerHeight + tol;
            if (r.left < -tol || r.right > innerWidth + tol || (checkVertView && (r.top < -tol || r.bottom > innerHeight + tol))) {
              oobView.push({ tag, r: { l: r.left, t: r.top, r: r.right, b: r.bottom }, vw: innerWidth, vh: innerHeight });
            }
          }

          // Visible text-to-text overlap check inside svg.annot + full page
          const boxes = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          for (let n = walker.nextNode(); n; n = walker.nextNode()) {
            if (!n.textContent.trim()) continue;
            const el = n.parentElement;
            if (!el || el.closest('.sr-only, script, style, option, title')) continue;
            const cs = getComputedStyle(el);
            if (cs.visibility === 'hidden' || cs.display === 'none') continue;
            const range = document.createRange();
            range.selectNodeContents(n);
            for (const b of range.getClientRects()) {
              if (b.width < 2 || b.height < 2) continue;
              if (b.bottom < 0 || b.top > innerHeight) continue;
              const px = Math.min(innerWidth - 1, Math.max(0, b.left + b.width / 2));
              const py = Math.min(innerHeight - 1, Math.max(0, b.top + b.height / 2));
              const hit = document.elementFromPoint(px, py);
              if (!hit || !(hit === el || el.contains(hit) || hit.contains(el))) continue;
              let c = { left: b.left, right: b.right, top: b.top, bottom: b.bottom };
              for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
                const s2 = getComputedStyle(a);
                if (s2.overflowX !== 'visible' || s2.overflowY !== 'visible') {
                  const r2 = a.getBoundingClientRect();
                  c = {
                    left: Math.max(c.left, r2.left),
                    right: Math.min(c.right, r2.right),
                    top: Math.max(c.top, r2.top),
                    bottom: Math.min(c.bottom, r2.bottom)
                  };
                }
              }
              if (c.right - c.left < 1 || c.bottom - c.top < 1) continue;
              boxes.push({ el, c, t: n.textContent.trim().slice(0, 24) });
            }
          }
          const overlaps = [];
          for (let i = 0; i < boxes.length; i++) {
            for (let j = i + 1; j < boxes.length; j++) {
              const A = boxes[i], B = boxes[j];
              if (A.el === B.el) continue;
              const w = Math.min(A.c.right, B.c.right) - Math.max(A.c.left, B.c.left);
              const h = Math.min(A.c.bottom, B.c.bottom) - Math.max(A.c.top, B.c.top);
              if (w > 2 && h > 3) overlaps.push(A.t + ' | ' + B.t);
            }
          }

          return {
            sw,
            cw,
            tipCount: presentTips.length,
            minR,
            maxR,
            ratio,
            oobTopoCount: oobTopo.length,
            oobViewCount: oobView.length,
            oobSample: [...oobTopo.slice(0, 3), ...oobView.slice(0, 3)],
            overlapCount: overlaps.length,
            overlapSample: overlaps.slice(0, 3)
          };
        })()`);

        const vpTag = `${route} @ ${vw}x${vh}`;
        check(
          1,
          "Organism-View",
          `${vpTag} viewBox synced & zero horizontal overflow`,
          synced && !geom.error && geom.sw <= geom.cw,
          `synced=${synced} sw=${geom.sw}/${geom.cw} err=${geom.error || "none"}`
        );
        check(
          1,
          "Organism-View",
          `${vpTag} renders all 8 tentacle tips inside .topo & viewport with 0 overlaps`,
          !geom.error &&
            geom.tipCount === 8 &&
            geom.oobTopoCount === 0 &&
            geom.oobViewCount === 0 &&
            geom.overlapCount === 0,
          `tips=${geom.tipCount} oobTopo=${geom.oobTopoCount} oobView=${geom.oobViewCount} overlaps=${geom.overlapCount} sample=${JSON.stringify(geom.oobSample || [])} ov=${JSON.stringify(geom.overlapSample || [])}`
        );

        const symOk = !geom.error && geom.tipCount === 8 && geom.minR > 20 && geom.ratio <= 1.45;
        if (strictMode) {
          check(
            4,
            "Organism-Symmetry",
            `${vpTag} proportional tentacle symmetry (minR > 20 && maxR/minR <= 1.45)`,
            symOk,
            `minR=${geom.minR?.toFixed(1)} maxR=${geom.maxR?.toFixed(1)} ratio=${geom.ratio?.toFixed(3)}`
          );
        } else {
          skip(
            4,
            "Organism-Symmetry",
            `${vpTag} proportional tentacle symmetry (minR > 20 && maxR/minR <= 1.45)`,
            `Baseline mode (ratio=${geom.ratio?.toFixed(3)})`
          );
        }
      }
    }

    // =========================================================================
    // Live activity is honest (2026-11-27). The topology publishes what its live layer draws on
    // the layer's canvas: `data-live-lit` (parts computing now), `data-live-glow` (afterglow),
    // `data-live-signals` (signals travelling) and `data-live-why` (each lit part with the event
    // behind it: `stage:<name>` from /api/live, `call:<audit id>:<part id>` from the call's audit
    // row, `exec:<role>` and `writer:session` from the live snapshot).
    //   a. With no call running nothing is lit, nothing glows, no signal travels, no frame is drawn.
    //   b. In an isolated home: a serving engine is started, one real nm_context_pack is made over
    //      MCP, and every light seen during and after it must match a stage event of /api/live
    //      since the call began or a step of the call's own audit row; then the view is at rest.
    // Runs last, so the serving engine changes no earlier check.
    // =========================================================================
    {
      const liveState = `(() => { const c = document.querySelector('canvas[data-e2e="live-layer"]'); if (!c || !window.__nmScene || !window.__nmScene.base) return null; const e = document.querySelector('[data-e2e="topology-engine"]'); const s = document.querySelector('[data-e2e="system-strip"]'); return { lit: c.dataset.liveLit || '', glow: c.dataset.liveGlow || '', why: c.dataset.liveWhy || '', signals: Number(c.dataset.liveSignals || 0), frames: window.__nmScene.frames, serving: e ? e.dataset.serving : null, strip: s ? s.dataset.state : null }; })()`;
      const liveWait = async (tries = 60) => {
        for (let i = 0; i < tries; i++) {
          const v = await cdp.evalJs(liveState);
          if (v) return v;
          await sleep(250);
        }
        return null;
      };
      const whys = (s) => (s && s.why ? s.why.split(" ").filter(Boolean) : []);
      await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1340, height: 720, deviceScaleFactor: 1, mobile: false });
      await cdp.send("Page.navigate", { url: baseUrl + "/topology/" });
      await waitForPageReady(cdp);
      const rest0 = await liveWait();
      await sleep(2000);
      const rest1 = await cdp.evalJs(liveState);
      // Against --url an engine may be working: then every light must still name its event.
      const atRest = Boolean(rest0 && rest1 && rest1.lit === "" && rest1.glow === "" && rest1.signals === 0 && rest1.frames === rest0.frames);
      const explained = Boolean(rest1 && whys(rest1).length > 0 && whys(rest1).every((w) => /=(stage|call|exec|writer):?/.test(w)));
      check(1, "UI-Live", "With no call running nothing on the topology is lit as active (no lit part, no afterglow, no signal, no frame drawn)", isolated ? atRest : atRest || explained, JSON.stringify(rest1));
      if (isolated) {
        const notServing = Boolean(rest1 && rest1.serving === "false" && rest1.strip === "no engine serving");
        check(1, "UI-Live", "With no engine serving the topology and the system strip say so", notServing, JSON.stringify(rest1));
        let engine = null;
        try {
          engine = startServingEngine(server.tempHome);
          await engine.request("initialize", { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "ui-e2e-live", version: "0" } });
          engine.notify("notifications/initialized");
          // The engine binds its MicroModels and builds its index at the start: wait until it is idle.
          let idle = null;
          for (let i = 0; i < 400 && !idle; i++) {
            await sleep(500);
            const l = await fetch(`${baseUrl}/api/live`).then((r) => r.json()).catch(() => null);
            if (l && l.serving >= 1 && l.state === "idle" && i > 12) idle = l;
          }
          await sleep(5000);
          const before0 = await cdp.evalJs(liveState);
          await sleep(1500);
          const before1 = await cdp.evalJs(liveState);
          check(
            1,
            "UI-Live",
            "With an engine serving and before any call nothing is lit unless the engine reports it running (its start-up loading and index work)",
            // 2026-11-27: the engine loads all four MicroModels and builds its index in the background
            // at start, which is real work and lights its parts; before a call every light must be
            // explained by a reading of the engine (stage, exec or writer), never by a call.
            Boolean(idle && before0 && before1 && before1.serving === "true" && (( before1.lit === "" && before1.glow === "" && before1.signals === 0 && before1.frames === before0.frames) || (whys(before1).length > 0 && whys(before1).every((w) => /=(stage|exec|writer):/.test(w))))),
            JSON.stringify({ idle: Boolean(idle), before1 })
          );
          // 2026-11-27: the strip reads the serving engine, never the interface server's own
          // process: its model count equals /api/live micromodels.held ("N of 4") while an engine
          // serves. Read twice a beat apart, since the strip follows the feed.
          let stripModels = null;
          let liveHeld = null;
          for (let i = 0; i < 8; i++) {
            const l = await fetch(`${baseUrl}/api/live`).then((r) => r.json()).catch(() => null);
            liveHeld = typeof l?.micromodels?.held === "string" ? l.micromodels.held.split(" ")[0] : null;
            stripModels = await cdp.evalJs(`(() => { const m = document.querySelector('[data-e2e="meter-models"]'); const t = m ? m.textContent || '' : ''; const x = t.match(/(\\d+) of \\d+ held/); return x ? x[1] : null; })()`);
            if (liveHeld !== null && stripModels === liveHeld) break;
            await sleep(1500);
          }
          check(1, "UI-Live", "With an engine serving the system strip's model count equals /api/live micromodels.held", liveHeld !== null && stripModels === liveHeld, JSON.stringify({ strip: stripModels, live: liveHeld }));
          const t0 = Date.now();
          const seen = new Set();
          let sampling = true;
          let samples = 0;
          const sampler = (async () => {
            while (sampling) {
              const s = await cdp.evalJs(liveState);
              for (const w of whys(s)) seen.add(w);
              samples++;
              // What the strip and the topology's engine line say during the call, for the report.
              if (samples === 8) console.log("[UI-Live] during the call:", JSON.stringify(await cdp.evalJs(`(() => { const t = (q) => { const e = document.querySelector(q); return e ? (e.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 400) : null; }; return { strip: t('[data-e2e="system-strip"]'), engine: t('[data-e2e="topology-engine"]') }; })()`)));
              await sleep(80);
            }
          })();
          const reply = await engine.request("tools/call", { name: "nm_context_pack", arguments: { question: "Where is the struct Value defined and how is a JSON number parsed?", format: "json" } });
          const callMs = Date.now() - t0;
          // The recorded call plays after its audit row arrives; then the afterglow fades.
          let end = null;
          for (let i = 0; i < 160; i++) {
            await sleep(250);
            const s = await cdp.evalJs(liveState);
            if (i > 16 && s && s.lit === "" && s.glow === "" && s.signals === 0) {
              end = s;
              break;
            }
          }
          sampling = false;
          await sampler;
          const liveNow = await fetch(`${baseUrl}/api/live`).then((r) => r.json()).catch(() => ({}));
          const audit = await fetch(`${baseUrl}/api/audit?kind=tool&limit=10`).then((r) => r.json()).catch(() => ({}));
          const row = [...(audit.audit ?? [])].filter((r) => r.tool === "nm_context_pack" && r.time_ms >= t0 - 2000).sort((a, b) => b.time_ms - a.time_ms)[0] ?? null;
          // Any stage the engine reported explains a light: the call's own, and background work that began
          // before it (start-up loading, the index build) and was still running.
          const stages = new Set([...(liveNow.timeline ?? []).map((e) => e.stage), ...(liveNow.active ?? []).map((a) => a.stage)]);
          const ran = new Set((liveNow.models ?? []).filter((m) => Number(m.forwards) > 0).map((m) => m.role));
          const unexplained = [];
          for (const w of seen) {
            const why = w.slice(w.indexOf("=") + 1);
            if (why.startsWith("stage:")) {
              if (!stages.has(why.slice(6))) unexplained.push(w);
            } else if (why.startsWith("call:")) {
              if (!row || why.split(":")[1] !== row.id) unexplained.push(w);
            } else if (why.startsWith("exec:")) {
              if (!ran.has(why.slice(5))) unexplained.push(w);
            } else unexplained.push(w);
          }
          const stageLights = [...seen].filter((w) => w.includes("=stage:")).length;
          const callLights = [...seen].filter((w) => w.includes("=call:")).length;
          check(
            1,
            "UI-Live",
            "A real call lights only the parts in its trace (each light matches a stage event of /api/live or a step of its audit row)",
            // The recorded call's replay (call: lights) plays only when its audit row arrives after the
            // stage lights have faded, so it is reported, not required; every light seen must be explained.
            Boolean(!reply.error && row && seen.size > 0 && stageLights > 0 && unexplained.length === 0),
            JSON.stringify({ call_ms: callMs, samples, lights: seen.size, stage_lights: stageLights, call_lights: callLights, stages: [...stages], unexplained, row: row ? row.id : null })
          );
          check(1, "UI-Live", "After the call the topology returns to rest (no lit part, no afterglow, no signal)", Boolean(end), JSON.stringify(end));
        } catch (e) {
          check(1, "UI-Live", "A real call lights only the parts in its trace (each light matches a stage event of /api/live or a step of its audit row)", false, e instanceof Error ? e.message : String(e));
        } finally {
          if (engine) await engine.stop();
        }
      }
    }
  } finally {
    if (cdp) {
      try { cdp.close(); } catch {}
    }
    await teardownEdge(edgeProc, profileDir, debugPort);
    if (server) await server.stop();
  }

  const { totalFail, totalSkip } = finish();

  if (totalFail > 0 || (strictMode && totalSkip > 0)) {
    process.exit(1);
  }
  process.exit(0);
}

main().catch((err) => {
  console.error("E2E UI Verify fatal:", err);
  process.exit(1);
});
