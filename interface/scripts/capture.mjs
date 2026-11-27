// Captures pages of the interface served by the engine (127.0.0.1:4870) with headless Edge over
// the DevTools protocol.
//
// Process hygiene: one browser at a time. Each capture gets its own profile folder inside the
// Nano folder (scratch/capture-profiles), never under the system TEMP folder. After each capture
// the whole process tree of that browser is ended (only processes whose command line names that
// profile folder, so no other Edge window is touched), the profile folder is removed, and both
// are confirmed before the next capture starts.
//
// Usage:
//   node scripts/capture.mjs <out-dir> <name>=<path> ... [--size=WxH] [--dpr=N] [--phone]
//        [--verify-signal] [--verify-dock] [--idle] [--base=http://127.0.0.1:<port>]
//        [--hold-api=<ms>]   every /api/ reply is held back <ms> and the shot is taken 1.2 s
//                            after navigation: the page's loading states, not its data
//   node scripts/capture.mjs --count      prints how many capture browsers are running
// Mock replies are used only by the --verify-* modes, and those captures are named verify-*. The
// verify-signal mock carries a `job` on three tentacle steps to show the live job label.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync, appendFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const HERE = dirname(fileURLToPath(import.meta.url));
const PROFILES = resolve(HERE, "../../scratch/capture-profiles");
const args = process.argv.slice(2);
// The server captured: --base=http://127.0.0.1:<port> (default the engine's own port, 4870).
const BASE = (args.find((a) => a.startsWith("--base=")) ?? "--base=http://127.0.0.1:4870").slice(7).replace(/\/$/, "");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Runs in the page: every visible run of text as boxes; counts pairs of text boxes from different
// elements that overlap, text cut off by the window's edge, and text wider than a clipping
// container that does not shorten it with an ellipsis.
const COLLIDE = `(() => {
  const boxes = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.textContent.trim()) continue;
    const el = n.parentElement;
    if (!el || el.closest('.sr-only, script, style, option, title')) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    const r = document.createRange();
    r.selectNodeContents(n);
    for (const b of r.getClientRects()) {
      if (b.width < 2 || b.height < 2) continue;
      const cx = Math.min(innerWidth - 1, Math.max(0, b.left + b.width / 2));
      const cy = Math.min(innerHeight - 1, Math.max(0, b.top + b.height / 2));
      if (b.bottom < 0 || b.top > innerHeight) continue;
      const hit = document.elementFromPoint(cx, cy);
      if (!hit || !(hit === el || el.contains(hit) || hit.contains(el))) continue;
      // What shows of the box: clipped by every ancestor that hides overflow.
      let c = { left: b.left, right: b.right, top: b.top, bottom: b.bottom };
      for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
        const s2 = getComputedStyle(a);
        if (s2.overflowX !== 'visible' || s2.overflowY !== 'visible') {
          const r2 = a.getBoundingClientRect();
          c = { left: Math.max(c.left, r2.left), right: Math.min(c.right, r2.right), top: Math.max(c.top, r2.top), bottom: Math.min(c.bottom, r2.bottom) };
        }
      }
      if (c.right - c.left < 1 || c.bottom - c.top < 1) continue;
      boxes.push({ el, b, c, t: n.textContent.trim().slice(0, 28) });
    }
  }
  const over = [];
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) {
      const A = boxes[i], B = boxes[j];
      if (A.el === B.el) continue;
      const w = Math.min(A.c.right, B.c.right) - Math.max(A.c.left, B.c.left);
      const h = Math.min(A.c.bottom, B.c.bottom) - Math.max(A.c.top, B.c.top);
      if (w > 2 && h > 3) over.push(A.t + ' | ' + B.t);
    }
  const cut = [];
  const shortened = (el) => { for (let a = el, k = 0; a && k < 3; a = a.parentElement, k++) if (getComputedStyle(a).textOverflow === 'ellipsis') return true; return false; };
  for (const { el, b, t } of boxes) {
    if (shortened(el)) continue;
    if (b.left < -1 || b.right > innerWidth + 1) { cut.push('edge: ' + t); continue; }
    for (let a = el; a && a !== document.body; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX !== 'visible' || cs.overflow === 'hidden') {
        const r = a.getBoundingClientRect();
        if ((b.right > r.right + 1 || b.left < r.left - 1) && getComputedStyle(el).textOverflow !== 'ellipsis' && cs.textOverflow !== 'ellipsis') cut.push('clip: ' + t);
        break;
      }
    }
  }
  // Leader lines of the scene's labels that run across another label's text.
  const crossed = [];
  for (const path of document.querySelectorAll('svg .leader')) {
    const m = (path.getAttribute('d') || '').match(/M\s*([-\d.]+)\s+([-\d.]+)\s*L\s*([-\d.]+)\s+([-\d.]+)/);
    const ctm = path.getScreenCTM();
    if (!m || !ctm) continue;
    const P = (x, y) => ({ x: ctm.a * x + ctm.c * y + ctm.e, y: ctm.b * x + ctm.d * y + ctm.f });
    const a = P(+m[1], +m[2]), b = P(+m[3], +m[4]);
    const own = path.closest('g');
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    for (const { el, c, t } of boxes) {
      if (own && own.contains(el)) continue;
      let hit = false;
      for (let d = 0; d <= len && !hit; d += 2) {
        const x = a.x + ((b.x - a.x) * d) / (len || 1), y = a.y + ((b.y - a.y) * d) / (len || 1);
        if (x > c.left + 1 && x < c.right - 1 && y > c.top + 1 && y < c.bottom - 1) hit = true;
      }
      if (hit) crossed.push('leader over: ' + t);
    }
  }
  return JSON.stringify({ texts: boxes.length, overlaps: over.length + crossed.length, cut: cut.length, examples: [...over.slice(0, 4), ...crossed.slice(0, 4), ...cut.slice(0, 4)] });
})()`;

/** Edge processes started by this script and still running: their command line names the
 *  profiles folder. A process that has ended but whose handle is not yet released is not counted. */
function ours() {
  const r = spawnSync("powershell", ["-NoProfile", "-Command", `@(Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | Where-Object { $_.CommandLine -like '*capture-profiles*' } | Where-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; $p -and -not $p.HasExited }) | ForEach-Object { $_.ProcessId }`], { encoding: "utf8" });
  return r.stdout.split(/\s+/).filter(Boolean).map(Number);
}

if (args[0] === "--count") {
  console.log(`capture browsers running: ${ours().length}`);
  process.exit(0);
}

const out = args[0];
const jobs = args.slice(1).filter((a) => a.includes("=") && !a.startsWith("--"));
const phone = args.includes("--phone");
const verifySignal = args.includes("--verify-signal");
const verifyDock = args.includes("--verify-dock");
const idle = args.includes("--idle");
const holdApi = Number(args.find((a) => a.startsWith("--hold-api="))?.slice(11) ?? 0);
const sizeArg = args.find((a) => a.startsWith("--size="));
const [vw, vh] = phone ? [390, 844] : sizeArg ? sizeArg.slice(7).split("x").map(Number) : [1920, 1080];
const dsf = Number(args.find((a) => a.startsWith("--dpr="))?.slice(6) ?? (phone ? 3 : 1.5));
mkdirSync(out, { recursive: true });
mkdirSync(PROFILES, { recursive: true });
const log = (s) => {
  console.log(s);
  appendFileSync(join(out, "log.txt"), s + "\n");
};

async function session(port) {
  for (let i = 0; i < 150; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (r.ok) break;
    } catch {
      /* not up yet */
    }
    await sleep(200);
  }
  const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const wait = new Map();
  const handlers = [];
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && wait.has(m.id)) {
      wait.get(m.id)(m);
      wait.delete(m.id);
    } else if (m.method) for (const h of handlers) h(m);
  });
  const send = (method, params = {}) =>
    new Promise((r) => {
      const n = ++id;
      wait.set(n, r);
      ws.send(JSON.stringify({ id: n, method, params }));
    });
  return { send, on: (h) => handlers.push(h), close: () => ws.close() };
}

/** Ends every process of one capture browser and removes its profile; confirms both. */
async function teardown(dir, port) {
  try {
    const v = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    const ws = new WebSocket(v.webSocketDebuggerUrl);
    await new Promise((r) => ws.addEventListener("open", r, { once: true }));
    ws.send(JSON.stringify({ id: 1, method: "Browser.close" }));
    await sleep(500);
  } catch {
    /* already closed */
  }
  const leaf = dir.split(/[\\/]/).pop();
  for (let k = 0; k < 20; k++) {
    const r = spawnSync("powershell", ["-NoProfile", "-Command", `@(Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | Where-Object { $_.CommandLine -like '*${leaf}*' }) | ForEach-Object { $_.ProcessId }`], { encoding: "utf8" });
    const pids = r.stdout.split(/\s+/).filter(Boolean);
    if (!pids.length) break;
    for (const pid of pids) spawnSync("taskkill", ["/PID", pid, "/T", "/F"], { stdio: "ignore" });
    await sleep(300);
  }
  for (let k = 0; k < 20 && existsSync(dir); k++) {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      await sleep(300);
    }
  }
  // A process that has ended can stay listed for a few seconds while its handles close.
  let left = ours().length;
  for (let k = 0; k < 30 && (left || existsSync(dir)); k++) {
    await sleep(500);
    if (existsSync(dir)) {
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch {
        /* still locked */
      }
    }
    left = ours().length;
  }
  if (left || existsSync(dir)) throw new Error(`teardown not confirmed: ${left} capture browsers running, profile ${existsSync(dir) ? "still present" : "removed"}`);
}

async function capture(name, path) {
  const dir = join(PROFILES, `shot-${Date.now()}-${Math.floor(Math.random() * 1e6)}`);
  mkdirSync(dir);
  const port = 9300 + Math.floor(Math.random() * 500);
  spawn(EDGE, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--hide-scrollbars", `--window-size=${vw},${vh}`, "about:blank"], { stdio: "ignore" });
  try {
    const s = await session(port);
    await s.send("Page.enable");
    await s.send("Runtime.enable");
    await s.send("Emulation.setDeviceMetricsOverride", { width: vw, height: vh, deviceScaleFactor: dsf, mobile: phone });
    if (phone) await s.send("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
    const mock = { audit: null, carts: null };
    if (verifySignal || verifyDock) {
      await s.send("Fetch.enable", { patterns: [{ urlPattern: "*/api/audit*" }, { urlPattern: "*/api/cartridges*" }] });
      s.on(async (m) => {
        if (m.method !== "Fetch.requestPaused") return;
        const url = m.params.request.url;
        const body = url.includes("/api/audit") ? mock.audit : mock.carts;
        if (body === null) return s.send("Fetch.continueRequest", { requestId: m.params.requestId });
        await s.send("Fetch.fulfillRequest", { requestId: m.params.requestId, responseCode: 200, responseHeaders: [{ name: "Content-Type", value: "application/json" }], body: Buffer.from(JSON.stringify(body)).toString("base64") });
      });
      const t0 = Date.now() - 600000;
      mock.audit = { audit: [{ id: "a-verify-0", kind: "tool", tool: "nm_status", label: "Status", ms: 0, ok: true, time_ms: t0, parts: ["nmapi.tools"] }], episodes: [] };
      mock.carts = [];
    }
    if (holdApi > 0) {
      await s.send("Fetch.enable", { patterns: [{ urlPattern: "*/api/*" }] });
      s.on((m) => {
        if (m.method === "Fetch.requestPaused") setTimeout(() => s.send("Fetch.continueRequest", { requestId: m.params.requestId }), holdApi);
      });
    }
    // Capture mode: everything maskable stays masked, whatever was revealed.
    await s.send("Page.addScriptToEvaluateOnNewDocument", { source: "window.__nmMaskAll = true;" });
    const nav = await s.send("Page.navigate", { url: BASE + path });
    if (nav.error || nav.result?.errorText) {
      log(`navigate failed ${name}: ${JSON.stringify(nav.result?.errorText ?? nav.error)}`);
      throw new Error("engine not answering");
    }
    // Wait for the data: no "Connecting" left, and any scene painted at least once. With the
    // replies held back, the shot is of the loading states instead.
    if (holdApi > 0) await sleep(1200);
    for (let i = 0; i < 40 && holdApi === 0; i++) {
      await sleep(500);
      const r = await s.send("Runtime.evaluate", { expression: "!document.body.innerText.includes('Connecting') && (!document.querySelector('canvas') || (window.__nmScene && window.__nmScene.base > 0))", returnByValue: true });
      if (r.result?.result?.value) break;
    }
    if (holdApi === 0) await sleep(2200);
    if (verifySignal) {
      // A mock call arrives after the page has settled: it plays the path the call took.
      mock.audit = {
        audit: [
          ...mock.audit.audit,
          {
            id: "a-verify-1", kind: "tool", tool: "nm_context_pack", label: "Context pack: verify capture (mock reply)", mode: "plain", ms: 212, ok: true, time_ms: Date.now(),
            steps: [
              { part: "nmsubs.views#g0", start_ms: 0, ms: 4 },
              { part: "nmpod.unit#d0", start_ms: 4, ms: 30 }, { part: "nmpod.unit#d3", start_ms: 4, ms: 30 }, { part: "nmpod.unit#d5", start_ms: 4, ms: 30 },
              { part: "nmapi.tentacle_core#t0", start_ms: 34, ms: 20, job: "structural" }, { part: "nmapi.tentacle_core#t3", start_ms: 34, ms: 20, job: "dependency" }, { part: "nmapi.tentacle_core#t5", start_ms: 34, ms: 20, job: "verification" },
              { part: "nmbrain.g0#t0", start_ms: 54, ms: 2 }, { part: "nmbrain.g0#t3", start_ms: 54, ms: 2 }, { part: "nmbrain.g0#t5", start_ms: 54, ms: 2 },
              { part: "nmbrain.g1#g1", start_ms: 56, ms: 6 },
              { part: "nmbrain.rivers#b0", start_ms: 62, ms: 8 },
              { part: "nmbrain.synthesis#b0", start_ms: 70, ms: 30 },
            ],
          },
        ],
        episodes: [],
      };
      await sleep(Number(process.env.SHOT_DELAY ?? 1900));
    }
    if (verifyDock) {
      mock.carts = [{ schema: "nm-cartridge/1", id: "verify-naming-pref", kind: "preference", data_class: "behavioural", base: "qwen3-0.6b", base_sha256: "", components: [], scopes: [], quantization: "4-bit", description: "Mock cartridge for a verify capture.", version: 1, check: "accept", date: "2026-11-15", loaded: true, model: "qwen3-0.6b", layers: [8, 19], class: "preference", verified: true }];
      await sleep(Number(process.env.SHOT_DELAY ?? 2600));
    }
    let report = null;
    if (idle) {
      const a = await s.send("Runtime.evaluate", { expression: "JSON.stringify(window.__nmScene||null)", returnByValue: true });
      await sleep(5000);
      const b = await s.send("Runtime.evaluate", { expression: "JSON.stringify(window.__nmScene||null)", returnByValue: true });
      report = { before: JSON.parse(a.result.result.value), after: JSON.parse(b.result.result.value) };
    }
    const sw = await s.send("Runtime.evaluate", { expression: "JSON.stringify({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, scene: window.__nmScene, build: (document.querySelector('[data-build]')||{}).textContent})", returnByValue: true });
    const build = JSON.parse(sw.result.result.value).build;
    if (build) writeFileSync(join(out, "build.txt"), `interface build stamp: ${build.trim().replace(/\s+/g, " ")}
`);
    const coll = await s.send("Runtime.evaluate", { expression: COLLIDE, returnByValue: true });
    const collide = coll.result?.result?.value ?? "{}";
    const shot = await s.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    if (!shot.result) throw new Error(`screenshot failed for ${name}: ${JSON.stringify(shot.error)}`);
    writeFileSync(join(out, `${name}.png`), Buffer.from(shot.result.data, "base64"));
    log(`${name} ${vw}x${vh} ${sw.result.result.value} text ${collide}${report ? " idle " + JSON.stringify(report) : ""}`);
    s.close();
  } finally {
    await teardown(dir, port);
  }
}

if (ours().length) {
  console.error("capture browsers are already running; not starting another");
  process.exit(1);
}
// Every capture run first checks that the drawn tree matches the engine's anatomy.
{
  const chk = spawnSync(process.execPath, ["--no-warnings", join(HERE, "check-anatomy.mjs"), "--quiet", `--base=${BASE}`], { encoding: "utf8" });
  log((chk.stdout || chk.stderr).trim());
  if (chk.status !== 0) {
    console.error("anatomy check failed; not capturing");
    process.exit(1);
  }
}
for (const j of jobs) {
  const i = j.indexOf("=");
  await capture(j.slice(0, i), j.slice(i + 1));
}
log(`done; capture browsers running: ${ours().length}`);
