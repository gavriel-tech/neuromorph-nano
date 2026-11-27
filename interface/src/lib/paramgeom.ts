// The geometry and colour of the parameter viewer (components/ParamOverview.tsx): editor paths read
// back to the rectangles they cover on a tensor's two-dimensional view, ruler ticks, the colour
// scales built from the theme's colours, and a value's percentile from a distribution's quantiles.
// Pure functions, tested in tests/paramgeom.test.mjs.

/** Rows `r0..r1` and columns `c0..c1` of a tensor's two-dimensional view (ends exclusive). */
export type Rect = { r0: number; r1: number; c0: number; c1: number };
/** What the viewer knows of a tensor's shape: its name, shape and two-dimensional view. */
export type Dims = { name: string; shape: number[]; rows: number; cols: number };

/** One bound pair of a slice (`a:b`, `a:`, `:b`, `:` or one index `a`) over `0..len`. */
function slice(s: string, len: number): [number, number] | null {
  const t = s.trim();
  if (t === ":" || t === "") return [0, len];
  const m = /^(-?\d*)\s*:\s*(-?\d*)$/.exec(t);
  const at = (x: string, dflt: number) => (x === "" ? dflt : Number(x) < 0 ? len + Number(x) : Number(x));
  if (m) {
    const a = at(m[1], 0);
    const b = at(m[2], len);
    return Number.isFinite(a) && Number.isFinite(b) && b > a ? [Math.max(0, a), Math.min(len, b)] : null;
  }
  if (/^-?\d+$/.test(t)) {
    const a = at(t, 0);
    return a >= 0 && a < len ? [a, a + 1] : null;
  }
  return null;
}

/** The rectangles a flat range `a..b` covers on a view of `cols` columns: the first row's tail, the
 *  whole rows between, the last row's head. */
export function flatRects(a: number, b: number, cols: number): Rect[] {
  if (b <= a || cols <= 0) return [];
  const ra = Math.floor(a / cols);
  const rb = Math.floor((b - 1) / cols);
  const ca = a % cols;
  const cb = ((b - 1) % cols) + 1;
  if (ra === rb) return [{ r0: ra, r1: ra + 1, c0: ca, c1: cb }];
  const out: Rect[] = [];
  let top = ra;
  if (ca > 0) {
    out.push({ r0: ra, r1: ra + 1, c0: ca, c1: cols });
    top = ra + 1;
  }
  const bottom = cb < cols ? rb : rb + 1;
  if (bottom > top) out.push({ r0: top, r1: bottom, c0: 0, c1: cols });
  if (cb < cols) out.push({ r0: rb, r1: rb + 1, c0: 0, c1: cb });
  return out;
}

/** The tensor an editor path names and the rectangles it covers, read as `pedit::parse_path` reads
 *  it: `name` (the whole tensor), `name[r0:r1, c0:c1]` (either side `:`), `name[a:b]` (a range of a
 *  vector, or a flat range of the view for a matrix or more dimensions). `null` when the path names
 *  no tensor of `dims` (a layer `L3`, the whole object `*`, an unknown name) or does not read. */
export function rectsOf(path: string, dims: Dims[]): { tensor: Dims; rects: Rect[] } | null {
  const p = path.trim();
  const open = p.indexOf("[");
  const name = (open < 0 ? p : p.slice(0, open)).trim();
  const t = dims.find((d) => d.name === name);
  if (!t) return null;
  if (open < 0) return { tensor: t, rects: [{ r0: 0, r1: t.rows, c0: 0, c1: t.cols }] };
  if (!p.endsWith("]")) return null;
  const idx = p.slice(open + 1, -1);
  const comma = idx.indexOf(",");
  if (comma >= 0) {
    if (t.shape.length !== 2) return null;
    const r = slice(idx.slice(0, comma), t.rows);
    const c = slice(idx.slice(comma + 1), t.cols);
    return r && c ? { tensor: t, rects: [{ r0: r[0], r1: r[1], c0: c[0], c1: c[1] }] } : null;
  }
  if (t.rows === 1) {
    const c = slice(idx, t.cols);
    return c ? { tensor: t, rects: [{ r0: 0, r1: 1, c0: c[0], c1: c[1] }] } : null;
  }
  const f = slice(idx, t.rows * t.cols);
  return f ? { tensor: t, rects: flatRects(f[0], f[1], t.cols) } : null;
}

/** Whether two rectangles share a weight. */
export function overlaps(a: Rect, b: Rect): boolean {
  return a.r0 < b.r1 && b.r0 < a.r1 && a.c0 < b.c1 && b.c0 < a.c1;
}

/** The bounding box of rectangles. */
export function boundsOf(rs: Rect[]): Rect | null {
  if (rs.length === 0) return null;
  return { r0: Math.min(...rs.map((r) => r.r0)), r1: Math.max(...rs.map((r) => r.r1)), c0: Math.min(...rs.map((r) => r.c0)), c1: Math.max(...rs.map((r) => r.c1)) };
}

/** A tick step of 1, 2 or 5 times a power of ten giving about `count` ticks over `span`; at least
 *  `min` (1 for weight indices). */
export function niceStep(span: number, count: number, min = 0): number {
  if (!(span > 0) || !(count > 0)) return Math.max(1, min);
  const raw = span / count;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  return Math.max(min, step);
}

/** The ticks of `step` inside `lo..hi`. */
export function ticks(lo: number, hi: number, step: number): number[] {
  if (!(step > 0) || !(hi > lo)) return [];
  const out: number[] = [];
  for (let x = Math.ceil(lo / step) * step; x <= hi + step * 1e-9 && out.length < 200; x += step) out.push(Number(x.toPrecision(12)));
  return out;
}

/** A value's percentile (0 to 100) from the values at each whole percentile (`q[p]`, 101 of them),
 *  interpolated between neighbours; a value held by a run of equal percentiles takes its middle. */
export function percentileOf(v: number, q: number[]): number | null {
  if (q.length < 2 || !Number.isFinite(v)) return null;
  const n = q.length - 1;
  if (v < q[0]) return 0;
  if (v > q[n]) return 100;
  let first = -1;
  let last = -1;
  for (let i = 0; i <= n; i++) {
    if (q[i] === v) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first >= 0) return ((first + last) / 2) * (100 / n);
  for (let i = 0; i < n; i++) {
    if (q[i] <= v && v <= q[i + 1]) return (i + (v - q[i]) / (q[i + 1] - q[i])) * (100 / n);
  }
  return null;
}

/* ---------- colour ---------- */

export type Rgb = [number, number, number];

/** A `#rrggbb` (or `#rgb`) colour; `null` when it does not read. */
export function parseHex(s: string): Rgb | null {
  const t = s.trim();
  const m6 = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(t);
  if (m6) return [parseInt(m6[1], 16), parseInt(m6[2], 16), parseInt(m6[3], 16)];
  const m3 = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(t);
  if (m3) return [parseInt(m3[1] + m3[1], 16), parseInt(m3[2] + m3[2], 16), parseInt(m3[3] + m3[3], 16)];
  return null;
}

const lin = (c: number) => {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
};
const gam = (x: number) => {
  const y = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(y * 255)));
};

/** sRGB to OKLab (perceptual lightness `L` from 0 to 1). */
export function toLab([r, g, b]: Rgb): [number, number, number] {
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

/** OKLab to sRGB. */
export function fromLab([L, a, b]: [number, number, number]): Rgb {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  return [gam(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), gam(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s), gam(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
}

/** A colour table of `n` entries along stops (`at` from 0 to 1, ascending), interpolated in OKLab so
 *  equal steps look equal. */
export function ramp(stops: { at: number; rgb: Rgb }[], n = 256): Rgb[] {
  const labs = stops.map((s) => ({ at: s.at, lab: toLab(s.rgb) }));
  const out: Rgb[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    let k = 0;
    while (k < labs.length - 2 && t > labs[k + 1].at) k++;
    const a = labs[k];
    const b = labs[Math.min(labs.length - 1, k + 1)];
    const u = b.at > a.at ? Math.max(0, Math.min(1, (t - a.at) / (b.at - a.at))) : 0;
    out.push(fromLab([a.lab[0] + (b.lab[0] - a.lab[0]) * u, a.lab[1] + (b.lab[1] - a.lab[1]) * u, a.lab[2] + (b.lab[2] - a.lab[2]) * u]));
  }
  return out;
}

/** `a` mixed toward `b` by `t` (0 to 1), in OKLab. */
export function mix(a: Rgb, b: Rgb, t: number): Rgb {
  const x = toLab(a);
  const y = toLab(b);
  return fromLab([x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t]);
}

/** The theme colours the scales are built from (the CSS variables of app/globals.css). */
export type Palette = { zero: Rgb; neg: Rgb; negHi: Rgb; pos: Rgb; posHi: Rgb; top: Rgb };

/** The diverging scale: index 0 is `-s`, the middle 0, the last `+s`. Each arm rises in lightness
 *  from the zero colour, blue below 0 and magenta above. */
export function diverging(p: Palette, n = 257): Rgb[] {
  const half = Math.floor(n / 2) + 1;
  const neg = ramp([{ at: 0, rgb: p.zero }, { at: 0.55, rgb: p.neg }, { at: 1, rgb: p.negHi }], half);
  const pos = ramp([{ at: 0, rgb: p.zero }, { at: 0.5, rgb: mix(p.zero, p.pos, 0.6) }, { at: 0.8, rgb: p.pos }, { at: 1, rgb: p.posHi }], half);
  return [...neg.slice(1).reverse(), p.zero, ...pos.slice(1)].slice(0, n);
}

/** The sequential scale for a magnitude: index 0 is 0, the last `s`. */
export function sequential(p: Palette, n = 256): Rgb[] {
  return ramp([{ at: 0, rgb: p.zero }, { at: 0.4, rgb: p.neg }, { at: 0.75, rgb: p.negHi }, { at: 1, rgb: p.top }], n);
}

/** The table index of `x` on a scale of `n` entries: diverging over `-s..s`, sequential over `0..s`
 *  (on its magnitude). Beyond `s` it is clipped to the end. */
export function indexOf(x: number, s: number, n: number, kind: "diverging" | "sequential"): number {
  if (kind === "sequential") {
    const t = s > 0 ? Math.min(1, Math.abs(x) / s) : 0;
    return Math.round(t * (n - 1));
  }
  const t = s > 0 ? Math.max(-1, Math.min(1, x / s)) : 0;
  return Math.round(((t + 1) / 2) * (n - 1));
}

/** A value in at most `chars` characters, for a cell: three, two, then one significant figure, the
 *  leading zero dropped (`-.044`), then an exponent (`-4e-2`); `null` when nothing fits. */
export function shortFmt(v: number, chars: number): string | null {
  if (!Number.isFinite(v)) return chars >= 3 ? "nan" : null;
  if (v === 0) return "0";
  const tries: string[] = [];
  for (const d of [3, 2, 1]) {
    const s = String(Number(v.toPrecision(d)));
    tries.push(s.replace(/^(-?)0\./, "$1."));
  }
  for (const d of [1, 0]) tries.push(v.toExponential(d).replace("e+", "e"));
  for (const s of tries) if (s.length <= chars && !/e/.test(s)) return s;
  for (const s of tries) if (s.length <= chars) return s;
  return null;
}
