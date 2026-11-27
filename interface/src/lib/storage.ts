/* Disk storage of the Nano (`GET /api/storage`): what it keeps on disk, by category, with every
 * file's exact path and size, and the safe controls (`POST /api/storage/prune`,
 * `POST /api/storage/delete`, `GET` and `POST /api/storage/settings`). Pure parsers and helpers,
 * shared by the storage page and the tests. Every destructive request carries `confirm: true`,
 * and the page asks the user first. */

export const STORAGE_API = "/api/storage";
export const STORAGE_PRUNE = "/api/storage/prune";
export const STORAGE_DELETE = "/api/storage/delete";
export const STORAGE_SETTINGS = "/api/storage/settings";

export type StorageKind = "rebuildable" | "removable" | "read-only" | "junk";
/** One item of a category (`items[]`): a file or a folder, its exact path, size, class (`kind`
 *  here), plain label, the project it belongs to and, for a folder, how many files it holds. */
export type StorageFile = {
  path: string;
  bytes: number;
  modified_ms: number | null;
  kind: StorageKind;
  label?: string | null;
  project?: string | null;
  entry?: "file" | "folder" | null;
  count?: number | null;
};
export type StorageCategory = {
  id: string;
  label: string;
  total_bytes: number;
  files: StorageFile[];
  /** Files on disk in the category (`files`, a count), summed from the items when not given. */
  file_count?: number;
  /** Bytes per class, as the engine sums them (null when not given). */
  rebuildable_bytes?: number | null;
  removable_bytes?: number | null;
  read_only_bytes?: number | null;
  note?: string | null;
};
export type StorageReply = {
  categories: StorageCategory[];
  root: string | null;
  total_bytes: number;
  free_bytes: number | null;
  disk_total_bytes?: number | null;
  total_files?: number | null;
  /** The free space kept on the drive, and whether the drive is below it. */
  floor_bytes?: number | null;
  below_floor?: boolean | null;
};
export type StorageSettings = Record<string, string | number | boolean>;

/** The four categories the roadmap names, in display order, with their plain description. Others
 *  the engine sends are listed after them. */
export const CATEGORIES: { id: string; label: string; about: string; hue: string }[] = [
  { id: "memory", label: "Memory and World Model", about: "world model notes, writer notes, examples, the G3 muscle store, episode and decision logs", hue: "var(--behav)" },
  { id: "vectors", label: "Dense Vectors and Caches", about: "dense chunk embeddings, auxiliary note vectors, repo index caches", hue: "var(--sense)" },
  { id: "compression", label: "Compression and Quantized Copies", about: "bmq8, q8 and q4 compiled copies, scratch and cache files that rebuild", hue: "var(--sense-pale)" },
  { id: "cartridges", label: "Parameter Cartridges and Weights", about: "base cartridges, parameter maps and job-path filters, lenses, preference cartridges, optional upstream open weights", hue: "var(--verified)" },
  // 2026-11-26 (Luis Jake Gabriel III: manage and delete junk files from the app).
  { id: "junk", label: "Junk", about: "what tests, experiments and stopped runs left behind and nothing uses: test leftovers, old experiment builds, stale copies, old logs", hue: "var(--attention)" },
];

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

function kindOf(v: unknown): StorageKind {
  if (v === "rebuildable" || v === "removable" || v === "read-only" || v === "junk") return v;
  if (v === "readonly" || v === "read_only") return "read-only";
  // An unknown kind is never offered for removal.
  return "read-only";
}

/** The category a reply's id belongs to among the four (by id or by words in its label). */
export function categoryOf(id: string, label = ""): string {
  const s = `${id} ${label}`.toLowerCase();
  if (id === "junk" || /^junk\b/.test(s)) return "junk";
  if (/memory|world|note|muscle|episode/.test(s)) return "memory";
  if (/vector|dense|index|embed/.test(s)) return "vectors";
  if (/compress|quant|bmq|q8|q4|scratch/.test(s)) return "compression";
  if (/cartridge|weight|param|filter|lens|prefer/.test(s)) return "cartridges";
  return id;
}

/** An item's class: the engine sends it as `class` (`rebuildable`, `removable`, `read-only`) and
 *  `kind` as `file` or `folder`; an earlier draft sent the class as `kind`. */
function classOf(y: Record<string, unknown>): StorageKind {
  if (y.class !== undefined) return kindOf(y.class);
  return kindOf(y.kind);
}

/** The reply, normalised: categories in the four's order, each item's class checked, totals summed
 *  where the engine left them out, items largest first. Reads the engine's shape (`name`, `bytes`,
 *  `items[]` with `class`, `files` as a count, `disk.free_bytes`) and the earlier draft's (`label`,
 *  `total_bytes`, `files[]` with `kind`, `free_bytes`). Null when the reply is not an object. */
export function parseStorage(raw: unknown): StorageReply | null {
  const o = obj(raw) ?? (Array.isArray(raw) ? { categories: raw } : null);
  if (!o) return null;
  const list = Array.isArray(o.categories) ? o.categories : [];
  const cats: StorageCategory[] = [];
  for (const c of list) {
    const x = obj(c);
    if (!x) continue;
    const name = str(x.name) ?? str(x.label);
    const id = str(x.id) ?? name;
    if (!id) continue;
    const files: StorageFile[] = [];
    const rows = Array.isArray(x.items) ? x.items : Array.isArray(x.files) ? x.files : [];
    for (const f of rows) {
      const y = obj(f);
      const path = y ? str(y.path) : null;
      if (!y || !path) continue;
      const entry = y.kind === "file" || y.kind === "folder" ? y.kind : null;
      files.push({ path, bytes: Math.max(0, num(y.bytes) ?? 0), modified_ms: num(y.modified_ms), kind: classOf(y), label: str(y.label), project: str(y.project), entry, count: num(y.files) });
    }
    files.sort((a, b) => b.bytes - a.bytes || a.path.localeCompare(b.path));
    const sum = files.reduce((a, f) => a + f.bytes, 0);
    const counted = files.reduce((a, f) => a + (f.count ?? 1), 0);
    cats.push({
      id,
      label: name ?? id,
      total_bytes: Math.max(0, num(x.bytes) ?? num(x.total_bytes) ?? sum),
      files,
      file_count: num(x.files) ?? counted,
      rebuildable_bytes: num(x.rebuildable_bytes),
      removable_bytes: num(x.removable_bytes),
      read_only_bytes: num(x.read_only_bytes),
      note: str(x.note),
    });
  }
  const rank = (c: StorageCategory) => {
    const i = CATEGORIES.findIndex((k) => k.id === categoryOf(c.id, c.label));
    return i < 0 ? CATEGORIES.length : i;
  };
  cats.sort((a, b) => rank(a) - rank(b));
  const total = cats.reduce((a, c) => a + c.total_bytes, 0);
  const disk = obj(o.disk);
  const floorGb = num(o.disk_floor_gb);
  return {
    categories: cats,
    root: str(o.root) ?? str(o.home) ?? str(disk?.path),
    total_bytes: num(o.total_bytes) ?? total,
    free_bytes: num(disk?.free_bytes) ?? num(o.free_bytes) ?? num(o.disk_free_bytes),
    disk_total_bytes: num(disk?.total_bytes),
    total_files: num(o.total_files),
    floor_bytes: num(o.disk_floor_bytes) ?? (floorGb !== null ? floorGb * 1024 ** 3 : null),
    below_floor: typeof o.below_floor === "boolean" ? o.below_floor : null,
  };
}

/** Each category's share of the whole, 0..1, for the breakdown bar. */
export function shares(r: StorageReply | null | undefined): { id: string; label: string; bytes: number; share: number; hue: string }[] {
  const cats = r?.categories ?? [];
  const total = cats.reduce((a, c) => a + c.total_bytes, 0);
  return cats.map((c, i) => {
    const def = CATEGORIES.find((k) => k.id === categoryOf(c.id, c.label));
    return { id: c.id, label: c.label, bytes: c.total_bytes, share: total > 0 ? c.total_bytes / total : 0, hue: def?.hue ?? `hsl(${(i * 67) % 360} 50% 70%)` };
  });
}

/** Bytes a prune of a category would free: its rebuildable files only. */
export function prunable(c: StorageCategory | null | undefined): number {
  if (typeof c?.rebuildable_bytes === "number") return Math.max(0, c.rebuildable_bytes);
  return (c?.files ?? []).filter((f) => f.kind === "rebuildable" || f.kind === "junk").reduce((a, f) => a + f.bytes, 0);
}

/** Whether a file may be deleted from the page: removable files only (read-only base weights and
 *  cartridges never, rebuildable files through prune). */
export function canDelete(f: StorageFile): boolean {
  return f.kind === "removable" || f.kind === "junk";
}

/** The request bodies. `confirm: true` is always sent; the page sends them only after the user
 *  confirmed in its dialog. */
export function pruneBody(category: string): { category: string; confirm: true } {
  return { category, confirm: true };
}
export function deleteBody(path: string): { path: string; confirm: true } {
  return { path, confirm: true };
}

/** The settings change body (`POST /api/storage/settings`): the key and the value as a string,
 *  which is what the engine reads. */
export function settingsBody(key: string, value: string | number | boolean): { key: string; value: string } {
  return { key, value: String(value) };
}

/** One storage setting the engine lets the user change (`settable[]`), with its value now. */
export type StorageSettable = { key: string; value: number | string | null; default: number | string | null; min: number | null; max: number | null; unit: string | null };
/** A folder the Nano keeps its files in (read only here: fixed by the Nano's layout). */
export type StoragePath = { id: string; label: string; path: string };
/** The storage settings view (`GET /api/storage/settings`): what can be changed, the folders, the
 *  memory folder of each project, and the drive with its floor. */
export type StorageSettingsView = {
  settable: StorageSettable[];
  paths: StoragePath[];
  memory_dirs: { project: string; path: string }[];
  memory_dir_note: string | null;
  free_bytes: number | null;
  total_bytes: number | null;
  floor_bytes: number | null;
  below_floor: boolean | null;
};

const PATH_LABELS: [string, string][] = [
  ["home", "Nano folder"],
  ["memory_dir", "memory folder"],
  ["projects_dir", "project memory folders"],
  ["library_dir", "cartridge library"],
  ["models_dir", "upstream open weights (optional)"],
];

/** The value now of a settable key: the reply's field named by the key's last part
 *  (`storage.disk_floor_gb` reads `disk_floor_gb`), else its `value`. */
function settableValue(o: Record<string, unknown>, row: Record<string, unknown>, key: string): number | string | null {
  const own = row.value;
  if (typeof own === "number" || (typeof own === "string" && own.trim())) return own as number | string;
  const tail = key.split(".").pop() ?? key;
  const v = o[tail] ?? o[key];
  return typeof v === "number" || (typeof v === "string" && v.trim()) ? (v as number | string) : null;
}

/** The settings reply, read as the engine sends it. Null when the reply is not an object. */
export function parseStorageSettingsView(raw: unknown): StorageSettingsView | null {
  const o = obj(raw);
  if (!o) return null;
  const settable: StorageSettable[] = [];
  for (const s of Array.isArray(o.settable) ? o.settable : []) {
    const row = obj(s);
    const key = row ? str(row.key) : null;
    if (!row || !key) continue;
    const d = row.default;
    settable.push({ key, value: settableValue(o, row, key), default: typeof d === "number" || typeof d === "string" ? d : null, min: num(row.min), max: num(row.max), unit: str(row.unit) });
  }
  const paths: StoragePath[] = [];
  for (const [id, label] of PATH_LABELS) {
    const p = str(o[id]);
    if (p) paths.push({ id, label, path: p });
  }
  const memory_dirs = (Array.isArray(o.memory_dirs) ? o.memory_dirs : [])
    .map((m) => obj(m))
    .filter((m): m is Record<string, unknown> => m !== null && str(m.path) !== null)
    .map((m) => ({ project: str(m.project) ?? "", path: str(m.path) as string }));
  const disk = obj(o.disk);
  const floorGb = num(o.disk_floor_gb);
  return {
    settable,
    paths,
    memory_dirs,
    memory_dir_note: str(o.memory_dir_note),
    free_bytes: num(disk?.free_bytes),
    total_bytes: num(disk?.total_bytes),
    floor_bytes: num(o.disk_floor_bytes) ?? (floorGb !== null ? floorGb * 1024 ** 3 : null),
    below_floor: typeof o.below_floor === "boolean" ? o.below_floor : null,
  };
}

/** The settings the user may change, as a flat map of key to value now: the engine's `settable`
 *  keys when it lists them; else (an earlier draft) the strings, numbers and flags of `settings`
 *  or of the reply. */
export function parseStorageSettings(raw: unknown): StorageSettings | null {
  const o = obj(raw);
  if (!o) return null;
  if (Array.isArray(o.settable)) {
    const out: StorageSettings = {};
    for (const s of parseStorageSettingsView(o)?.settable ?? []) if (s.value !== null) out[s.key] = s.value;
    return out;
  }
  const src = obj(o.settings) ?? o;
  const out: StorageSettings = {};
  for (const [k, v] of Object.entries(src)) if (typeof v === "string" || typeof v === "boolean" || (typeof v === "number" && Number.isFinite(v))) out[k] = v;
  return out;
}

/** Plain words for the kinds. */
export const KIND_WORDS: Record<StorageKind, string> = {
  rebuildable: "rebuilds on demand",
  removable: "removable",
  "read-only": "read only",
  junk: "junk",
};

/** Bytes on disk of the files whose path names `id` (a cartridge, filter or lens), from the
 *  storage reply; null when storage is not reported or no file names it. */
export function bytesFor(r: StorageReply | null | undefined, id: string): number | null {
  if (!r || !id) return null;
  const needle = id.toLowerCase();
  let sum = 0;
  let found = false;
  for (const c of r.categories)
    for (const f of c.files)
      if (f.path.toLowerCase().replace(/\\/g, "/").includes(needle)) {
        sum += f.bytes;
        found = true;
      }
  return found ? sum : null;
}
