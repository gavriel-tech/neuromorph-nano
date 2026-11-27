import { levelOf } from "./parse.js";

/** Keeps records at or above a level, optionally from one source only. */
export class LevelFilter {
  constructor(minLevel = 2, source = null) {
    this.minLevel = minLevel;
    this.source = source;
  }

  accepts(rec) {
    if (rec.level < this.minLevel) {
      return false;
    }
    return this.source === null || rec.source === this.source;
  }

  /** From command line words: `--level warn --source disk`. */
  static fromArgs(args) {
    const at = (flag) => {
      const i = args.indexOf(flag);
      return i >= 0 ? args[i + 1] : undefined;
    };
    const level = at("--level");
    return new LevelFilter(level ? levelOf(level) : 2, at("--source") ?? null);
  }
}

/** Records whose text matches every word of a search, case ignored. */
export const matchAll = (records, words) =>
  records.filter((r) => words.every((w) => r.text.toLowerCase().includes(w.toLowerCase())));

/** At most `n` records per source per minute; the rest are counted, not shown. */
export function rateLimit(records, n) {
  const seen = new Map();
  let dropped = 0;
  const kept = records.filter((r) => {
    const key = `${r.source}@${Math.floor(r.time.getTime() / 60000)}`;
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);
    if (count > n) {
      dropped += 1;
      return false;
    }
    return true;
  });
  return { kept, dropped };
}
