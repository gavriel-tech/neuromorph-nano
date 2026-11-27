// Parsing of one log line into a record.
// function commentedOut() {} is not code.

/** Severity names in rising order; the index is the numeric level. */
export const LEVELS = ["trace", "debug", "info", "warn", "error", "fatal"];

/** `2026-03-01T10:22:05Z [warn] disk: 91% used` */
const LINE = /^(\d{4}-\d\d-\d\dT[\d:.]+Z?)\s+\[(\w+)\]\s+(?:([\w.-]+):\s+)?(.*)$/;

/**
 * One line to `{ time, level, source, text }`, or null when the line has no
 * timestamp (continuation lines of a stack trace).
 */
export function parseLine(line) {
  const m = LINE.exec(line);
  if (!m) {
    return null;
  }
  const level = LEVELS.indexOf(m[2].toLowerCase());
  return { time: new Date(m[1]), level: level < 0 ? 2 : level, source: m[3] ?? "", text: m[4] };
}

/** Lines to records, joining continuation lines onto the record before them. */
export const parseAll = (text) => {
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    const rec = parseLine(line);
    if (rec) {
      out.push(rec);
    } else if (out.length > 0 && line.trim() !== "") {
      out[out.length - 1].text += `\n${line}`;
    }
  }
  return out;
};

export function levelOf(name) {
  const i = LEVELS.indexOf(String(name).toLowerCase());
  if (i < 0) {
    throw new RangeError(`unknown level "${name}"; known: ${LEVELS.join(", ")}`);
  }
  return i;
}
