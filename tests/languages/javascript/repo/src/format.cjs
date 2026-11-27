"use strict";
// Terminal output, CommonJS so older tools can require it.
const { pad } = require("./util.cjs");

const COLORS = {
  warn: "\x1b[33m",
  error: "\x1b[31m",
  fatal: "\x1b[41m",
};
const RESET = "\x1b[0m";

function colorize(levelName, text) {
  const c = COLORS[levelName];
  return c ? c + text + RESET : text;
}

/** One record as a single terminal line: time, level, source, text. */
function formatRecord(rec, levelNames) {
  const name = levelNames[rec.level] || "info";
  const time = rec.time.toISOString().slice(11, 19);
  return `${time} ${colorize(name, pad(name, 5))} ${rec.source ? rec.source + ": " : ""}${rec.text}`;
}

module.exports = { colorize, formatRecord };
