"use strict";

/** Right-pad with spaces to `width` characters. */
function pad(s, width) {
  return s.length >= width ? s : s + " ".repeat(width - s.length);
}

/** Milliseconds as `1.2 s` or `340 ms`. */
const humanMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);

module.exports = { pad, humanMs };
