#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { parseAll, LEVELS } from "../src/parse.js";
import { LevelFilter, rateLimit } from "../src/filter.js";
import { follow } from "../src/follow.js";

const require = createRequire(import.meta.url);
const { formatRecord } = require("../src/format.cjs");

/** Most lines one source may print per minute before the rest are dropped. */
const RATE_PER_MINUTE = 120;

async function main(argv) {
  const [path, ...rest] = argv;
  if (!path) {
    console.error("usage: logtail <file> [--level warn] [--source name] [--follow]");
    return 2;
  }
  const filter = LevelFilter.fromArgs(rest);
  const show = (records) => {
    const { kept, dropped } = rateLimit(records.filter((r) => filter.accepts(r)), RATE_PER_MINUTE);
    for (const r of kept) console.log(formatRecord(r, LEVELS));
    if (dropped > 0) console.log(`(${dropped} lines dropped by the rate limit)`);
  };
  show(parseAll(await readFile(path, "utf8")));
  if (rest.includes("--follow")) {
    await follow(path, show);
  }
  return 0;
}

main(process.argv.slice(2)).then((code) => process.exit(code));
