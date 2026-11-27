import assert from "node:assert/strict";
import { test } from "node:test";
import { parseAll, parseLine } from "../src/parse.js";

test("a stack trace joins the record before it", () => {
  const recs = parseAll("2026-03-01T10:22:05Z [error] api: boom\n    at handler (api.js:3)\n");
  assert.equal(recs.length, 1);
  assert.match(recs[0].text, /at handler/);
});

test("a line without a timestamp is not a record", () => {
  assert.equal(parseLine("plain words"), null);
});
