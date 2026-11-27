import { open, stat } from "node:fs/promises";
import { parseAll } from "./parse.js";

/** How often a followed file is checked for new lines, in milliseconds. */
export const POLL_MS = 500;

/**
 * Follows a growing file like `tail -f`: calls `onRecords` with the records of
 * each new chunk. A file that shrank was rotated and is read again from the top.
 */
export async function follow(path, onRecords, signal) {
  let offset = (await stat(path)).size;
  while (!signal?.aborted) {
    const size = (await stat(path)).size;
    if (size < offset) {
      offset = 0;
    }
    if (size > offset) {
      const fh = await open(path, "r");
      try {
        const buf = Buffer.alloc(size - offset);
        await fh.read(buf, 0, buf.length, offset);
        offset = size;
        onRecords(parseAll(buf.toString("utf8")));
      } finally {
        await fh.close();
      }
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
}
