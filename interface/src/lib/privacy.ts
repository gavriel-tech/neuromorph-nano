/** What may be shown. The interface is a display that others may see, share or capture, so by
 *  default it shows nothing personal: paths are shown relative to the Nano folder or with the home
 *  part as `~`, project names are shortened, the text of questions and prompts is replaced by the
 *  kind of call, and anything that looks like a token, key or password is never shown.
 *
 *  A reveal is per item and per session: it lives in memory only and is never saved. In capture
 *  mode (`?mask=all` in the address, or `window.__nmMaskAll`) nothing can be revealed. */
import { useSyncExternalStore } from "react";

declare global {
  interface Window {
    __nmMaskAll?: boolean;
  }
}

const revealed = new Set<string>();
const subs = new Set<() => void>();
let version = 0;

export function captureMask(): boolean {
  if (typeof window === "undefined") return true;
  return window.__nmMaskAll === true || new URLSearchParams(window.location.search).get("mask") === "all";
}

export function reveal(id: string) {
  if (captureMask()) return;
  revealed.add(id);
  version++;
  subs.forEach((f) => f());
}

export function hide(id: string) {
  revealed.delete(id);
  version++;
  subs.forEach((f) => f());
}

/** Whether an item is revealed now; re-renders when reveals change. */
export function useRevealed(id: string): boolean {
  useSyncExternalStore(
    (f) => {
      subs.add(f);
      return () => subs.delete(f);
    },
    () => version,
    () => 0,
  );
  return !captureMask() && revealed.has(id);
}

const SECRET = /\b(?:sk|pk|ghp|gho|xox[abp]|hf)_[A-Za-z0-9_-]{8,}|\b[A-Fa-f0-9]{32,}\b|\b(?:password|passwd|secret|token|api[_-]?key|bearer)\s*[:=]\s*\S+/gi;

/** Removes anything that looks like a token, key, password or long hex secret. Never revealed. */
export function scrub(text: string): string {
  return text.replace(SECRET, "[hidden]");
}

/** A path as it may be shown: relative to the Nano folder when inside it, else with the home
 *  part (`C:\Users\name`, `/home/name`, `/Users/name`) as `~`, else only its last two parts. */
export function maskPath(p: string | null | undefined, root?: string | null): string {
  if (!p) return "";
  const norm = (s: string) => s.replace(/\\/g, "/").replace(/\/+$/, "");
  const P = norm(p);
  if (root) {
    const R = norm(root);
    if (P.toLowerCase() === R.toLowerCase()) return ".";
    if (P.toLowerCase().startsWith(R.toLowerCase() + "/")) return "./" + P.slice(R.length + 1);
  }
  const home = P.match(/^(?:[A-Za-z]:\/Users\/[^/]+|\/home\/[^/]+|\/Users\/[^/]+)(\/.*)?$/);
  if (home) return "~" + (home[1] ?? "");
  const parts = P.split("/").filter(Boolean);
  return parts.length > 2 ? `…/${parts.slice(-2).join("/")}` : P;
}

/** Every absolute path inside a block of text (for design files and records) shown as above. */
export function maskPathsIn(text: string, root?: string | null): string {
  const re = /(?:[A-Za-z]:\\\\|[A-Za-z]:\\|[A-Za-z]:\/|\/(?:home|Users)\/)[^"\n,}\]]*/g;
  return scrub(text.replace(re, (m) => maskPath(m.replace(/\\\\/g, "\\"), root)));
}

/** A project name shortened: its first characters and its length, e.g. `e1-r… (16)`. */
export function maskName(name: string | null | undefined): string {
  if (!name) return "";
  return name.length <= 4 ? "…" : `${name.slice(0, 4)}…`;
}

/** The label of a call as it may be shown: the kind of call ("Context pack"), with the question
 *  or prompt text after the colon held back. */
export function callLabel(label: string | undefined | null, tool?: string | null): { short: string; hidden: string | null } {
  const l = label?.trim() || tool?.trim() || "call with no tool name";
  const i = l.indexOf(": ");
  if (i < 0) return { short: scrub(l), hidden: null };
  return { short: l.slice(0, i), hidden: scrub(l.slice(i + 2)) };
}
