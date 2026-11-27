// Stated preference rules as the interface shows them: the kind's label (the stored kind `ban`
// reads "avoid"), a want's strength as text, and a typed strength read and checked against the
// engine's range before it is sent to `/api/preferences/edit`.

/** A want's strength range and default, as the engine holds them (`nmlearn::stated`). */
export const STRENGTH_MIN = 0.1;
export const STRENGTH_MAX = 2.0;
export const STRENGTH_DEFAULT = 0.5;

/** The kind as people read it: "avoid" for the stored kind `ban`, else the kind's own name. */
export function kindLabel(kind: string, label?: string): string {
  if (label) return label;
  return kind === "ban" ? "avoid" : kind;
}

/** A strength as the engine writes it: 0.5, 1, 0.25. */
export function strengthText(x: number): string {
  return String(Math.round(x * 100) / 100);
}

/** A typed strength in the engine's range, rounded to hundredths; null when it is not one. */
export function parseStrength(t: string): number | null {
  const s = t.trim();
  if (!/^\d+(\.\d+)?$|^\.\d+$/.test(s)) return null;
  const x = Math.round(Number(s) * 100) / 100;
  return x >= STRENGTH_MIN && x <= STRENGTH_MAX ? x : null;
}

/** A want's strength from a rule row (a number or its text); null for every other kind. */
export function ruleStrength(r: { kind: string; strength?: number | string }): number | null {
  if (r.kind !== "want" || r.strength === undefined) return null;
  const x = typeof r.strength === "number" ? r.strength : Number(r.strength);
  return Number.isFinite(x) ? x : null;
}
