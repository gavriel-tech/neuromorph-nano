"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "also you have the micromodels named as such when
// thats wrong. the oipen weights used is not the micromodel. that should be clearly diferentiated but
// not overyly through text"): one way to show a MicroModel on every page. Its name is its role
// (Reader, Writer, Embedder, Reranker); beside it, two marks tell the parts apart at a glance: a
// filled violet square for the Neuromorph base cartridge it runs on, a hollow ring for the upstream
// open weights that cartridge was built from (quieter, since the MicroModel is not them). Hovering
// either says what it is.

import { useApi } from "@/lib/api";
import { useRoleMap } from "@/lib/roles";

type Bases = { models?: { model: string; serving: string }[] };

/** "reader" -> "Reader"; two roles on one model read "Reader + Writer". */
export function roleTitle(roles: string): string {
  return roles
    .split(/\s*\+\s*/)
    .filter(Boolean)
    .map((r) => r.charAt(0).toUpperCase() + r.slice(1))
    .join(" + ");
}

/** The serving base cartridge of each model id, from the engine. */
export function useServingBases(): Map<string, string> {
  const q = useApi<Bases>("/api/cartridges/bases", 30000);
  return new Map((q.data?.models ?? []).map((b) => [b.model, b.serving]));
}

/** The base cartridge mark: a filled violet square and its key. */
export function BaseMark({ base, quiet = false, labeled = false }: { base: string | null | undefined; quiet?: boolean; labeled?: boolean }) {
  if (!base || base === "open weights") {
    return (
      <span className="mono inline-flex items-center gap-1.5 text-[10.5px] text-attention" title="No Neuromorph base cartridge serves: this MicroModel runs straight on the upstream open weights (open mode)">
        <i aria-hidden className="inline-block h-1.5 w-1.5 border border-attention" />
        open mode (no base cartridge)
      </span>
    );
  }
  return (
    <span className={`mono inline-flex items-center gap-1.5 text-[10.5px] ${quiet ? "text-faint" : "text-muted"}`} title={`${base}: the Neuromorph base cartridge this MicroModel runs on`}>
      <i aria-hidden className="inline-block h-1.5 w-1.5 bg-learned" />
      {labeled && <span className="text-quiet">base cartridge</span>}
      <span>{base}</span>
    </span>
  );
}

/** The upstream open weights mark: a hollow ring and the model id, quieter than the cartridge. */
export function WeightsMark({ model, labeled = false }: { model: string; labeled?: boolean }) {
  return (
    <span className="mono inline-flex items-center gap-1.5 text-[10px] text-quiet" title={`${model}: the upstream open weights (Apache-2.0) the base cartridge was built from; not the MicroModel itself`}>
      <i aria-hidden className="inline-block h-1.5 w-1.5 rounded-full border border-quiet" />
      {labeled && <span className="text-quiet/80">open weights</span>}
      <span>{model}</span>
    </span>
  );
}

/** A MicroModel by its role, with its base cartridge and, quieter, its upstream open weights. */
export default function MicroModelName({ model, layout = "inline", nameClass = "text-text" }: { model: string; layout?: "inline" | "stack"; nameClass?: string }) {
  const roles = useRoleMap();
  const bases = useServingBases();
  const role = roles.roleOf(model);
  const name = role ? roleTitle(role) : "MicroModel (no role)";
  return (
    <span className={`min-w-0 ${layout === "stack" ? "grid gap-0.5" : "inline-flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5"}`} data-e2e="micromodel" data-model={model}>
      <span className={`mono truncate ${nameClass}`}>{name}</span>
      <span className="inline-flex flex-wrap items-baseline gap-x-2.5">
        <BaseMark base={bases.get(model)} />
        <WeightsMark model={model} />
      </span>
    </span>
  );
}
