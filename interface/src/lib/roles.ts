/* Which MicroModel runs each role, from the engine's role choice (GET /api/models/roles: `roles[]`
 * with each role's `model`, and `models[]` with each model's current `role`). Roles are
 * reassignable (a model may serve two roles, the writer may share the reader's), so nothing here
 * is fixed: the stated default map is used only while the reply has not come yet. */

import { useMemo } from "react";
import { useApi } from "@/lib/api";

export type RolesReply = {
  applies?: string;
  roles?: { role: string; model?: string | null }[];
  models?: { id: string; kind?: string; role?: string | null }[];
  error?: string;
};

/** The released defaults, used only until /api/models/roles answers. */
const DEFAULT_ROLE_OF: Record<string, string> = { "qwen3-0.6b": "reader", "qwen3-1.7b": "writer", "qwen3-embedding-0.6b": "embedder", "qwen3-reranker-0.6b": "reranker" };

export type RoleMap = {
  /** The model a role runs on now; null when the engine names none. */
  modelOf: (role: string) => string | null;
  /** Every role a model serves now (empty when none). */
  rolesOf: (model: string) => string[];
  /** The roles a model serves, joined ("reader + writer"), or "" when none. */
  roleOf: (model: string) => string;
  /** True once the engine's reply is in hand (then the defaults are no longer used). */
  reported: boolean;
};

/** The role map from a reply; `loading` true means it has not come yet, so the defaults stand in. */
export function roleMap(data: RolesReply | null | undefined, loading: boolean): RoleMap {
  const roles = data && !data.error && Array.isArray(data.roles) ? data.roles : null;
  const models = data && !data.error && Array.isArray(data.models) ? data.models : null;
  const reported = roles !== null || models !== null;
  const rolesOf = (model: string): string[] => {
    if (roles) {
      const out = roles.filter((r) => r.model === model).map((r) => r.role);
      if (out.length > 0 || !models) return out;
    }
    if (models) {
      const r = models.find((m) => m.id === model)?.role;
      return r ? [r] : [];
    }
    return loading && DEFAULT_ROLE_OF[model] ? [DEFAULT_ROLE_OF[model]] : [];
  };
  const modelOf = (role: string): string | null => {
    if (roles) {
      const m = roles.find((r) => r.role === role)?.model;
      if (m) return m;
    }
    if (models) return models.find((m) => m.role === role)?.id ?? null;
    if (!loading) return null;
    return Object.entries(DEFAULT_ROLE_OF).find(([, r]) => r === role)?.[0] ?? null;
  };
  return { modelOf, rolesOf, roleOf: (model) => rolesOf(model).join(" + "), reported };
}

/** The live role map, refreshed with the roles panel. */
export function useRoleMap(): RoleMap & { error: string | null; answered: boolean } {
  const q = useApi<RolesReply>("/api/models/roles", 10000);
  return useMemo(() => ({ ...roleMap(q.data, q.loading), error: q.error, answered: !q.loading }), [q.data, q.loading, q.error]);
}
