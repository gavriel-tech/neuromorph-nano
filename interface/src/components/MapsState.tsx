"use client";

// The live state every lab builder reads: the maps of each MicroModel, the job filters and the one
// build running (GET /api/parameter-maps, GET /api/surgery), the layer count of each model, and the
// write path with its note. The Parameters page and the Cartridges page's Build tab each hold one, so
// the same builder behaves the same on either page.

import { useMemo, useState } from "react";
import type { BuildState, JobFilter, MapsApi, Note, RoleMaps } from "@/components/MapCore";
import { post, refreshAll, useApi, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";

type MapsPayload = { roles?: RoleMaps[] };
export type Surgery = { roles?: Record<string, { job_filters?: JobFilter[] }>; job_filter_build?: BuildState };

/** What a write refreshes when it lands; the caller adds its own page's reads. */
const REFRESH = ["/api/parameter-maps", "/api/surgery", "/api/audit", "/api/live"];

export function useMapsState(extra: readonly string[] = []) {
  const q = useApi<MapsPayload>("/api/parameter-maps", 15000);
  const surgery = useApi<Surgery>("/api/surgery", 3000);
  const status = useApi<Status>("/api/status", 15000);
  const now = useNow();
  const [note, setNoteState] = useState<{ note: Note; at: string | null } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [role, setRole] = useState("reranker");
  const [job, setJob] = useState("code-slice");
  const [layers, setLayers] = useState("all");
  const roles = useMemo(() => q.data?.roles ?? [], [q.data]);
  const build = surgery.data?.job_filter_build && surgery.data.job_filter_build.status !== "idle" ? surgery.data.job_filter_build : null;

  const api: MapsApi = {
    roles,
    layerCount: (model) => {
      const n = status.data?.models.find((m) => m.id === model)?.layers;
      return typeof n === "number" && n > 0 ? n : null;
    },
    filtersOf: (r) => surgery.data?.roles?.[r]?.job_filters ?? [],
    now,
    busy: busy !== null,
    run: async (id, path, body, done) => {
      setBusy(id);
      try {
        await post(path, body);
        for (const p of [...REFRESH, ...extra]) refreshAll(p);
        setNoteState({ note: { text: done, tone: "verified" }, at: id });
      } catch (e) {
        setNoteState({ note: { text: e instanceof Error ? e.message : String(e), tone: "error" }, at: id });
      }
      setBusy(null);
    },
    onNote: (n) => setNoteState({ note: n, at: null }),
    buildRunning: build?.status === "running",
    rebuild: (r, j) => {
      setRole(r);
      setJob(j);
    },
  };

  return { q, surgery, roles, build, note, api, form: { role, setRole, job, setJob, layers, setLayers } };
}
