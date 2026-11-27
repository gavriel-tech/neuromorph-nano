import { useEffect, useState } from "react";
import type { Task, TaskStatus } from "@taskboard/core";
import { ApiClient } from "@/api/client";

/** The tasks of one column, loaded from the server and refreshed every `refreshMs`. */
export function useTasks(client: ApiClient, status: TaskStatus, refreshMs = 30000) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const load = () =>
      client
        .fetchTasks(status)
        .then((t) => live && setTasks(t))
        .catch((e) => live && setError(String(e.message ?? e)));
    load();
    const timer = setInterval(load, refreshMs);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [client, status, refreshMs]);

  return { tasks, error };
}
