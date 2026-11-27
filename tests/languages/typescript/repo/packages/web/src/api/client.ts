import type { Task, TaskStatus } from "@taskboard/core";

/** Where the board's server answers; every request path starts here. */
export const API_BASE = "/api/v2";

/** How long a request may take before it is abandoned, in milliseconds. */
const REQUEST_TIMEOUT_MS = 8000;

export interface ApiError {
  status: number;
  message: string;
}

/**
 * The board's server, one method per endpoint. Requests time out after
 * REQUEST_TIMEOUT_MS and failed responses become ApiError values.
 */
export class ApiClient {
  constructor(private readonly token: string, private readonly base: string = API_BASE) {}

  async fetchTasks(status?: TaskStatus): Promise<Task[]> {
    const query = status ? `?status=${encodeURIComponent(status)}` : "";
    return this.request<Task[]>(`/tasks${query}`);
  }

  async saveTask(task: Task): Promise<Task> {
    return this.request<Task>(`/tasks/${task.id}`, { method: "PUT", body: JSON.stringify(task) });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(`${this.base}${path}`, {
        ...init,
        signal: ctl.signal,
        headers: { authorization: `Bearer ${this.token}`, "content-type": "application/json" },
      });
      if (!res.ok) {
        const err: ApiError = { status: res.status, message: `${init.method ?? "GET"} ${path} failed: ${res.statusText}` };
        throw err;
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }
}
