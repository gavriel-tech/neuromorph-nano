/**
 * Tasks: what one card on the board holds.
 * function notADefinition() {} stays inside this comment.
 */

export type TaskStatus = "todo" | "doing" | "done";

export enum Priority {
  Low = 1,
  Normal = 2,
  Urgent = 3,
}

export const DEFAULT_PRIORITY = Priority.Normal;

export interface Task {
  id: string;
  title: string;
  status: TaskStatus;
  priority: Priority;
  due?: Date;
  tags: string[];
}

/** Raised when a task fails validation; carries the field that failed. */
export class TaskValidationError extends Error {
  constructor(public readonly field: keyof Task, message: string) {
    super(`task field ${String(field)}: ${message}`);
  }
}

let nextNumber = 1;

/** A new task with a fresh id, normal priority and no tags unless given. */
export function createTask(title: string, extra: Partial<Task> = {}): Task {
  const trimmed = title.trim();
  if (trimmed.length === 0) {
    throw new TaskValidationError("title", "must not be empty");
  }
  const id = `T-${String(nextNumber++).padStart(4, "0")}`;
  return { id, title: trimmed, status: "todo", priority: DEFAULT_PRIORITY, tags: [], ...extra };
}

/** Tags are lower case words of letters, digits and dashes. */
const TAG_PATTERN = /^[a-z0-9-]+$/;

export const normalizeTag = (tag: string): string => {
  const t = tag.trim().toLowerCase().replace(/\s+/g, "-");
  if (!TAG_PATTERN.test(t)) {
    throw new TaskValidationError("tags", `bad tag "${tag}"`);
  }
  return t;
};

export function withTags(task: Task, tags: string[]): Task {
  return { ...task, tags: Array.from(new Set(tags.map(normalizeTag))) };
}
