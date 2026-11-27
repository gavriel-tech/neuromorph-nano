import type { Task } from "./task";

const DAY_MS = 24 * 60 * 60 * 1000;

/** Days added to a due date when a recurring task is completed. */
export const RECUR_DAYS: Record<string, number> = {
  daily: 1,
  weekly: 7,
  fortnightly: 14,
};

/** The next due date of a recurring task, or undefined for a one-off task. */
export function nextDueDate(task: Task, rule: string): Date | undefined {
  const days = RECUR_DAYS[rule];
  if (days === undefined || task.due === undefined) {
    return undefined;
  }
  return new Date(task.due.getTime() + days * DAY_MS);
}

/** True when the task is past due and not done. */
export const isOverdue = (task: Task, now: Date = new Date()): boolean =>
  task.status !== "done" && task.due !== undefined && task.due.getTime() < now.getTime();

export namespace Calendar {
  /** Monday of the week holding `d`, at midnight. */
  export function weekStart(d: Date): Date {
    const day = (d.getDay() + 6) % 7;
    const start = new Date(d.getTime() - day * DAY_MS);
    start.setHours(0, 0, 0, 0);
    return start;
  }

  export const daysBetween = (a: Date, b: Date): number => Math.round((b.getTime() - a.getTime()) / DAY_MS);
}
