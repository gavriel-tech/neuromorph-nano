import { Task, TaskStatus, TaskValidationError } from "./task";

/** The most tasks one board keeps; adding past it is refused. */
export const MAX_TASKS = 500;

type Listener = (tasks: readonly Task[]) => void;

/**
 * Holds the board's tasks in insertion order and tells listeners after each change.
 * The index maps an id to its place so lookups do not scan.
 */
export class TaskStore {
  private tasks: Task[] = [];
  private index = new Map<string, number>();
  private listeners: Listener[] = [];

  add(task: Task): void {
    if (this.tasks.length >= MAX_TASKS) {
      throw new TaskValidationError("id", `board is full (${MAX_TASKS} tasks)`);
    }
    if (this.index.has(task.id)) {
      throw new TaskValidationError("id", "duplicate id");
    }
    this.index.set(task.id, this.tasks.length);
    this.tasks.push(task);
    this.emit();
  }

  remove(id: string): boolean {
    const at = this.index.get(id);
    if (at === undefined) {
      return false;
    }
    this.tasks.splice(at, 1);
    this.reindex();
    this.emit();
    return true;
  }

  /** Move a task to another column; the board's only state change besides add and remove. */
  public moveTo(id: string, status: TaskStatus): Task | undefined {
    const at = this.index.get(id);
    if (at === undefined) {
      return undefined;
    }
    const moved = { ...this.tasks[at], status };
    this.tasks[at] = moved;
    this.emit();
    return moved;
  }

  byStatus(status: TaskStatus): Task[] {
    return this.tasks.filter((t) => t.status === status);
  }

  count = (): number => this.tasks.length;

  subscribe(listener: Listener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private reindex(): void {
    this.index.clear();
    this.tasks.forEach((t, i) => this.index.set(t.id, i));
  }

  private emit(): void {
    const snapshot = Object.freeze([...this.tasks]);
    for (const l of this.listeners) {
      l(snapshot);
    }
  }
}
