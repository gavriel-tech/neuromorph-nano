import { describe, expect, it } from "vitest";
import { TaskStore, MAX_TASKS } from "./store";
import { createTask } from "./task";

describe("TaskStore", () => {
  it("refuses a duplicate id", () => {
    const s = new TaskStore();
    const t = createTask("write the report");
    s.add(t);
    expect(() => s.add(t)).toThrow();
  });

  it("stops at the board limit", () => {
    const s = new TaskStore();
    for (let i = 0; i < MAX_TASKS; i++) {
      s.add(createTask(`task ${i}`));
    }
    expect(() => s.add(createTask("one too many"))).toThrow();
  });
});
