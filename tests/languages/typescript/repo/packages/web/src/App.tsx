import { useMemo } from "react";
import { isOverdue, type Task, type TaskStatus } from "@taskboard/core";
import { ApiClient } from "./api/client";
import { useTasks } from "./hooks/useTasks";
import { priorityLabel, relativeDue, shortTitle } from "./format";
import "./app.css";

const COLUMNS: TaskStatus[] = ["todo", "doing", "done"];

interface RowProps {
  task: Task;
  now: Date;
}

/** One card: title, priority and due date, marked when overdue. */
const TaskRow = ({ task, now }: RowProps) => (
  <li className={isOverdue(task, now) ? "card card--overdue" : "card"}>
    <span className="card__title">{shortTitle(task.title)}</span>
    <span className="card__priority">{priorityLabel(task.priority)}</span>
    {task.due && <time>{relativeDue(task.due, now)}</time>}
  </li>
);

function Column({ client, status }: { client: ApiClient; status: TaskStatus }) {
  const { tasks, error } = useTasks(client, status);
  const now = useMemo(() => new Date(), [tasks]);
  if (error) {
    return <p className="column__error">{error}</p>;
  }
  return (
    <section className="column">
      <h2>{status}</h2>
      <ul>{tasks.map((t) => <TaskRow key={t.id} task={t} now={now} />)}</ul>
    </section>
  );
}

/** The whole board: one column per status. */
export default function App({ token }: { token: string }) {
  const client = useMemo(() => new ApiClient(token), [token]);
  return (
    <main className="board">
      {COLUMNS.map((s) => (
        <Column key={s} client={client} status={s} />
      ))}
    </main>
  );
}
