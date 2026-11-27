import { Calendar, Priority } from "@taskboard/core";

const LABELS: Record<Priority, string> = {
  [Priority.Low]: "low",
  [Priority.Normal]: "normal",
  [Priority.Urgent]: "urgent",
};

/** The word shown for a priority on a card. */
export function priorityLabel(p: Priority): string {
  return LABELS[p];
}

/** "today", "tomorrow", "in 3 days" or "2 days ago" for a due date seen from `now`. */
export const relativeDue = (due: Date, now: Date): string => {
  const days = Calendar.daysBetween(now, due);
  if (days === 0) return "today";
  if (days === 1) return "tomorrow";
  return days > 0 ? `in ${days} days` : `${-days} days ago`;
};

/** A title cut to `max` characters at a word boundary, with an ellipsis. */
export function shortTitle(title: string, max = 40): string {
  if (title.length <= max) {
    return title;
  }
  const cut = title.slice(0, max).replace(/\s+\S*$/, "");
  return `${cut}…`;
}
