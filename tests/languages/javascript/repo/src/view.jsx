import { useState } from "react";
import { LEVELS } from "./parse.js";

/** A browser view of records with a level picker. */
export function RecordTable({ records }) {
  const [min, setMin] = useState(2);
  const shown = records.filter((r) => r.level >= min);
  return (
    <div>
      <select value={min} onChange={(e) => setMin(Number(e.target.value))}>
        {LEVELS.map((name, i) => (
          <option key={name} value={i}>{name}</option>
        ))}
      </select>
      <table>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i} className={`level-${LEVELS[r.level]}`}>
              <td>{r.time.toISOString()}</td>
              <td>{r.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
