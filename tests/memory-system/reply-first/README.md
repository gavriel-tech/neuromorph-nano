# Reply first, memory write after

The memory write is split around the reply. Before the reply: the answer's episode is appended and
its journal event is flushed to disk (the two records memory is rebuilt from). After the reply
(`nmapi::session::settle`): the hierarchy's item and the journal's views. Does a process that ends
between the reply and the second half lose anything?

Run (under a minute, processor only):

```
python tests/memory-system/reply-first/run.py --exe <neuromorph-nano.exe>
```

It uses a scratch Nano folder under `scratch/lane-c/reply-first/` (never the real `memory/`) and the
frozen tree `scratch/frozen-p2`, read only. The process is ended right after its reply by the
environment variable `NANO_TEST_STOP_AFTER_REPLY` (read by the MCP loop only for this test).

Pass rule (set before the first run): after the stop, the episode and its journal record are on
disk, and the views and the hierarchy do not have it yet; after a new process opens the folder, the
views reach the journal's last event and the hierarchy holds the answer. The report, with the
reply's server time and the write's two halves from four normal calls, goes to
`runs/memory-reply-first-<ms>.json`.

Files: `run.py` (uses the second-session harness's helpers in `../second-session/run.py`).
