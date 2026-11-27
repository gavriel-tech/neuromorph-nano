# nmnte

The Neural Topology Executor and the RNAM foundations. Depends on `nmjson` and `nmtype` only.

- `graph` (nodes, typed ports, edges, five connection modes), `topology` (reachability, the Core
  Brain bypass relation), `registry` (Component Registry), `integrity` (the anatomy check run
  before every publish), `runtime` (worker pool; panics become faults), `frame`, `snapshot`
  (immutable snapshots published atomically, rollback on a failed publish), `scheduler` (waves,
  feedback, conditional routing, fault isolation).
- `compose` (an assembly from the `designs/compositions.json` row the objective kind picks),
  `binding`, `lifecycle` (seven logged phases per anatomy change), `plasticity` (anatomy changes as
  data), `homeo`.
- No block type, layer count or forward order: the executor runs whatever graph a snapshot holds.

Tests: `cargo test --release -p nmnte --test acceptance` (twelve acceptance tests). Section checks:
`neuromorph-nano check nmnte`; the design anatomy: `neuromorph-nano check anatomy`.
