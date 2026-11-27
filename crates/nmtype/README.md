# nmtype

The contract crate of the Neuromorph Nano. Every other crate depends on it; it depends only on
`nmjson` and `nmhash`.

- Traits: `Component`, `ContextProcessor`, `TensorProcessor`, `MemoryComponent`, `Neuropod`,
  `FromCard`, `Resolve`, `FrameView`.
- Sub-components, one file each: `packet` (the Context Packet), `component`, `port`, `card`
  (`part.json`), `hop`, `tensor`, `task`, `memory`, `view` (`SubstrateView`, `SlotView`,
  `source_weight`), `class` (sensory and behavioural data classes), `message` (the in-house
  envelope `nm-msg/1`), `settings`, `order`, `project` (per-project memory folders), `doc` (the document model: derived
  text and blocks with heading paths and locators, shared by `nmdoc` and `nmsubs`).
- Functions in `lib.rs`: `nano_root()` (the Nano folder found at run time), `models_dir()` (where
  the released weights are read from) and `tests_dir(category)` (`tests/<category>`; the engine
  never studies `tests/`).
- Every public type has `to_value` and `from_value` over `nmjson::Value`.

Section checks: `neuromorph-nano check nmtype`.
