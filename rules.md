# Rules for this folder

The Neuromorph Nano's own rules. Gavriel Technologies; designs credited to Luis Jake Gabriel III. `nmsubs::constraint` reads one rule per line under `## rules`; the Nano finds its own folder by the marker file `nano.json` (or `NANO_HOME`), and in a development tree also by this file and `crates/nmcore/Cargo.toml`.

## terms
Neuromorph is a digital organism; the Neuromorph Nano is its first commercial release. Core Brain (singular: one brain; MicroModels are its regions, parameter sets its neurons); tentacle (each one a small brain); drone; Globalized Context Stack, written in full or as GCX, or called the spinal cord or brain stem, and no other short form or name; Global Context Substrate (GCS); Context Packet; muscle memory; cartridge; MicroModel. No em dashes.

## rules
- The released open weights are never altered; they are only read. Quantized copies, cartridges and learned pieces are separate files.
- Reads of the user's code base live in RAM while the code is unchanged; they are never written to disk. Muscle memory is the exception, and the dense index may keep vectors and content hashes (never the material's text) in the Nano's own memory folders.
- Context is never cut to improve a number; it is selected and routed.
- Nothing leaves the user's machine.
- The Nano reads model weights from the models folder (`nmtype::models_dir`) and never writes there.
- No outside crates: the standard library and this workspace's own crates only; `unsafe` is forbidden and `unwrap` is denied.
- Write the Globalized Context Stack in full or as GCX (or the spinal cord or brain stem); no other short form or name is ever used.
- Say MicroModel; describe the Neuromorph as a digital organism; the Core Brain is singular.
- No em dashes in any text.
- Deleting anything is previewed first and confirmed in more than one step.
- Memory of a studied repository is kept per project under `memory/projects/`; the audit log `memory/audit.jsonl` is one file for every project and never holds file contents.
