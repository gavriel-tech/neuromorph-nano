# Cartridges, parameter maps and the cartridge libraries

This document describes the cartridge system of Neuromorph Nano: every kind of cartridge, the
cartridge library on this machine and how cartridges plug in and out, the role of the untouched open weights as
reference and fallback, parameter-level mapping and editing with surgical add-back, versions and
rollback, the official and third-party cartridge libraries, and the longer-term roadmap.

Architecture and directions by Luis Jake Gabriel III (Gavriel Technologies).

For the commands that built the v0.1.0 base cartridges, see `docs/build-flow.md`. For the
interface, see `docs/user-guide.md`, Cartridges.

---

## 1. Core principle: separate weights, structure and specialization

Neuromorph Nano separates a MicroModel's fixed architecture (its assembly of attention,
feed-forward, norm and readout blocks) from its parameter sources and from the cartridges stacked on
top. The four MicroModels are named by role: Reader, Writer, Embedder and Reranker. Each is built
from upstream Qwen3 open weights (Apache-2.0); those open weights are not the MicroModel.

1. **Upstream open weights (`weights/`)**: optional and always read-only when installed, never
   modified. Neuromorph Nano runs standalone on its four base cartridges plus only the small
   `config.json` and `tokenizer.json` files (about 44 MB total). The full 8 GB upstream `.safetensors`
   open weights are optional and used for **fallback** (open mode), **output-parity reference
   checks**, **parameter mapping**, and **cartridge surgery** (donor parameter values when a base
   cartridge needs paths restored). They are never copied whole into a base cartridge.
2. **Base cartridges (`library/cartridges/base/`)**: the parameter set each MicroModel serves,
   mapped and edited for its jobs from the open weights. Base cartridge editing is operational and
   surgical, never continual learning: it serves only each MicroModel's own jobs, is verified
   against held-out checks, and every version can be rolled back.
3. **Cartridges stacked on top** (compensation, job filters, adapters and lenses, learned units,
   preference and engine cartridges): plugged in and out without changing the base parameters.
   All continual learning lives outside base cartridges, in these modules and in the learning
   engine's own state, gated by its supervisor with validation and rollback snapshots
   (`docs/continuous-learning.md`).

---

## 2. Kinds of cartridges

| Kind | Folder | Form and mechanism | Scope and role | Plug and unplug |
|---|---|---|---|---|
| **Base cartridge** | `library/cartridges/base/<role>/` | `kind: "tensors"` (`nmengn::base`) | The complete parameter source of one MicroModel (`reader`, `embedder`, `reranker`, `writer`), mapped and edited for its jobs | Seating checks the sets are complete and connected for the job's assembly, binds the model over them and compares its check outputs with the open weights' on the same input; only then are the open weights released from memory and the card. A refusal names its reason and the open weights keep serving. Pulling the base binds the open weights again |
| **Engine (settings) cartridge** | `library/cartridges/hardware/<piece>/` | `kind: "settings"` (`nmcore`) | Operating numbers for one piece of hardware (for example the GeForce GTX 1650 with 4 GB: precision per model, batch sizes, the attention kernel, how much of each piece the judge reads, when the second look runs), or values for the whole engine. Sets values, not parameters | Matched to this machine's probed pieces (graphics card, processor, RAM, disk); a command-line flag wins |
| **Adapter and lens** | `library/cartridges/adapter/<name>/` | Low-rank additive `y += scale * up(down(x))` after a linear node (`nmengn::adapter`) | Learned from one project's verified places (for example `embedder-lens-*`, `judge-code-*`); also the form for specializing a base cartridge for a job | Applied unmerged; bound to the exact base it was fitted on; loads only for its project; unloading returns the base outputs exactly |
| **Compensation and job filter** | `library/cartridges/learned/<name>/` and the job filters built from parameter maps | `kind: "affine"`, per channel `y = x * gain + bias` at named sockets (`L27.swigate`, `L20.o`, `final_norm`) (`nmengn::filter`) | Representation editing in the (IA)3 and RED style. Compensation adds back what one base cartridge version leaves out and fits only that version; a job filter serves one job over some of a MicroModel's layers | Stacked filters at one socket are folded in float64 into one gain and bias, so a socket costs one pass however many filter it; swapped between calls with no restart; with gain 1 and no bias a channel returns its input bit for bit |
| **Preference** | `library/cartridges/preference/<name>/` | Preference overlay | Your way of working, from edits you kept or reverted, corrections and rules you stated | Always plugged last, so your way wins |
| **Learned units** | the learning stores (`learned/` in each project's memory folder, `units/` in the Nano folder; plugs in `units/plugs.json`) | Lenses, read-outs, learned pieces, learning units | What the learning cycle keeps, grouped by the compartment it was fitted in (the Nano itself, shared, or one project) | Each unit has its own plug and each compartment a switch for all its units (Cartridges page, Learned; MCP `nm_learning` `op: unit_on|unit_off|compartment_on|compartment_off`) |

The learning engine's own state is kept as six versioned records per project in
`<project memory>/continual/` (`clc-knowledge`, `clc-procedural`, `clc-behavioral`,
`clc-retrieval`, `clc-interaction`, `clc-evaluation`; `nmlearn::engine`). They are not weight files.
Each moves through `observed`, `candidate`, `under_evaluation`, then `validated` or `rejected`, with
rollback snapshots in `continual/snapshots/`.

---

## 3. The cartridge library on this machine and how cartridges plug in and out

`library/` holds only the libraries: the parameter map library and the cartridge library, whose
checked catalogue lists every cartridge and stack. This machine's stack state is kept beside the
Nano's memory, outside the libraries:

```text
library/
  parameter-maps/
    reader/                   # per-job parameter maps of the Reader
    embedder/                 # job activation and parameter-level maps of the Embedder
    reranker/                 # role and per-weight maps of the Reranker
    writer/                   # job maps and active output vocabulary rows of the Writer
  cartridges/
    catalogue.json            # every cartridge, version, parent, provenance and stack, with hashes (nm-library/1)
    base/                     # base cartridge versions per MicroModel
    hardware/                 # engine (settings) cartridges per hardware piece
    adapter/                  # project adapters and lenses, base specialization adapters
    learned/                  # affine filters and compensation cartridges
    preference/               # preference cartridges (plugged last)
    set-aside/                # edited versions set aside by an undo (nothing is deleted)
memory/stacks/
  machine.json                # this machine's probed pieces (graphics card, processor, RAM, disk)
  in-use.json                 # what a serving process plugged at its start: bases, stacks, sockets, refusals
incoming/                     # files fetched from an online cartridge library, held until they pass the checks
```

An installation made before this layout is moved on its first start: the records it kept loose
at the root of `library/` go to the places above, and nothing is deleted.

### The base cartridges in v0.1.0

The serving version of each base cartridge is listed, with its SHA-256 and parent version, in
`designs/base-cartridges.json`:

| MicroModel | Built from | Serving base cartridge | Size on disk | How it was made |
|---|---|---|---|---|
| Reader (also the drones and tentacles) | `qwen3-0.6b` | `reader-base@3` | about 1,137 MiB | Parameter-level map with relevance heads pinned; parameters left out across the attention and feed-forward tensors at full shapes, then parameters restored from the open weights across all 28 layers |
| Embedder | `qwen3-embedding-0.6b` | `embedder-base@4` | about 1,136 MiB | The same two steps, no compensation needed |
| Reranker | `qwen3-reranker-0.6b` | `reranker-base@5` | about 1,136 MiB | The same two steps, no compensation needed |
| Writer | `qwen3-1.7b` | `writer-base@4` | about 3,282 MiB | The same two steps over the Writer's jobs (notes, summaries, file notes, outlines), no compensation needed |

The files are stored at 16 bits (BF16) with the open weights' full shapes, so the zeroed parameters
take space on disk; on the card, 8-bit linears with enough exact zeros are compressed
(`bmq8.cu`, below). Each label records the version's parent, its hashes and its measured seat check.

### Plug and unplug rules

- **Integrity on every load**: each cartridge file is checked against its record in
  `library/cartridges/catalogue.json`; a parameter cartridge is hashed when it is seated.
- **Base identity**: an adapter or affine filter made for base `X` is refused by name over base
  `Y`. A cartridge made for another MicroModel or other open weights is refused too.
- **No restart**: a serving process watches the catalogue and applies a stack switch (Cartridges
  page, `POST /api/cartridges`, `neuromorph-nano library switch`, MCP `nm_cartridges`) where its
  next call starts. A base cartridge version applies from the next start.
- **Exact unplug**: unplugging an overlay returns the base outputs exactly; switching a MicroModel
  to `open` serves its open weights.
- **Job filters and their base**: each job filter records the base cartridge it was built on.
  `/api/surgery` and MCP `nm_cartridges` `op: filters` report `built_on`, `serving_base` and
  `base_match`; a mismatch is reported only, and the filter loads either way.

---

## 4. From coarse channel removal to parameter-level mapping and editing

### Why coarse channel removal hits a ceiling

Cutting whole feed-forward channels removes `3 x hidden_width` weights at once (3,072 weights per
channel in the Reader's 0.6B open weights). On a MicroModel with several jobs, like the Reader
(drones, tentacle integration, joint reads, yes/no checks, request reads), middle-layer channels are
shared across jobs, so few whole channels can go; yet inside those shared channels many single
parameter paths are unused by every job. The first base cartridges trimmed whole channels; the
v0.1.0 ones are edited at the level of single parameters.

### Parameter-level mapping, both ways

A parameter map is the coordinate system for both removing and adding parameter paths in a
cartridge. For each MicroModel and each job `t`:

1. **Per-job calibration**: real job prompts (`D_t`, split into a map half and a held-out half)
   run through the untouched open weights.
2. **Per-weight importance** (Wanda-style, with an optional Fisher read-out score):
   - the activation norm `||x_j||_t` of every input feature of every linear (`q_proj`, `k_proj`,
     `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), plus which token rows a job uses;
   - per-weight job importance `s_t(w_ij) = |w_ij| * ||x_j||_t`;
   - layers above a job's execution depth get no keep votes from that job;
   - the tentacle relevance heads are pinned and protected.
3. **Layer budgets**: OWL layerwise sparsity (`pmap::owl_layer_sparsities`) gives each layer a
   share inversely to its activation outlier ratio, so sensitive layers stay denser while the mean
   stays on target.
4. **Surgical operations over masks**:
   - **Removal** (`M_next = M_current AND NOT M_unused`): parameter paths no job uses are left out.
   - **Addition** (`M_next = M_current OR M_target`): exact parameter paths a new job needs, or that
     recover a job after too much was removed (`Delta = M_target AND NOT M_current`), are copied from
     the untouched open weights into the cartridge as version `N+1`.

### Making cartridges lighter and faster on a 4 GB card

The GeForce GTX 1650 (4 GB, Turing `sm_75`) has no sparse tensor cores, so parameter-level editing
combines several mechanisms:

1. **Compressed 8-bit weights on the card** (`kernels/cukern/bmq8.cu`): per-weight bitmasks,
   64-weight group offsets, row offsets, packed `int8` values and `f16` row scales, decompressed
   with `__popc()` inside the kernel. One-token decode runs straight over the compressed weights;
   longer prefill unpacks into one pooled buffer and uses the existing 8-bit linear kernels.
   `nmengn::model` uses it for any 8-bit linear with at least 20% exact zeros (`NANO_BMQ8=off`
   turns it off); the cartridges stay 16-bit on disk.
2. **Map-guided structured edits**: the per-weight map scores finer structured units (feed-forward
   sub-blocks, input and output rows) more accurately than channel energy alone.
3. **Job-conditioned token rows**: the Writer's idle notes load only the active rows of its output
   head (the workspace's tokens, the 16,384 most frequent BPE tokens and the end token) instead of
   all 151,936 rows; `NANO_WRITER_FULL_HEAD=on` loads them all.
4. **Bias compensation and specialization adapters**: per-output mean-activation compensation
   (`b_i += sum_{removed j} w_ij * E[x_j]`) stored as a small separate affine cartridge, and low-rank
   adapters for targeted recovery or specialization. The v0.1.0 base cartridges need no
   compensation.

### Surgical add-back when too much was removed

If a new base cartridge version `vN` falls short on a held-out job `t`:

1. Do not discard it or start the cut again.
2. Map the untouched open weights on the failing job cases.
3. Compute the missing parameter paths: `Delta = M_failing AND NOT M_current`.
4. Copy only those parameter values from the open weights into the cartridge to make `v(N+1)`,
   recording its parent, the paths added and its new SHA-256.
5. Check it again on the held-out cases before it serves.

### Reading and editing a cartridge's parameters

Parameter maps describe the open weights; cartridge parameters are the learned values an object
holds. Both are on the Parameters page and form one system: a map says where a job's parameters
are, and the parameters view shows what a cartridge changes there. Parameters are not tied to the
cartridge form: a job filter file, a project adapter in the adapter library and a learning unit hold
parameters and are read and edited the same way, while an engine (settings) cartridge holds none.

One read format, `nm-parameters/1` (`crates/nmcore/src/bin/cparams.rs`), covers every object:

| Object | Tensors | Edits |
|---|---|---|
| Socket cartridge (`nmengn-sockets/1`, any family) and job filter | `x * gain + bias` per point: `<node>.gain` (identity 1) and `<node>.bias` (identity 0), layer from the node id | Strength, layers (off drops the point), values |
| Embedder lens (adapter library) | `flat [k, d]`, `beta [k]`, `down [r, d]`, `up [d, r]` (identity 0) | Strength on `beta` and `up`, values; no layers |
| Judge code read-out (adapter library) | `w [d]` | Strength, values; no layers |
| Learning unit | `down [rank, k]` (fixed), `up [n, rank]` (the change), at the target node's layer | Strength on `up`, its layer (off zeroes `up`), values |
| Preference cartridge | `[good, signals]` per `pref.*` key | Set as values, not here |
| Base cartridge | Every parameter set by name, layer, block, shape and type; values of parts up to 256 KiB, larger parts paged from the file | Every operation on a selection of at most 32 Mi values; written as a new base version beside the others, on trial, checked against the open weights when seated |
| Settings cartridge | None | Edited as values |

Per tensor the section gives the L2 norm, the largest magnitude, `effect_l2` (the norm of the
difference from the identity, so a strength of 0 gives 0), and a summary (minimum, maximum, mean
and an 8-bin histogram); per layer, the summed `effect_l2`. Values are paged, at most 4,096 per
read, and a large base part is never loaded whole. Another builder exposing a new kind of
parameters (for example distilled cartridges) adds it to `held` in that module, as a list of
`Tensor` rows with their identity values and an `Origin` that writes its next version; the
format, the list, the edits and the interface then cover it.

An edit is a strength `s` (every scalable tensor becomes `identity + s * (x - identity)`), per-layer
switches (`off`, `on` or a scale), and direct values for tensors of at most 65,536 elements, shape
checked. It always makes a new version beside the source, through each object's own write path:
the library's `add_cartridge` (a new version on trial; stacks are not changed), a new job filter
file plugged out, the next version in the adapter library, or a new candidate unit in the registry.
The source is never written, and neither are the open weights. The new version's compatibility is
checked (`/api/cartridges/compat` for library cartridges and job filters, the shape and values-key
check for an adapter, the base hash for a unit) and the edit is one `cartridge-parameters` audit
row, which Audit undoes by setting the new version aside, deleting nothing.

#### Selections and operations

Every edit, from the interface, MCP or the API, is a list of operations on selections, run by one
engine (`crates/nmcore/src/bin/pedit.rs`) for every kind.

A selection names tensors (all, one by name, the tensors at a layer, at a target node, or of one
part such as `gain`) and, inside each, elements: a flat range, one index, a block of rows and
columns of a matrix, one attention head (`head` with `head_dim`), or one channel (a column of a
matrix, an element of a vector). A path gives the same in one string:

| Path | Selects |
|---|---|
| `*` | The whole object |
| `L3` | Every tensor at layer 3 |
| `L3.o.gain` | One tensor |
| `L3.o.gain[4]`, `L3.o.gain[0:16]` | One element, a range |
| `down[2:4, 0:8]`, `down[2:3, :]`, `down[:, 5:6]` | A block, a row, a channel |

| Operation | Effect on the selected elements |
|---|---|
| `set`, `add`, `mul` (`value`) | Set, add, multiply |
| `scale` (`value`, 0 to 4) | Toward the identity: `identity + s * (x - identity)`; without a named tensor, only the tensors that carry the change |
| `clamp` (`lo`, `hi`), `zero`, `identity`, `negate` | Clamp; set 0; set the value that changes nothing; negate |
| `paste` (`values`) | Values in selection order, as many as selected |
| `import` (`file`) | A paste from a values file (`nm-parameter-values/1`, as an export writes, or `{"values": [...]}`) |
| `copy`, `interpolate` (`source`, `t`) | From another object (`{"key": ...}`: another version, another cartridge) or a job filter map (`{"map": "<role>/job-filter-<job>.json"}`) at the same place; interpolate moves `t` (0 to 1) of the way |
| `drop` | Removes the selected points (a list of points only) |
| `add_point` (`target`, `width`) | Adds a point (gain 1, bias 0); a node the MicroModel's assembly does not have is refused |
| `rank` (`rank`) | Truncates a low-rank pair (`down` keeps its first rows, `up` its first columns) |

`name` and `description` label the new version. Every operation is checked against the tensor's
shape (a selection outside it, a paste of the wrong length, a source of another shape are refused
by name) and every value must be finite. `preview: true` returns the difference and writes nothing:
per tensor the values changed, the L2 norm and the norm of the change before and after, the
largest change and a sample of old and new values, and the tensors added or dropped.

What each kind takes, as a new version and in its overlay (below):

| Kind | Values and ranges | Strength, layers | Structure (new version only) | Label | Overlay (edit, mask, freeze) |
|---|---|---|---|---|---|
| Open weights (`open:<model>`) | Overlay only: every operation but structure, by region (read by page, never whole) | Strength toward 0 (a norm weight toward 1); a layer masked is an exact identity | No | No | Yes; the only way to change them |
| Base cartridge | Every operation on a selection of at most 32 Mi values, written as a new base version on trial | Strength and layers as values | No | Yes | Yes; applied after its seat check |
| Socket cartridge, job filter | Yes | Yes (a layer off drops its points; in the overlay it zeroes its change) | Drop and add points | Yes | Yes |
| Embedder lens, judge read-out | Yes | Strength; no layers | Rank (lens) | Yes | Yes |
| Learning unit | Yes | Strength and its layer (off zeroes `up`) | Rank | No (named by its id) | Yes |
| Preference counts | Whole counts, good at most signals; written through the cartridge's values | No | No | No | Yes (rounded to whole counts; masked: no evidence) |
| Operating settings | Holds no parameters; edited as values | No | No | No | No |

Rank is reduced to the best approximation of that rank of `up * down` (a truncated singular value
decomposition through the small `r x r` problem: the two factors' QR, then a one-sided Jacobi
decomposition, in float64), each new factor carrying the square root of the kept singular values.
`interpolate` with `source` and `source_b` blends any two versions (`source + t * (source_b -
source)`); with `source` alone it moves the values here toward it. A copy, interpolation or blend
reads a whole tensor of another object from its file where it is above the read cap, up to 16 Mi
values.

#### The parameter overlay

Besides a new version, every object that holds parameters takes edits, masks and freezes held in
its parameter overlay, `memory/overlays/<key>-<hash>.json` (`nm-overlay/1`,
`crates/nmengn/src/overlay.rs` for the format and the replay, `crates/nmcore/src/bin/overlay.rs` for
the surface). The upstream open weights take edits only this way: `weights/<model>/files/*.safetensors`
and every cartridge file stay as they are, and the engine applies the overlay when it seats the
object. An overlay is an ordered list of entries:

| Entry | What it holds | Replay |
|---|---|---|
| `edit` | One operation (`set`, `add`, `mul`, `scale`, `clamp`, `zero`, `identity`, `negate`, or `paste` with captured values) on resolved regions: a flat range, or rows and columns of a matrix, per tensor | In entry order, each result rounded to the type the engine holds the values in (BF16 for the open weights and base cartridges) |
| `mask` | Regions whose contribution is off, set to their identity: 0 for a weight and a bias, 1 for a gain and for a norm weight of the open weights or a base cartridge; `whole` for the object | After every edit, whatever the edits did: unmasking returns the edited values exactly, since the originals stay in the files |
| `freeze` | Regions made read-only | Changes no value; an edit, mask, unmask or clear touching it is refused with its reason until `unfreeze` |

A copy, interpolation, paste or import is kept as the values it gives (at most 65,536 per region), so
it replays the same whatever happens to its source; the other operations are pure functions of the
released values. Structural operations stay with new versions.

Where the engine applies it:

| Object | Applied in | Takes effect |
|---|---|---|
| Open weights | `nmapi::neural`, when the MicroModel's population opens and the open weights serve it: each touched tensor read from its file, patched in a copy in RAM and offered to the provisioner under the copy's content hash; outputs kept on disk are keyed `open weights <sha> with overlay <id>` | The next call: a serving process binds the MicroModel again with its overlays where the call starts (`settle_overlays`); the Writer at its next load |
| Base cartridge | After its seat check against the open weights (the cartridge as written), the same patch over its parts; masked whole, it is left out where offered | The next call, as for the open weights |
| Job filter | `nmengn::filter::load_enabled_job_filters`, at each forward pass (checked once a second, the overlays' stamp in its cache key); masked whole, it is left out while plugged | The next call |
| Socket and adapter-family cartridge | The rack (`library::rack_of`), restaged by the serving process's catalogue watch, which also watches the overlays | The next call |
| Lens, judge read-out | `nmapi::adapter`, loaded again at the next look after an overlay changes; masked whole, left out (no earlier version takes its place) | The next call |
| Learning unit | `nmapi::learn::load`, and for a unit loaded on the Reader `learn::apply_units` where each call starts; masked whole, it is taken off its node | The next call |
| Preference counts | The served preference values (`library`), re-read by the catalogue watch | The next call |

A running server binds a MicroModel again when the overlays that decide its seat change (its open
weights', and those of every base cartridge offered to it), where its next call starts: a bound
model holds packed copies of every tensor made at bind for its precision (8-bit, 4-bit and 16-bit
forms, and their card copies), so rewriting a touched tensor in place would have to rebuild those
for every form; the whole MicroModel is bound again instead, the slots that use it are bound to the
new model, and the old one is dropped when the calls running on it end. The time it took is
recorded on this machine (`memory/overlays/state/reloads.json`) and stated in the reply. When the
Embedder is bound again with other served values, the dense lane follows without a restart: its
vectors, whose header names the values that made them, are marked stale, the meaning ranking is
withheld (questions run on the lexical and structural paths), and a new index is built with the
new Embedder at idle, preempted by calls as the idle Writer is, then switched in at once when it
covers every current piece (see the user guide, The parameter overlay). Lab measurements and
builds open every population with its overlay by default and say so in their first line;
`--no-overlay` reads the released files. Their reports under `runs/` record it in a line
`overlay: applied (<n> objects)`, `overlay: not applied (--no-overlay)` or `overlay: none present`.
Every change is one `parameter-overlay` audit row naming the entries it added and removed and a
snapshot of the overlay before it (`memory/overlays/history/`); undo takes out what was added and
puts back what was removed at its place.

#### Mapping beside the parameters

The parameter maps relate to a cartridge's parameters on the same layers and nodes. For any object,
`maps` lists the maps of its MicroModel that relate to it, each with the reason: its lineage says
it was measured on the object or the base the object was built on, or names the object among what
it was used by; it is a job filter at the same points; or it holds values on the same layers (keys
`L<n>`, read for maps up to 4 MiB). A region read returns, beside the selected values, the values a
job filter map holds at the same point and indices and a per-layer map's value at the tensor's
layer.

---

## 5. Versions, upgrades and rollback

- Every base cartridge version is labelled with its parent, its SHA-256, the SHA-256 of the open
  weights it was made over, and its measured seat check.
- An upgrade adds new base cartridge versions to the cartridge library (`library merge`, run by the
  installer). The control `library.upgrade` decides what happens to the older versions: `keep`
  (the default) keeps them beside the new one, `replace` removes them once the new one is in the
  cartridge library and usable. A version you chose for a MicroModel is never removed.
- Choose which version serves, per MicroModel, on the Cartridges page (Base versions), with
  `neuromorph-nano library base <model> <key|auto|open>`, or MCP `nm_cartridges` `op: base_serve`.
  `auto` (also what serves when nothing is chosen) is the newest usable version; `open` serves the
  open weights.
- `neuromorph-nano library rollback <model>` goes back to the version the serving one was made
  over, when that version is in the cartridge library. Both apply from the next start and are logged in the
  audit, with undo.
- `library base-remove <key> --confirm` (or the Cartridges page) deletes one version that is
  neither serving nor chosen. A missing serving version can be downloaded again from the official
  library (Models page, Downloads).

---

## 6. Cartridge libraries: official and third party

Neuromorph Nano takes cartridges and parameter maps from the official library on Hugging Face,
[`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)
(built in, public, no token required), and from any third-party Hugging Face library you add by its repository name (Models
page, side panel, Libraries; MCP `nm_settings` `op: sources|source_add|source_browse|source_fetch|source_fetch_all|source_publish|source_remove`).
The Qwen3 open weights are the optional upstream fallback, from Downloads. The official library is under
the same PolyForm Noncommercial License 1.0.0 as the Nano (`LICENSE` at root), with the upstream Apache-2.0 notices
at `NOTICE.md` and `third_party/APACHE-2.0.txt`.

- **Nothing is fetched or uploaded until you ask.** Browse lists a library's files (one request to
  huggingface.co); fetch downloads one file (and `fetch all` downloads every active parameter map and non-base stack) through the same downloader as the base cartridges (its
  SHA-256 checked when the library publishes one, partial files resumed) and installs it through
  the checks of the cartridge library on this machine. The air gap blocks browsing, downloading and publishing.
- **What a library can share, and how it is checked:**
  - stacks: the Nano's own stack export (`neuromorph-nano library export`), under `stacks/` or named
    `<name>.stack.json`; every file and the whole body carry hashes; an imported stack and its
    cartridges arrive on trial and serve only once their evidence allows;
  - parameter maps and job filters: under `parameter-maps/<role>/` (`reader`, `writer`, `embedder`,
    `reranker`); each is checked by its kind and never replaces a map you have;
  - base cartridges: from the official library only (a third-party library's base
    cartridges are listed, not installed);
  - archived / decommissioned versions: superseded base cartridges, parameter maps, job filters, and stacks are stored under `archived/` mirroring the active layout (`archived/base/<role>/`, `archived/parameter-maps/<role>/`, `archived/stacks/`). The downloader automatically ignores `archived/` (and `archive/`) so users only ever see healthy, current, up-to-date downloadable content with no duplicates or decommissioned versions.
- **Publish to Hugging Face (`source_publish`):** From **Models -> Libraries**, API `POST /api/cartridge-sources` (`action: "publish"`), or MCP `nm_settings` (`op: "source_publish"`), you can upload a local parameter map or non-base stack directly to your own Hugging Face repository or open a Pull Request against `gavriel-tech/neuromorph-nano-cartridges` using native WinHTTP `POST`. Publishing is confirmation-gated, audited, blocked in Air-Gap Mode, and prints exact `hf upload` command guidance for files larger than 10 MiB that need Git LFS.
- **Your Hugging Face token** (`HF_TOKEN`, or the `hf` login) is not sent for public official library reads by default; it is sent only to `huggingface.co` when you explicitly publish or tick "send my token" on a private or gated third-party library you trust.
- **The list of sources** is `cartridge-sources.json` in the Nano folder. Adding or removing a source
  is logged in the audit and can be undone; removing one goes through a preview and a confirmation
  and keeps whatever was already imported from it.
- **Contributing to the official library:** see `CONTRIBUTING.md`, Contributing weight
  cartridges, parameter maps and techniques. Accepted cartridges are published to the official
  library by the maintainers.

---

## 7. Stacking order

- **Order of authority** (`nmlearn::library::family_rank`): base, then hardware and settings, then
  adapters (base specialization and project), then learned (job filters, compensation,
  fundamentals), then preference.
- The design rule for combining (settled 2026-11-17; v0.1.0 applies cartridges in the order
  above): where stacked cartridges can combine at a read-out, evidence is added in log-odds,
  weighted by each cartridge's validated record for the scope; where they cannot, the order
  decides. A preference cartridge is meant never to change which facts or source lines are found.
- When stacks for several uses apply together: general first, then hardware from the most general
  piece to the most specific, then project, then job, and preference last.

---

## 8. Roadmap: learning the parameter system

The parameter-level pipeline above is the first of three phases. The later two are plans, not
built in v0.1.0:

1. **Phase 1 (v0.1.0): parameter-level map, edit, add-back and base specialization.** Per-job
   parameter maps, parameter-edited base cartridges, surgical add-back from the open weights,
   specialization adapters, versions and rollback.
2. **Phase 2 (planned): a parameter-to-outcome atlas.** Structured records linking parameter
   paths, subspaces and adapter ranks in a job's context to measured outcome changes, across every
   cut, add-back, adapter and learning cycle; circuits and subspaces in the parameter map library organized by
   capability, job and hardware.
3. **Phase 3 (long term): direct parameter printing.** A predictive model of the parameter system,
   so Neuromorph can synthesize parameter paths, low-rank adapter matrices and affine gains and
   biases directly into labelled cartridges for a target behaviour, gated by the same checks and
   supervisor validation.

One part of the data collection already runs: with continuous learning on, every served context
call logs a compact row (task, retrieval weights, active cartridges, outcome, tokens, time) to
`<project memory>/continual/param-telemetry.jsonl`.
