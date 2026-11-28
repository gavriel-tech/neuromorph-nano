# Dead-Skin Biomimetic Vector Store

Architecture and concept by Luis Jake Gabriel III (Gavriel Technologies).  
Implementation: `crates/nmsubs/src/dense_store.rs` and `crates/nmsubs/src/dense.rs`.

## Overview

A local dense semantic index for an active software repository faces a recurring maintenance problem: as files are edited, refactored, reverted, or switched across Git branches, old chunk vectors become stale ("dead rows"). Conventional vector stores either leave dead rows accumulating indefinitely or pause to bulk-rewrite the entire vector file once tombstone ratios cross a fixed threshold.

Neuromorph Nano's dense meaning index (`<project memory>/dense/`) solves this through **epidermal desquamation and barrier homeostasis**, modeled on how biological epidermis manages non-viable cells:

- Living epidermis never undergoes catastrophic bulk sloughing; it maintains **continuous micro-desquamation** in small, bounded batches.
- The outer non-viable layer (the **stratum corneum**) remains structurally intact as a functional **protective barrier** before detachment.
- Turnover is regulated by **enzymatic age-calibrated kinetics** (analogous to proteolytic desmosomal cleavage loosening only the oldest outer cells) rather than arbitrary fixed-ratio flushes.
- Renewal proceeds via **basal-to-apical chronological stratification**, appending new layers at the base and detaching fully expired outer strata whole.

## Five Core Mechanisms

| Mechanism | Bioengineering Principle | Implementation in `dense_store.rs` |
|---|---|---|
| **1. Continuous micro-desquamation** | Gradual bounded cellular shedding instead of disruptive bulk compaction | Each write/sync sheds at most `SHED_BATCH = 64` expired dead rows, touching at most the affected segment instead of rewriting the full store. |
| **2. Stratum corneum barrier / undo layer (`skin.json`)** | Non-viable outer layer serves a functional barrier before detachment | Recently dead rows remain on disk (tracked in `skin.json` by opaque 64-hex chunk content hash and 16-hex file-path digest only; **zero code text and zero file paths**). When an undo, `git checkout`, or branch switch restores an earlier chunk, its vector is **revived immediately with zero GPU re-embedding**. |
| **3. Enzymatic age-calibrated turnover kinetics** | Proteolytic turnover timed to observed tissue-revival dynamics | Edited-file retention TTL (`ttl_edit_ms`, default 7 days, bounded between `MIN_TTL_MS = 10 min` and `MAX_TTL_MS = 30 days`) auto-calibrates once `CALIBRATION_MIN_DEATHS = 16` chunk deaths are observed, using `p90 * 2` of the repository's own sliding window of revival ages (up to `MAX_REVIVAL_SAMPLES = 256`). Low-revival repos shrink toward 10 minutes; undo- and branch-heavy repos expand up to 30 days so the barrier pays for its bytes. |
| **4. Basal-to-apical stratified segments (`NMDENSE1`)** | Basal renewal with whole-layer apical detachment | New vectors append at the newest basal segment (`FLUSH_ROWS = 128`, `SEGMENT_ROWS = 512` per file: `vectors.bin`, `seg-000001.bin`, ...). Whenever every row in an oldest outer segment expires, the entire segment file detaches in `O(1)` via `remove_file` with **zero bytes rewritten**. |
| **5. Accelerated clearance of detached tissue** | Rapid clearance of non-returning detached structures | Rows whose 16-hex file digest disappeared from the workspace receive `1 / 4` of the edited-file TTL (`DELETED_TTL_DIV = 4`, floor `MIN_DELETED_TTL_MS = 2 min`), shedding four times faster than edited-file rows. |

## On-Disk Format, Privacy, and Dynamic Embedder Following

Each segment uses the atomic `NMDENSE1` binary format (written to `<file>.tmp`, synced, and renamed over the target):

- **Header:** sorted `key=value` metadata naming the Embedder's weight hash, precision, arithmetic switches, dimension (`dim`), row count (`rows`), and embedding recipe (`recipe` / `parts_within`, or a per-row `row_recipe` table when mixed recipes coexist).
- **Integrity:** SHA-256 checksum of the binary body (64 hex characters).
- **Body:** `rows` packed records, each holding a 64-byte opaque content hash and `dim` little-endian `binary16` (`f16`) values (plus 1 recipe-index byte when `row_recipe` is active).
- **Zero code on disk:** Neither the segment files nor `skin.json` ever store source code, prompt text, or file paths.
- **Automatic Embedder-change following:** Because every segment header stamps the serving Embedder's weight hash and live parameter overlay id, switching base cartridges or editing Embedder weights via a live parameter overlay automatically marks older vectors stale, falls back safely to lexical and structural retrieval during interactive calls, and rebuilds the dense index at idle before swapping in atomically with zero restart.
