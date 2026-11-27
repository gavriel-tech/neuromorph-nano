# Third-party notices

Neuromorph Nano is source-available under the PolyForm Noncommercial License 1.0.0 (`LICENSE`).
It uses the third-party components below; each keeps its own licence.

## Models

The upstream models are Qwen3 open weights, used exactly as released: read only, never altered,
never stored in git. They are downloaded into `weights/<model>/files/` from the Hugging Face
revision each weight card pins (see `docs/user-guide.md`); each card records the SHA-256 of the
files, and `INTEGRITY.md` lists one SHA-256 per file.

| Model | Author | Licence | Source | Pinned revision |
|---|---|---|---|---|
| Qwen3-0.6B | Qwen team, Alibaba Cloud | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3-1.7B | Qwen team, Alibaba Cloud | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen3-Reranker-0.6B | Qwen team, Alibaba Cloud | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Qwen3-Embedding-0.6B | Qwen team, Alibaba Cloud | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |

The Apache License 2.0 text is distributed with each model (`LICENSE` in the model's files).

## Cartridge library

The official cartridge library (`gavriel-tech/neuromorph-nano-cartridges` on Hugging Face) holds
base cartridges, parameter maps and job filters derived from the models above. The library is
licensed under the PolyForm Noncommercial License 1.0.0, the same licence as the Nano, with
commercial use only under an individually negotiated licence (`COMMERCIAL.md`). The upstream
Apache-2.0 notices ship beside it (`NOTICE.md` naming each model and its pinned revision, and a
copy of the Apache License 2.0 as `LICENSE-APACHE-2.0`).

A third-party cartridge library you add as a cartridge source keeps its own licence; check it
before you use what you fetch from it.

## Engine

The Rust engine uses no third-party crates: `Cargo.lock` lists only the 21 workspace crates
(`nmapi` to `nmtype`, including `nmmusc` and `nmtool`), and the engine is built on the Rust
standard library. At run time it loads the NVIDIA driver (`nvcuda.dll`) already installed on the
machine; the driver is not distributed with the Nano. Downloads you ask for are made through
Windows' own HTTPS service (WinHTTP), which is part of Windows (with `curl.exe`, also part of
Windows, as the fallback); neither is distributed with the Nano. The CUDA kernels in
`kernels/` are the Nano's own code, compiled once with the NVIDIA CUDA toolkit into the
`kernels/build/*.fatbin` files; the toolkit is not distributed with the Nano.

## Interface

The interface (`interface/`) is built with npm. Only its static export (`interface/out/`) is
served; Node.js is not needed to run it.

Direct dependencies, from `interface/package.json` and `interface/package-lock.json`:

| Package | Version | Licence | Used for |
|---|---|---|---|
| next | 16.4.6 | MIT | Build and runtime |
| react | 19.3.0 | MIT | Runtime |
| react-dom | 19.3.0 | MIT | Runtime |
| tailwindcss | 4.3.3 | MIT | Build |
| @tailwindcss/turbopack | 4.3.3 | MIT | Build |
| typescript | 5.9.3 | Apache-2.0 | Build |
| eslint | 9.39.5 | MIT | Development |
| eslint-config-next | 16.4.6 | MIT | Development |
| @types/node | 20.19.43 | MIT | Development |
| @types/react | 19.3.0 | MIT | Development |
| @types/react-dom | 19.3.0 | MIT | Development |

All packages in the lock file (431), by licence: MIT 339, Apache-2.0 34, ISC 16, MPL-2.0 13,
LGPL-3.0-or-later 10, BSD-2-Clause 7, Apache-2.0 AND LGPL-3.0-or-later 3, BSD-3-Clause 2,
BlueOak-1.0.0 2, Apache-2.0 AND LGPL-3.0-or-later AND MIT 1, CC-BY-4.0 1, CC0-1.0 1, 0BSD 1,
Python-2.0 1.

The LGPL-3.0-or-later packages are the optional `sharp` and `libvips` image binaries that Next.js
can use at build time. The interface sets `images.unoptimized` and is a static export, so they are
not part of what the Nano serves.

The served pages include code from next, react, react-dom and their runtime dependencies
(`@next/env`, `@swc/helpers` Apache-2.0, `scheduler`, `styled-jsx`, `client-only` MIT, `tslib`
0BSD). Their licence texts are in `interface/node_modules/<package>/LICENSE` after `npm ci`.
