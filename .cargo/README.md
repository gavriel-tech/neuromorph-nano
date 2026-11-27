# .cargo: Cargo settings for this workspace

`config.toml`: build output in `target/`, incremental cache off, processor flags AVX2, FMA, F16C
(never AVX-512). Read by Cargo on every build in this folder.
