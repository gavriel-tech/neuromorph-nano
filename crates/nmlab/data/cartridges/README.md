# crates/nmlab/data/cartridges: base cartridge manifests (lab data)

One manifest per base cartridge (`nm-cartridge/1`, `crates/nmnpcs/src/cartridge.rs`): the
population and its base hash, atlas version, job, layer span, mask hash, closeness result, date,
and its packets (cluster, precision, bytes, content hash). A manifest is a record about the
weights: names, sizes and hashes only, never parameter values. Written by `nano-lab run cart`
(steps `f`, `e8`, `p`; `crates/nmlab/src/cart.rs`).

Lab data only. The Neuromorph Nano loads the released models whole, with quantization only, and
uses cartridges only for learned adapter cartridges (`nmlearn::cartridge`); nothing in the product
reads this folder. The released weights stay unchanged.
