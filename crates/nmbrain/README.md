# nmbrain

The Core Brain and the layers of the Globalized Context Stack (GCX). Depends on `nmjson`,
`nmlex`, `nmtype`.

- `slot`: the four slots (router, writer, embedder, reranker) on exact controls, and engine
  binding (`slot::Engine`, `Slots::bind`); `nmapi::neural` binds the released embedder and reranker.
- `arbiter` (statuses, stance, certainty, constraint authority), `contradiction` (Tension Buffer),
  `synthesis` (cited reduce, relevance floor, evidence-kind ordering, withholding), `rivers`
  (bounded mailboxes by kind), `stack` (the context stack as a Neuropod).
- `objectives`, `homeostasis` (reflexes from `designs/homeostasis.json`), `development`
  (repeated findings become anatomy proposals).
- `gmap`: the GCX context map of one question. `g0`, `g1`, `g2`: the GCX layers (tentacle state,
  coordinating loop, upstream selection). `evidence`: the evidence set of the GCX. `chain`:
  MicroModel chains as data. `reflex`, `instruct`, `allocation`.

Section checks: `neuromorph-nano check nmbrain`.
