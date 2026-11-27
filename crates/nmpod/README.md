# nmpod

The Neuropod: tentacles, drone tips and territories. Depends on `nmjson`, `nmlex`, `nmtype`; a unit
sees only `SubstrateView` and `SlotView`.

- `unit`: one type for tentacles and drone tips, any of the twelve jobs, three phases (assign,
  sense, process). `tip`: readers (lexical, symbols, semantic, git, build, tests, uses, deep, rules,
  interface). `pipeline`: reflex table from `designs/reflexes.json`, confidence by reader agreement.
- `territory`: non-overlapping regions covering every file. `attention`: demand budgets.
  `recursion`: subordinate requests as question packets to the Core Brain. `pod`: 8 tentacles and
  8 tips by default. `record`, `fixture` (in-memory views for this crate's section checks).

Section checks: `neuromorph-nano check nmpod`.
