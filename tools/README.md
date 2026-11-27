# tools: developer scripts

- `verify.ps1`: the release checks in order, stopping at the first failure (build, clippy, section
  checks, cards, anatomy, designs, integrity, MCP smoke test, engine self-test, interface). Builds
  into `scratch/gate/target` and stops no process. Card steps (engine self-test, the `nmengn` and
  `nmffi` section checks) run only with `-Card` and a free card; otherwise they are listed as
  skipped. `-SkipInterface` leaves out the interface steps.
- `mcp-smoke.ps1`: starts `neuromorph-nano serve` on a fixture repository and checks `initialize`,
  `tools/list` and one cited `nm_context_pack` over standard input and output.
- `gen-cards.ps1`: writes `crates/<crate>/part.json` for every library crate; then
  `neuromorph-nano seal crates/*/part.json` and `neuromorph-nano check cards`.
- `pack-compare.py`: compares context pack sizes between reply forms.
