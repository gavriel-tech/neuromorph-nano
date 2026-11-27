# nmtool

Build help, Step Check and improvement. `run` (child process with a deadline), `lex`, `build`
(rustc diagnostics into packets), `explain` (first error located and explained), `stepcheck` (a
diff applied and tested on a scratch copy in `scratch/`), `improve` (findings and one-line
proposals), `approval` (the proposal queue; nothing is applied here), `verify` (verification
dimensions of a diff, MCP tool `nm_verify`), `conformance` (rules in `designs/conformance.json`,
`nano-lab conform`) and `scoreboard` (`designs/scoreboard.json`, `nano-lab scoreboard`).

Section checks: `neuromorph-nano check nmtool`.
