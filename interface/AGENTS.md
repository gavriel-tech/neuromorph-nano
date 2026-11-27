<!-- BEGIN:nextjs-agent-rules -->

## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## Rules for agents working on this interface

The interface of the Neuromorph Nano: a Next.js static export served by `neuromorph-nano ui` on
this machine only. These hold for every agent and every contributor.

- **Every page is an interactive system.** Use `src/components/Workspace.tsx`: the page nav on the
  left (Shell), the working system in the middle (live visuals and controls, no paragraphs), and the
  right rail for libraries, references, descriptions (the About tab) and the detail of whatever is
  picked (`useRail().show`). Descriptions stay mostly hidden: tooltips or the rail.
- **Real data only.** Every figure comes from the engine's API. A figure that was not measured is
  named as not measured, never given a number. Nothing is animated as activity unless it is.
- **One feature, three surfaces.** Every feature is in the interface, MCP and the API, in sync.
- **Names.** MicroModels are named by their role (Reader, Writer, Embedder, Reranker) through
  `src/components/MicroModel.tsx`: a filled violet square marks the base cartridge, a hollow ring the
  upstream open weights it was built from. The upstream weights are never called the MicroModel.
- **Cartridges are visual and plug and play**: decks per MicroModel, the library beside them,
  compatibility shown, decks accept only what fits.
- **Deleting** goes through `DeleteFlow` (a preview of what goes, then more than one confirmation).
- **Style**: the established dark instrument-panel style (`src/components/ui.tsx`, `globals.css`
  tokens), hairlines, monospace figures, no em dashes, motion off under reduced motion.
- **Checks**: `npm run lint` (0 problems), `npm test`, `npm run build`, and the end-to-end script
  `scripts/e2e-ui-verify.mjs` against a built binary. Keep the `data-e2e` attributes it uses.
