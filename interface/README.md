# interface: Neuromorph Nano Local Observability UI

The 11-page local web interface for Neuromorph Nano (Next.js, React, TypeScript, Tailwind CSS), exported as static files (`interface/out/`) and served directly by `neuromorph-nano ui` at `http://127.0.0.1:4870/`. Node.js is only required to build or develop the interface; running Neuromorph Nano requires no Node.js runtime.

## Build (Static Export)

Requires Node.js `20.9+`.

```powershell
cd interface
npm ci
npm run build
```

This writes the static bundle to `interface/out/`, which `neuromorph-nano launch ui` serves locally.

## Development Server

Start the Rust API server in `--dev` mode in one terminal and the Next.js dev server in another:

```powershell
# Terminal 1 (from repo root)
neuromorph-nano ui --dev

# Terminal 2 (from interface/)
npm run dev
```

## Checks & Tests

```powershell
npm run lint
npm test
```

See [docs/user-guide.md](../docs/user-guide.md) and [docs/interface-api.md](../docs/interface-api.md) for the page overview and local JSON API reference.
