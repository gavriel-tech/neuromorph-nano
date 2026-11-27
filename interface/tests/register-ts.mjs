import { register } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const globalKey = "__nmTsRegistered";
const g = globalThis;

if (!g[globalKey]) {
  const here = dirname(fileURLToPath(import.meta.url));
  const srcUrl = pathToFileURL(resolve(here, "../src")).href;
  register(
    "data:text/javascript," +
      encodeURIComponent(`
        export async function resolve(spec, ctx, next) {
          if (spec.startsWith("@/")) {
            spec = ${JSON.stringify(srcUrl)} + "/" + spec.slice(2);
          }
          if ((spec.startsWith("./") || spec.startsWith("../") || spec.startsWith("file:")) && !/\\.[cm]?[jt]sx?$/.test(spec)) {
            for (const ext of [".ts", ".tsx"]) {
              try { return await next(spec + ext, ctx); } catch {}
            }
          }
          return next(spec, ctx);
        }
      `),
    import.meta.url,
  );
  g[globalKey] = true;
}
