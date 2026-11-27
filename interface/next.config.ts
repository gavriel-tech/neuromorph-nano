import type { NextConfig } from "next";

// The build stamp: the moment this interface was built, in UTC, fixed into the bundle so every
// capture shows which build it came from.
const t = new Date();
const two = (n: number) => String(n).padStart(2, "0");
const BUILD_STAMP = `${t.getUTCFullYear()}-${two(t.getUTCMonth() + 1)}-${two(t.getUTCDate())} ${two(t.getUTCHours())}:${two(t.getUTCMinutes())}`;

const nextConfig: NextConfig = {
  env: { NEXT_PUBLIC_BUILD_STAMP: BUILD_STAMP },
  // Static export: the Nano engine serves out/ on localhost; Node is only needed to build.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
