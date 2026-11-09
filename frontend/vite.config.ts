import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  // `--mode lite`: the browser-only build (Python in the page, src/lite), for any static host.
  ...(mode === "lite" && {
    base: "./",
    define: { "import.meta.env.VITE_LITE": JSON.stringify("1") },
    build: { outDir: "dist-lite" },
    worker: { format: "es" as const },
  }),
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // The Guide page imports the user guide from ../docs (docs/USER_GUIDE.md).
    fs: { allow: [".", "../docs"] },
    // Override with DE_API_URL (e.g. in .env.local) if the backend runs on another port.
    proxy: { "/api": loadEnv(mode, ".", "DE_").DE_API_URL ?? "http://127.0.0.1:8000" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    testTimeout: 20_000,
  },
}));
