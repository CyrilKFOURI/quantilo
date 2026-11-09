import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./styles/index.css";

import { createHashHistory } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./app/App";
import { createAppRouter } from "./app/router";
import { loadPlotly } from "./charts/Plot";
import { startLite } from "./lite/bridge";
import { EngineGate } from "./lite/EngineGate";
import { LITE } from "./lite/flag";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

if (LITE) {
  // Browser-only build: Python runs in a worker; addresses use # (a static host has no routes).
  startLite();
  createRoot(root).render(
    <StrictMode>
      <EngineGate>
        <App router={createAppRouter({ history: createHashHistory() })} />
      </EngineGate>
    </StrictMode>,
  );
} else {
  createRoot(root).render(
    <StrictMode>
      <App router={createAppRouter()} />
    </StrictMode>,
  );
}

// plotly.js takes ~2 s to read: do it while the first page is read, not at the first chart.
window.addEventListener(
  "load",
  () => {
    const idle = window.requestIdleCallback ?? ((run: () => void) => window.setTimeout(run, 1000));
    idle(() => void loadPlotly(), { timeout: 3000 });
  },
  { once: true },
);
