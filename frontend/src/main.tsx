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
import { loadLocale } from "./lib/i18n";
import { installPaletteStyles } from "./lib/palettes";
import { languagePreference, readPreference } from "./lib/preferences";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");
installPaletteStyles();

if (LITE) {
  // Browser-only build: Python runs in a worker; addresses use # (a static host has no routes).
  // The engine starts while the texts load: it does not wait for the language.
  startLite();
}

// French texts (127) are read before the first render; a failed read leaves the English ones.
void loadLocale(readPreference(languagePreference))
  .catch(() => undefined)
  .then(() => render(root));

function render(root: HTMLElement) {
  if (LITE) {
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
