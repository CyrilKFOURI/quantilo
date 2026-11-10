import "@testing-library/jest-dom/vitest";

import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

import { resetCurrentDataset } from "../features/datasets/current";

// The suite runs files in parallel: on a busy machine, 1 s is too short to wait for a render.
configure({ asyncUtilTimeout: 5_000 });

afterEach(() => {
  cleanup();
  localStorage.clear();
  resetCurrentDataset(); // 098: the dataset kept open while browsing Home does not leak between tests
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.density;
});

if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

window.scrollTo = () => {};
