/// <reference lib="webworker" />
/**
 * The lite build's backend: Python (Pyodide) in a Web Worker runs the real FastAPI app
 * (lite/boot.py). The page sends it each /api request (bridge.ts) and gets the response back.
 * 119: Pillow is loaded only before the first request that needs it (Vision, image routes), and a
 * folder of images picked in the page is written into Python's file system (R-VIS-WEB-04).
 */
import type { MessageKey } from "../lib/i18n";
import { needsPillow, PILLOW } from "./images";

export const PYODIDE_VERSION = "314.0.7";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// The database and the web layer first: on a phone memory can run out while the large
// scientific libraries load, and then the message says which one did not fit.
const CORE = ["duckdb", "numpy", "pandas", "pydantic", "pydantic-core", "fastapi", "micropip"];
const SCIENCE = ["pyarrow", "scipy", "statsmodels", "scikit-learn"];
// What the engine imports, checked before it starts (a failed load is only logged by Pyodide).
const MODULES = ["duckdb", "numpy", "pandas", "pyarrow", "scipy", "statsmodels", "sklearn", "pydantic", "fastapi"];

type PyProxy = { toJs: () => unknown; destroy: () => void };
type Pyodide = {
  loadPackage: (names: string[], options?: object) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  unpackArchive: (buffer: ArrayBuffer, format: string, options?: object) => void;
  toPy: (value: unknown) => unknown;
  globals: { get: (name: string) => (...args: unknown[]) => Promise<PyProxy> };
  FS: {
    mkdirTree: (path: string) => void;
    writeFile: (path: string, data: Uint8Array) => void;
    mount: (type: unknown, options: object, path: string) => void;
    filesystems: { IDBFS: unknown };
    syncfs: (populate: boolean, done: (error: unknown) => void) => void;
  };
};

export type LiteRequest = {
  id: number;
  method: string;
  url: string;
  headers: Record<string, string>;
  body: ArrayBuffer | null;
};
/** 119: a folder of images picked in the page, written under the workspace's uploads. */
export type LiteFolder = {
  type: "folder";
  id: number;
  name: string;
  files: { path: string; data: ArrayBuffer }[];
};
/** 127: the worker has no French texts, so it sends the key of a step or of a failure and the
 *  page translates it where it is shown (EngineGate). */
export type LiteStep = Extract<MessageKey, `lite.step.${string}`>;
export type LiteReason = { key: Extract<MessageKey, `lite.error.${string}`>; vars: Record<string, string> };
export type LiteMessage =
  | { type: "folder"; id: number; path: string | null; error: string | null }
  | { type: "progress"; step: LiteStep; fraction: number }
  | { type: "ready" }
  | { type: "failed"; error: string; reason: LiteReason | null }
  | {
      type: "response";
      id: number;
      status: number;
      headers: Record<string, string>;
      body: ArrayBuffer;
    };

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = (message: LiteMessage, transfer: Transferable[] = []) =>
  scope.postMessage(message, transfer);

/** A failure the page can say in its own language. */
class LiteError extends Error {
  constructor(readonly reason: LiteReason) {
    super(reason.key);
  }
}

function syncfs(py: Pyodide, populate: boolean): Promise<void> {
  return new Promise((resolve) =>
    py.FS.syncfs(populate, () => resolve()),
  );
}

async function boot(bundleUrl: string): Promise<Pyodide> {
  post({ type: "progress", step: "lite.step.python", fraction: 0.05 });
  const { loadPyodide } = (await import(
    /* @vite-ignore */ `${INDEX_URL}pyodide.mjs`
  )) as { loadPyodide: (options: object) => Promise<Pyodide> };
  const py = await loadPyodide({ indexURL: INDEX_URL });
  post({ type: "progress", step: "lite.step.libraries", fraction: 0.15 });
  let loaded = 0;
  const errors: string[] = [];
  const options = {
    messageCallback: (message: string) => {
      if (message.startsWith("Loaded")) {
        loaded += 1;
        post({
          type: "progress",
          step: "lite.step.libraries",
          fraction: 0.15 + 0.55 * Math.min(1, loaded / 40),
        });
      }
    },
    errorCallback: (message: string) => errors.push(message),
  };
  await py.loadPackage(CORE, options);
  await py.loadPackage(SCIENCE, options);
  const missing = (await py.runPythonAsync(
    `import importlib.util
[name for name in ${JSON.stringify(MODULES)} if importlib.util.find_spec(name) is None]`,
  )) as PyProxy;
  const absent = missing.toJs() as string[];
  missing.destroy();
  if (absent.length > 0) {
    const said = errors.slice(-2).join(" · ");
    throw new LiteError({
      key: said ? "lite.error.librariesSaid" : "lite.error.libraries",
      vars: { names: absent.join(", "), said },
    });
  }
  post({ type: "progress", step: "lite.step.helpers", fraction: 0.72 });
  // Pure-Python wheels Pyodide lacks, shipped with the site (lite/bundle.py): no call to PyPI.
  const wheelsUrl = new URL("wheels/", bundleUrl);
  const names = (await (await fetch(new URL("index.json", wheelsUrl))).json()) as string[];
  const urls = names.map((name) => new URL(name, wheelsUrl).href);
  await py.runPythonAsync(
    `import micropip\nawait micropip.install(${JSON.stringify(urls)}, deps=False)`,
  );
  post({ type: "progress", step: "lite.step.quantilo", fraction: 0.78 });
  const bundle = await fetch(bundleUrl);
  if (!bundle.ok) throw new LiteError({ key: "lite.error.bundle", vars: { url: bundleUrl, status: String(bundle.status) } });
  py.unpackArchive(await bundle.arrayBuffer(), "zip", { extractDir: "/lite" });
  // The workspace (datasets, settings) is kept in the browser between visits.
  py.FS.mkdirTree("/home");
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/home");
  await syncfs(py, true);
  post({ type: "progress", step: "lite.step.engine", fraction: 0.85 });
  await py.runPythonAsync(
    "import sys\nsys.path.insert(0, '/lite')\nimport boot\nawait boot.start()",
  );
  return py;
}

let ready: Promise<Pyodide> | null = null;
let queue: Promise<unknown> = Promise.resolve();
let vision: Promise<void> | null = null;

/** R-VIS-WEB-04: Pillow (about 5 MB) and the Vision routes, once, when first needed. */
function loadVision(py: Pyodide): Promise<void> {
  vision ??= (async () => {
    const errors: string[] = [];
    await py.loadPackage([PILLOW], { errorCallback: (message: string) => errors.push(message) });
    const found = (await py.runPythonAsync("import importlib.util\nimportlib.util.find_spec('PIL') is not None")) as boolean;
    if (!found) {
      const reason = errors.length > 0 ? ` The browser said: ${errors.slice(-2).join(" · ")}` : "";
      throw new Error(`The browser could not load Pillow, which reads the images.${reason} Reload the page to try again.`);
    }
    await py.runPythonAsync("import boot\nboot.enable_vision()");
  })();
  vision.catch(() => (vision = null)); // a failed load is tried again by the next request
  return vision;
}

/** Where a file of a picked folder goes: its parts (made safe by the page) under the folder. */
const joined = (root: string, path: string) =>
  `${root}/${path.split("/").filter((part) => part !== "" && part !== "." && part !== "..").join("/")}`;

async function writeFolder(py: Pyodide, folder: LiteFolder) {
  try {
    // the app makes the folder (uploads/<id>/<name>): it is found and checked like an upload
    const root = (await py.runPythonAsync(`import boot\nboot.upload_folder(${JSON.stringify(folder.name)})`)) as string;
    for (const file of folder.files) {
      const target = joined(root, file.path);
      py.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
      py.FS.writeFile(target, new Uint8Array(file.data));
    }
    await syncfs(py, false);
    post({ type: "folder", id: folder.id, path: root, error: null });
  } catch (error) {
    post({ type: "folder", id: folder.id, path: null, error: String(error) });
  }
}

async function answer(py: Pyodide, request: LiteRequest) {
  if (needsPillow(request.url)) await loadVision(py);
  const handle = py.globals.get("boot_handle");
  const body = request.body ? py.toPy(new Uint8Array(request.body)) : undefined;
  const result = await handle(
    request.method,
    request.url,
    JSON.stringify(request.headers),
    body,
  );
  const [status, headers, bytes] = result.toJs() as [number, string, Uint8Array];
  result.destroy();
  if (request.method !== "GET") await syncfs(py, false);
  const copy = bytes.slice().buffer;
  post(
    {
      type: "response",
      id: request.id,
      status,
      headers: JSON.parse(headers) as Record<string, string>,
      body: copy,
    },
    [copy],
  );
}

scope.onmessage = (event: MessageEvent) => {
  const data = event.data as { type: "boot"; bundleUrl: string } | LiteFolder | LiteRequest;
  if ("type" in data && data.type === "boot") {
    ready = boot(data.bundleUrl).then(async (py) => {
      await py.runPythonAsync("from boot import handle as boot_handle");
      post({ type: "ready" });
      return py;
    });
    ready.catch((error: unknown) =>
      post({ type: "failed", error: String(error), reason: error instanceof LiteError ? error.reason : null }),
    );
    return;
  }
  if ("type" in data && data.type === "folder") {
    const folder = data;
    queue = queue
      .then(async () => writeFolder(await (ready as Promise<Pyodide>), folder))
      .catch((error: unknown) => post({ type: "folder", id: folder.id, path: null, error: String(error) }));
    return;
  }
  const request = data as LiteRequest;
  // One request at a time: Python runs on this single thread.
  queue = queue
    .then(async () => answer(await (ready as Promise<Pyodide>), request))
    .catch((error: unknown) =>
      post({
        type: "response",
        id: request.id,
        status: 500,
        headers: { "content-type": "application/json" },
        body: new TextEncoder().encode(
          JSON.stringify({ detail: `Lite engine error: ${String(error)}` }),
        ).buffer,
      }),
    );
};
