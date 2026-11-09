/// <reference lib="webworker" />
/**
 * The lite build's backend: Python (Pyodide) in a Web Worker runs the real FastAPI app
 * (lite/boot.py). The page sends it each /api request (bridge.ts) and gets the response back.
 */

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
export type LiteMessage =
  | { type: "progress"; step: string; fraction: number }
  | { type: "ready" }
  | { type: "failed"; error: string }
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

function syncfs(py: Pyodide, populate: boolean): Promise<void> {
  return new Promise((resolve) =>
    py.FS.syncfs(populate, () => resolve()),
  );
}

async function boot(bundleUrl: string): Promise<Pyodide> {
  post({ type: "progress", step: "Loading Python", fraction: 0.05 });
  const { loadPyodide } = (await import(
    /* @vite-ignore */ `${INDEX_URL}pyodide.mjs`
  )) as { loadPyodide: (options: object) => Promise<Pyodide> };
  const py = await loadPyodide({ indexURL: INDEX_URL });
  post({ type: "progress", step: "Loading the scientific libraries", fraction: 0.15 });
  let loaded = 0;
  const errors: string[] = [];
  const options = {
    messageCallback: (message: string) => {
      if (message.startsWith("Loaded")) {
        loaded += 1;
        post({
          type: "progress",
          step: "Loading the scientific libraries",
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
    const reason = errors.length > 0 ? ` The browser said: ${errors.slice(-2).join(" · ")}` : "";
    throw new Error(
      `The browser could not load ${absent.join(", ")}.${reason} On a phone or a tablet this is usually ` +
        "memory: Python and its scientific libraries take a lot of it in one tab. Close other tabs and reload, " +
        "or use a computer.",
    );
  }
  post({ type: "progress", step: "Installing small helpers", fraction: 0.72 });
  // Pure-Python wheels Pyodide lacks, shipped with the site (lite/bundle.py): no call to PyPI.
  const wheelsUrl = new URL("wheels/", bundleUrl);
  const names = (await (await fetch(new URL("index.json", wheelsUrl))).json()) as string[];
  const urls = names.map((name) => new URL(name, wheelsUrl).href);
  await py.runPythonAsync(
    `import micropip\nawait micropip.install(${JSON.stringify(urls)}, deps=False)`,
  );
  post({ type: "progress", step: "Loading Quantilo", fraction: 0.78 });
  const bundle = await fetch(bundleUrl);
  if (!bundle.ok) throw new Error(`Could not load ${bundleUrl} (${bundle.status}).`);
  py.unpackArchive(await bundle.arrayBuffer(), "zip", { extractDir: "/lite" });
  // The workspace (datasets, settings) is kept in the browser between visits.
  py.FS.mkdirTree("/home");
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/home");
  await syncfs(py, true);
  post({ type: "progress", step: "Starting the analysis engine", fraction: 0.85 });
  await py.runPythonAsync(
    "import sys\nsys.path.insert(0, '/lite')\nimport boot\nawait boot.start()",
  );
  return py;
}

let ready: Promise<Pyodide> | null = null;
let queue: Promise<unknown> = Promise.resolve();

async function answer(py: Pyodide, request: LiteRequest) {
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
  const data = event.data as { type: "boot"; bundleUrl: string } | LiteRequest;
  if ("type" in data && data.type === "boot") {
    ready = boot(data.bundleUrl).then(async (py) => {
      await py.runPythonAsync("from boot import handle as boot_handle");
      post({ type: "ready" });
      return py;
    });
    ready.catch((error: unknown) =>
      post({ type: "failed", error: String(error) }),
    );
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
