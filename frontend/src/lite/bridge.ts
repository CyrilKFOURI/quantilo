/**
 * Lite build: the page has no server. Every request to /api/... (fetch, download links) is
 * answered by the Python worker (worker.ts) instead of the network.
 */
import type { LiteMessage, LiteRequest } from "./worker";

export type EngineState =
  | { status: "loading"; step: string; fraction: number }
  | { status: "ready" }
  | { status: "failed"; error: string };

let state: EngineState = { status: "loading", step: "Starting", fraction: 0 };
const listeners = new Set<() => void>();
const pending = new Map<number, (response: Response) => void>();
let nextId = 1;
let worker: Worker | null = null;

export const engineState = () => state;
export function onEngineState(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
const startedAt = performance.now();
function setState(next: EngineState) {
  if (next.status !== "loading" || state.status !== "loading" || next.step !== state.step)
    console.info(
      `[lite] ${((performance.now() - startedAt) / 1000).toFixed(1)} s: ${next.status === "loading" ? next.step : next.status}`,
    );
  state = next;
  listeners.forEach((listener) => listener());
}

const isApi = (url: string) => url.startsWith("/api/");

async function viaWorker(url: string, init: RequestInit = {}): Promise<Response> {
  if (!worker) throw new Error("The lite engine is not started.");
  const headers: Record<string, string> = {};
  new Headers(init.headers).forEach((value, key) => (headers[key] = value));
  let body: ArrayBuffer | null = null;
  if (init.body instanceof Blob) body = await init.body.arrayBuffer();
  else if (typeof init.body === "string")
    body = new TextEncoder().encode(init.body).buffer as ArrayBuffer;
  else if (init.body instanceof ArrayBuffer) body = init.body;
  const id = nextId++;
  const message: LiteRequest = {
    id,
    method: (init.method ?? "GET").toUpperCase(),
    url,
    headers,
    body,
  };
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker?.postMessage(message, body ? [body] : []);
  });
}

function fileName(response: Response, fallback: string): string {
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  return match?.[1] ? decodeURIComponent(match[1]) : fallback;
}

async function download(href: string, suggested: string | null) {
  const response = await viaWorker(href);
  if (!response.ok) {
    window.alert(`Download failed (${response.status}).`);
    return;
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName(response, suggested || "download");
  nativeClick.call(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const nativeClick = HTMLAnchorElement.prototype.click;

export function startLite() {
  worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
  worker.onmessage = (event: MessageEvent<LiteMessage>) => {
    const message = event.data;
    if (message.type === "progress")
      setState({
        status: "loading",
        step: message.step,
        fraction: message.fraction,
      });
    else if (message.type === "ready") setState({ status: "ready" });
    else if (message.type === "failed")
      setState({ status: "failed", error: message.error });
    else {
      const resolve = pending.get(message.id);
      pending.delete(message.id);
      resolve?.(
        new Response(message.status === 204 ? null : message.body, {
          status: message.status,
          headers: message.headers,
        }),
      );
    }
  };
  worker.postMessage({
    type: "boot",
    bundleUrl: new URL("python.zip", document.baseURI).href,
  });

  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : null;
    return url && isApi(url) ? viaWorker(url, init) : nativeFetch(input, init);
  };

  // Download links (<a href="/api/..." download>) and links clicked from code.
  document.addEventListener(
    "click",
    (event) => {
      const link = (event.target as Element | null)?.closest?.("a");
      const href = link?.getAttribute("href");
      if (link && href && isApi(href)) {
        event.preventDefault();
        void download(href, link.getAttribute("download"));
      }
    },
    true,
  );
  HTMLAnchorElement.prototype.click = function (this: HTMLAnchorElement) {
    const href = this.getAttribute("href");
    if (href && isApi(href)) void download(href, this.getAttribute("download"));
    else nativeClick.call(this);
  };
}
