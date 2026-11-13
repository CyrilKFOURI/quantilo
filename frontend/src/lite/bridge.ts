/**
 * Lite build: the page has no server. Every request to /api/... (fetch, download links) is
 * answered by the Python worker (worker.ts) instead of the network.
 */
import { t } from "../lib/i18n";
import type { ImageFolder } from "./images";
import type { LiteFolder, LiteMessage, LiteReason, LiteRequest, LiteStep } from "./worker";

export type EngineState =
  | { status: "loading"; step: LiteStep; fraction: number }
  | { status: "ready" }
  | { status: "failed"; error: string; reason: LiteReason | null };

let state: EngineState = { status: "loading", step: "lite.step.starting", fraction: 0 };
const listeners = new Set<() => void>();
const pending = new Map<number, (response: Response) => void>();
const folders = new Map<number, { resolve: (path: string) => void; reject: (error: Error) => void }>();
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

/**
 * 119: a folder of images picked in the page, written into Python's file system under the
 * workspace's uploads; resolves with its path, which is imported like a folder on disk.
 */
export async function writeImageFolder(folder: ImageFolder): Promise<string> {
  if (!worker) throw new Error("The lite engine is not started.");
  const files: LiteFolder["files"] = [];
  for (const item of folder.files) files.push({ path: item.path, data: await item.file.arrayBuffer() });
  const id = nextId++;
  const message: LiteFolder = { type: "folder", id, name: folder.name, files };
  return new Promise((resolve, reject) => {
    folders.set(id, { resolve, reject });
    worker?.postMessage(message, files.map((file) => file.data));
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
    window.alert(t("lite.error.download", { status: String(response.status) }));
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

// 119: images of an image set (<img src="/api/...">, thumbnails drawn on the map) come from the
// worker too: their bytes become a blob URL, kept for the next time the same image is shown.
const IMAGE_CACHE = 4_000; // 2,000 thumbnails and as many originals at most (R-VIS-WEB-01)
const images = new Map<string, Promise<string>>();

/**
 * The blob URL of an image the worker serves. `<img src>` goes through it on its own; a CSS
 * `url(...)` (the anomaly mask, R-VIS-LAY-04) does not, so its component asks here (useApiSrc).
 */
export function apiImage(url: string): Promise<string> {
  const found = images.get(url);
  if (found) return found;
  const made = viaWorker(url).then(async (response) =>
    response.ok ? URL.createObjectURL(await response.blob()) : "data:,",
  );
  images.set(url, made);
  if (images.size > IMAGE_CACHE) {
    const [oldest, blob] = images.entries().next().value as [string, Promise<string>];
    images.delete(oldest);
    void blob.then((href) => href.startsWith("blob:") && URL.revokeObjectURL(href));
  }
  return made;
}

function serveImagesFromWorker() {
  const descriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
  if (!descriptor?.set) return;
  const nativeSet = descriptor.set;
  const load = (image: HTMLImageElement, url: string) => {
    image.dataset.apiSrc = url;
    void apiImage(url).then((href) => {
      if (image.dataset.apiSrc === url) nativeSet.call(image, href);
    });
  };
  Object.defineProperty(HTMLImageElement.prototype, "src", {
    ...descriptor,
    set(this: HTMLImageElement, value: string) {
      if (isApi(String(value))) return load(this, String(value));
      delete this.dataset.apiSrc;
      nativeSet.call(this, value);
    },
  });
  // React writes `src` as an attribute
  const nativeSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (this: Element, name: string, value: string) {
    if (this instanceof HTMLImageElement && name.toLowerCase() === "src" && isApi(String(value)))
      return load(this, String(value));
    nativeSetAttribute.call(this, name, value);
  };
}

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
      setState({ status: "failed", error: message.error, reason: message.reason });
    else if (message.type === "folder") {
      const waiting = folders.get(message.id);
      folders.delete(message.id);
      if (message.path) waiting?.resolve(message.path);
      else waiting?.reject(new Error(message.error ?? t("lite.error.copy")));
    } else {
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
  serveImagesFromWorker();

  HTMLAnchorElement.prototype.click = function (this: HTMLAnchorElement) {
    const href = this.getAttribute("href");
    if (href && isApi(href)) void download(href, this.getAttribute("download"));
    else nativeClick.call(this);
  };
}
