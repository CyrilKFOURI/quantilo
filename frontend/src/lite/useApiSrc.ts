import { useEffect, useState } from "react";

import { apiImage } from "./bridge";
import { LITE } from "./flag";

const isApi = (url: string) => url.startsWith("/api/");

/**
 * 119: an image address usable where the page itself fetches it, such as a CSS `url(...)` (the
 * anomaly mask). The full app returns it as it is; the browser version has no server, so an
 * /api/... address becomes the blob URL of the bytes the worker returns, `null` until they arrive.
 */
export function useApiSrc(src: string | null | undefined): string | null {
  const viaWorker = LITE && Boolean(src) && isApi(src ?? "");
  const [resolved, setResolved] = useState<{ url: string; href: string } | null>(null);
  useEffect(() => {
    if (!viaWorker || !src) return;
    let live = true;
    void apiImage(src).then((href) => {
      if (live) setResolved({ url: src, href });
    });
    return () => {
      live = false;
    };
  }, [viaWorker, src]);
  if (!viaWorker || !src) return src ?? null;
  return resolved && resolved.url === src ? resolved.href : null;
}
