import { useSyncExternalStore } from "react";

/** Widths of the layout (041): phone below 640 px, tablet below 1024 px, laptop above. */
export const PHONE = "(max-width: 639px)";
export const BELOW_LAPTOP = "(max-width: 1023px)";

/** Whether a media query matches, following its changes. False where matchMedia is missing (tests). */
export function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof window.matchMedia !== "function") return () => {};
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    () => typeof window.matchMedia === "function" && window.matchMedia(query).matches,
    () => false,
  );
}
