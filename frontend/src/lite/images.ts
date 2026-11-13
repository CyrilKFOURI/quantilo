/**
 * 119: images in the browser version. Pure helpers shared by the page (a folder picked in the
 * import panel) and the worker (which loads Pillow before the first request that needs it).
 * The engine applies the same limits (quantilo/vision/browser.py, R-VIS-WEB-01).
 */
import type { MessageKey } from "../lib/i18n";

/** The Python package the worker loads on demand, never at the start (R-VIS-WEB-04). */
export const PILLOW = "pillow";

/** R-VIS-WEB-01 to 03, as in quantilo/vision/browser.py. */
export const MAX_IMAGES = 2_000;
export const MAX_MB = 300;
export const MEASURE_SIDE = 256;
export const BLUR_SHARPNESS = 300; // R-VQ-04 at 256 pixels (100 at 512)
/** What only the desktop app runs: models by name, then kinds of methods (their words translated). */
export const DESKTOP_ONLY = ["DINOv2", "PatchCore", "PaDiM"] as const;
export const DESKTOP_ONLY_KINDS: readonly MessageKey[] = ["lite.autoencoders", "lite.gradients"];

const VISION_ROUTES = [
  /^\/api\/v1\/modules\/vision\//,
  /^\/api\/v1\/images\//,
  /^\/api\/v1\/datasets\/[^/?]+\/images(\/|\?|$)/,
  // an image set reimported reads its pixels again (a table's reimport loads Pillow once too)
  /^\/api\/v1\/datasets\/[^/?]+\/reimport(\?|$)/,
];

/** R-VIS-WEB-04: the requests that need Pillow: Vision, the image routes, an image set's reimport. */
export function needsPillow(url: string): boolean {
  return VISION_ROUTES.some((route) => route.test(url));
}

const IMAGE_SUFFIXES = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"]);
// A labels file at the root of the folder (R-VIS-LAB-01) travels with the images.
const SIDE_SUFFIXES = new Set([".csv", ".tsv", ".parquet"]);

const suffix = (name: string) => {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
};

/** A path inside the folder, made safe: no empty, `.` or `..` part, no drive, no hidden part. */
export function safeParts(path: string): string[] | null {
  const parts = path.split(/[\\/]+/).filter((part) => part !== "" && part !== ".");
  if (parts.length === 0) return null;
  if (parts.some((part) => part === ".." || part.includes(":") || part.includes("\0"))) return null;
  if (parts.some((part) => part.startsWith(".") || part === "__MACOSX")) return null;
  return parts;
}

export type PickedFile = { path: string; file: File };
export type ImageFolder = { name: string; files: PickedFile[]; images: number; bytes: number };
/** Why a folder is refused: a text key and its values, translated where it is shown (127). */
export type FolderRefusal = { error: MessageKey; vars: Record<string, string> };

/** The relative path the browser gives a file of a picked folder (its name for loose files). */
export const relativePath = (file: File) => file.webkitRelativePath || file.name;

/**
 * The files of a picked folder (or of several images picked together) that an image set uses:
 * the images and the labels files, under the folder's own name. Refused with R-VIS-WEB-01's
 * reason beyond 2,000 images or 300 MB, so nothing large is copied into the page's memory.
 */
export function imageFolder(files: File[]): ImageFolder | FolderRefusal {
  const picked: PickedFile[] = [];
  for (const file of files) {
    const parts = safeParts(relativePath(file));
    if (!parts) continue;
    const kind = suffix(parts[parts.length - 1] ?? "");
    if (IMAGE_SUFFIXES.has(kind) || SIDE_SUFFIXES.has(kind)) picked.push({ path: parts.join("/"), file });
  }
  const tops = new Set(picked.map((item) => item.path.split("/")[0]));
  const folder = tops.size === 1 && picked.every((item) => item.path.includes("/"));
  // a picked folder: its name, and the paths under it; loose images: one folder named "images"
  const name = folder ? [...tops][0]! : "images";
  const inside = folder
    ? picked.map((item) => ({ file: item.file, path: item.path.slice(name.length + 1) }))
    : picked;
  const images = inside.filter((item) => IMAGE_SUFFIXES.has(suffix(item.path))).length;
  const bytes = inside.reduce((total, item) => total + item.file.size, 0);
  if (images === 0) return { error: "lite.error.noImage", vars: {} };
  if (images > MAX_IMAGES || bytes > MAX_MB * 2 ** 20)
    return {
      error: "lite.error.tooLarge",
      vars: {
        images: images.toLocaleString("en-US"),
        mb: Math.round(bytes / 2 ** 20).toLocaleString("en-US"),
        maxImages: MAX_IMAGES.toLocaleString("en-US"),
        maxMb: String(MAX_MB),
      },
    };
  return { name, files: inside, images, bytes };
}
