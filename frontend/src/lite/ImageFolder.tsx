import { FolderOpen } from "lucide-react";
import { useRef, useState } from "react";

import { formatCount } from "../lib/format";
import { t } from "../lib/i18n";
import { Button } from "../ui/Button";
import { writeImageFolder } from "./bridge";
import { imageFolder } from "./images";

export type ImageFolderPick = {
  /** Several images, or a picked folder's files: copied into the page, then imported as a folder. */
  pick: (files: File[]) => void;
  copying: number | null; // images being copied into the page
  error: string | null;
};

/** 119: a folder (or several images) picked in the browser version, written into Python's file
 *  system (`uploads/<id>/<name>`), then opened in the import dialog like a folder on disk. */
export function useImageFolder(onReady: (path: string) => void): ImageFolderPick {
  const [copying, setCopying] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pick = (files: File[]) => {
    const found = imageFolder(files);
    if ("error" in found) {
      setError(t(found.error, found.vars));
      return;
    }
    setError(null);
    setCopying(found.images);
    writeImageFolder(found)
      .then(onReady)
      .catch((failure: unknown) => setError(failure instanceof Error ? failure.message : String(failure)))
      .finally(() => setCopying(null));
  };
  return { pick, copying, error };
}

/** The import panel's folder button in the browser version: the browser reads a folder the user
 *  picks (an `<input webkitdirectory>`), never a path on disk. */
export function ImageFolderButton({ folder, disabled }: { folder: ImageFolderPick; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <Button icon={FolderOpen} disabled={disabled || folder.copying !== null} onClick={() => inputRef.current?.click()} title={t("lite.imageFolderHint")}>
        {folder.copying !== null ? t("lite.imageFolderCopying", { count: formatCount(folder.copying) }) : t("lite.imageFolder")}
      </Button>
      <input
        ref={(node) => {
          node?.setAttribute("webkitdirectory", ""); // a folder, not files (no React prop for it)
          inputRef.current = node;
        }}
        type="file"
        multiple
        className="hidden"
        aria-label={t("lite.imageFolder")}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length > 0) folder.pick(files);
        }}
      />
    </>
  );
}
