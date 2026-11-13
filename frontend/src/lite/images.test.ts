import { describe, expect, it } from "vitest";

import { t } from "../lib/i18n";
import { imageFolder, MAX_IMAGES, needsPillow, safeParts } from "./images";

const file = (path: string, size = 10) => {
  const made = new File([new Uint8Array(size)], path.split("/").at(-1) ?? path);
  Object.defineProperty(made, "webkitRelativePath", { value: path.includes("/") ? path : "" });
  return made;
};

describe("119: images in the browser version", () => {
  it("loads Pillow only for Vision, the image routes and a reimport (R-VIS-WEB-04)", () => {
    expect(needsPillow("/api/v1/modules/vision/datasets/abc/task")).toBe(true);
    expect(needsPillow("/api/v1/images/preview")).toBe(true);
    expect(needsPillow("/api/v1/images/rules")).toBe(true);
    expect(needsPillow("/api/v1/datasets/abc/images/index")).toBe(true);
    expect(needsPillow("/api/v1/datasets/abc/images/x1/thumb")).toBe(true);
    expect(needsPillow("/api/v1/datasets/abc/reimport")).toBe(true);
    expect(needsPillow("/api/v1/datasets")).toBe(false);
    expect(needsPillow("/api/v1/datasets/abc")).toBe(false);
    expect(needsPillow("/api/v1/datasets/abc/rows?limit=50")).toBe(false);
    expect(needsPillow("/api/v1/uploads?filename=beans.zip")).toBe(false);
    expect(needsPillow("/api/v1/modules/ml/catalog")).toBe(false);
    expect(needsPillow("/api/v1/health")).toBe(false);
  });

  it("refuses unsafe paths", () => {
    expect(safeParts("beans/train/a.jpg")).toEqual(["beans", "train", "a.jpg"]);
    expect(safeParts("beans\\test\\b.jpg")).toEqual(["beans", "test", "b.jpg"]);
    expect(safeParts("../etc/passwd")).toBeNull();
    expect(safeParts("C:/x.png")).toBeNull();
    expect(safeParts("beans/.DS_Store")).toBeNull();
    expect(safeParts("__MACOSX/beans/a.jpg")).toBeNull();
    expect(safeParts("")).toBeNull();
  });

  it("keeps a picked folder's images and labels files under its name", () => {
    const found = imageFolder([
      file("beans/train/healthy/a.jpg", 100),
      file("beans/test/rust/b.PNG", 50),
      file("beans/labels.csv", 5),
      file("beans/notes.txt"),
      file("beans/.hidden/c.jpg"),
    ]);
    expect(found).toEqual({
      name: "beans",
      images: 2,
      bytes: 155,
      files: [
        { path: "train/healthy/a.jpg", file: expect.any(File) },
        { path: "test/rust/b.PNG", file: expect.any(File) },
        { path: "labels.csv", file: expect.any(File) },
      ],
    });
  });

  it("puts loose images in one folder named images", () => {
    const found = imageFolder([file("a.jpg"), file("b.webp")]);
    expect("error" in found ? found.error : found.name).toBe("images");
    expect("files" in found && found.files.map((item) => item.path)).toEqual(["a.jpg", "b.webp"]);
  });

  it("refuses an empty folder and one beyond the browser's limits (R-VIS-WEB-01)", () => {
    // 127: the refusal is a text key, said in the chosen language where it is shown
    const said = (found: ReturnType<typeof imageFolder>) => ("error" in found ? t(found.error, found.vars) : "");
    expect(said(imageFolder([file("docs/readme.txt")]))).toContain("No image");
    const many = Array.from({ length: MAX_IMAGES + 1 }, (_, i) => file(`big/${i}.png`, 1));
    const refused = said(imageFolder(many));
    expect(refused).toContain("R-VIS-WEB-01");
    expect(refused).toContain("2,001 images");
    const heavy = said(imageFolder([file("big/a.png", 301 * 2 ** 20)]));
    expect(heavy).toContain("300 MB");
  });
});
