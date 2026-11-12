import type { ImageIndex, ImageReport } from "./api";

/** A small image set for the tests (107, 108): two exact copies, a blurry image, a broken file. */
type Image = {
  rel: string;
  id: string;
  label: string | null;
  split: string | null;
  width: number | null;
  height: number | null;
  brightness: number | null;
  sharpness: number | null;
  issues: string[];
  excluded: boolean;
  group: number | null;
  ok?: boolean;
};

const IMAGES: Image[] = [
  { rel: "cat/0.png", id: "aaaaaaaaaaaaaaaa", label: "cat", split: "train", width: 200, height: 200, brightness: 0.5, sharpness: 500, issues: ["R-VQ-09"], excluded: false, group: 0 },
  { rel: "cat/copy.png", id: "aaaaaaaaaaaaaaaa", label: "cat", split: "train", width: 200, height: 200, brightness: 0.5, sharpness: 500, issues: ["R-VQ-09"], excluded: true, group: 0 },
  { rel: "cat/blur.png", id: "bbbbbbbbbbbbbbbb", label: "cat", split: "train", width: 400, height: 300, brightness: 0.3, sharpness: 12, issues: ["R-VQ-04"], excluded: false, group: null },
  { rel: "cat/broken.png", id: "cccccccccccccccc", label: "cat", split: "train", width: null, height: null, brightness: null, sharpness: null, issues: ["R-VQ-01"], excluded: true, group: null, ok: false },
  { rel: "dog/0.png", id: "dddddddddddddddd", label: "dog", split: "train", width: 640, height: 480, brightness: 0.55, sharpness: 800, issues: [], excluded: false, group: null },
  { rel: "dog/1.png", id: "eeeeeeeeeeeeeeee", label: "dog", split: "test", width: 2000, height: 1500, brightness: 0.7, sharpness: 4000, issues: [], excluded: false, group: null },
];

export function imageIndex(images: Image[] = IMAGES): ImageIndex {
  const pick = <T,>(read: (image: Image) => T) => images.map(read);
  const nulls = pick(() => null);
  return {
    count: images.length,
    image_id: pick((image) => image.id),
    rel_path: pick((image) => image.rel),
    file_name: pick((image) => image.rel.split("/").at(-1) ?? image.rel),
    label: pick((image) => image.label),
    split: pick((image) => image.split),
    format: pick((image) => (image.ok === false ? null : "png")),
    color_mode: pick(() => "RGB"),
    camera: nulls,
    dominant_colour: pick(() => "#806040"),
    decode_error: pick((image) => (image.ok === false ? "cannot identify image file" : null)),
    decision: nulls,
    width: pick((image) => image.width),
    height: pick((image) => image.height),
    file_bytes: pick((image) => (image.width ? image.width * 40 : 120)),
    bit_depth: pick(() => 8),
    frames: pick(() => 1),
    duplicate_group: pick((image) => image.group),
    megapixels: pick((image) => (image.width && image.height ? (image.width * image.height) / 1e6 : null)),
    aspect_ratio: pick((image) => (image.width && image.height ? image.width / image.height : null)),
    brightness: pick((image) => image.brightness),
    luma_p05: nulls,
    luma_p95: nulls,
    contrast: pick((image) => (image.brightness === null ? null : 0.2)),
    entropy: pick((image) => (image.brightness === null ? null : 7)),
    saturation: pick((image) => (image.brightness === null ? null : 0.3)),
    colourfulness: pick((image) => (image.brightness === null ? null : 40)),
    sharpness: pick((image) => image.sharpness),
    clipped_dark: nulls,
    clipped_light: nulls,
    is_normal: nulls,
    is_grayscale: pick(() => false),
    has_location: pick((image) => image.rel === "dog/1.png"),
    decode_ok: pick((image) => image.ok !== false),
    excluded: pick((image) => image.excluded),
    rule_excluded: pick((image) => image.excluded),
    has_mask: pick(() => false),
    issues: pick((image) => image.issues),
    date_taken: nulls,
  };
}

export const imageReport: ImageReport = {
  sentence: "6 images (5 decoded): 2 labels.",
  source_kind: "folder",
  n_images: 6,
  n_decoded: 5,
  labels: { cat: 4, dog: 2 },
  splits: { train: 5, test: 1 },
  rules: [{ code: "R-VIS-LAY-01", message: "Labels read from folder names.", reason: "ImageFolder convention.", count: 6 }],
  skipped: [],
  skipped_count: 0,
  conflicts: [],
  conflicts_count: 0,
  quality: {
    sentence: "4 images with an issue (R-VQ-01, R-VQ-04, R-VQ-09).",
    findings: [
      { code: "R-VQ-01", title: "Could not be decoded", message: "1 file could not be decoded.", reason: "A broken file makes a whole training batch fail.", effect: "excluded", count: 1, rows: [3], parameters: {}, details: {} },
      { code: "R-VQ-04", title: "Blurry", message: "1 image is blurry.", reason: "Two conditions: relative and absolute.", effect: "flagged", count: 1, rows: [2], parameters: { blur_z: 3, blur_sharpness: 100 }, details: {} },
      { code: "R-VQ-09", title: "Exact duplicate", message: "1 exact copy: the first is kept.", reason: "The same pixels twice inflate the set.", effect: "excluded", count: 1, rows: [0, 1], parameters: {}, details: {} },
    ],
    groups: [{ group: 0, kind: "exact", rows: [0, 1], keep: 0 }],
    leaks: [],
    thresholds: {},
    notes: [],
    recommended_metric: "accuracy",
    class_weight: null,
    excluded: 2,
    flagged: 4,
  },
};

export const qualityRules = [
  { code: "R-VQ-01", title: "Could not be decoded", reason: "A broken file makes a whole training batch fail.", effect: "excluded", parameters: {} },
  { code: "R-VQ-04", title: "Blurry", reason: "Two conditions: relative and absolute.", effect: "flagged", parameters: { blur_z: 3, blur_sharpness: 100 } },
  { code: "R-VQ-05", title: "Too dark", reason: "Underexposed.", effect: "flagged", parameters: {} },
  { code: "R-VQ-09", title: "Exact duplicate", reason: "The same pixels twice inflate the set.", effect: "excluded", parameters: {} },
];
