import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderAt } from "../../test/render";
import { dataset, health, mockApi } from "../../test/mockApi";
import { importStepFor } from "./stepTypes";

const tablePreview = {
  kind: "folder",
  format: "other",
  files: [],
  columns: [],
  rows: [],
  row_estimate: null,
  suggested_name: "pets",
  detected: {
    kind: "image",
    format: "images",
    label: "folder of images",
    rule: "R-VIS-SRC-01",
    reason: "11 of its 12 files are images (R-VIS-SRC-01): it is imported as an image set, one row per image.",
    mismatch: false,
  },
};

const imagePreview = (ignored: string[]) => ({
  source_kind: "folder",
  suggested_name: "pets",
  sentence: "11 images found (png 11): 2 labels.",
  n_images: 11,
  bytes: 40_000,
  formats: { png: 11 },
  labels: ignored.includes("R-VIS-LAY-01") ? {} : { cat: 6, dog: 5 },
  splits: {},
  normal: null,
  masks: 0,
  rules: ignored.includes("R-VIS-LAY-01")
    ? []
    : [
        {
          code: "R-VIS-LAY-01",
          message: "Labels read from folder names: cat, dog (R-VIS-LAY-01). Check that your data was read correctly.",
          reason: "ImageFolder convention.",
          count: 11,
        },
      ],
  skipped: [],
  skipped_count: 1,
  conflicts: 0,
  sample: ["cat/0.png"],
  label_file: null,
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("image set import step (106)", () => {
  it("routes an image set to its own step, unless a Parquet table is asked for", () => {
    expect(importStepFor(tablePreview as never, "auto")).toBe("image");
    expect(importStepFor(tablePreview as never, "simple")).toBe("image");
    const parquet = {
      ...tablePreview,
      format: "parquet",
      columns: [{ name: "image", dtype: "STRUCT" }],
      detected: { ...tablePreview.detected, format: "parquet", rule: "R-VIS-SRC-03" },
    };
    expect(importStepFor(parquet as never, "auto")).toBe("image");
    expect(importStepFor(parquet as never, "simple")).toBe("default");
  });

  it("shows how the set is read, undoes a rule and starts the import job", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "pets", path: "/s/pets", format: "images", size: 1 }],
      "POST /sources/preview": tablePreview,
      "POST /images/preview": (body: unknown) =>
        imagePreview((body as { options: { ignore_rules: string[] } }).options.ignore_rules),
      "POST /images/import": {
        dataset: dataset({ id: "d9", format: "images", status: "importing" }),
        job: { id: "j1", status: "running", progress: 0.1, message: "Starting" },
        reused: false,
      },
      "GET /jobs/j1": { id: "j1", status: "running", progress: 0.3, message: "Reading images: 3 of 11" },
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "pets" }));

    expect(await screen.findByText("11 images found (png 11): 2 labels.")).toBeInTheDocument();
    const rules = screen.getByRole("region", { name: "How the file is read" });
    expect(rules).toHaveTextContent("Labels read from folder names: cat, dog (R-VIS-LAY-01).");
    expect(screen.getByText("Images per label")).toBeInTheDocument();
    expect(screen.getByText("Files left out: 1, each with its rule")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(await screen.findByText("R-VIS-LAY-01 undone: not applied to this import.")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("Images per label")).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Import 11 images" }));
    expect(await screen.findByText(/Reading images: 3 of 11/)).toBeInTheDocument();
    const request = calls.find((call) => call.key === "POST /images/import")?.body as {
      path: string;
      options: { ignore_rules: string[] };
    };
    expect(request.path).toBe("/s/pets");
    expect(request.options.ignore_rules).toEqual(["R-VIS-LAY-01"]);
  });

  it("reads a labels file given in the step (c3, R-VIS-LAB-05)", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "pets", path: "/s/pets", format: "images", size: 1 }],
      "POST /sources/preview": tablePreview,
      "POST /images/preview": (body: unknown) => {
        const options = (body as { options: { ignore_rules: string[]; labels: string | null } }).options;
        return { ...imagePreview(options.ignore_rules), label_file: options.labels ? "labels.csv" : null };
      },
      "POST /images/import": {
        dataset: dataset({ id: "d9", format: "images", status: "importing" }),
        job: { id: "j1", status: "running", progress: 0.1, message: "Starting" },
        reused: false,
      },
      "GET /jobs/j1": { id: "j1", status: "running", progress: 0.3, message: "Reading images: 3 of 11" },
    });
    renderAt("/");
    fireEvent.click(await screen.findByRole("button", { name: "pets" }));
    fireEvent.click(await screen.findByText("Labels file"));
    fireEvent.change(screen.getByRole("textbox", { name: "Path of the file" }), { target: { value: "C:/data/labels.csv" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Label column" }), { target: { value: "breed" } });
    fireEvent.click(screen.getByRole("button", { name: "Read with this file" }));
    expect(await screen.findByText("labels.csv read (R-VIS-LAB-01)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Import 11 images" }));
    await waitFor(() => expect(calls.some((call) => call.key === "POST /images/import")).toBe(true));
    const request = calls.find((call) => call.key === "POST /images/import")?.body as {
      options: { labels: string; label_column: string };
    };
    expect(request.options.labels).toBe("C:/data/labels.csv");
    expect(request.options.label_column).toBe("breed");
  });
});
