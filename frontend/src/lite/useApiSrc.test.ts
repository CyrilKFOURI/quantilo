import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const apiImage = vi.fn((url: string) => Promise.resolve(`blob:worker${url}`));
vi.mock("./bridge", () => ({ apiImage: (url: string) => apiImage(url) }));
vi.mock("./flag", () => ({ LITE: true, FULL_APP_URL: "https://example.test" }));

describe("An image address read by CSS in the browser version (119)", () => {
  it("becomes the blob URL of the worker's bytes, nothing before they arrive", async () => {
    const { useApiSrc } = await import("./useApiSrc");
    const { result } = renderHook(() => useApiSrc("/api/v1/images/d1/3/mask"));

    expect(result.current).toBeNull();
    await waitFor(() => expect(result.current).toBe("blob:worker/api/v1/images/d1/3/mask"));
    expect(apiImage).toHaveBeenCalledWith("/api/v1/images/d1/3/mask");
  });

  it("leaves an address that is not the API, and no address, as they are", async () => {
    const { useApiSrc } = await import("./useApiSrc");
    expect(renderHook(() => useApiSrc("blob:already")).result.current).toBe("blob:already");
    expect(renderHook(() => useApiSrc(null)).result.current).toBeNull();
  });
});
