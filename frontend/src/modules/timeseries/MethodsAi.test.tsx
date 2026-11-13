import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MethodsRow } from "./Methods";

describe("AI models in + Add method (028 phase B)", () => {
  it("are folded, never ticked, and a model to download asks first instead of being ticked", () => {
    const onChange = vi.fn();
    const onPick = vi.fn((key: string) => key === "chronos_bolt_small");
    render(
      <MethodsRow
        section="forecast"
        chosen={["naive"]}
        onChange={onChange}
        onPick={onPick}
        tag={(key) => (key === "chronos_bolt_small" ? "Download · 191 MB" : null)}
        unavailable={{ nhits: "Needs at least 500 periods" }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Add method" }));

    expect(screen.getByText("Pretrained (AI)")).toBeInTheDocument();
    expect(screen.getByText("Download · 191 MB")).toBeInTheDocument();
    expect(screen.getByText("Needs at least 500 periods")).toBeInTheDocument();
    expect(screen.getByRole("menuitemcheckbox", { name: /N-HiTS/ })).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: /Chronos-Bolt Small/ }));
    expect(onPick).toHaveBeenCalledWith("chronos_bolt_small");
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Add method" }));
    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: /Chronos-Bolt Tiny/ }));
    expect(onChange).toHaveBeenCalledWith(["naive", "chronos_bolt_tiny"]);
  });
});
