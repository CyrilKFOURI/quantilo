import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";

/** Pinned rows (099): routes /api/v1/datasets/{id}/pins (backend/app/api/pins.py). Shared by the
 *  Data page, Unusual rows (Segments) and Clean, so it lives in features/. */

type Schemas = components["schemas"];
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

export type PinnedRows = Complete<Schemas["PinnedRows"]>;
export type Pin = PinnedRows["pins"][number];
export type PinLabel = PinnedRows["labels"][number];
export type PinnedComparison = Complete<Schemas["PinnedComparison"]>;
export type PinnedTrait = PinnedComparison["traits"][number];

const path = (id: string) => `/datasets/${id}/pins`;

export const pinsApi = {
  list: (id: string) => request<PinnedRows>("GET", path(id)),
  pin: (id: string, rows: number[], label: string, color: number) => request<PinnedRows>("PUT", path(id), { rows, label, color }),
  unpin: (id: string, keys: string[]) => request<PinnedRows>("POST", `${path(id)}/remove`, { keys }),
  changeLabel: (id: string, change: { label: string; to?: string; color?: number }) => request<PinnedRows>("PATCH", `${path(id)}/labels`, change),
  compare: (id: string, label: string | null) =>
    request<PinnedComparison>("GET", `${path(id)}/compare${label === null ? "" : `?${new URLSearchParams({ label })}`}`),
};

export const pinsKey = (id: string) => ["datasets", id, "pins"];

/** The dataset's pins: a SQLite read, plus a key check of the pinned rows only (none: nothing). */
export function usePins(id: string, enabled = true) {
  return useQuery({ queryKey: pinsKey(id), queryFn: () => pinsApi.list(id), enabled, staleTime: 30_000 });
}

export function usePinActions(id: string) {
  const client = useQueryClient();
  const done = (found: PinnedRows) => {
    client.setQueryData(pinsKey(id), found);
    void client.invalidateQueries({ queryKey: [...pinsKey(id), "compare"] });
    // 099 bis: a table narrowed to the pinned rows (R-PIN-17) reads them again
    void client.invalidateQueries({ queryKey: ["datasets", id, "rows", { pinned: true }] });
  };
  return {
    pin: useMutation({
      mutationFn: ({ rows, label, color }: { rows: number[]; label: string; color: number }) => pinsApi.pin(id, rows, label, color),
      onSuccess: done,
    }),
    unpin: useMutation({ mutationFn: (keys: string[]) => pinsApi.unpin(id, keys), onSuccess: done }),
    changeLabel: useMutation({
      mutationFn: (change: { label: string; to?: string; color?: number }) => pinsApi.changeLabel(id, change),
      onSuccess: done,
    }),
  };
}

/** R-PIN-05: computed only once asked (`label` undefined: not asked; null: every pinned row). */
export function usePinComparison(id: string, label: string | null | undefined) {
  return useQuery({
    queryKey: [...pinsKey(id), "compare", label],
    queryFn: () => pinsApi.compare(id, label ?? null),
    enabled: label !== undefined,
    retry: false,
  });
}
