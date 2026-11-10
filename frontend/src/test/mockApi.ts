import { vi } from "vitest";

type Handler = (body: unknown) => unknown;
type Routes = Record<string, Handler | unknown>;

/**
 * Stub `fetch` with a table of "METHOD /path" → response body (or function of the request body).
 * Unknown routes answer 404 so a missing mock is visible.
 */
export function mockApi(routes: Routes) {
  const calls: { key: string; body: unknown; params: URLSearchParams }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://test");
    const key = `${init?.method ?? "GET"} ${url.pathname.replace("/api/v1", "")}`;
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body;
    calls.push({ key, body, params: url.searchParams });
    if (!(key in routes)) return Response.json({ error: { message: `No mock for ${key}` } }, { status: 404 });
    const route = routes[key];
    const payload = typeof route === "function" ? (route as Handler)(body) : route;
    if (payload instanceof Error) throw payload;
    if (payload instanceof Response) return payload;
    return Response.json(payload);
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

export const health = { status: "ok", version: "0.1.0", db: "ok" };

export function dataset(overrides: Record<string, unknown> = {}) {
  return {
    id: "d1",
    name: "sales",
    source_path: "C:\\data\\sales.csv",
    source_kind: "file",
    format: "csv",
    status: "ready",
    error: null,
    row_count: 1200,
    col_count: 3,
    size_bytes: 20480,
    sample_pct: 100,
    created_at: "2026-10-12T10:00:00Z",
    imported_at: "2026-10-12T10:00:00Z",
    last_opened_at: "2026-10-12T10:00:00Z",
    source_state: "ok",
    job: null,
    ...overrides,
  };
}
