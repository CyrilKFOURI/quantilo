"""Run the Quantilo backend inside the browser (Pyodide), without a server.

The frontend's requests to /api/... are handed to `handle()` by the lite worker
(frontend/src/lite/worker.ts), which calls the FastAPI app directly through ASGI.
WebAssembly has no threads: background jobs and sync routes run inline.
Vision waits for Pillow, loaded by the worker when first needed (vision_gate.py, 119).
"""

import concurrent.futures
import json
import os
import sys
from pathlib import Path

sys.path[:0] = ["/lite/engine", "/lite/backend"]

# The bundle is unpacked again on every visit, stamping the samples with that moment: a fixed
# date keeps an imported sample "Ready" instead of "Source changed" on the next visit.
SAMPLES_MTIME = 1_767_225_600  # 2026-01-01
for _sample in Path("/lite/samples").glob("*"):
    os.utime(_sample, (SAMPLES_MTIME, SAMPLES_MTIME))


class InlineExecutor:
    """Background jobs run at once, inside the request that starts them."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


# Patched before the app imports them.
concurrent.futures.ThreadPoolExecutor = InlineExecutor  # type: ignore[misc]

import fastapi.concurrency  # noqa: E402
import fastapi.dependencies.utils  # noqa: E402
import fastapi.routing  # noqa: E402
import starlette.concurrency  # noqa: E402


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


for _module in (
    fastapi.concurrency,
    fastapi.dependencies.utils,
    fastapi.routing,
    starlette.concurrency,
):
    if hasattr(_module, "run_in_threadpool"):
        _module.run_in_threadpool = _run_inline

import anyio.to_thread  # noqa: E402


async def _run_sync_inline(func, *args, abandon_on_cancel=False, cancellable=None, limiter=None):
    return func(*args)


# File and streaming responses read through anyio's worker threads.
anyio.to_thread.run_sync = _run_sync_inline

import vision_gate  # noqa: E402

# 119, R-VIS-WEB-04: Pillow is not loaded at the start; Vision is mounted by `enable_vision()`.
vision_gate.defer()

from app.config import Settings  # noqa: E402
from app.core.jobs import InlineJobRunner  # noqa: E402
from app.main import API_PREFIX, create_app  # noqa: E402

app = create_app(
    Settings(
        home="/home/workspace",
        samples_dir="/lite/samples",
        max_upload_mb=1024,
        keep_version_limit_mb=200,  # IndexedDB space: "Under 200 MB" (031)
        lite=True,  # machine learning: quantilo/ml/browser.py holds back what needs a computer (102)
    ),
    job_runner=InlineJobRunner(),
)
_lifespan = None


def enable_vision() -> None:
    """Called by the worker once Pillow is loaded (R-VIS-WEB-04)."""
    vision_gate.enable(app, API_PREFIX)


def upload_folder(name: str) -> str:
    """A new folder under the workspace's uploads for a folder of images picked in the page (119):
    the worker writes its files there, and it is imported like any upload."""
    folder = app.state.datasets.new_upload_path(name)
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


async def start() -> None:
    global _lifespan
    _lifespan = app.router.lifespan_context(app)
    await _lifespan.__aenter__()


async def handle(method: str, target: str, headers_json: str, body) -> list:
    """One HTTP request through the ASGI app: returns [status, headers as JSON, body bytes]."""
    path, _, query = target.partition("?")
    if body is None or type(body).__name__ == "JsNull":
        payload = b""
    else:
        payload = bytes(body.to_py() if hasattr(body, "to_py") else body)
    headers = [(k.lower().encode(), v.encode()) for k, v in json.loads(headers_json).items()]
    headers.append((b"host", b"localhost"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 0),
        "server": ("localhost", 80),
    }
    sent = False
    status = 500
    response_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
            for key, value in message.get("headers", []):
                response_headers[key.decode()] = value.decode()
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return [status, json.dumps(response_headers), b"".join(chunks)]
