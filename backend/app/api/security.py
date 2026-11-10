"""Protect a local-only API from the browser: other websites must not be able to call it.

- Host check: blocks DNS rebinding (an attacker domain resolving to 127.0.0.1).
- Cross-site check: blocks requests initiated by another site (CSRF, including simple GETs
  such as <img src="http://localhost:8000/api/...">), using Sec-Fetch-Site and Origin.
- Response headers (081): no MIME sniffing, never shown in another site's frame (clickjacking,
  for browsers without Sec-Fetch-*), no address leaked in the Referer.
"""

from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
}


def hostname(value: str) -> str:
    """Host header or origin → bare host name ("[::1]:8000" → "::1")."""
    netloc = urlsplit(value).netloc if "//" in value else value
    if netloc.startswith("["):
        return netloc[1 : netloc.find("]")]
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def forbidden(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=403, content={"error": {"code": "forbidden", "message": message}}
    )


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, allowed_hosts: tuple[str, ...]) -> None:
        super().__init__(app)
        self.allowed_hosts = allowed_hosts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await self.check(request, call_next)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    async def check(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if hostname(request.headers.get("host", "")) not in self.allowed_hosts:
            return forbidden("Unknown host.")
        if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
            return forbidden("Cross-site requests are not allowed.")
        origin = request.headers.get("origin")
        if origin and hostname(origin) not in self.allowed_hosts:
            return forbidden("Cross-origin requests are not allowed.")
        return await call_next(request)


def protect(app: FastAPI, allowed_hosts: tuple[str, ...] = LOCAL_HOSTS) -> None:
    app.add_middleware(LocalOnlyMiddleware, allowed_hosts=allowed_hosts)  # pyright: ignore[reportArgumentType]
