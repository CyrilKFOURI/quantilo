"""The web site's static pages and headers (lite/static_site.py, feature 081)."""

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

LITE = Path(__file__).resolve().parents[3] / "lite"
sys.path.insert(0, str(LITE))

import static_site  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_markdown_subset_matches_the_app_reader():
    out = static_site.markdown_to_html(
        "# Title\n\nSome **bold**, *soft* and `code`.\nSame paragraph.\n\n## Two\n## Two\n\n"
        "- [Terms](terms)\n- [Web](https://example.org)\n- [Bad](javascript:void)\n\n"
        "| A | B |\n|---|---|\n| 1 | x \\| y |\n\n<b>not html</b>"
    )
    assert '<h1 id="title">Title</h1>' in out
    assert (
        "<p>Some <strong>bold</strong>, <em>soft</em> and <code>code</code>. Same paragraph.</p>"
        in out
    )
    assert '<h2 id="two">Two</h2>' in out and '<h2 id="two-1">Two</h2>' in out
    assert '<li><a href="/terms">Terms</a></li>' in out
    assert '<a href="https://example.org">Web</a>' in out
    assert "<li>Bad</li>" in out  # a non-https link is shown as plain text
    assert "<td>x | y</td>" in out
    assert "&lt;b&gt;not html&lt;/b&gt;" in out


def test_script_hashes_cover_inline_scripts_only():
    page = (
        '<script>var a = 1;</script><script type="module" src="/x.js"></script><script> </script>'
    )
    digest = base64.b64encode(hashlib.sha256(b"var a = 1;").digest()).decode()
    assert static_site.script_hashes(page) == [f"'sha256-{digest}'"]


def test_pyodide_origin_is_read_from_the_worker():
    source = static_site.WORKER.read_text(encoding="utf-8")
    assert static_site.pyodide_origin(source) == "https://cdn.jsdelivr.net"
    with pytest.raises(ValueError):
        static_site.pyodide_origin("const nothing = 1;")


def test_security_headers():
    headers = {
        h["key"]: h["value"] for h in static_site.security_headers(["'sha256-x'"], "https://cdn")
    }
    csp = dict(part.split(" ", 1) for part in headers["Content-Security-Policy"].split("; "))
    assert csp["script-src"].split() == [
        "'self'",
        "'wasm-unsafe-eval'",
        "'unsafe-eval'",
        "https://cdn",
        "'sha256-x'",
    ]
    assert "'unsafe-inline'" not in csp["script-src"]
    assert csp["connect-src"] == "'self' https://cdn"
    assert csp["frame-ancestors"] == "'none'"
    assert csp["object-src"] == "'none'"
    assert headers["Strict-Transport-Security"].startswith("max-age=63072000")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in headers["Permissions-Policy"]


def test_write_site(tmp_path: Path):
    (tmp_path / "index.html").write_text("<script>theme()</script>", encoding="utf-8")
    written = static_site.write_site(tmp_path)
    assert written == ["privacy.html", "terms.html", "legal.html", "404.html", "vercel.json"]

    config = json.loads((tmp_path / "vercel.json").read_text(encoding="utf-8"))
    assert config["cleanUrls"] is True
    assert "rewrites" not in config  # an unknown path must stay a 404 (no soft 404)
    csp = next(
        h["value"] for h in config["headers"][0]["headers"] if h["key"] == "Content-Security-Policy"
    )
    assert "https://cdn.jsdelivr.net" in csp and "'sha256-" in csp

    lost = (tmp_path / "404.html").read_text(encoding="utf-8")
    assert "<h1>Page not found</h1>" in lost and 'href="/"' in lost
    assert "<script" not in lost  # works without JavaScript, nothing for the CSP to allow

    notice = (tmp_path / "legal.html").read_text(encoding="utf-8")
    assert "Vercel Inc." in notice and "440 N Barranca Avenue" in notice
    assert "Mentions légales" in notice
    assert notice.count("LCEN, art. 1-1 II") == 2  # anonymity option: no postal address published
    assert 'aria-current="page"' in notice
    privacy = (tmp_path / "privacy.html").read_text(encoding="utf-8")
    assert "Whole rows are never sent" in privacy
    assert '<a href="/legal">Legal notice</a>' in privacy
