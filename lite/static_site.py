"""The static part of the web site (feature 081), written into dist-lite after the Vite build:

- privacy.html, terms.html, legal.html: the legal texts of docs/legal/*.md (the app reads the same
  files), readable without JavaScript;
- 404.html: what Vercel serves, with the 404 status, for any address that is not a file;
- vercel.json: clean URLs (/privacy) and the security headers. The Content-Security-Policy gets the
  sha256 of every inline script of index.html and the origin Pyodide is loaded from (worker.ts), so
  it cannot drift from the build.

Usage: called by lite/bundle.py (`npm run build:lite`), or
`python lite/static_site.py <dist-lite folder>`.
"""

import base64
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ROOT / "docs/legal"
TOKENS = ROOT / "frontend/src/styles/tokens.css"
WORKER = ROOT / "frontend/src/lite/worker.ts"
REPO = "https://github.com/CyrilKFOURI/quantilo"

# slug of docs/legal/<slug>.md -> (published file, title)
PAGES = {
    "privacy": ("privacy.html", "Privacy"),
    "terms": ("terms.html", "Terms of use"),
    "notice": ("legal.html", "Legal notice"),
}

# --- Markdown (the subset of frontend/src/features/help/markdown.tsx) ----------------------------

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")
LIST_ITEM = re.compile(r"^\s*(-|\d+\.)\s+")


def slug(text: str) -> str:
    text = re.sub(r"[`*_]", "", text.lower())
    return re.sub(r"[\W_]+", "-", text).strip("-")


def href(target: str) -> str | None:
    """Another legal text -> its clean URL; an https address as is; anything else -> plain text."""
    if target in PAGES:
        return "/" + PAGES[target][0].removesuffix(".html")
    return target if target.startswith("https://") else None


def inline(text: str) -> str:
    out: list[str] = []
    for part in INLINE.split(text.replace("\\|", "|")):
        if not part:
            continue
        if part.startswith("**"):
            out.append(f"<strong>{html.escape(part[2:-2])}</strong>")
        elif part.startswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        elif part.startswith("[") and (match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)):
            label, target = match.groups()
            url = href(target)
            out.append(
                f'<a href="{html.escape(url)}">{html.escape(label)}</a>'
                if url
                else html.escape(label)
            )
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(f"<em>{html.escape(part[1:-1])}</em>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def cells(line: str) -> list[str]:
    inner = re.sub(r"^\||\|$", "", line.strip())
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", inner)]


def markdown_to_html(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    used: dict[str, int] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if heading := re.match(r"^(#{1,3})\s+(.*)$", line):
            level, text = len(heading.group(1)), heading.group(2)
            base = slug(text)
            seen = used.get(base, 0)
            used[base] = seen + 1
            anchor = f"{base}-{seen}" if seen else base
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            i += 1
        elif line.strip().startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = cells(lines[i])
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row):
                    rows.append(row)
                i += 1
            head = "".join(f'<th scope="col">{inline(c)}</th>' for c in rows[0]) if rows else ""
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in rows[1:]
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif LIST_ITEM.match(line):
            tag = "ol" if re.match(r"^\s*\d+\.", line) else "ul"
            items: list[str] = []
            while i < len(lines) and LIST_ITEM.match(lines[i]):
                items.append(f"<li>{inline(LIST_ITEM.sub('', lines[i]))}</li>")
                i += 1
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
        elif line.strip():
            joined: list[str] = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"^(#|\||\s*(-|\d+\.)\s)", lines[i])
            ):
                joined.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(joined))}</p>")
        else:
            i += 1
    return "\n".join(out)


# --- Pages ----------------------------------------------------------------------------------------

LOGO = (
    '<svg viewBox="2 2 58 58" width="20" height="20" fill="none" stroke-linecap="round" '
    'aria-hidden="true">'
    '<path d="M47.54 23.51A20 20 0 1 1 36.49 12.46" stroke="var(--ink-1)" stroke-width="7.5"/>'
    '<path d="M37.49 39.49 52.33 54.33" stroke="var(--ink-1)" stroke-width="7.5"/>'
    '<path d="M29 31 53.04 6.96" stroke="var(--accent)" stroke-width="7"/></svg>'
)

# The empty chart of the app's not-found page (NotFoundPage.tsx).
EMPTY_AXES = (
    '<svg class="axes" viewBox="0 0 160 96" width="160" height="96" fill="none" aria-hidden="true">'
    + "".join(
        f'<path d="M24 {y}H152" stroke="var(--line)" stroke-dasharray="3 4"/>' for y in (20, 42, 64)
    )
    + '<path d="M24 8V84H152" stroke="var(--ink-3)" stroke-width="1.25" stroke-linecap="round"/>'
    + "".join(
        f'<path d="M{x} 84v4" stroke="var(--ink-3)" stroke-width="1.25" stroke-linecap="round"/>'
        for x in (56, 88, 120, 152)
    )
    + '<circle cx="88" cy="84" r="3" fill="var(--accent)"/></svg>'
)

STYLE = """
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--ink-1);font:14px/1.6 var(--font-ui);
-webkit-font-smoothing:antialiased;text-size-adjust:100%}
a{color:var(--accent);text-underline-offset:2px}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.bar{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
padding:10px var(--gutter);border-bottom:1px solid var(--line);background:var(--surface-0)}
.brand{display:flex;align-items:center;gap:8px;color:var(--ink-1);font-weight:600;text-decoration:none}
.bar nav{display:flex;gap:4px 16px;flex-wrap:wrap;font-size:13px}
.bar nav a{color:var(--ink-2);text-decoration:none;padding:4px 0}
.bar nav a[aria-current=page]{color:var(--ink-1);font-weight:500;
box-shadow:inset 0 -2px var(--accent)}
main{max-width:46rem;margin:0 auto;padding:32px var(--gutter) 64px}
h1{font-size:20px;line-height:1.3;letter-spacing:-.01em;margin:0 0 4px}
h2{font-size:15px;margin:40px 0 0;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:24px 0 0}
p,ul,ol{color:var(--ink-2);margin:12px 0 0}
ul,ol{padding-left:20px}
li+li{margin-top:6px}
li::marker{color:var(--ink-3)}
strong{color:var(--ink-1);font-weight:600}
code{font-family:var(--font-code);font-size:12px;background:var(--surface-2);color:var(--ink-1);
padding:1px 4px;border-radius:4px}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
th,td{text-align:left;padding:6px 12px;border-top:1px solid var(--line);vertical-align:top}
.foot{max-width:46rem;margin:0 auto;padding:16px var(--gutter) 40px;
border-top:1px solid var(--line);font-size:12px;color:var(--ink-3)}
.foot a{color:var(--ink-3)}
.lost{padding-top:64px}
.lost .axes{display:block;margin-bottom:24px}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}
.button{display:inline-flex;align-items:center;min-height:32px;padding:0 12px;border-radius:6px;
font-size:13px;text-decoration:none;border:1px solid var(--line);background:var(--surface-1);
color:var(--ink-1)}
.button:hover{background:var(--surface-2);text-decoration:none}
.button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);
font-weight:500}
.button.primary:hover{filter:brightness(.95);background:var(--accent)}
@media (pointer:coarse){.bar nav a,.foot a{display:inline-block;padding:10px 0}
.button{min-height:44px}}
"""


def page(title: str, body: str, *, active: str | None = None, description: str = "") -> str:
    tokens = TOKENS.read_text(encoding="utf-8")
    nav = "".join(
        f'<a href="/{file.removesuffix(".html")}"'
        + (' aria-current="page"' if slug_ == active else "")
        + f">{html.escape(name)}</a>"
        for slug_, (file, name) in PAGES.items()
    )
    meta = f'<meta name="description" content="{html.escape(description)}">' if description else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{html.escape(title)} · Quantilo</title>
{meta}
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{tokens}{STYLE}</style>
</head>
<body>
<header class="bar">
<a class="brand" href="/">{LOGO}<span>Quantilo</span></a>
<nav aria-label="Legal">{nav}</nav>
</header>
<main>
{body}
</main>
<footer class="foot">Quantilo is open-source software under the MIT licence ·
<a href="/">Open the app</a> · <a href="{REPO}">Source code</a></footer>
</body>
</html>
"""


def not_found_page() -> str:
    body = f"""<div class="lost">
{EMPTY_AXES}
<h1>Page not found</h1>
<p>There is no page at this address. The link may be mistyped or out of date.</p>
<div class="actions">
<a class="button primary" href="/">Open Quantilo</a>
<a class="button" href="/#/help">Read the guide</a>
<a class="button" href="{REPO}/issues">Report a broken link</a>
</div>
</div>"""
    return page("Page not found", body)


# --- Headers --------------------------------------------------------------------------------------

INLINE_SCRIPT = re.compile(
    r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE
)


def script_hashes(index_html: str) -> list[str]:
    """CSP sources allowing exactly the inline scripts of a page (the theme, set before paint)."""
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"
        for body in INLINE_SCRIPT.findall(index_html)
        if body.strip()
    ]


def pyodide_origin(worker_source: str) -> str:
    match = re.search(r"INDEX_URL = `(https://[^/`]+)/", worker_source)
    if not match:
        raise ValueError("Pyodide's INDEX_URL not found in worker.ts")
    return match.group(1)


def content_security_policy(hashes: list[str], cdn: str) -> str:
    directives = {
        "default-src": ["'self'"],
        # wasm-unsafe-eval: Pyodide compiles WebAssembly; unsafe-eval: plotly.js's WebGL traces
        # (regl, scattergl) build their shaders' code with Function. cdn: pyodide.mjs (worker).
        "script-src": ["'self'", "'wasm-unsafe-eval'", "'unsafe-eval'", cdn, *hashes],
        # plotly.js and React set inline styles; inline scripts stay forbidden.
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "blob:"],
        "font-src": ["'self'", "data:"],
        # Pyodide's packages and lockfile (cdn); python.zip and wheels/ (self).
        "connect-src": ["'self'", cdn],
        "worker-src": ["'self'", "blob:"],
        "manifest-src": ["'self'"],
        "object-src": ["'none'"],
        "base-uri": ["'self'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'none'"],
    }
    return "; ".join(f"{name} {' '.join(values)}" for name, values in directives.items())


def security_headers(hashes: list[str], cdn: str) -> list[dict[str, str]]:
    pairs = {
        "Content-Security-Policy": content_security_policy(hashes, cdn),
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), "
            "bluetooth=(), browsing-topics=(), interest-cohort=()"
        ),
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    return [{"key": key, "value": value} for key, value in pairs.items()]


def vercel_config(hashes: list[str], cdn: str) -> dict:
    """No rewrite to index.html: the app's routes are after # and an unknown path must be a 404."""
    return {
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [
            {"source": "/(.*)", "headers": security_headers(hashes, cdn)},
            {
                "source": "/assets/(.*)",
                "headers": [
                    {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}
                ],
            },
        ],
    }


def write_site(dist: Path) -> list[str]:
    written: list[str] = []
    for slug_, (file, title) in PAGES.items():
        text = (LEGAL / f"{slug_}.md").read_text(encoding="utf-8")
        (dist / file).write_text(
            page(title, markdown_to_html(text), active=slug_, description=f"Quantilo: {title}."),
            encoding="utf-8",
        )
        written.append(file)
    (dist / "404.html").write_text(not_found_page(), encoding="utf-8")
    # read_text turns CRLF into LF, as the browser's HTML parser does before hashing a script
    hashes = script_hashes((dist / "index.html").read_text(encoding="utf-8"))
    cdn = pyodide_origin(WORKER.read_text(encoding="utf-8"))
    (dist / "vercel.json").write_text(
        json.dumps(vercel_config(hashes, cdn), indent=2), encoding="utf-8"
    )
    return [*written, "404.html", "vercel.json"]


if __name__ == "__main__":
    print(", ".join(write_site(Path(sys.argv[1]))))
