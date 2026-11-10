# Security

Quantilo runs on your own computer and is meant to be reached from it only: Docker publishes it on `127.0.0.1`, and the API refuses requests from other sites (Host, `Sec-Fetch-Site` and `Origin` checks) and network paths.

- **Your data** stays on your computer. Only the optional assistant sends anything out: your messages, column names and types, and analysis results — never whole rows — to the AI provider you configure.
- **API keys** are read from the environment or saved by the app outside the repository; the API never returns them.
- **Headers**: the local app sends `nosniff`, `X-Frame-Options: DENY`, `frame-ancestors 'none'` and `Referrer-Policy: no-referrer`. The web version (Vercel) adds a Content-Security-Policy (scripts from the site, Pyodide's CDN and the hashed theme script only), HSTS and a Permissions-Policy, written into `vercel.json` at build time by `lite/static_site.py`.

To report a vulnerability, please open a [private security advisory](https://github.com/CyrilKFOURI/quantilo/security/advisories/new) rather than a public issue.
