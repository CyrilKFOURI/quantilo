"""Start the app on this computer and open it in the browser (run.bat / run.sh call this).

Serves the API and the frontend built into backend/static, on the first free port from 8000.
"""

import os
import socket
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from app.config import adopt_legacy_environment

HERE = Path(__file__).resolve().parent


def free_port(start: int = 8000) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("No free port between 8000 and 8049.")


def open_when_ready(url: str) -> None:
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/v1/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    if os.environ.get("QUANTILO_NO_BROWSER") != "1":
        webbrowser.open(url)


def main() -> None:
    adopt_legacy_environment()  # DATA_EXPLORER_PORT… from before the rename (043)
    os.environ.setdefault("QUANTILO_STATIC_DIR", str(HERE / "static"))
    os.environ.setdefault("QUANTILO_SAMPLES_DIR", str(HERE.parent / "samples"))
    port = int(os.environ.get("QUANTILO_PORT") or free_port())
    url = f"http://localhost:{port}"
    print(f"\n  Quantilo: {url}  (Ctrl+C to stop)\n")
    threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, app_dir=str(HERE), log_level="warning")


if __name__ == "__main__":
    main()
