"""Pack the Python the lite (browser) build runs: the engine, the backend app, the samples, and
the few pure-Python wheels Pyodide does not ship (so a visit never calls PyPI). Then write the
site's static pages and vercel.json (static_site.py, feature 081).

Usage: python lite/bundle.py <dist-lite folder>   (run by `npm run build:lite`, with pip)
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from static_site import write_site

ROOT = Path(__file__).resolve().parents[1]
PARTS = {
    "engine/quantilo": ROOT / "packages/quantilo/src/quantilo",
    "backend/app": ROOT / "backend/app",
    "samples": ROOT / "samples",
}
SKIPPED = {"__pycache__", "static"}
# Not in Pyodide's distribution; installed with micropip from the site itself.
WHEELS = ["openpyxl", "et-xmlfile", "pydantic-settings", "python-dotenv"]


def pack(out: Path) -> int:
    count = 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "lite/boot.py", "boot.py")
        for prefix, folder in PARTS.items():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not SKIPPED.intersection(path.relative_to(folder).parts):
                    archive.write(path, f"{prefix}/{path.relative_to(folder).as_posix()}")
                    count += 1
    return count


def wheels(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--quiet",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(folder),
            *WHEELS,
        ],
        check=True,
    )
    names = sorted(path.name for path in folder.glob("*.whl"))
    (folder / "index.json").write_text(json.dumps(names))
    return names


def main(dist: Path) -> None:
    zipped = dist / "python.zip"
    count = pack(zipped)
    print(f"{zipped}: {count} files, {zipped.stat().st_size / 1e6:.1f} MB")
    print(f"{dist / 'wheels'}: {', '.join(wheels(dist / 'wheels'))}")
    print(f"{dist}: {', '.join(write_site(dist))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
