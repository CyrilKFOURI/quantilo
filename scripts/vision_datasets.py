"""Download the vision test sets (120) outside the repository, checked file by file.

Direct HTTPS from Hugging Face (`resolve/<pinned revision>/...`), standard library only: no Kaggle
CLI, no account, no token. Every file is checked against its size and its SHA-256 (Git LFS
object id) or its Git blob hash; a dataset above 1 GB is refused. Each dataset folder gets a
README with its licence and citation, and a manifest.json of the checked files.

The default destination is `<parent of the repository>/data-corpus/vision/`, never the
repository itself: these images are never committed, and MVTec AD (CC BY-NC-SA 4.0) is never
redistributed (local, non-commercial test use only).

Usage:
    python scripts/vision_datasets.py list
    python scripts/vision_datasets.py get beans
    python scripts/vision_datasets.py get mvtec-bottle --accept-noncommercial
    python scripts/vision_datasets.py verify beans mvtec-bottle
Options: --dest DIR (outside the repository).
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT.parent / "data-corpus" / "vision"
HF = "https://huggingface.co"
MAX_BYTES = 1_000_000_000  # a test set above 1 GB is refused
USER_AGENT = "quantilo-vision-datasets/1"
ATTEMPTS = 30  # Hugging Face connections are often reset: retry, resume with Range
WORKERS = 4  # files downloaded at once (many small files for MVTec)
TIMEOUT = 60
CHUNK = 1 << 20

# Mirrors known to carry MVTec AD under a wrong licence (MIT): never used.
MISLABELLED = ("MSherbinii/mvtec-ad",)


@dataclass(frozen=True)
class FileSpec:
    path: str  # path in the repository (and under the dataset folder)
    size: int
    sha256: str | None = None  # Git LFS files
    git_sha1: str | None = None  # small files kept in Git


@dataclass(frozen=True)
class Dataset:
    name: str
    repo: str  # Hugging Face dataset id
    revision: str  # pinned commit: the files cannot change under us
    licence: str  # Hugging Face licence id, checked on the dataset card
    licence_name: str
    task: str
    citation: str
    use: str  # what the licence allows, written in the README
    files: tuple[FileSpec, ...] = ()  # pinned list, or…
    prefix: str | None = None  # …a folder listed at the pinned revision
    expected_files: int | None = None
    expected_bytes: int | None = None
    noncommercial: bool = False
    licence_marker: str | None = None  # text the downloaded licence file must contain
    notes: tuple[str, ...] = field(default_factory=tuple)


BEANS = Dataset(
    name="beans",
    repo="AI-Lab-Makerere/beans",
    revision="27aa014ce09b193e1a6f58112d4a66e0eddb69c5",
    licence="mit",
    licence_name="MIT",
    task="classification, 3 classes (angular_leaf_spot, bean_rust, healthy), splits provided",
    citation=(
        "@ONLINE {beansdata,\n"
        '    author="Makerere AI Lab",\n'
        '    title="Bean disease dataset",\n'
        '    month="January",\n'
        '    year="2020",\n'
        '    url="https://github.com/AI-Lab-Makerere/ibean/"\n'
        "}"
    ),
    use="MIT: free use with the copyright notice. Still kept out of the repository (size).",
    files=(
        FileSpec(
            "data/train-00000-of-00001.parquet",
            143_773_054,
            "7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19",
        ),
        FileSpec(
            "data/validation-00000-of-00001.parquet",
            18_499_710,
            "33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f",
        ),
        FileSpec(
            "data/test-00000-of-00001.parquet",
            17_706_070,
            "534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27",
        ),
    ),
)

MVTEC_BOTTLE = Dataset(
    name="mvtec-bottle",
    repo="foersben/mvtec-ad",
    revision="c75b39616f84db43677bcc8228caaafaf5096d7f",
    licence="cc-by-nc-sa-4.0",
    licence_name="CC BY-NC-SA 4.0",
    task="anomaly detection, one class (bottle): 209 normal training images, 83 test images "
    "(good, broken_large, broken_small, contamination), pixel masks",
    citation=(
        "Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: "
        "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for "
        "Unsupervised Anomaly Detection. International Journal of Computer Vision "
        "129(4):1038-1059, 2021.\n"
        "Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD - A "
        "Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR 2019."
    ),
    use=(
        "CC BY-NC-SA 4.0: local, non-commercial test use only, with the citation above. NEVER "
        "commit these files, never put them in a Docker image, never redistribute them "
        "(including derived images). Mirrors labelled MIT (e.g. MSherbinii/mvtec-ad-*) are "
        "mislabelled: the licence of MVTec AD is CC BY-NC-SA 4.0 whatever the mirror says."
    ),
    prefix="bottle",
    expected_files=357,
    expected_bytes=156_570_806,
    noncommercial=True,
    licence_marker="Attribution-NonCommercial-ShareAlike 4.0",
)

DATASETS = {d.name: d for d in (BEANS, MVTEC_BOTTLE)}


class DatasetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# HTTP


def _open(url: str, start: int = 0):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)


def _retry(action, what: str):
    delay = 2.0
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 404):
                raise DatasetError(f"{what}: HTTP {error.code}") from error
            if attempt == ATTEMPTS:
                raise DatasetError(f"{what}: {error}") from error
            time.sleep(delay)
            delay = min(delay * 1.5, 15)
    raise AssertionError("unreachable")


def _json(url: str):
    def fetch():
        with _open(url) as response:
            return json.loads(response.read().decode("utf-8"))

    return _retry(fetch, url)


def _listing(dataset: Dataset) -> list[FileSpec]:
    """Files under `prefix` at the pinned revision, with their sizes and hashes."""
    url = (
        f"{HF}/api/datasets/{dataset.repo}/tree/{dataset.revision}/{dataset.prefix}?recursive=true"
    )
    found: list[FileSpec] = []
    while url:
        page_url = url

        def fetch(page_url: str = page_url):
            with _open(page_url) as response:
                link = response.headers.get("Link") or ""
                return json.loads(response.read().decode("utf-8")), link

        entries, link = _retry(fetch, page_url)
        for entry in entries:
            if entry.get("type") != "file":
                continue
            lfs = entry.get("lfs")
            found.append(
                FileSpec(
                    path=entry["path"],
                    size=int(entry["size"]),
                    sha256=lfs["oid"] if lfs else None,
                    git_sha1=None if lfs else entry["oid"],
                )
            )
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1 : part.find(">")]
    return found


# ---------------------------------------------------------------------------------------------
# Checks


def _hashes(path: Path) -> tuple[str, str]:
    """(SHA-256 of the file, Git blob SHA-1)."""
    size = path.stat().st_size
    sha256, blob = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def _matches(path: Path, spec: FileSpec) -> bool:
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    sha256, blob = _hashes(path)
    if spec.sha256 is not None:
        return sha256 == spec.sha256
    return spec.git_sha1 is not None and blob == spec.git_sha1


def check_licence(dataset: Dataset) -> None:
    """The dataset card at the pinned revision must carry the expected licence."""
    if dataset.repo.startswith(MISLABELLED):
        raise DatasetError(f"{dataset.repo} carries a wrong licence: not used.")
    info = _json(f"{HF}/api/datasets/{dataset.repo}/revision/{dataset.revision}")
    licence = (info.get("cardData") or {}).get("license")
    found = licence if isinstance(licence, list) else [licence]
    if dataset.licence not in found:
        raise DatasetError(
            f"{dataset.repo} says licence {licence!r}, expected {dataset.licence!r}: refused "
            "(a mirror with a wrong licence cannot be trusted)."
        )


def check_destination(dest: Path) -> Path:
    dest = dest.resolve()
    if dest.is_relative_to(ROOT):
        raise DatasetError(f"{dest} is inside the repository: images are never committed.")
    return dest


def check_space(dest: Path, needed: int) -> None:
    if needed > MAX_BYTES:
        raise DatasetError(f"{needed / 1e6:,.0f} MB is above the 1 GB limit of a test set.")
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < needed * 2:
        raise DatasetError(f"{free / 1e9:.1f} GB free: not enough for {needed / 1e6:,.0f} MB.")


# ---------------------------------------------------------------------------------------------
# Download


def _download(dataset: Dataset, spec: FileSpec, folder: Path) -> str:
    target = folder / spec.path
    if _matches(target, spec):
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = f"{HF}/datasets/{dataset.repo}/resolve/{dataset.revision}/{spec.path}"

    def fetch() -> None:
        start = part.stat().st_size if part.exists() else 0
        if start >= spec.size:
            part.unlink()
            start = 0
        with _open(url, start) as response:
            if start and response.status != 206:  # the server ignored Range: start over
                start = 0
            with part.open("ab" if start else "wb") as handle:
                while chunk := response.read(CHUNK):
                    handle.write(chunk)
        if part.stat().st_size != spec.size:
            raise ConnectionError(f"{part.stat().st_size} of {spec.size} bytes")

    _retry(fetch, spec.path)
    part.replace(target)
    if not _matches(target, spec):
        target.unlink()
        raise DatasetError(f"{spec.path}: hash differs from the pinned one, file removed.")
    return "downloaded"


def files_of(dataset: Dataset) -> list[FileSpec]:
    files = list(dataset.files) if dataset.files else _listing(dataset)
    total = sum(f.size for f in files)
    if dataset.expected_files is not None and len(files) != dataset.expected_files:
        raise DatasetError(f"{len(files)} files listed, {dataset.expected_files} expected.")
    if dataset.expected_bytes is not None and total != dataset.expected_bytes:
        raise DatasetError(f"{total:,} bytes listed, {dataset.expected_bytes:,} expected.")
    return files


def readme(dataset: Dataset, files: list[FileSpec]) -> str:
    total = sum(f.size for f in files)
    return "\n".join(
        [
            f"# {dataset.name}",
            "",
            f"Source: {HF}/datasets/{dataset.repo} (revision `{dataset.revision}`)",
            f"Task: {dataset.task}",
            f"Files: {len(files)}, {total / 1e6:,.1f} MB, each checked by size and hash "
            "(manifest.json).",
            f"Downloaded: {datetime.now(UTC):%Y-%m-%d} by scripts/vision_datasets.py "
            "(Quantilo, 120).",
            "",
            f"## Licence: {dataset.licence_name}",
            "",
            dataset.use,
            "",
            "## Citation",
            "",
            "```",
            dataset.citation,
            "```",
            "",
        ]
    )


def get(dataset: Dataset, dest: Path, accept_noncommercial: bool = False) -> Path:
    if dataset.noncommercial and not accept_noncommercial:
        raise DatasetError(
            f"{dataset.name} is {dataset.licence_name}: local, non-commercial test use only. "
            "Re-run with --accept-noncommercial to download it for that use."
        )
    dest = check_destination(dest)
    check_licence(dataset)
    files = files_of(dataset)
    folder = dest / dataset.name
    check_space(dest, sum(f.size for f in files))
    folder.mkdir(parents=True, exist_ok=True)
    done = {"kept": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [pool.submit(_download, dataset, spec, folder) for spec in files]
        for i, job in enumerate(as_completed(jobs), 1):
            done[job.result()] += 1
            if i % 50 == 0 or i == len(files):
                print(f"  {dataset.name}: {i}/{len(files)} files", flush=True)
    if dataset.licence_marker is not None:
        texts = [
            (folder / f.path).read_text(encoding="utf-8", errors="replace")
            for f in files
            if f.path.endswith("license.txt")
        ]
        if not any(dataset.licence_marker in text for text in texts):
            raise DatasetError(f"No licence file mentions {dataset.licence_marker!r}.")
    manifest = {
        "name": dataset.name,
        "repo": dataset.repo,
        "revision": dataset.revision,
        "licence": dataset.licence,
        "files": [f.__dict__ for f in files],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (folder / "README.md").write_text(readme(dataset, files), encoding="utf-8")
    print(
        f"{dataset.name}: {done['downloaded']} downloaded, {done['kept']} already there -> {folder}"
    )
    return folder


def verify(dataset: Dataset, dest: Path) -> bool:
    folder = check_destination(dest) / dataset.name
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        print(f"{dataset.name}: not downloaded ({folder})")
        return False
    specs = [FileSpec(**f) for f in json.loads(manifest.read_text(encoding="utf-8"))["files"]]
    bad = [s.path for s in specs if not _matches(folder / s.path, s)]
    print(
        f"{dataset.name}: {len(specs) - len(bad)}/{len(specs)} files match"
        + (f"; differ: {bad[:5]}" if bad else "")
    )
    return not bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("action", choices=["list", "get", "verify"])
    parser.add_argument("names", nargs="*", default=[])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--accept-noncommercial", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "list":
        for d in DATASETS.values():
            size = f"{d.expected_bytes or sum(f.size for f in d.files):,} bytes"
            print(f"{d.name:14} {d.licence_name:16} {size:>20}  {d.repo}  {d.task}")
        return 0
    names = args.names or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
    try:
        if args.action == "verify":
            return 0 if all([verify(DATASETS[n], args.dest) for n in names]) else 1
        for n in names:
            get(DATASETS[n], args.dest, args.accept_noncommercial)
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
