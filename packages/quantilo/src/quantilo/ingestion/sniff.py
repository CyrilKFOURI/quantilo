"""What a file is, told by its first bytes (082, rules R-DET-01 to R-DET-04).

The content decides, not the name (R-DET-01): an API export saved as `.txt` is JSON, a Parquet
file renamed `.csv` is Parquet. Every answer carries its rule and a sentence for the user, and a
file that is recognised but not read names what it is (R-DET-03).

Standard library only, bytes in and a dataclass out: the same code runs in the browser (Pyodide).
"""

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quantilo.ingestion.images import _LABELS as IMAGE_LABELS
from quantilo.ingestion.images import sniff_image

HEAD_BYTES = 64 * 1024
Kind = Literal["table", "document", "image", "unsupported"]

# What the name says, for the R-DET-01 comparison.
NAME_FORMATS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".dat": "csv",
    ".data": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".npy": "npy",
    ".json": "json",
    ".geojson": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
    ".png": "image",
    ".apng": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".heic": "image",
    ".heif": "image",
    ".avif": "image",
}
LABELS = {
    "csv": "delimited text (CSV)",
    "parquet": "Parquet",
    "excel": "Excel workbook (.xlsx)",
    "npy": "NumPy array (.npy)",
    "json": "JSON",
    "jsonl": "JSON Lines",
    "xml": "XML",
    "image": "image",
}
JSONL_SHARE = 0.95  # R-DET-02
JSONL_LINES = 50

# (signature, format, label, why it is not read): recognised, never imported (R-DET-03). Short
# text-like marks (ID3) are checked with a binary byte after them, so a CSV never matches.
UNSUPPORTED: list[tuple[bytes, str, str, str]] = [
    (b"%PDF-", "pdf", "PDF", "PDF files are not supported yet."),
    (
        bytes.fromhex("D0CF11E0A1B11AE1"),
        "xls",
        "Excel 97-2003 workbook (.xls) or another Office 97 file",
        "Open it in Excel or LibreOffice and save it as .xlsx (File > Save As), then import the "
        ".xlsx.",
    ),
    (b"\x1f\x8b", "gzip", "gzip archive", "Unpack it first, then import the file inside."),
    (b"7z\xbc\xaf\x27\x1c", "7z", "7-Zip archive", "Unpack it first, then import the file inside."),
    (b"Rar!\x1a\x07", "rar", "RAR archive", "Unpack it first, then import the file inside."),
    (b"SQLite format 3\x00", "sqlite", "SQLite database", "Export its tables first."),
    (b"ID3\x03", "mp3", "MP3 audio file", "Audio files are not supported."),
    (b"ID3\x04", "mp3", "MP3 audio file", "Audio files are not supported."),
    (b"fLaC\x00", "flac", "FLAC audio file", "Audio files are not supported."),
    (b"OggS\x00", "ogg", "Ogg audio or video file", "Audio and video files are not supported."),
    (b"\x1aE\xdf\xa3", "webm", "Matroska / WebM video", "Video files are not supported."),
]


@dataclass(frozen=True)
class Sniff:
    """What the first bytes say: the kind of content, its format, the rule and why."""

    kind: Kind
    format: str  # csv, parquet, excel, npy, json, jsonl, xml, an image format, or what was seen
    label: str  # for the user: "Parquet", "PNG image", "PDF"
    rule: str
    reason: str
    name_format: str | None = None  # what the extension says (None: an unknown extension)

    @property
    def mismatch(self) -> bool:
        """R-DET-01: the name says one thing, the content another (shown, the content wins)."""
        if self.name_format is None:
            return False
        if self.kind == "image":
            return self.name_format != "image"
        if self.format in ("json", "jsonl") and self.name_format in ("json", "jsonl"):
            return False
        return self.name_format != self.format

    def note(self, name: str) -> str | None:
        """The sentence shown at import when the name and the content disagree."""
        if not self.mismatch or self.name_format is None:
            return None
        suffix = Path(name).suffix or "without extension"
        read = f": read as {self.label}" if self.kind in ("table", "document") else ""
        return (
            f"{name} is named {suffix}, but its content is {self.label}{read}. The content "
            "decides (R-DET-01)."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "format": self.format,
            "label": self.label,
            "rule": self.rule,
            "reason": self.reason,
            "name_format": self.name_format,
            "mismatch": self.mismatch,
        }


def _strip(head: bytes) -> bytes:
    for bom in (b"\xef\xbb\xbf",):
        head = head.removeprefix(bom)
    return head.lstrip(b" \t\r\n")


def _utf16(head: bytes) -> bool:
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    sample = head[:2048]
    if len(sample) < 8:
        return False
    odd = sample[1::2].count(0) / max(len(sample) // 2, 1)
    even = sample[0::2].count(0) / max(len(sample) // 2, 1)
    return odd > 0.4 or even > 0.4


def _text_of(head: bytes) -> str:
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return head.decode("utf-16", errors="ignore")
    return head.decode("utf-8", errors="ignore")


def _json_kind(text: str, complete: bool) -> Literal["json", "jsonl"] | None:
    """R-DET-02: one JSON value (possibly cut by the head), or one per line."""
    body = text.lstrip("﻿ \t\r\n")
    try:
        json.loads(body)
        return "json"
    except json.JSONDecodeError:
        # the head cut a big document: its opening must still be a JSON value's, not a CSV cell
        # in brackets ("[a],b")
        rest = body[1:].lstrip()
        opening = rest[:1] in (('"', "}") if body[:1] == "{" else tuple('{["-0123456789tfn]'))
        cut_at_end = not complete and opening
        lines = [line for line in body.splitlines() if line.strip()][:JSONL_LINES]
        if not complete and lines:
            lines = lines[:-1] or lines  # the last line may be cut by the head
        good = 0
        for line in lines:
            try:
                json.loads(line)
                good += 1
            except (ValueError, RecursionError):
                pass
        if len(lines) >= 2 and good / len(lines) >= JSONL_SHARE:
            return "jsonl"
        if cut_at_end:
            return "json"
    except RecursionError:
        return "json"
    return None


def _xml_root(text: str, complete: bool) -> bool:
    """A root element that holds the document, not a CSV whose first cell is markup
    ("<b>x</b>,y"): the whole file ends by closing the root, or (cut by the head) the first
    line does not go on with a delimiter after a closed element."""
    match = re.match(r"<([A-Za-z_][\w.:-]*)[\s>/]", text)
    if match is None:
        return False
    name = match.group(1)
    first_line = text.split("\n", 1)[0]
    if re.search(rf"</{re.escape(name)}\s*>\s*[,;\t|]", first_line):
        return False
    if complete:
        end = text.rstrip()
        return end.endswith(f"</{name}>") or (end.endswith("/>") and end.count("<") == 1)
    return True


def _zip_kind(head: bytes, path: Path | None) -> tuple[str, str] | None:
    """An Office Open XML package, told by its entries: (format, label), or None for a plain zip."""
    names: list[str] = []
    if path is not None:
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()[:2000]
        except (zipfile.BadZipFile, OSError):
            names = []
    if not names:  # bytes only (no path): the local headers in the head name the entries
        found = re.findall(rb"(?:xl|word|ppt)/[\w./]+|content\.xml", head)
        names = [name.decode("latin-1") for name in found]
    if any(name.startswith("xl/") for name in names):
        return "excel", LABELS["excel"]
    if any(name.startswith("word/") for name in names):
        return "docx", "Word document (.docx)"
    if any(name.startswith("ppt/") for name in names):
        return "pptx", "PowerPoint presentation (.pptx)"
    if "content.xml" in names or "mimetype" in names:
        return "odf", "OpenDocument file (.ods, .odt)"
    return None


def sniff_bytes(head: bytes, name: str = "", path: Path | None = None) -> Sniff:
    """The kind and format of a file from its first bytes (64 KiB are enough) and its name.

    `path`, when given, lets a zip be opened to read its entries (a workbook or not)."""
    suffix = Path(name).suffix.lower()
    said = NAME_FORMATS.get(suffix)
    complete = len(head) < HEAD_BYTES

    def found(kind: Kind, fmt: str, label: str, rule: str, reason: str) -> Sniff:
        return Sniff(kind, fmt, label, rule, reason, said)

    if not head:
        return found("unsupported", "empty", "empty file", "R-DET-03", "The file is empty.")
    if head.startswith(b"PAR1"):
        return found("table", "parquet", "Parquet", "R-DET-01", "Starts with PAR1, Parquet's mark.")
    if head.startswith(b"\x93NUMPY"):
        return found(
            "table", "npy", LABELS["npy"], "R-DET-01", "Starts with NumPy's \\x93NUMPY mark."
        )
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        office = _zip_kind(head, path)
        if office and office[0] == "excel":
            return found(
                "table", "excel", office[1], "R-DET-01", "A zip package holding xl/ parts."
            )
        label = office[1] if office else "zip archive"
        reason = (
            f"{label[0].upper()}{label[1:]} detected: only Excel workbooks are read among Office "
            "files."
            if office
            else "Zip archive detected: unpack it first, then import the file inside."
        )
        return found("unsupported", office[0] if office else "zip", label, "R-DET-03", reason)
    image = sniff_image(head[:64])
    if image is not None:
        label = f"{IMAGE_LABELS.get(image, image.upper())} image"
        return found("image", image, label, "R-IMG-01", f"Starts with the {label} signature.")
    for signature, fmt, label, why in UNSUPPORTED:
        if head.startswith(signature):
            return found(
                "unsupported",
                fmt,
                label,
                "R-DET-03",
                f"{label[0].upper()}{label[1:]} detected. {why}",
            )
    if head[4:8] == b"ftyp":
        return found(
            "unsupported",
            "video",
            "video or audio file",
            "R-DET-03",
            "Video or audio file (MP4, QuickTime) detected. Video and audio files are not "
            "supported.",
        )
    if head.startswith(b"RIFF"):
        return found(
            "unsupported",
            "riff",
            "audio or video file",
            "R-DET-03",
            "Audio or video file (WAV, AVI) detected. Audio and video files are not supported.",
        )
    utf16 = _utf16(head)
    if b"\x00" in head[:4096] and not utf16:
        return found(
            "unsupported",
            "binary",
            "binary file",
            "R-DET-03",
            "Binary content that is none of the known formats (CSV, Parquet, Excel, .npy, JSON, "
            "XML, images).",
        )
    text = _text_of(head)
    start = text.lstrip("﻿ \t\r\n")
    if start[:1] in ("{", "["):
        kind = _json_kind(text, complete)
        if kind is not None:
            label = LABELS[kind]
            reason = (
                "Starts with '{' or '[' and parses as one JSON value."
                if kind == "json"
                else "Not one JSON value, but at least 95 % of the first lines are one each "
                "(R-DET-02)."
            )
            return found(
                "document", kind, label, "R-DET-02" if kind == "jsonl" else "R-DET-01", reason
            )
    if start[:1] == "<":
        lowered = start[:4096].lower()
        if re.search(r"<svg[\s>/]", lowered):
            return found(
                "unsupported",
                "svg",
                "SVG image",
                "R-DET-04",
                "SVG is XML and can carry scripts: it is not read as an image.",
            )
        if lowered.startswith("<!doctype html") or re.match(r"<html[\s>]", lowered):
            return found(
                "unsupported",
                "html",
                "web page (HTML)",
                "R-DET-03",
                "Web page (HTML) detected. Save the table it shows as CSV, or export the data "
                "from the site.",
            )
        if lowered.startswith("<?xml") or _xml_root(start, complete):
            return found("document", "xml", "XML", "R-DET-01", "Starts with an XML tag.")
    if said == "jsonl":  # a JSON Lines file whose first line is not JSON: let the reader say why
        return found("document", "jsonl", LABELS["jsonl"], "R-DET-02", "Named .jsonl.")
    return found(
        "table",
        "csv",
        LABELS["csv"],
        "R-DET-01",
        "Plain text that is neither JSON nor XML: read as delimited text, the dialect detected.",
    )


def sniff_path(path: Path) -> Sniff:
    """`sniff_bytes` on the first 64 KiB of a file."""
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)
    return sniff_bytes(head, path.name, path)
