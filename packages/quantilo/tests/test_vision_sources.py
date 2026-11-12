"""106 (engine side): an image set told from its names (R-VIS-SRC), and the import preview."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quantilo.ingestion.image_sets import image_source
from quantilo.ingestion.options import SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.vision.ingest import preview_image_set
from tests.vision_fixtures import encode, scene


def _folder(root: Path) -> Path:
    for split in ("train", "test"):
        for label in ("cat", "dog"):
            (root / split / label).mkdir(parents=True)
            for i in range(3):
                (root / split / label / f"{i}.png").write_bytes(encode(scene(seed=i)))
    (root / "notes.txt").write_text("hello")
    return root


def test_folder_of_images_is_an_image_set(tmp_path: Path) -> None:
    found = image_source(_folder(tmp_path / "set"))
    assert found is not None and found.rule == "R-VIS-SRC-01" and found.images == 12
    tables = tmp_path / "tables"
    tables.mkdir()
    for i in range(3):
        (tables / f"{i}.csv").write_text("a,b\n1,2\n")
    (tables / "logo.png").write_bytes(encode(scene()))
    assert image_source(tables) is None


def test_zip_of_images_and_office_files(tmp_path: Path) -> None:
    archive = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a/1.png", encode(scene()))
        z.writestr("__MACOSX/a/._1.png", b"x")
    assert image_source(archive).rule == "R-VIS-SRC-02"  # type: ignore[union-attr]
    office = tmp_path / "book.zip"
    with zipfile.ZipFile(office, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("xl/media/1.png", encode(scene()))
    assert image_source(office) is None


def test_parquet_with_an_image_column(tmp_path: Path) -> None:
    path = tmp_path / "train-00000-of-00001.parquet"
    image = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    pq.write_table(
        pa.table({"image": pa.array([{"bytes": encode(scene()), "path": "x.png"}], image)}), path
    )
    assert image_source(path).rule == "R-VIS-SRC-03"  # type: ignore[union-attr]
    plain = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": [1]}), plain)
    assert image_source(plain) is None
    # the table preview still works, and says the file can be an image set
    preview = preview_source(path, SourceOptions())
    assert preview.detected is not None and preview.detected.rule == "R-VIS-SRC-03"
    assert [c.name for c in preview.columns] == ["image"]


def test_table_preview_routes_a_folder_of_images(tmp_path: Path) -> None:
    preview = preview_source(_folder(tmp_path / "set"), SourceOptions())
    assert preview.detected is not None
    assert (preview.detected.kind, preview.detected.format) == ("image", "images")
    assert preview.columns == [] and preview.suggested_name == "set"


def test_image_set_preview_decodes_nothing(tmp_path: Path) -> None:
    root = _folder(tmp_path / "set")
    preview = preview_image_set(root)
    assert preview.source_kind == "folder" and preview.n_images == 12
    assert preview.labels == {"cat": 6, "dog": 6}
    assert preview.splits == {"test": 6, "train": 6}
    assert preview.formats == {"png": 12}
    codes = {r.code for r in preview.rules}
    assert {"R-VIS-LAY-01", "R-VIS-LAY-02", "R-VIS-ING-02"} <= codes
    assert len(preview.sample) == 12 and preview.sample[0].endswith(".png")
    assert "12 images found (png 12): 2 labels" in preview.sentence
    ignored = preview_image_set(root, ignore_rules={"R-VIS-LAY-01"})
    assert ignored.labels == {}
    assert not (tmp_path / "set" / "images").exists()
