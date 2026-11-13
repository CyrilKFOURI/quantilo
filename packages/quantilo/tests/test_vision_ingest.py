"""104: an image set from a folder, a ZIP, a labels file or a Hugging Face Parquet file; layout
rules (R-VIS-LAY), safe archives (R-VIS-ZIP), storage by content, thumbnails, the table."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from PIL import Image

import quantilo as dx
from quantilo.ingestion.errors import IngestionError
from quantilo.vision.ingest import COLUMNS, SCHEMA, image_set
from quantilo.vision.layout import read_layout
from quantilo.vision.store import ImageStore, check_dataset_id
from quantilo.vision.zipsafe import UnsafeArchive, check_zip
from tests.vision_fixtures import (
    SVG,
    encode,
    flat,
    hf_parquet,
    image_folder,
    jpeg_with_exif,
    noise,
    png_bomb,
    scene,
    truncated,
    write,
    zip_of,
)


def _codes(images) -> set[str]:
    return {r.code for r in images.rules}


# --- layout rules (paths only) ------------------------------------------------------------------


def test_imagefolder_labels() -> None:
    layout = read_layout(["cat/1.png", "cat/2.png", "dog/1.png", "dog/2.png"])
    assert [f.label for f in layout.files] == ["cat", "cat", "dog", "dog"]
    note = next(r for r in layout.rules if r.code == "R-VIS-LAY-01")
    assert "Check that your data was read correctly." in note.message
    assert note.reason and note.count == 4


def test_common_parent_folders_are_not_labels() -> None:
    layout = read_layout(["data/images/cat/1.png", "data/images/dog/1.png"])
    assert [f.label for f in layout.files] == ["cat", "dog"]
    layout = read_layout(["images/1.png", "images/2.png"])
    assert all(f.label is None for f in layout.files)
    assert not layout.rules


def test_labels_need_eighty_percent_of_the_images() -> None:
    paths = ["a/1.png", "b/1.png"] + [f"{i}.png" for i in range(8)]
    layout = read_layout(paths)
    assert all(f.label is None for f in layout.files)
    note = next(r for r in layout.rules if r.code == "R-VIS-LAY-01")
    assert note.count == 0 and "not read as labels" in note.message


def test_splits_and_labels_under_them() -> None:
    paths = ["Train/cat/1.png", "train/dog/2.png", "VALIDATION/cat/3.png", "test/dog/4.png"]
    layout = read_layout(paths)
    assert [(f.split, f.label) for f in layout.files] == [
        ("train", "cat"),
        ("train", "dog"),
        ("val", "cat"),
        ("test", "dog"),
    ]
    assert {"R-VIS-LAY-01", "R-VIS-LAY-02"} <= {r.code for r in layout.rules}


def test_mvtec_layout_normal_defects_and_masks() -> None:
    paths = [
        "bottle/train/good/000.png",
        "bottle/train/good/001.png",
        "bottle/test/good/000.png",
        "bottle/test/broken_large/000.png",
        "bottle/test/contamination/000.png",
        "bottle/ground_truth/broken_large/000_mask.png",
        "bottle/ground_truth/contamination/000_mask.png",
        "bottle/ground_truth/scratch/009_mask.png",
    ]
    layout = read_layout(paths)
    by = layout.by_path()
    assert len(layout.files) == 5  # the masks are not images of the set
    assert by["bottle/train/good/000.png"].is_normal is True
    assert by["bottle/test/broken_large/000.png"].is_normal is False
    assert by["bottle/test/broken_large/000.png"].label == "broken_large"
    assert (
        by["bottle/test/broken_large/000.png"].mask
        == "bottle/ground_truth/broken_large/000_mask.png"
    )
    assert by["bottle/test/good/000.png"].mask is None  # same name, but no "good" masks folder
    assert layout.orphan_masks == ["bottle/ground_truth/scratch/009_mask.png"]
    assert {"R-VIS-LAY-01", "R-VIS-LAY-02", "R-VIS-LAY-03", "R-VIS-LAY-04"} <= {
        r.code for r in layout.rules
    }


def test_rules_can_be_ignored() -> None:
    layout = read_layout(["train/cat/1.png", "train/dog/1.png"], ignore={"R-VIS-LAY-01"})
    assert all(f.label is None for f in layout.files)
    assert all(f.split == "train" for f in layout.files)
    layout = read_layout(["train/cat/1.png", "train/dog/1.png"], ignore={"R-VIS-LAY-02"})
    assert all(f.split is None for f in layout.files)


# --- ZIP safety ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["../evil.png", "a/../../evil.png", "/abs/evil.png", "C:/evil.png"]
)
def test_zip_slip_refuses_the_archive(tmp_path: Path, name: str) -> None:
    path = zip_of(tmp_path / "bad.zip", {"ok.png": encode(noise()), name: encode(noise())})
    with pytest.raises(UnsafeArchive) as raised:
        check_zip(path)
    assert raised.value.rule == "R-VIS-ZIP-01"
    with pytest.raises(UnsafeArchive):
        image_set(path, workspace=tmp_path / "ws")
    assert not (tmp_path / "evil.png").exists()


def test_zip_bomb_member_is_skipped(tmp_path: Path) -> None:
    path = zip_of(tmp_path / "bomb.zip", {"ok.png": encode(noise()), "bomb.bin": b"\0" * 5_000_000})
    plan = check_zip(path)
    assert [m.name for m in plan.members] == ["ok.png"]
    assert plan.skipped[0].code == "R-VIS-ZIP-04"


def test_too_many_members_and_too_large(tmp_path: Path, monkeypatch) -> None:
    import quantilo.vision.zipsafe as zipsafe

    path = zip_of(tmp_path / "many.zip", {f"{i}.png": encode(noise(seed=i)) for i in range(5)})
    monkeypatch.setattr(zipsafe, "MAX_MEMBERS", 3)
    with pytest.raises(UnsafeArchive) as raised:
        check_zip(path)
    assert raised.value.rule == "R-VIS-ZIP-02"
    monkeypatch.setattr(zipsafe, "MAX_MEMBERS", 50_000)
    monkeypatch.setattr(zipsafe, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(UnsafeArchive) as raised:
        check_zip(path)
    assert raised.value.rule == "R-VIS-ZIP-03"


def test_nested_hidden_and_damaged_members(tmp_path: Path) -> None:
    path = zip_of(
        tmp_path / "mixed.zip",
        {
            "cat/1.png": encode(noise(seed=1)),
            "dog/1.png": encode(noise(seed=2)),
            "inner.zip": b"PK\x03\x04" + b"\0" * 30,
            "__MACOSX/._cat.png": b"junk",
            "notes.txt": b"hello",
            "chart.svg": SVG,
        },
    )
    images = image_set(path, workspace=tmp_path / "ws", dataset_id="mixed")
    assert images.source_kind == "zip"
    assert images.n_images == 2
    assert images.labels == {"cat": 1, "dog": 1}
    codes = {s.code for s in images.skipped}
    assert {"R-VIS-ZIP-05", "R-VIS-ING-06"} <= codes
    assert "R-VIS-ING-02" in _codes(images)  # notes.txt
    assert "R-VIS-ING-04" in _codes(images)  # __MACOSX


def test_zip_member_lying_about_its_size_cannot_overflow(tmp_path: Path) -> None:
    path = tmp_path / "liar.zip"
    zip_of(path, {"a.png": encode(noise((200, 200)))}, compression=zipfile.ZIP_STORED)
    plan = check_zip(path)
    member = plan.members[0]
    from quantilo.vision.zipsafe import read_member

    with zipfile.ZipFile(path) as archive:
        assert len(read_member(archive, member, limit=100)) == 100
        assert len(read_member(archive, member)) == member.size


# --- store --------------------------------------------------------------------------------------


def test_store_names_files_by_content_and_confines_paths(tmp_path: Path) -> None:
    store = ImageStore(tmp_path, "set-1")
    digest, rel = store.put_file(b"abc", ".png")
    assert rel == f"files/{digest[:2]}/{digest}.png"
    assert store.put_file(b"abc", ".png") == (digest, rel)  # stored once
    assert store.path(rel).read_bytes() == b"abc"
    with pytest.raises(ValueError):
        store.path("../../outside.txt")
    with pytest.raises(ValueError):
        store.file_rel("../x", ".png")
    for bad in ("../x", "a/b", "", "CON.jpg", "x" * 65):
        with pytest.raises(ValueError):
            check_dataset_id(bad)


def test_thumbnail_is_webp_without_exif(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "in", {})
    write(folder / "photo.jpg", jpeg_with_exif(noise((600, 300)), orientation=6, gps=True))
    images = image_set(folder, workspace=tmp_path / "ws", dataset_id="thumbs")
    row = images.table.to_pylist()[0]
    thumb = images.store.path(row["thumb_path"])
    data = thumb.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert b"EXIF" not in data and b"Exif" not in data
    with Image.open(thumb) as opened:
        assert opened.size == (128, 256)  # oriented (portrait), long side 256
    assert row["has_location"] is True
    assert row["width"] == 300 and row["height"] == 600
    assert row["camera"] == "Canon EOS R5"
    assert "R-VIS-DEC-01" in _codes(images)


# --- folder import ------------------------------------------------------------------------------


@pytest.fixture
def beans_like(tmp_path: Path) -> Path:
    layout: dict[str, Image.Image | bytes] = {}
    for split in ("train", "test"):
        for k, label in enumerate(("healthy", "rust", "spot")):
            for i in range(3):
                layout[f"{split}/{label}/{label}_{i}.png"] = scene(k * 10 + i + (split == "test"))
    layout["train/rust/broken.png"] = truncated(encode(noise((200, 200)), "PNG"), 0.4)
    layout["train/spot/chart.svg"] = SVG
    layout["readme.txt"] = b"hello"
    return image_folder(tmp_path / "beans", layout)


def test_folder_import_table_and_rules(beans_like: Path, tmp_path: Path) -> None:
    images = image_set(beans_like, workspace=tmp_path / "ws", dataset_id="beans")
    table = images.table
    assert tuple(table.column_names[: len(COLUMNS)]) == COLUMNS
    assert table.schema.field("phash").type == SCHEMA.field("phash").type
    assert images.n_images == 19
    assert images.n_decoded == 18
    assert images.splits == {"train": 10, "test": 9}
    assert images.labels == {"rust": 7, "healthy": 6, "spot": 6}
    assert {"R-VIS-LAY-01", "R-VIS-LAY-02", "R-VIS-ING-02"} <= _codes(images)
    assert [s.code for s in images.skipped] == ["R-VIS-ING-06"]
    broken = next(r for r in table.to_pylist() if r["file_name"] == "broken.png")
    assert broken["decode_ok"] is False and broken["excluded"] is True
    assert broken["issues"] == ["R-VQ-01"]
    assert broken["brightness"] is None and broken["decode_error"]
    good = next(r for r in table.to_pylist() if r["file_name"] == "healthy_0.png")
    assert good["image_id"] == good["sha256"][:16]
    assert 0 <= good["brightness"] <= 1 and good["sharpness"] > 0
    assert good["dominant_colour"].startswith("#")
    assert images.store.path(good["stored_path"]).is_file()
    assert images.table_path == tmp_path / "ws" / "datasets" / "beans.parquet"
    assert pq.read_table(images.table_path).num_rows == 19
    assert "19 images from a folder" in images.sentence
    assert "could not be decoded (R-VQ-01)" in images.sentence


def test_exact_copies_are_stored_once(tmp_path: Path) -> None:
    data = encode(scene(1))
    folder = image_folder(
        tmp_path / "dup", {"a/x.png": data, "b/y.png": data, "a/z.png": encode(scene(2))}
    )
    images = image_set(folder, workspace=tmp_path / "ws", dataset_id="dup")
    rows = images.table.to_pylist()
    assert rows[0]["image_id"] == rows[2]["image_id"] or rows[0]["image_id"] == rows[1]["image_id"]
    assert len(list((images.root / "files").rglob("*.png"))) == 2


def test_ignore_folder_labels(beans_like: Path, tmp_path: Path) -> None:
    images = image_set(beans_like, workspace=tmp_path / "ws", ignore_rules={"R-VIS-LAY-01"})
    assert images.labels == {}
    assert images.splits == {"train": 10, "test": 9}


def test_hidden_links_and_depth(tmp_path: Path, monkeypatch) -> None:
    import quantilo.vision.ingest as ingest

    folder = image_folder(
        tmp_path / "deep",
        {".hidden/a.png": noise(), "a/b/c/d.png": noise(seed=1), "top.png": noise(seed=2)},
    )
    monkeypatch.setattr(ingest, "MAX_DEPTH", 2)
    images = image_set(folder, workspace=tmp_path / "ws")
    assert images.table["rel_path"].to_pylist() == ["top.png"]
    assert {"R-VIS-ING-04", "R-VIS-ING-05"} <= _codes(images)


def test_image_limit(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "many", {f"{i}.png": noise(seed=i) for i in range(5)})
    images = image_set(folder, workspace=tmp_path / "ws", max_images=3)
    assert images.n_images == 3
    assert "R-VIS-ING-01" in _codes(images)


def test_bomb_in_a_folder_is_a_failed_row_not_a_crash(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "b", {"ok.png": noise(), "bomb.png": png_bomb(12_000, 10_000)})
    images = image_set(folder, workspace=tmp_path / "ws")
    rows = {r["file_name"]: r for r in images.table.to_pylist()}
    assert rows["bomb.png"]["decode_ok"] is False
    assert "too large to decode safely" in rows["bomb.png"]["decode_error"]
    assert rows["bomb.png"]["width"] == 12_000  # still read from the header (084)
    assert rows["ok.png"]["decode_ok"] is True


def test_mvtec_import_links_masks(tmp_path: Path) -> None:
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (10, 10, 20, 20))
    folder = image_folder(
        tmp_path / "bottle",
        {
            "train/good/000.png": scene(1),
            "train/good/001.png": scene(2),
            "test/good/000.png": scene(3),
            "test/broken/000.png": scene(4),
            "ground_truth/broken/000_mask.png": mask,
        },
    )
    images = image_set(folder, workspace=tmp_path / "ws", dataset_id="bottle")
    rows = {r["rel_path"]: r for r in images.table.to_pylist()}
    broken = rows["test/broken/000.png"]
    assert broken["is_normal"] is False and broken["label"] == "broken"
    assert broken["mask_path"] == f"masks/{broken['sha256']}.png"
    assert images.store.path(broken["mask_path"]).is_file()
    assert rows["train/good/000.png"]["is_normal"] is True
    assert images.n_images == 4


# --- labels file --------------------------------------------------------------------------------


def test_labels_csv_by_file_stem_with_metadata_and_conflicts(tmp_path: Path) -> None:
    folder = image_folder(
        tmp_path / "img", {"cat/a.png": scene(1), "cat/b.png": scene(2), "dog/c.png": scene(3)}
    )
    csv = write(
        tmp_path / "labels.csv",
        b"id,breed,weight\na,siamese,4.2\nb,persian,5.0\nc,siamese,30.1\na,again,1\n",
    )
    images = image_set(folder, workspace=tmp_path / "ws", labels=csv)
    rows = {r["file_name"]: r for r in images.table.to_pylist()}
    assert rows["a.png"]["label"] == "siamese"
    assert rows["c.png"]["label"] == "siamese"
    assert rows["c.png"]["weight"] == pytest.approx(30.1)
    assert {"R-VIS-LAB-01", "R-VIS-LAB-02", "R-VIS-LAB-04", "R-VIS-LAY-05"} <= _codes(images)
    assert len(images.conflicts) == 3  # every folder label differs from the file's
    assert images.conflicts[0].folder_label == "cat"


def test_labels_csv_without_a_key_is_refused(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "img", {"a.png": scene(1), "b.png": scene(2)})
    csv = write(tmp_path / "labels.csv", b"name,label\nx,1\ny,2\n")
    with pytest.raises(IngestionError, match="R-VIS-LAB-01"):
        image_set(folder, workspace=tmp_path / "ws", labels=csv)


def test_csv_at_the_root_is_read_when_it_names_the_images(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "img", {"images/a.png": scene(1), "images/b.png": scene(2)})
    write(folder / "train.csv", b"image,label\nimages/a.png,good\nimages/b.png,bad\n")
    images = image_set(folder, workspace=tmp_path / "ws")
    assert images.labels == {"bad": 1, "good": 1}
    assert "R-VIS-LAB-03" in _codes(images)
    assert "R-VIS-LAY-03" in _codes(images)  # "good" is a normal label


def test_csv_at_the_root_that_names_nothing_is_said(tmp_path: Path) -> None:
    folder = image_folder(tmp_path / "img", {"a.png": scene(1)})
    write(folder / "notes.csv", b"x,y\n1,2\n")
    images = image_set(folder, workspace=tmp_path / "ws")
    note = next(r for r in images.rules if r.code == "R-VIS-LAB-03")
    assert "not read as labels" in note.message


# --- Hugging Face Parquet -----------------------------------------------------------------------


def test_hf_parquet_with_class_names_and_split(tmp_path: Path) -> None:
    images_bytes = [encode(scene(i)) for i in range(7)]
    path = hf_parquet(
        tmp_path / "train-00000-of-00001.parquet",
        images_bytes,
        [0, 1, 2, 0, 1, 2, 2],
        ["angular_leaf_spot", "bean_rust", "healthy"],
    )
    images = image_set(path, workspace=tmp_path / "ws", dataset_id="beans-hf")
    assert images.source_kind == "parquet"
    assert images.n_images == 7
    assert images.labels == {"healthy": 3, "angular_leaf_spot": 2, "bean_rust": 2}
    assert images.splits == {"train": 7}
    assert {"R-VIS-HF-01", "R-VIS-HF-02", "R-VIS-HF-03"} <= _codes(images)
    rows = images.table.to_pylist()
    assert rows[0]["file_name"] == "img_0.png"  # base name only, never the producer's path
    assert all("/home/someone" not in (r["rel_path"] or "") for r in rows)
    assert rows[0]["image_file_path"] == "/data/img_0.png"  # other columns kept as metadata


def _hf_folder(root: Path) -> Path:
    """The Hugging Face layout of Beans: data/{train,validation,test}-00000-of-00001.parquet, the
    same image names in every file (img_0.png ...)."""
    names = ["angular_leaf_spot", "bean_rust", "healthy"]
    data = root / "data"
    data.mkdir(parents=True)
    for split, n in (("train", 5), ("validation", 2), ("test", 3)):
        hf_parquet(
            data / f"{split}-00000-of-00001.parquet",
            [encode(scene(i + 10 * n)) for i in range(n)],
            [i % 3 for i in range(n)],
            names,
        )
    (root / "README.md").write_text("# beans")
    return data


def test_folder_of_hf_parquet_files_reads_each_file_and_its_split(tmp_path: Path) -> None:
    from quantilo.vision.ingest import preview_image_set

    data = _hf_folder(tmp_path / "beans")
    images = image_set(data, workspace=tmp_path / "ws", dataset_id="beans-folder")
    assert images.source_kind == "parquet" and images.n_images == 10
    assert images.splits == {"train": 5, "val": 2, "test": 3}
    assert images.labels == {"angular_leaf_spot": 4, "bean_rust": 4, "healthy": 2}
    codes = _codes(images)
    assert {"R-VIS-LAY-06", "R-VIS-HF-01", "R-VIS-HF-02"} <= codes
    assert "R-VIS-HF-03" not in codes  # the folder's note replaces each file's
    note = next(r for r in images.rules if r.code == "R-VIS-LAY-06")
    assert "3 Parquet files" in note.message and "train 5" in note.message and note.count == 10
    hf01 = next(r for r in images.rules if r.code == "R-VIS-HF-01")
    assert hf01.count == 10  # one note for the three files, their rows added up
    rows = images.table.to_pylist()
    paths = [r["rel_path"] for r in rows]
    assert len(set(paths)) == 10  # the same image names in three files stay apart
    assert "train-00000-of-00001/img_0.png" in paths
    assert {r["file_name"] for r in rows} >= {"img_0.png", "img_4.png"}
    assert all(r["image_file_path"].startswith("/data/") for r in rows)  # metadata kept
    assert "folder of 3 Parquet files" in images.sentence
    # the preview reads the same, decoding nothing
    preview = preview_image_set(data)
    assert preview.n_images == 10 and preview.splits == images.splits
    # undone: no split read from the names
    undone = preview_image_set(data, ignore_rules={"R-VIS-LAY-06"})
    assert undone.n_images == 10 and undone.splits == {}
    assert "R-VIS-LAY-06" not in {r.code for r in undone.rules}


def test_a_folder_with_images_ignores_its_parquet_files(tmp_path: Path) -> None:
    data = _hf_folder(tmp_path / "mixed")
    write(data / "photo.png", encode(scene(1)))
    images = image_set(data, workspace=tmp_path / "ws", dataset_id="mixed")
    assert images.source_kind == "folder" and images.n_images == 1


def test_parquet_without_images_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    with pytest.raises(IngestionError, match="No column"):
        image_set(path, workspace=tmp_path / "ws")


# --- one image, the facade ----------------------------------------------------------------------


def test_single_image_and_refused_files(tmp_path: Path) -> None:
    path = write(tmp_path / "one.jpg", encode(scene(1), "JPEG"))
    images = dx.image_set(path, workspace=tmp_path / "ws")
    assert images.source_kind == "image" and images.n_images == 1
    assert images.sentence.startswith("1 image from an image file")
    with pytest.raises(IngestionError, match="scripts"):
        dx.image_set(write(tmp_path / "x.svg", SVG), workspace=tmp_path / "ws")
    with pytest.raises(dx.NotFoundError):
        dx.image_set(tmp_path / "missing", workspace=tmp_path / "ws")


def test_table_opens_as_a_dataset(tmp_path: Path) -> None:
    folder = image_folder(
        tmp_path / "img",
        {
            f"{lab}/{i}.png": flat((40 * i, 80, 200 - 30 * i))
            for lab in ("a", "b")
            for i in range(3)
        },
    )
    images = dx.image_set(folder, workspace=tmp_path / "ws", dataset_id="open-me")
    ds = images.open(cache_dir=tmp_path / "cache")
    assert len(ds) == 6
    assert "brightness" in ds.columns
