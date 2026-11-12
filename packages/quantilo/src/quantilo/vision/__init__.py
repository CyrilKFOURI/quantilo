"""Computer vision: an image set as a dataset (103, 104; study 2026-11-11 vision-et-xai).

- `decode`: safe decoding with Pillow (bombs, truncated files, SVG refused, no GPS kept).
- `measures`: brightness, contrast, sharpness... at a fixed 512-pixel scale (§ 3.4, § 4.1).
- `hashes`: pHash, dHash and the pixel SHA (§ 4.1).
- `descriptors`: HSV histogram, HOG and uniform LBP in numpy (§ 4.4).
- `layout`, `zipsafe`, `store`, `ingest`: a folder, a ZIP, a single image or a Hugging Face Parquet
  file as a table of images, every reading announced by its rule (§ 3).

    images = dx.image_set("beans/")
"""

from quantilo.vision.decode import (
    DecodedImage,
    DecodeFailure,
    ImageDecodeError,
    decode_image,
    try_decode,
)
from quantilo.vision.descriptors import (
    ClassicDescriptor,
    classic_descriptor,
    hog,
    hsv_histogram,
    lbp,
)
from quantilo.vision.hashes import dhash, hamming, phash, pixel_sha
from quantilo.vision.ingest import ImageSet, LabelConflict, Skipped, image_set
from quantilo.vision.layout import Layout, RuleNote, read_layout
from quantilo.vision.measures import ImageMeasures, measure, working_copy
from quantilo.vision.store import ImageStore
from quantilo.vision.zipsafe import UnsafeArchive, check_zip

__all__ = [
    "ClassicDescriptor",
    "DecodeFailure",
    "DecodedImage",
    "ImageDecodeError",
    "ImageMeasures",
    "ImageSet",
    "ImageStore",
    "LabelConflict",
    "Layout",
    "RuleNote",
    "Skipped",
    "UnsafeArchive",
    "check_zip",
    "classic_descriptor",
    "decode_image",
    "dhash",
    "hamming",
    "hog",
    "hsv_histogram",
    "image_set",
    "lbp",
    "measure",
    "phash",
    "pixel_sha",
    "read_layout",
    "try_decode",
    "working_copy",
]
