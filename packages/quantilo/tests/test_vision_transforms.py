"""110: preparing images step by step (transforms)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from quantilo.vision import transforms as tf
from tests.vision_fixtures import flat, gradient, noise, scene


def framed(size: tuple[int, int] = (120, 90), border: int = 12) -> Image.Image:
    inner = scene(0, (size[0] - 2 * border, size[1] - 2 * border))
    out = Image.new("RGB", size, (255, 255, 255))
    out.paste(inner, (border, border))
    return out


def test_default_recipes_follow_the_task() -> None:
    assert tf.Recipe.default().augment and tf.Recipe.default().flip
    anomaly = tf.Recipe.default("anomaly")
    assert not anomaly.enabled("augment")  # R-VPR-07: a rotation can be the defect
    assert tf.Recipe.default().to_dict()["kind"] == "vision-prepare"


def test_recipe_round_trip_and_bounds() -> None:
    recipe = tf.Recipe(size=256, equalise="clahe", denoise="median")
    assert tf.Recipe.from_dict(recipe.to_dict()) == recipe
    with pytest.raises(tf.RecipeError, match="size"):
        tf.Recipe.from_dict({"size": 225})
    with pytest.raises(tf.RecipeError, match="sigma"):
        tf.Recipe.from_dict({"sigma": 50})
    with pytest.raises(tf.RecipeError, match="Unknown setting"):
        tf.Recipe.from_dict({"blur": 1})
    with pytest.raises(tf.RecipeError, match="equalise"):
        tf.Recipe.from_dict({"equalise": "loud"})


def test_resize_crop_and_pad() -> None:
    image = noise((300, 200))
    assert tf.resize(image, 224, "crop").size == (224, 224)
    padded = np.asarray(tf.resize(image, 224, "pad"))
    assert padded.shape == (224, 224, 3)
    assert tuple(padded[0, 0]) == tf.PAD_COLOUR  # letterbox on the ImageNet mean grey


def test_odd_aspect_ratio_is_padded_in_crop_mode() -> None:
    wide = gradient((500, 100))  # 5:1, R-VQ-03
    applied = tf.apply(wide, tf.Recipe(), train=False)
    assert applied.image.size == (224, 224)
    assert any("R-VPR-04" in note for note in applied.notes)
    pixels = np.asarray(applied.image)
    assert tuple(pixels[0, 112]) == tf.PAD_COLOUR  # the padding, not a crop of the gradient


def test_trim_removes_a_uniform_frame() -> None:
    image = framed()
    assert tf.has_uniform_border(image)
    assert tf.trim(image).size == (96, 66)
    assert not tf.has_uniform_border(noise())
    assert tf.trim(noise()).size == noise().size


def test_white_balance_equalises_channel_means() -> None:
    a = np.asarray(noise((64, 64)), dtype=np.float32) * np.array([1.0, 0.6, 0.3])
    out = np.asarray(tf.white_balance(Image.fromarray(a.astype(np.uint8))), dtype=np.float32)
    means = out.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 4


def test_denoise_lowers_the_noise() -> None:
    image = noise((64, 64))
    for method in ("gaussian", "median"):
        out = np.asarray(tf.denoise(image, method, 1.5), dtype=np.float32)
        assert out.std() < np.asarray(image, dtype=np.float32).std()


def test_equalise_spreads_a_dull_image_and_clahe_is_bounded() -> None:
    dull = np.full((80, 80, 3), 120, dtype=np.uint8)
    dull[:, 40:] = 136
    image = Image.fromarray(dull)
    out = np.asarray(tf.equalise(image, "global").convert("L"), dtype=np.float32)
    assert out.max() - out.min() > 200  # the 16 levels are stretched to the full range
    # CLAHE limits the stretch (clip 2): a low-contrast gradient gains contrast, within bounds
    low = np.asarray(gradient((96, 96)), dtype=np.float32) * 0.2 + 100
    local = np.asarray(tf.equalise(Image.fromarray(low.astype(np.uint8)), "clahe").convert("L"))
    assert float(low.std()) - 1 < float(local.std()) < 128
    y = np.asarray(noise((50, 70)).convert("L"))
    assert tf.clahe(y, 2.0, 8).shape == y.shape
    assert tf.clahe(y).min() >= 0 and tf.clahe(y).max() <= 255


def test_sharpen_raises_the_laplacian_variance() -> None:
    from quantilo.vision.measures import measure

    image = scene(1)
    assert measure(tf.sharpen(image, 1.5, 2.0)).sharpness > measure(image).sharpness


def test_augment_is_seeded() -> None:
    recipe = tf.Recipe(rotate=10, jitter=0.2)
    image = scene(2, (224, 224))
    a = np.asarray(tf.augment(image, recipe, seed=3))
    b = np.asarray(tf.augment(image, recipe, seed=3))
    c = np.asarray(tf.augment(image, recipe, seed=4))
    assert (a == b).all() and not (a == c).all()
    assert a.shape == (224, 224, 3)


def test_apply_skips_augmentation_outside_training_and_order_is_fixed() -> None:
    image = scene(3, (300, 240))
    recipe = tf.Recipe(rotate=20, jitter=0.3)
    a = np.asarray(tf.apply(image, recipe).image)
    b = np.asarray(tf.apply(image, recipe).image)
    assert (a == b).all()
    up_to_resize = tf.apply(image, recipe, upto="resize").image
    assert up_to_resize.size == (224, 224)


def test_preview_gives_the_input_and_output_of_a_step() -> None:
    image = scene(4, (320, 240))
    recipe = tf.Recipe(denoise="gaussian", sigma=2.0)
    shown = tf.preview(image, recipe, "denoise")
    assert shown.before.size == shown.after.size == (224, 224)
    assert not shown.geometric and shown.enabled
    off = tf.preview(image, tf.Recipe(), "equalise")
    assert not off.enabled and off.after.size == (224, 224)  # previewed with its settings anyway
    resized = tf.preview(image, recipe, "resize")
    assert resized.geometric and resized.before.size == (320, 240)
    draws = {
        np.asarray(tf.preview(image, recipe, "augment", seed=s).after).tobytes() for s in range(6)
    }
    assert len(draws) > 1
    with pytest.raises(tf.RecipeError):
        tf.preview(image, recipe, "glow")


def test_to_tensor_is_normalised() -> None:
    x = tf.to_tensor(flat(tf.PAD_COLOUR, (300, 300)), tf.Recipe.default("anomaly"))  # type: ignore[arg-type]
    assert x.shape == (3, 224, 224)
    assert np.abs(x).max() < 0.02  # the mean grey is zero once normalised
