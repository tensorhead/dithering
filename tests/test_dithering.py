"""End-to-end tests for the dithering Python module."""

import numpy as np
import pytest

import dithering
from dithering import (
    available_matrices,
    available_methods,
    dither,
    error_diffusion,
    ordered_dither,
    random_dither,
)

GRAY = np.full((64, 64), 128, dtype=np.uint8)
RGB = np.dstack([np.full((32, 32), v, dtype=np.uint8) for v in (10, 128, 245)])
GRADIENT = np.tile(np.arange(256, dtype=np.uint8), (64, 1))


def frac_white(a):
    return (a == 255).mean()


# ---------------------------------------------------------------- ordered

@pytest.mark.parametrize("matrix", ["bayer2x2", "bayer4x4", "bayer8x8", "bayer16x16"])
def test_ordered_mid_gray_is_half_white(matrix):
    out = ordered_dither(GRAY, matrix)
    assert out.shape == GRAY.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert abs(frac_white(out) - 0.5) <= 0.02


@pytest.mark.parametrize("matrix", ["Bayer2x2", "BAYER8X8", "bayer-4x4", 8, "16x16"])
def test_ordered_matrix_spellings(matrix):
    ordered_dither(GRAY, matrix)  # should not raise


def test_ordered_default_matrix():
    out = ordered_dither(GRAY)
    assert abs(frac_white(out) - 0.5) <= 0.02


def test_ordered_gradient_preserves_mean():
    out = ordered_dither(GRADIENT, "bayer8x8")
    assert abs(out.mean() - GRADIENT.mean()) < 4


def test_ordered_rgb():
    out = ordered_dither(RGB, "bayer8x8")
    assert out.shape == RGB.shape
    # Each channel dithers to its own density.
    assert frac_white(out[..., 0]) < 0.1
    assert abs(frac_white(out[..., 1]) - 0.5) <= 0.05
    assert frac_white(out[..., 2]) > 0.9


def test_ordered_levels():
    out = ordered_dither(GRAY, "bayer8x8", levels=4)
    assert set(np.unique(out)) <= {0, 85, 170, 255}
    assert abs(out.mean() - 128) < 3


def test_ordered_invalid_matrix_raises_value_error():
    with pytest.raises(ValueError, match="bayer8x8"):
        ordered_dither(GRAY, "bogus")
    with pytest.raises(ValueError):
        ordered_dither(GRAY, 3)


def test_ordered_invalid_levels():
    with pytest.raises(ValueError):
        ordered_dither(GRAY, levels=1)
    with pytest.raises(ValueError):
        ordered_dither(GRAY, levels=257)


# ---------------------------------------------------------------- error diffusion

@pytest.mark.parametrize("method", [
    "floyd_steinberg", "jarvis_judice_ninke", "stucki", "atkinson",
    "burkes", "sierra", "sierra_two_row", "sierra_lite",
])
@pytest.mark.parametrize("serpentine", [False, True])
def test_error_diffusion_mid_gray(method, serpentine):
    out = error_diffusion(GRAY, method, serpentine=serpentine)
    assert out.shape == GRAY.shape
    assert set(np.unique(out)) <= {0, 255}
    tol = 0.15 if method == "atkinson" else 0.02
    assert abs(frac_white(out) - 0.5) <= tol


@pytest.mark.parametrize("method", ["Floyd-Steinberg", "FS", "jjn", "SIERRA2", "sierra3"])
def test_error_diffusion_method_spellings(method):
    error_diffusion(GRAY, method)  # should not raise


def test_error_diffusion_default_method():
    out = error_diffusion(GRAY)
    assert abs(frac_white(out) - 0.5) <= 0.02


def test_error_diffusion_gradient_preserves_mean():
    out = error_diffusion(GRADIENT, "floyd_steinberg")
    assert abs(out.mean() - GRADIENT.mean()) < 3


def test_error_diffusion_rgb_levels():
    out = error_diffusion(RGB, "floyd_steinberg", levels=4)
    assert out.shape == RGB.shape
    assert set(np.unique(out)) <= {0, 85, 170, 255}


def test_error_diffusion_unknown_method():
    with pytest.raises(ValueError, match="floyd_steinberg"):
        error_diffusion(GRAY, "not_a_method")


# ---------------------------------------------------------------- random

def test_random_dither_is_reproducible_with_seed():
    a = random_dither(GRAY, seed=42)
    b = random_dither(GRAY, seed=42)
    c = random_dither(GRAY, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(a)) <= {0, 255}
    assert abs(frac_white(a) - 0.5) < 0.05


# ---------------------------------------------------------------- alpha & shapes

def test_rgba_alpha_preserved_by_default():
    rgba = np.dstack([RGB, np.full((32, 32), 100, dtype=np.uint8)])
    for out in (
        ordered_dither(rgba, "bayer4x4"),
        error_diffusion(rgba, "floyd_steinberg"),
        random_dither(rgba, seed=1),
    ):
        assert np.array_equal(out[..., 3], rgba[..., 3])
        assert set(np.unique(out[..., :3])) <= {0, 255}


def test_rgba_alpha_dithered_when_disabled():
    rgba = np.dstack([RGB, np.full((32, 32), 100, dtype=np.uint8)])
    out = ordered_dither(rgba, "bayer4x4", preserve_alpha=False)
    assert set(np.unique(out[..., 3])) <= {0, 255}


def test_la_alpha_preserved():
    la = np.dstack([GRAY, np.full((64, 64), 77, dtype=np.uint8)])
    out = ordered_dither(la, "bayer2x2")
    assert np.array_equal(out[..., 1], la[..., 1])


def test_single_channel_3d():
    img = GRAY[..., None]
    out = ordered_dither(img, "bayer2x2")
    assert out.shape == img.shape


def test_1d_input_raises_value_error():
    with pytest.raises(ValueError, match="2-D"):
        ordered_dither(np.zeros(10, dtype=np.uint8), "bayer2x2")


def test_4d_input_raises_value_error():
    with pytest.raises(ValueError):
        ordered_dither(np.zeros((2, 2, 2, 2), dtype=np.uint8), "bayer2x2")


def test_empty_image_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        ordered_dither(np.zeros((0, 5), dtype=np.uint8), "bayer2x2")


def test_float_input_raises_type_error_with_guidance():
    with pytest.raises(TypeError, match="astype"):
        ordered_dither(GRAY.astype(np.float32) / 255.0, "bayer2x2")


def test_non_contiguous_view():
    big = np.tile(np.arange(256, dtype=np.uint8), (128, 2))
    view = big[::2, ::4]
    out = ordered_dither(view, "bayer8x8")
    assert out.shape == view.shape
    ref = ordered_dither(np.ascontiguousarray(view), "bayer8x8")
    assert np.array_equal(out, ref)


def test_input_is_not_mutated():
    src = GRADIENT.copy()
    ordered_dither(src, "bayer8x8")
    error_diffusion(src, "fs")
    assert np.array_equal(src, GRADIENT)


def test_list_input_rejected_with_guidance():
    # np.asarray of an int list is int64, not uint8 -> friendly TypeError.
    with pytest.raises(TypeError, match="uint8"):
        ordered_dither([[0, 128], [255, 64]], "bayer2x2")


def test_pil_image_accepted():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.fromarray(GRAY)
    out = ordered_dither(img, "bayer8x8")
    assert isinstance(out, PIL.Image)  # PIL in -> PIL out
    arr = np.asarray(out)
    assert arr.shape == GRAY.shape
    assert abs(frac_white(arr) - 0.5) <= 0.02


# ---------------------------------------------------------------- palette

GB_PALETTE = ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]  # Game Boy greens


def test_palette_output_only_contains_palette_colors():
    rgb = np.random.default_rng(0).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    expected = {(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)}
    for out in (
        error_diffusion(rgb, "fs", palette=GB_PALETTE),
        ordered_dither(rgb, "bayer8x8", palette=GB_PALETTE),
        random_dither(rgb, palette=GB_PALETTE, seed=7),
    ):
        colors = {tuple(c) for c in out.reshape(-1, 3)}
        assert colors <= expected


def test_palette_accepts_tuples_and_short_hex():
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    a = error_diffusion(rgb, "fs", palette=[(0, 0, 0), (255, 255, 255)])
    b = error_diffusion(rgb, "fs", palette=["#000", "#ffffff"])
    assert np.array_equal(a, b)


def test_palette_bw_matches_levels_on_gray_ramp():
    ramp = np.repeat(GRADIENT[..., None], 3, axis=2)
    a = error_diffusion(ramp, "fs")
    b = error_diffusion(ramp, "fs", palette=["#000000", "#ffffff"])
    assert np.array_equal(a, b)


def test_palette_mean_preservation_rgb_cube():
    rgb = np.dstack([np.full((64, 64), v, dtype=np.uint8) for v in (64, 128, 200)])
    corners = [(r, g, b) for r in (0, 255) for g in (0, 255) for b in (0, 255)]
    out = error_diffusion(rgb, "fs", palette=corners)
    for c, want in [(0, 64), (1, 128), (2, 200)]:
        assert abs(out[..., c].mean() - want) < 3


def test_palette_preserves_alpha():
    rgba = np.dstack([RGB, np.full((32, 32), 123, dtype=np.uint8)])
    out = error_diffusion(rgba, "fs", palette=GB_PALETTE)
    assert np.array_equal(out[..., 3], rgba[..., 3])


def test_palette_on_grayscale_raises():
    with pytest.raises(ValueError, match="RGB"):
        error_diffusion(GRAY, "fs", palette=GB_PALETTE)


def test_palette_and_levels_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        error_diffusion(RGB, "fs", levels=4, palette=GB_PALETTE)


def test_palette_validation_errors():
    with pytest.raises(ValueError, match="at least 2"):
        error_diffusion(RGB, "fs", palette=["#000"])
    with pytest.raises(ValueError, match="hex"):
        error_diffusion(RGB, "fs", palette=["#00", "#fff"])
    with pytest.raises(ValueError, match="0..=255"):
        error_diffusion(RGB, "fs", palette=[(0, 0, 0), (256, 0, 0)])
    with pytest.raises(TypeError, match="sequence"):
        error_diffusion(RGB, "fs", palette="#000#fff")


# ---------------------------------------------------------------- linear light

def test_linear_mid_gray_density():
    # sRGB 128 is ~21.6% linear light; gamma-correct dithering must produce
    # ~21.6% white, not 50%.
    for out in (
        ordered_dither(GRAY, "bayer16x16", linear=True),
        error_diffusion(GRAY, "fs", linear=True),
    ):
        assert abs(frac_white(out) - 0.216) < 0.02, frac_white(out)


def test_linear_levels_are_gamma_encoded():
    # Gray 128 is ~55 in linear light, which sits between the linear levels
    # 0 and 85. Level 85-linear encodes to sRGB 156 (not 85), and the mix
    # must average to 55 linear: fraction ~= 55/85.
    out = error_diffusion(GRAY, "fs", levels=4, linear=True)
    values = set(np.unique(out))
    assert values == {0, 156}, values
    assert abs((out == 156).mean() - 55 / 85) < 0.02


# ---------------------------------------------------------------- strength

def test_strength_zero_is_pure_quantization():
    out = ordered_dither(GRAY, "bayer8x8", strength=0.0)
    assert set(np.unique(out)) == {255}  # 128 >= 127.5 -> all white
    out = error_diffusion(GRADIENT, "fs", strength=0.0)
    assert np.array_equal(out, np.where(GRADIENT < 128, 0, 255))


def test_strength_out_of_range_raises():
    with pytest.raises(ValueError, match="strength"):
        ordered_dither(GRAY, strength=1.5)
    with pytest.raises(ValueError, match="strength"):
        error_diffusion(GRAY, strength=-0.1)


# ---------------------------------------------------------------- custom matrix

def test_custom_integer_matrix_matches_builtin_bayer():
    bayer2 = np.array([[0, 2], [3, 1]], dtype=np.int64)
    a = ordered_dither(GRADIENT, bayer2)
    b = ordered_dither(GRADIENT, "bayer2x2")
    assert np.array_equal(a, b)


def test_custom_float_threshold_matrix():
    m = np.array([[0.25, 0.75]], dtype=np.float64)  # non-square 1x2
    out = ordered_dither(GRAY, m)
    assert set(np.unique(out)) <= {0, 255}
    # 128/255 = 0.502: above 0.25, below 0.75 -> alternating columns.
    assert abs(frac_white(out) - 0.5) < 0.01


def test_custom_matrix_validation():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ordered_dither(GRAY, np.array([[0.5, 1.5]]))
    with pytest.raises(ValueError, match="2-D"):
        ordered_dither(GRAY, np.zeros((2, 2, 2)))


# ---------------------------------------------------------------- dither()

def test_dither_dispatches_by_method():
    assert np.array_equal(dither(GRAY, "fs"), error_diffusion(GRAY, "fs"))
    assert np.array_equal(dither(GRAY, "bayer4x4"), ordered_dither(GRAY, "bayer4x4"))
    assert np.array_equal(
        dither(GRAY, "random", seed=5), random_dither(GRAY, seed=5)
    )


def test_dither_default_is_floyd_steinberg():
    assert np.array_equal(dither(GRAY), error_diffusion(GRAY))


def test_dither_unknown_method_lists_all():
    with pytest.raises(ValueError, match="random"):
        dither(GRAY, "bogus")


# ---------------------------------------------------------------- PIL round trip

def test_pil_in_pil_out():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.fromarray(GRAY)
    out = dither(img, "atkinson")
    assert isinstance(out, PIL.Image)
    assert out.mode == "L"
    assert out.size == img.size
    arr = np.asarray(out)
    assert set(np.unique(arr)) <= {0, 255}


def test_pil_rgba_alpha_preserved():
    PIL = pytest.importorskip("PIL.Image")
    rgba = np.dstack([RGB, np.full((32, 32), 100, dtype=np.uint8)])
    out = dither(PIL.fromarray(rgba), "fs")
    assert out.mode == "RGBA"
    assert np.array_equal(np.asarray(out)[..., 3], rgba[..., 3])


def test_pil_palette_mode_converted():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.fromarray(RGB).quantize(16)  # mode "P"
    out = dither(img, "fs")
    assert out.mode == "RGB"


# ---------------------------------------------------------------- review regressions

def test_hex_color_with_multibyte_chars_raises_value_error():
    # Byte-indexed slicing of a multi-byte UTF-8 hex string used to panic
    # (uncatchable PanicException) instead of raising ValueError.
    for bad in ["#ééé", "#fféé", "#zzzzzz", "#ﬀﬀﬀ"]:
        with pytest.raises(ValueError, match="hex"):
            error_diffusion(RGB, "fs", palette=[bad, "#fff"])


def test_levels_not_dividing_255_preserve_mean():
    # levels=3 has an unrepresentable ideal mid level (127.5); the error must
    # be accounted against the emitted byte (128) or the mean drifts.
    img = np.full((256, 256), 127, dtype=np.uint8)
    for lv in (3, 5, 7, 9):
        out = error_diffusion(img, "fs", levels=lv)
        assert abs(out.mean() - 127.0) < 0.35, f"levels={lv}: mean {out.mean()}"


def test_threshold_matrix_of_one_keeps_black_black():
    # A 1.0 threshold cell (common after m / m.max()) must not flip black.
    black = np.zeros((8, 8), dtype=np.uint8)
    out = ordered_dither(black, np.array([[1.0]]))
    assert set(np.unique(out)) == {0}
    white = np.full((8, 8), 255, dtype=np.uint8)
    out = ordered_dither(white, np.array([[0.0]]))
    assert set(np.unique(out)) == {255}


def test_huge_matrix_index_values_rejected():
    with pytest.raises(ValueError, match="2\\^24"):
        ordered_dither(GRAY, np.array([[0, 2**40]], dtype=np.int64))
    with pytest.raises(ValueError):  # uint64 overflow wraps negative
        ordered_dither(GRAY, np.array([[0, 2**63]], dtype=np.uint64))


def test_pil_subclass_recognized():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.fromarray(GRAY)

    class ExternalImage(PIL.Image):
        pass

    ExternalImage.__module__ = "someones_toolkit"
    img.__class__ = ExternalImage
    out = dither(img, "fs")
    assert isinstance(out, PIL.Image)


def test_negative_seed_accepted():
    a = random_dither(GRAY, seed=-1)
    b = random_dither(GRAY, seed=-1)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- misc API

def test_discovery_helpers():
    assert "bayer8x8" in available_matrices()
    assert "floyd_steinberg" in available_methods()
    assert dithering.__version__


def test_docstrings_present():
    assert "Bayer" in ordered_dither.__doc__
    assert "serpentine" in error_diffusion.__doc__
