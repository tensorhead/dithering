from typing import Any, Sequence, TypeVar, Union, overload

import numpy as np
from numpy.typing import NDArray

__version__: str

_Color = Union[str, Sequence[int]]
_Palette = Sequence[_Color]
_Matrix = Union[str, int, NDArray[Any]]
_ImageT = TypeVar("_ImageT")

@overload
def dither(
    image: NDArray[Any],
    method: str = "floyd_steinberg",
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    serpentine: bool = False,
    seed: int | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> NDArray[np.uint8]: ...
@overload
def dither(
    image: _ImageT,
    method: str = "floyd_steinberg",
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    serpentine: bool = False,
    seed: int | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> _ImageT:
    """Dither an image — the one-stop entry point.

    Dispatches on `method`: an error-diffusion name ('floyd_steinberg', 'fs',
    'atkinson', ...), an ordered matrix name ('bayer8x8', ...), or 'random'.
    Parameters that don't apply to the chosen method are ignored.

    Args:
        image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
            image (a PIL image is returned in that case).
        method: what to dither with; default 'floyd_steinberg'.
        levels: number of output levels per channel (2 = black & white).
        palette: dither to these colors instead of gray levels — a sequence
            of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
            image; the alpha channel is always passed through. Mutually
            exclusive with levels.
        serpentine: (error diffusion) alternate scan direction every row.
        seed: (random) optional seed for reproducible output.
        strength: dithering strength in [0, 1]; 0 = plain quantization.
        linear: do the math in linear light instead of raw sRGB values
            (gamma-correct; preserves perceived brightness).
        preserve_alpha: leave the last channel untouched for 2- or 4-channel
            images (LA / RGBA).

    Returns:
        A new uint8 array (or PIL image) of the same shape.
    """

@overload
def ordered_dither(
    image: NDArray[Any],
    matrix: _Matrix | None = None,
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> NDArray[np.uint8]: ...
@overload
def ordered_dither(
    image: _ImageT,
    matrix: _Matrix | None = None,
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> _ImageT:
    """Apply ordered (Bayer / threshold-matrix) dithering.

    Args:
        image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
            image (a PIL image is returned in that case).
        matrix: 'bayer2x2', 'bayer4x4', 'bayer8x8', 'bayer16x16'
            (case-insensitive), the size as an int (2, 4, 8, 16), or a custom
            2-D numpy matrix: integer dtype = Bayer-style indices, float
            dtype = thresholds in [0, 1]. None means 'bayer8x8'.
        levels: number of output levels per channel (2 = black & white).
        palette: dither to these colors instead of gray levels — a sequence
            of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
            image; the alpha channel is always passed through. Mutually
            exclusive with levels.
        strength: dithering strength in [0, 1]; 0 = plain quantization.
        linear: do the math in linear light instead of raw sRGB values
            (gamma-correct; preserves perceived brightness).
        preserve_alpha: leave the last channel untouched for 2- or 4-channel
            images (LA / RGBA).

    Returns:
        A new uint8 array (or PIL image) of the same shape.
    """

@overload
def error_diffusion(
    image: NDArray[Any],
    method: str = "floyd_steinberg",
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    serpentine: bool = False,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> NDArray[np.uint8]: ...
@overload
def error_diffusion(
    image: _ImageT,
    method: str = "floyd_steinberg",
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    serpentine: bool = False,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> _ImageT:
    """Apply error-diffusion dithering (Floyd-Steinberg and friends).

    Args:
        image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
            image (a PIL image is returned in that case).
        method: 'floyd_steinberg', 'false_floyd_steinberg',
            'jarvis_judice_ninke', 'stucki', 'atkinson', 'burkes', 'sierra',
            'sierra_two_row' or 'sierra_lite' (case-insensitive; '-' and '_'
            interchangeable; aliases like 'fs', 'jjn', 'sierra2' work).
        levels: number of output levels per channel (2 = black & white).
        palette: dither to these colors instead of gray levels — a sequence
            of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
            image; the alpha channel is always passed through. Mutually
            exclusive with levels.
        serpentine: alternate scan direction every row; reduces directional
            worm artifacts.
        strength: fraction of the quantization error diffused, in [0, 1].
        linear: do the math in linear light instead of raw sRGB values
            (gamma-correct; preserves perceived brightness).
        preserve_alpha: leave the last channel untouched for 2- or 4-channel
            images (LA / RGBA).

    Returns:
        A new uint8 array (or PIL image) of the same shape.
    """

@overload
def random_dither(
    image: NDArray[Any],
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    seed: int | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> NDArray[np.uint8]: ...
@overload
def random_dither(
    image: _ImageT,
    *,
    levels: int = 2,
    palette: _Palette | None = None,
    seed: int | None = None,
    strength: float = 1.0,
    linear: bool = False,
    preserve_alpha: bool = True,
) -> _ImageT:
    """Apply random (white-noise) threshold dithering.

    Args:
        image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
            image (a PIL image is returned in that case).
        levels: number of output levels per channel (2 = black & white).
        palette: dither to these colors instead of gray levels — a sequence
            of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
            image; the alpha channel is always passed through. Mutually
            exclusive with levels.
        seed: optional seed for reproducible output; random when omitted.
        strength: dithering strength in [0, 1]; 0 = plain quantization.
        linear: do the math in linear light instead of raw sRGB values
            (gamma-correct; preserves perceived brightness).
        preserve_alpha: leave the last channel untouched for 2- or 4-channel
            images (LA / RGBA).

    Returns:
        A new uint8 array (or PIL image) of the same shape.
    """

def available_matrices() -> list[str]:
    """List the valid `matrix` values for `ordered_dither`."""

def available_methods() -> list[str]:
    """List the valid `method` values for `error_diffusion`."""
