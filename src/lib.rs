//! Fast image dithering for Python, written in Rust.

mod diffusion;
mod matrices;
mod ordered;
mod palette;
mod space;

use diffusion::Kernel;
use matrices::ThresholdMatrix;
use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use palette::Palette;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use space::{Levels, Space};

const BAYER_SIZES: &[usize] = &[2, 4, 8, 16];

// ---------------------------------------------------------------- parsing

/// Normalize a user-facing name: lowercase, drop '-', '_' and spaces.
fn normalize(name: &str) -> String {
    name.to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect()
}

/// Parse a matrix spec: "bayer8x8" / "Bayer8x8" / "8x8" / 8 / a 2-D numpy
/// array of Bayer-style integer indices or float thresholds in [0, 1].
fn parse_matrix(spec: &Bound<'_, PyAny>) -> PyResult<ThresholdMatrix> {
    if let Ok(n) = spec.extract::<usize>() {
        return bayer_of_size(n, &n.to_string());
    }
    if let Ok(s) = spec.extract::<String>() {
        let norm = normalize(&s);
        let body = norm.strip_prefix("bayer").unwrap_or(&norm);
        let head = body.split('x').next().unwrap_or("");
        return match head.parse::<usize>() {
            Ok(n) if body == format!("{n}x{n}") || body == format!("{n}") => bayer_of_size(n, &s),
            _ => Err(invalid_matrix_error(&s)),
        };
    }
    // Custom matrix: anything numpy can view as a 2-D array.
    let py = spec.py();
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (spec,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(format!(
            "custom dither matrix must be a non-empty 2-D array, got shape {shape:?}; \
             pass 'bayerNxN', the size as an int (2, 4, 8, 16), or a 2-D numpy array"
        )));
    }
    let (h, w) = (shape[0], shape[1]);
    let result = match kind.as_str() {
        "i" | "u" => {
            let vals: Vec<i64> = arr
                .call_method1("astype", ("int64",))?
                .call_method0("flatten")?
                .call_method0("tolist")?
                .extract()?;
            ThresholdMatrix::from_indices(h, w, &vals)
        }
        "f" => {
            let vals: Vec<f64> = arr
                .call_method1("astype", ("float64",))?
                .call_method0("flatten")?
                .call_method0("tolist")?
                .extract()?;
            ThresholdMatrix::from_thresholds(h, w, &vals)
        }
        k => {
            return Err(PyTypeError::new_err(format!(
                "custom dither matrix must have an integer dtype (Bayer-style \
                 indices) or float dtype (thresholds in [0, 1]), got kind {k:?}"
            )))
        }
    };
    result.map_err(PyValueError::new_err)
}

fn bayer_of_size(n: usize, got: &str) -> PyResult<ThresholdMatrix> {
    if BAYER_SIZES.contains(&n) {
        Ok(ThresholdMatrix::bayer(n))
    } else {
        Err(invalid_matrix_error(got))
    }
}

fn invalid_matrix_error(got: &str) -> PyErr {
    PyValueError::new_err(format!(
        "unknown dither matrix {got:?}; valid values: {}, the size as an int \
         (2, 4, 8, 16), or a custom 2-D numpy matrix",
        BAYER_SIZES
            .iter()
            .map(|n| format!("'bayer{n}x{n}'"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Parse an error-diffusion method name (case/format insensitive, aliases).
fn parse_method(name: &str) -> Option<&'static Kernel> {
    match normalize(name).as_str() {
        "floydsteinberg" | "fs" => Some(&diffusion::FLOYD_STEINBERG),
        "falsefloydsteinberg" | "ffs" => Some(&diffusion::FALSE_FLOYD_STEINBERG),
        "jarvisjudiceninke" | "jarvis" | "jjn" => Some(&diffusion::JARVIS_JUDICE_NINKE),
        "stucki" => Some(&diffusion::STUCKI),
        "atkinson" => Some(&diffusion::ATKINSON),
        "burkes" => Some(&diffusion::BURKES),
        "sierra" | "sierra3" => Some(&diffusion::SIERRA),
        "sierratworow" | "tworowsierra" | "sierra2" => Some(&diffusion::SIERRA_TWO_ROW),
        "sierralite" | "sierra24a" => Some(&diffusion::SIERRA_LITE),
        _ => None,
    }
}

fn invalid_method_error(got: &str) -> PyErr {
    PyValueError::new_err(format!(
        "unknown error-diffusion method {got:?}; valid values: {}",
        diffusion::ALL_KERNELS
            .iter()
            .map(|k| format!("'{}'", k.name))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Parse a palette given as a sequence of (r, g, b) tuples/lists or hex
/// strings like "#0f380f" / "0f380f" / "#fff".
fn parse_palette(spec: &Bound<'_, PyAny>) -> PyResult<Vec<[u8; 3]>> {
    if spec.extract::<String>().is_ok() {
        return Err(PyTypeError::new_err(
            "palette must be a sequence of colors, e.g. [\"#000\", \"#fff\"] \
             or [(0, 0, 0), (255, 255, 255)]",
        ));
    }
    let mut entries = Vec::new();
    for item in spec.try_iter()? {
        let item = item?;
        if let Ok(s) = item.extract::<String>() {
            entries.push(parse_hex_color(&s)?);
        } else if let Ok(rgb) = item.extract::<Vec<i64>>() {
            if rgb.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "palette color must have exactly 3 components (r, g, b), got {}",
                    rgb.len()
                )));
            }
            if rgb.iter().any(|&v| !(0..=255).contains(&v)) {
                return Err(PyValueError::new_err(format!(
                    "palette color components must be in 0..=255, got {rgb:?}"
                )));
            }
            entries.push([rgb[0] as u8, rgb[1] as u8, rgb[2] as u8]);
        } else {
            return Err(PyTypeError::new_err(
                "palette entries must be hex strings like '#0f380f' or (r, g, b) tuples",
            ));
        }
    }
    if entries.len() < 2 {
        return Err(PyValueError::new_err(
            "palette must contain at least 2 colors",
        ));
    }
    if entries.len() > 4096 {
        return Err(PyValueError::new_err(
            "palette must contain at most 4096 colors",
        ));
    }
    Ok(entries)
}

fn parse_hex_color(s: &str) -> PyResult<[u8; 3]> {
    let hex = s.strip_prefix('#').unwrap_or(s);
    let err = || {
        PyValueError::new_err(format!(
            "invalid hex color {s:?}; expected '#rgb' or '#rrggbb'"
        ))
    };
    // Reject non-hex bytes up front; this also keeps the byte-index slicing
    // below from landing inside a multi-byte UTF-8 character (which would
    // panic instead of raising).
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(err());
    }
    match hex.len() {
        3 => {
            let mut out = [0u8; 3];
            for (i, c) in hex.chars().enumerate() {
                let v = c.to_digit(16).ok_or_else(err)? as u8;
                out[i] = v * 17;
            }
            Ok(out)
        }
        6 => {
            let mut out = [0u8; 3];
            for i in 0..3 {
                out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| err())?;
            }
            Ok(out)
        }
        _ => Err(err()),
    }
}

// ---------------------------------------------------------------- image io

/// The validated, owned image ready for processing.
struct Img {
    data: Vec<u8>,
    shape: Vec<usize>,
    height: usize,
    width: usize,
    channels: usize,
    /// Channel index to pass through untouched (alpha), if any.
    skip_channel: Option<usize>,
    /// The original was a PIL image (mode kept for the return conversion).
    pil_mode: Option<String>,
}

/// Accept a uint8 numpy array (2-D gray or 3-D H×W×C) or a PIL image.
/// Produces an owned standard-layout copy.
fn extract_image(py: Python<'_>, image: &Bound<'_, PyAny>, preserve_alpha: bool) -> PyResult<Img> {
    let mut pil_mode = None;
    let mut source = image.clone();

    let arr: PyReadonlyArrayDyn<'_, u8> = match source.extract() {
        Ok(a) => a,
        Err(_) => {
            // Not a uint8 ndarray. PIL image? (isinstance check so that
            // subclasses defined outside the PIL package are recognized too;
            // skipped entirely when Pillow isn't installed.)
            if let Ok(pil) = py.import("PIL.Image") {
                let cls = pil.getattr("Image")?;
                if image.is_instance(&cls).unwrap_or(false) {
                    let mode: String = image.getattr("mode")?.extract()?;
                    if matches!(mode.as_str(), "L" | "LA" | "RGB" | "RGBA") {
                        pil_mode = Some(mode);
                    } else {
                        // Palette, 1-bit, float, CMYK... -> convert to RGB first.
                        source = image.call_method1("convert", ("RGB",))?;
                        pil_mode = Some("RGB".to_string());
                    }
                }
            }
            // Let numpy try (handles PIL, memoryviews, buffers...).
            let np = py.import("numpy")?;
            let converted = np.call_method1("asarray", (&source,))?;
            match converted.extract::<PyReadonlyArrayDyn<'_, u8>>() {
                Ok(a) => a,
                Err(_) => {
                    let dtype: String = converted
                        .getattr("dtype")
                        .and_then(|d| d.str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| "<unknown>".into());
                    return Err(PyTypeError::new_err(format!(
                        "expected a uint8 image array, got dtype {dtype}; convert with \
                         np.asarray(img, dtype=np.uint8), or (img * 255).clip(0, 255).astype(np.uint8) \
                         for float images in [0, 1]"
                    )));
                }
            }
        }
    };

    let shape = arr.shape().to_vec();
    let (height, width, channels) = match shape.len() {
        2 => (shape[0], shape[1], 1),
        3 => (shape[0], shape[1], shape[2]),
        d => {
            return Err(PyValueError::new_err(format!(
                "expected a 2-D grayscale or 3-D (H, W, C) image, got a {d}-D array with shape {shape:?}"
            )))
        }
    };
    if height == 0 || width == 0 || channels == 0 {
        return Err(PyValueError::new_err(format!(
            "image has an empty dimension: shape {shape:?}"
        )));
    }
    // Last channel is treated as alpha for LA (2) and RGBA (4) layouts.
    let skip_channel = if preserve_alpha && shape.len() == 3 && (channels == 2 || channels == 4) {
        Some(channels - 1)
    } else {
        None
    };
    let view = arr.as_array();
    let data = if let Some(slice) = view.as_slice() {
        slice.to_vec()
    } else {
        view.iter().copied().collect()
    };
    Ok(Img {
        data,
        shape,
        height,
        width,
        channels,
        skip_channel,
        pil_mode,
    })
}

/// Build the output object: a numpy array, or a PIL image if one came in.
fn into_output<'py>(py: Python<'py>, img: Img) -> PyResult<Bound<'py, PyAny>> {
    let pil_mode = img.pil_mode.clone();
    let arr = ArrayD::from_shape_vec(img.shape.clone(), img.data)
        .expect("shape/data length mismatch")
        .into_pyarray(py);
    match pil_mode {
        None => Ok(arr.into_any()),
        Some(mode) => {
            let pil = py.import("PIL.Image")?;
            // fromarray infers L/RGB/RGBA from the shape; LA needs the
            // explicit mode.
            if mode == "LA" {
                pil.call_method1("fromarray", (arr, "LA"))
            } else {
                pil.call_method1("fromarray", (arr,))
            }
        }
    }
}

// ---------------------------------------------------------------- shared setup

enum Target {
    Levels(Levels),
    Palette(Palette),
}

struct Job {
    img: Img,
    space: Space,
    target: Target,
    strength: f32,
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    levels: u32,
    palette: Option<&Bound<'_, PyAny>>,
    strength: f32,
    linear: bool,
    preserve_alpha: bool,
) -> PyResult<Job> {
    if !(0.0..=1.0).contains(&strength) {
        return Err(PyValueError::new_err(format!(
            "strength must be between 0.0 and 1.0, got {strength}"
        )));
    }
    if !(2..=256).contains(&levels) {
        return Err(PyValueError::new_err(format!(
            "levels must be between 2 and 256, got {levels}"
        )));
    }
    if palette.is_some() && levels != 2 {
        return Err(PyValueError::new_err(
            "pass either levels or palette, not both",
        ));
    }
    let entries = palette.map(parse_palette).transpose()?;
    let img = extract_image(py, image, preserve_alpha)?;
    if entries.is_some() && img.channels < 3 {
        return Err(PyValueError::new_err(
            "palette dithering needs an RGB or RGBA image; for grayscale use \
             `levels`, or stack to RGB with np.dstack([img] * 3)",
        ));
    }
    let space = Space::new(linear);
    let target = match entries {
        Some(e) => Target::Palette(Palette::new(e, &space)),
        None => Target::Levels(Levels::new(levels, &space)),
    };
    Ok(Job {
        img,
        space,
        target,
        strength,
    })
}

fn random_seed(seed: Option<i128>) -> u64 {
    // Truncating cast so any Python int (including negatives, e.g. hash()
    // results) is a valid seed.
    seed.map(|s| s as u64).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
    }) ^ 0xD1B5_4A32_D192_ED03
}

// ---------------------------------------------------------------- pyfunctions

/// Apply ordered (Bayer / threshold-matrix) dithering.
///
/// Args:
///     image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
///         image (a PIL image is returned in that case).
///     matrix: 'bayer2x2', 'bayer4x4', 'bayer8x8', 'bayer16x16'
///         (case-insensitive), the size as an int (2, 4, 8, 16), or a custom
///         2-D numpy matrix: integer dtype = Bayer-style indices,
///         float dtype = thresholds in [0, 1].
///     levels: number of output levels per channel (2 = black & white).
///     palette: dither to these colors instead of gray levels — a sequence
///         of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
///         image; the alpha channel is always passed through. Mutually
///         exclusive with levels.
///     strength: dithering strength in [0, 1]; 0 = plain quantization.
///     linear: do the math in linear light instead of raw sRGB values
///         (gamma-correct; preserves perceived brightness).
///     preserve_alpha: leave the last channel untouched for 2- or 4-channel
///         images (LA / RGBA).
///
/// Returns:
///     A new uint8 array (or PIL image) of the same shape.
#[pyfunction]
#[pyo3(signature = (image, matrix = None, *, levels = 2, palette = None, strength = 1.0, linear = false, preserve_alpha = true))]
#[allow(clippy::too_many_arguments)]
fn ordered_dither<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    matrix: Option<&Bound<'py, PyAny>>,
    levels: u32,
    palette: Option<&Bound<'py, PyAny>>,
    strength: f32,
    linear: bool,
    preserve_alpha: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let m = match matrix {
        Some(spec) => parse_matrix(spec)?,
        None => ThresholdMatrix::bayer(8),
    };
    let mut job = prepare(py, image, levels, palette, strength, linear, preserve_alpha)?;
    py.detach(|| {
        let img = &mut job.img;
        match &job.target {
            Target::Levels(l) => ordered::ordered_levels(
                &mut img.data,
                img.width,
                img.height,
                img.channels,
                img.skip_channel,
                &m,
                l,
                &job.space,
                job.strength,
            ),
            Target::Palette(p) => ordered::ordered_palette(
                &mut img.data,
                img.width,
                img.height,
                img.channels,
                &m,
                p,
                &job.space,
                job.strength,
            ),
        }
    });
    into_output(py, job.img)
}

/// Apply error-diffusion dithering (Floyd–Steinberg and friends).
///
/// Args:
///     image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
///         image (a PIL image is returned in that case).
///     method: 'floyd_steinberg', 'false_floyd_steinberg',
///         'jarvis_judice_ninke', 'stucki', 'atkinson', 'burkes', 'sierra',
///         'sierra_two_row' or 'sierra_lite' (case-insensitive; '-' and '_'
///         interchangeable; aliases like 'fs', 'jjn', 'sierra2' work).
///     levels: number of output levels per channel (2 = black & white).
///     palette: dither to these colors instead of gray levels — a sequence
///         of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
///         image; the alpha channel is always passed through. Mutually
///         exclusive with levels.
///     serpentine: alternate scan direction every row; reduces directional
///         worm artifacts.
///     strength: fraction of the quantization error diffused, in [0, 1].
///     linear: do the math in linear light instead of raw sRGB values
///         (gamma-correct; preserves perceived brightness).
///     preserve_alpha: leave the last channel untouched for 2- or 4-channel
///         images (LA / RGBA).
///
/// Returns:
///     A new uint8 array (or PIL image) of the same shape.
#[pyfunction]
#[pyo3(signature = (image, method = "floyd_steinberg", *, levels = 2, palette = None, serpentine = false, strength = 1.0, linear = false, preserve_alpha = true))]
#[allow(clippy::too_many_arguments)]
fn error_diffusion<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    method: &str,
    levels: u32,
    palette: Option<&Bound<'py, PyAny>>,
    serpentine: bool,
    strength: f32,
    linear: bool,
    preserve_alpha: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel = parse_method(method).ok_or_else(|| invalid_method_error(method))?;
    let mut job = prepare(py, image, levels, palette, strength, linear, preserve_alpha)?;
    py.detach(|| {
        let img = &mut job.img;
        match &job.target {
            Target::Levels(l) => diffusion::diffuse_levels(
                &mut img.data,
                img.width,
                img.height,
                img.channels,
                img.skip_channel,
                kernel,
                l,
                &job.space,
                job.strength,
                serpentine,
            ),
            Target::Palette(p) => diffusion::diffuse_palette(
                &mut img.data,
                img.width,
                img.height,
                img.channels,
                kernel,
                p,
                &job.space,
                job.strength,
                serpentine,
            ),
        }
    });
    into_output(py, job.img)
}

/// Apply random (white-noise) threshold dithering.
///
/// Args:
///     image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
///         image (a PIL image is returned in that case).
///     levels: number of output levels per channel (2 = black & white).
///     palette: dither to these colors instead of gray levels — a sequence
///         of '#rrggbb' hex strings or (r, g, b) tuples. Needs an RGB(A)
///         image; the alpha channel is always passed through. Mutually
///         exclusive with levels.
///     seed: optional seed for reproducible output; random when omitted.
///     strength: dithering strength in [0, 1]; 0 = plain quantization.
///     linear: do the math in linear light instead of raw sRGB values
///         (gamma-correct; preserves perceived brightness).
///     preserve_alpha: leave the last channel untouched for 2- or 4-channel
///         images (LA / RGBA).
///
/// Returns:
///     A new uint8 array (or PIL image) of the same shape.
#[pyfunction]
#[pyo3(signature = (image, *, levels = 2, palette = None, seed = None, strength = 1.0, linear = false, preserve_alpha = true))]
#[allow(clippy::too_many_arguments)]
fn random_dither<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    levels: u32,
    palette: Option<&Bound<'py, PyAny>>,
    seed: Option<i128>,
    strength: f32,
    linear: bool,
    preserve_alpha: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let mut job = prepare(py, image, levels, palette, strength, linear, preserve_alpha)?;
    let seed = random_seed(seed);
    py.detach(|| {
        let img = &mut job.img;
        match &job.target {
            Target::Levels(l) => ordered::random_levels(
                &mut img.data,
                img.channels,
                img.skip_channel,
                l,
                &job.space,
                job.strength,
                seed,
            ),
            Target::Palette(p) => ordered::random_palette(
                &mut img.data,
                img.channels,
                p,
                &job.space,
                job.strength,
                seed,
            ),
        }
    });
    into_output(py, job.img)
}

/// Dither an image — the one-stop entry point.
///
/// Dispatches on `method`: an error-diffusion name ('floyd_steinberg', 'fs',
/// 'atkinson', ...), an ordered matrix name ('bayer8x8', ...), or 'random'.
/// See `error_diffusion`, `ordered_dither` and `random_dither` for the
/// method-specific parameters; parameters that don't apply to the chosen
/// method are ignored.
///
/// Args:
///     image: uint8 numpy array — 2-D grayscale or 3-D (H, W, C) — or a PIL
///         image (a PIL image is returned in that case).
///     method: what to dither with; default 'floyd_steinberg'.
///     levels: number of output levels per channel (2 = black & white).
///     palette: dither to these colors instead of gray levels — a sequence
///         of '#rrggbb' hex strings or (r, g, b) tuples; the alpha channel
///         is always passed through.
///     serpentine: (error diffusion) alternate scan direction every row.
///     seed: (random) optional seed for reproducible output.
///     strength: dithering strength in [0, 1]; 0 = plain quantization.
///     linear: do the math in linear light instead of raw sRGB values
///         (gamma-correct; preserves perceived brightness).
///     preserve_alpha: leave the last channel untouched for 2- or 4-channel
///         images (LA / RGBA).
///
/// Returns:
///     A new uint8 array (or PIL image) of the same shape.
#[pyfunction]
#[pyo3(signature = (image, method = "floyd_steinberg", *, levels = 2, palette = None, serpentine = false, seed = None, strength = 1.0, linear = false, preserve_alpha = true))]
#[allow(clippy::too_many_arguments)]
fn dither<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    method: &str,
    levels: u32,
    palette: Option<&Bound<'py, PyAny>>,
    serpentine: bool,
    seed: Option<i128>,
    strength: f32,
    linear: bool,
    preserve_alpha: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if parse_method(method).is_some() {
        return error_diffusion(
            py,
            image,
            method,
            levels,
            palette,
            serpentine,
            strength,
            linear,
            preserve_alpha,
        );
    }
    if normalize(method) == "random" {
        return random_dither(
            py,
            image,
            levels,
            palette,
            seed,
            strength,
            linear,
            preserve_alpha,
        );
    }
    let as_str = pyo3::types::PyString::new(py, method);
    if parse_matrix(as_str.as_any()).is_ok() {
        return ordered_dither(
            py,
            image,
            Some(as_str.as_any()),
            levels,
            palette,
            strength,
            linear,
            preserve_alpha,
        );
    }
    Err(PyValueError::new_err(format!(
        "unknown dither method {method:?}; valid values: {}, {}, 'random'",
        diffusion::ALL_KERNELS
            .iter()
            .map(|k| format!("'{}'", k.name))
            .collect::<Vec<_>>()
            .join(", "),
        BAYER_SIZES
            .iter()
            .map(|n| format!("'bayer{n}x{n}'"))
            .collect::<Vec<_>>()
            .join(", "),
    )))
}

/// List the valid `matrix` values for `ordered_dither`.
#[pyfunction]
fn available_matrices() -> Vec<String> {
    BAYER_SIZES
        .iter()
        .map(|n| format!("bayer{n}x{n}"))
        .collect()
}

/// List the valid `method` values for `error_diffusion`.
#[pyfunction]
fn available_methods() -> Vec<String> {
    diffusion::ALL_KERNELS
        .iter()
        .map(|k| k.name.to_string())
        .collect()
}

#[pymodule]
fn dithering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(dither, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_dither, m)?)?;
    m.add_function(wrap_pyfunction!(error_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(random_dither, m)?)?;
    m.add_function(wrap_pyfunction!(available_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(available_methods, m)?)?;
    Ok(())
}
