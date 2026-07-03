//! Error-diffusion dithering: Floyd–Steinberg and friends.

use crate::palette::Palette;
use crate::space::{Levels, Space};

/// One tap of a diffusion kernel: (dx, dy, weight). Weights are divided by `div`.
type Tap = (i32, i32, f32);

pub struct Kernel {
    pub name: &'static str,
    pub taps: &'static [Tap],
    pub div: f32,
}

pub const FLOYD_STEINBERG: Kernel = Kernel {
    name: "floyd_steinberg",
    taps: &[(1, 0, 7.0), (-1, 1, 3.0), (0, 1, 5.0), (1, 1, 1.0)],
    div: 16.0,
};

pub const FALSE_FLOYD_STEINBERG: Kernel = Kernel {
    name: "false_floyd_steinberg",
    taps: &[(1, 0, 3.0), (0, 1, 3.0), (1, 1, 2.0)],
    div: 8.0,
};

pub const JARVIS_JUDICE_NINKE: Kernel = Kernel {
    name: "jarvis_judice_ninke",
    taps: &[
        (1, 0, 7.0),
        (2, 0, 5.0),
        (-2, 1, 3.0),
        (-1, 1, 5.0),
        (0, 1, 7.0),
        (1, 1, 5.0),
        (2, 1, 3.0),
        (-2, 2, 1.0),
        (-1, 2, 3.0),
        (0, 2, 5.0),
        (1, 2, 3.0),
        (2, 2, 1.0),
    ],
    div: 48.0,
};

pub const STUCKI: Kernel = Kernel {
    name: "stucki",
    taps: &[
        (1, 0, 8.0),
        (2, 0, 4.0),
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 8.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
        (-2, 2, 1.0),
        (-1, 2, 2.0),
        (0, 2, 4.0),
        (1, 2, 2.0),
        (2, 2, 1.0),
    ],
    div: 42.0,
};

/// Atkinson intentionally diffuses only 6/8 of the error (loses detail in
/// deep shadows/highlights but looks crisp — the classic Mac look).
pub const ATKINSON: Kernel = Kernel {
    name: "atkinson",
    taps: &[
        (1, 0, 1.0),
        (2, 0, 1.0),
        (-1, 1, 1.0),
        (0, 1, 1.0),
        (1, 1, 1.0),
        (0, 2, 1.0),
    ],
    div: 8.0,
};

pub const BURKES: Kernel = Kernel {
    name: "burkes",
    taps: &[
        (1, 0, 8.0),
        (2, 0, 4.0),
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 8.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
    ],
    div: 32.0,
};

pub const SIERRA: Kernel = Kernel {
    name: "sierra",
    taps: &[
        (1, 0, 5.0),
        (2, 0, 3.0),
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 5.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
        (-1, 2, 2.0),
        (0, 2, 3.0),
        (1, 2, 2.0),
    ],
    div: 32.0,
};

pub const SIERRA_TWO_ROW: Kernel = Kernel {
    name: "sierra_two_row",
    taps: &[
        (1, 0, 4.0),
        (2, 0, 3.0),
        (-2, 1, 1.0),
        (-1, 1, 2.0),
        (0, 1, 3.0),
        (1, 1, 2.0),
        (2, 1, 1.0),
    ],
    div: 16.0,
};

pub const SIERRA_LITE: Kernel = Kernel {
    name: "sierra_lite",
    taps: &[(1, 0, 2.0), (-1, 1, 1.0), (0, 1, 1.0)],
    div: 4.0,
};

pub const ALL_KERNELS: &[&Kernel] = &[
    &FLOYD_STEINBERG,
    &FALSE_FLOYD_STEINBERG,
    &JARVIS_JUDICE_NINKE,
    &STUCKI,
    &ATKINSON,
    &BURKES,
    &SIERRA,
    &SIERRA_TWO_ROW,
    &SIERRA_LITE,
];

/// Left padding of the rolling error rows; must cover the widest kernel
/// reach (2), so out-of-image taps land in discarded pad slots instead of
/// needing bounds checks.
const PAD: usize = 2;
/// Weight-window width. Taps span dx = -2..=2 (lanes 0..=4); the window is
/// padded to 8 zero-weighted lanes so the inner loop is two 4-wide SIMD
/// multiply-adds. Error rows carry `KW - PAD` slots of right padding.
const KW: usize = 8;

/// The kernel as three rows of KW weights (lane = dx + PAD) so the inner
/// loop is straight multiply-adds with no per-tap branching, plus the
/// horizontally mirrored version for right-to-left (serpentine) rows and a
/// used-rows mask to skip empty rows. `strength` is folded into the weights.
type KernelRows = [[f32; KW]; 3];

fn kernel_rows(kernel: &Kernel, strength: f32) -> (KernelRows, KernelRows, [bool; 3]) {
    let mut rows = [[0f32; KW]; 3];
    let mut rev = [[0f32; KW]; 3];
    let mut used = [false; 3];
    for &(dx, dy, w) in kernel.taps {
        let wn = w / kernel.div * strength;
        rows[dy as usize][(dx + PAD as i32) as usize] = wn;
        rev[dy as usize][(-dx + PAD as i32) as usize] = wn;
        used[dy as usize] |= wn != 0.0;
    }
    (rows, rev, used)
}

/// Error-diffuse one channel of an interleaved (H, W, C) u8 buffer in place.
///
/// Kernel taps that fall outside the image are dropped (their error is
/// discarded), matching the common convention. `serpentine` alternates the
/// scan direction per row, mirroring the kernel on right-to-left rows.
#[allow(clippy::too_many_arguments)]
fn diffuse_channel(
    buf: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
    kernel: &Kernel,
    levels: &Levels,
    space: &Space,
    strength: f32,
    serpentine: bool,
) {
    let (rows, rows_rev, used) = kernel_rows(kernel, strength);
    let binary = levels.encoded.len() == 2;
    let (enc_lo, enc_hi) = (levels.encoded[0], *levels.encoded.last().unwrap());
    let (work_lo, work_hi) = (levels.work[0], *levels.work.last().unwrap());

    // Rolling error rows for the current and next two scanlines, padded so
    // the KW-wide weight-window writes never go out of bounds.
    let rw = width + KW;
    let mut err0 = vec![0f32; rw];
    let mut err1 = vec![0f32; rw];
    let mut err2 = vec![0f32; rw];

    macro_rules! process_pixel {
        ($row:expr, $k:expr, $x:expr) => {{
            let x = $x;
            let p = &mut $row[x * channels + channel];
            let v = space.to_work[*p as usize] + err0[x + PAD];
            // The error is accounted against the working-space value of the
            // byte actually emitted (levels.work), not the ideal grid level —
            // for level counts that don't divide 255 they differ and the
            // output mean would drift otherwise.
            let (q, out) = if binary {
                // Branchless two-level fast path.
                if v >= 127.5 {
                    (work_hi, enc_hi)
                } else {
                    (work_lo, enc_lo)
                }
            } else {
                let (_, idx) = levels.quantize(v);
                (levels.work[idx], levels.encoded[idx])
            };
            *p = out;
            let e = v - q;
            // err index for dx=-2 is (x + PAD) - 2 = x, so the KW-slot
            // windows start at x.
            if used[0] {
                let e0 = &mut err0[x..x + KW];
                for j in 0..KW {
                    e0[j] += e * $k[0][j];
                }
            }
            if used[1] {
                let e1 = &mut err1[x..x + KW];
                for j in 0..KW {
                    e1[j] += e * $k[1][j];
                }
            }
            if used[2] {
                let e2 = &mut err2[x..x + KW];
                for j in 0..KW {
                    e2[j] += e * $k[2][j];
                }
            }
        }};
    }

    for y in 0..height {
        let reverse = serpentine && (y % 2 == 1);
        let row = &mut buf[y * width * channels..(y + 1) * width * channels];
        if reverse {
            for x in (0..width).rev() {
                process_pixel!(row, rows_rev, x);
            }
        } else {
            for x in 0..width {
                process_pixel!(row, rows, x);
            }
        }
        // Advance one scanline: err1 becomes current, err2 becomes next.
        std::mem::swap(&mut err0, &mut err1);
        std::mem::swap(&mut err1, &mut err2);
        err2.fill(0.0);
    }
}

/// Error-diffuse an interleaved (H, W, C) u8 buffer in place to evenly
/// spaced levels, channel by channel. `skip_channel` (alpha) is untouched.
#[allow(clippy::too_many_arguments)]
pub fn diffuse_levels(
    buf: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    skip_channel: Option<usize>,
    kernel: &Kernel,
    levels: &Levels,
    space: &Space,
    strength: f32,
    serpentine: bool,
) {
    debug_assert_eq!(buf.len(), width * height * channels);
    for c in 0..channels {
        if skip_channel == Some(c) {
            continue;
        }
        diffuse_channel(
            buf, width, height, channels, c, kernel, levels, space, strength, serpentine,
        );
    }
}

/// Error-diffuse to a fixed palette: quantize each pixel to the nearest
/// palette entry and diffuse the 3-channel error vector jointly. Only the
/// first three channels are dithered; any further channels are untouched.
#[allow(clippy::too_many_arguments)]
pub fn diffuse_palette(
    buf: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    kernel: &Kernel,
    palette: &Palette,
    space: &Space,
    strength: f32,
    serpentine: bool,
) {
    debug_assert!(channels >= 3);
    let (rows, rows_rev, used) = kernel_rows(kernel, strength);

    let rw = width + KW;
    let mut err0 = vec![[0f32; 3]; rw];
    let mut err1 = vec![[0f32; 3]; rw];
    let mut err2 = vec![[0f32; 3]; rw];

    macro_rules! process_pixel {
        ($row:expr, $k:expr, $x:expr) => {{
            let x = $x;
            let base = x * channels;
            let acc = err0[x + PAD];
            let v = [
                space.to_work[$row[base] as usize] + acc[0],
                space.to_work[$row[base + 1] as usize] + acc[1],
                space.to_work[$row[base + 2] as usize] + acc[2],
            ];
            let i = palette.nearest(v);
            let out = palette.srgb[i];
            $row[base] = out[0];
            $row[base + 1] = out[1];
            $row[base + 2] = out[2];
            let pw = palette.work[i];
            let e = [v[0] - pw[0], v[1] - pw[1], v[2] - pw[2]];
            if used[0] {
                let e0 = &mut err0[x..x + KW];
                for j in 0..KW {
                    let w = $k[0][j];
                    for c in 0..3 {
                        e0[j][c] += e[c] * w;
                    }
                }
            }
            if used[1] {
                let e1 = &mut err1[x..x + KW];
                for j in 0..KW {
                    let w = $k[1][j];
                    for c in 0..3 {
                        e1[j][c] += e[c] * w;
                    }
                }
            }
            if used[2] {
                let e2 = &mut err2[x..x + KW];
                for j in 0..KW {
                    let w = $k[2][j];
                    for c in 0..3 {
                        e2[j][c] += e[c] * w;
                    }
                }
            }
        }};
    }

    for y in 0..height {
        let reverse = serpentine && (y % 2 == 1);
        let row = &mut buf[y * width * channels..(y + 1) * width * channels];
        if reverse {
            for x in (0..width).rev() {
                process_pixel!(row, rows_rev, x);
            }
        } else {
            for x in 0..width {
                process_pixel!(row, rows, x);
            }
        }
        std::mem::swap(&mut err0, &mut err1);
        std::mem::swap(&mut err1, &mut err2);
        err2.fill([0.0; 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diffuse_gray(plane: &mut [u8], w: usize, h: usize, k: &Kernel, levels: u32, serp: bool) {
        let s = Space::new(false);
        let l = Levels::new(levels, &s);
        diffuse_levels(plane, w, h, 1, None, k, &l, &s, 1.0, serp);
    }

    fn mean(p: &[u8]) -> f64 {
        p.iter().map(|&v| v as f64).sum::<f64>() / p.len() as f64
    }

    #[test]
    fn kernel_weights_sum_to_divisor_except_atkinson() {
        for k in ALL_KERNELS {
            let sum: f32 = k.taps.iter().map(|t| t.2).sum();
            if k.name == "atkinson" {
                assert_eq!(sum, 6.0);
                assert_eq!(k.div, 8.0);
            } else {
                assert_eq!(sum, k.div, "kernel {} does not sum to its divisor", k.name);
            }
        }
    }

    #[test]
    fn all_taps_are_causal_and_within_pad() {
        // Error must only flow to not-yet-visited pixels: dy > 0, or
        // dy == 0 && dx > 0 — and every offset must fit the pad/rows layout.
        for k in ALL_KERNELS {
            for &(dx, dy, _) in k.taps {
                assert!(
                    dy > 0 || (dy == 0 && dx > 0),
                    "non-causal tap in {}",
                    k.name
                );
                assert!((0..=2).contains(&dy), "tap beyond two rows in {}", k.name);
                assert!(
                    dx.unsigned_abs() as usize <= PAD,
                    "tap beyond pad in {}",
                    k.name
                );
            }
        }
    }

    #[test]
    fn constant_gray_mean_is_preserved() {
        for k in ALL_KERNELS {
            for serp in [false, true] {
                let (w, h) = (128usize, 128usize);
                let mut plane = vec![100u8; w * h];
                diffuse_gray(&mut plane, w, h, k, 2, serp);
                let m = mean(&plane);
                // Kernels that diffuse less than 100% of the error by design
                // (Atkinson) get a wide band.
                let tol = if k.taps.iter().map(|t| t.2).sum::<f32>() < k.div {
                    30.0
                } else {
                    2.0
                };
                assert!(
                    (m - 100.0).abs() < tol,
                    "{} serpentine={serp}: mean {m} not ~100",
                    k.name
                );
                assert!(plane.iter().all(|&v| v == 0 || v == 255));
            }
        }
    }

    #[test]
    fn gradient_mean_is_preserved() {
        let (w, h) = (256usize, 64usize);
        let mut plane: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
        let expected = mean(&plane);
        diffuse_gray(&mut plane, w, h, &FLOYD_STEINBERG, 2, false);
        let m = mean(&plane);
        assert!(
            (m - expected).abs() < 2.0,
            "mean {m} vs expected {expected}"
        );
    }

    #[test]
    fn multi_level_quantization_snaps_to_levels() {
        let (w, h) = (64usize, 64usize);
        let mut plane = vec![100u8; w * h];
        diffuse_gray(&mut plane, w, h, &FLOYD_STEINBERG, 4, false);
        for &v in &plane {
            assert!([0u8, 85, 170, 255].contains(&v), "unexpected level {v}");
        }
        assert!((mean(&plane) - 100.0).abs() < 2.0);
    }

    #[test]
    fn pure_black_and_white_are_fixed_points() {
        for k in ALL_KERNELS {
            let (w, h) = (32usize, 32usize);
            let mut black = vec![0u8; w * h];
            diffuse_gray(&mut black, w, h, k, 2, false);
            assert!(black.iter().all(|&v| v == 0));
            let mut white = vec![255u8; w * h];
            diffuse_gray(&mut white, w, h, k, 2, true);
            assert!(white.iter().all(|&v| v == 255));
        }
    }

    #[test]
    fn serpentine_differs_from_raster_on_gradient() {
        let (w, h) = (64usize, 64usize);
        let src: Vec<u8> = (0..w * h).map(|i| (i % w * 4) as u8).collect();
        let mut a = src.clone();
        diffuse_gray(&mut a, w, h, &FLOYD_STEINBERG, 2, false);
        let mut b = src;
        diffuse_gray(&mut b, w, h, &FLOYD_STEINBERG, 2, true);
        assert_ne!(a, b);
        assert!((mean(&a) - mean(&b)).abs() < 3.0);
    }

    #[test]
    fn strength_zero_is_pure_threshold() {
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let (w, h) = (32usize, 32usize);
        let mut plane = vec![100u8; w * h];
        diffuse_levels(
            &mut plane,
            w,
            h,
            1,
            None,
            &FLOYD_STEINBERG,
            &l,
            &s,
            0.0,
            false,
        );
        assert!(plane.iter().all(|&v| v == 0)); // 100 < 127.5, no error carried
    }

    #[test]
    fn rgb_channels_are_independent_in_levels_mode() {
        // A channel that is constant 0 or 255 must stay exact even when other
        // channels diffuse heavily.
        let (w, h, ch) = (32usize, 32usize, 3usize);
        let mut buf = vec![0u8; w * h * ch];
        for i in 0..w * h {
            buf[i * ch] = 0;
            buf[i * ch + 1] = 128;
            buf[i * ch + 2] = 255;
        }
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        diffuse_levels(
            &mut buf,
            w,
            h,
            ch,
            None,
            &FLOYD_STEINBERG,
            &l,
            &s,
            1.0,
            false,
        );
        for i in 0..w * h {
            assert_eq!(buf[i * ch], 0);
            assert_eq!(buf[i * ch + 2], 255);
        }
    }

    #[test]
    fn palette_bw_matches_levels_mode_on_gray_ramp() {
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let p = crate::palette::Palette::new(vec![[0, 0, 0], [255, 255, 255]], &s);
        let (w, h) = (64usize, 64usize);
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i / 3) % 256) as u8).collect();
        let mut a = src.clone();
        diffuse_levels(&mut a, w, h, 3, None, &FLOYD_STEINBERG, &l, &s, 1.0, false);
        let mut b = src;
        diffuse_palette(&mut b, w, h, 3, &FLOYD_STEINBERG, &p, &s, 1.0, false);
        assert_eq!(a, b);
    }

    #[test]
    fn palette_output_only_contains_palette_colors() {
        let s = Space::new(false);
        let pal = vec![[0u8, 0, 0], [255, 0, 0], [0, 0, 255], [255, 255, 255]];
        let p = crate::palette::Palette::new(pal.clone(), &s);
        let (w, h) = (32usize, 32usize);
        let mut buf: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        diffuse_palette(&mut buf, w, h, 3, &FLOYD_STEINBERG, &p, &s, 1.0, true);
        for px in buf.chunks_exact(3) {
            assert!(
                pal.contains(&[px[0], px[1], px[2]]),
                "non-palette color {px:?}"
            );
        }
    }

    #[test]
    fn palette_mean_is_preserved_per_channel() {
        // With a palette rich enough (8 corners of the RGB cube), each
        // channel's mean must be preserved like independent BW dithering.
        let s = Space::new(false);
        let mut corners = Vec::new();
        for r in [0u8, 255] {
            for g in [0u8, 255] {
                for b in [0u8, 255] {
                    corners.push([r, g, b]);
                }
            }
        }
        let p = crate::palette::Palette::new(corners, &s);
        let (w, h) = (128usize, 128usize);
        let mut buf = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            buf.extend_from_slice(&[64, 128, 200]);
        }
        diffuse_palette(&mut buf, w, h, 3, &FLOYD_STEINBERG, &p, &s, 1.0, false);
        for (c, want) in [(0usize, 64.0), (1, 128.0), (2, 200.0)] {
            let m = buf.chunks_exact(3).map(|px| px[c] as f64).sum::<f64>() / (w * h) as f64;
            assert!(
                (m - want).abs() < 2.5,
                "channel {c}: mean {m}, want ~{want}"
            );
        }
    }
}
