//! Ordered (Bayer/threshold-matrix) and random (white-noise) dithering.

use crate::matrices::ThresholdMatrix;
use crate::palette::Palette;
use crate::space::{Levels, Space};

/// Ordered-dither an interleaved (H, W, C) u8 buffer in place, quantizing to
/// evenly spaced levels. Each pixel is perturbed by the tiled threshold
/// matrix and snapped to the nearest level:
/// out = quantize(work(p) + (t - 0.5) * step * strength). `skip_channel`
/// (alpha) is passed through untouched.
#[allow(clippy::too_many_arguments)]
pub fn ordered_levels(
    buf: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    skip_channel: Option<usize>,
    matrix: &ThresholdMatrix,
    levels: &Levels,
    space: &Space,
    strength: f32,
) {
    debug_assert_eq!(buf.len(), width * height * channels);
    let amp = levels.step * strength;
    for y in 0..height {
        let row = &mut buf[y * width * channels..(y + 1) * width * channels];
        for x in 0..width {
            let off = (matrix.at(x, y) - 0.5) * amp;
            for c in 0..channels {
                if skip_channel == Some(c) {
                    continue;
                }
                let p = &mut row[x * channels + c];
                let (_, idx) = levels.quantize(space.to_work[*p as usize] + off);
                *p = levels.encoded[idx];
            }
        }
    }
}

/// Ordered-dither to a fixed palette: perturb each color channel by the
/// threshold offset scaled to that channel's palette spread, then snap the
/// pixel to the nearest palette entry. Only the first three channels are
/// dithered; `skip_channel` (alpha) is passed through.
#[allow(clippy::too_many_arguments)]
pub fn ordered_palette(
    buf: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    matrix: &ThresholdMatrix,
    palette: &Palette,
    space: &Space,
    strength: f32,
) {
    debug_assert!(channels >= 3);
    let amp = [
        palette.spread[0] * strength,
        palette.spread[1] * strength,
        palette.spread[2] * strength,
    ];
    for y in 0..height {
        let row = &mut buf[y * width * channels..(y + 1) * width * channels];
        for x in 0..width {
            let t = matrix.at(x, y) - 0.5;
            let base = x * channels;
            let v = [
                space.to_work[row[base] as usize] + t * amp[0],
                space.to_work[row[base + 1] as usize] + t * amp[1],
                space.to_work[row[base + 2] as usize] + t * amp[2],
            ];
            let e = palette.srgb[palette.nearest(v)];
            row[base] = e[0];
            row[base + 1] = e[1];
            row[base + 2] = e[2];
        }
    }
}

/// SplitMix64: tiny, seedable, high-quality-enough PRNG for white noise.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Random (white-noise) dithering to evenly spaced levels. One random
/// perturbation per pixel, shared across channels so gray stays gray.
#[allow(clippy::too_many_arguments)]
pub fn random_levels(
    buf: &mut [u8],
    channels: usize,
    skip_channel: Option<usize>,
    levels: &Levels,
    space: &Space,
    strength: f32,
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let amp = levels.step * strength;
    for px in buf.chunks_exact_mut(channels) {
        let off = (rng.uniform() - 0.5) * amp;
        for (c, p) in px.iter_mut().enumerate() {
            if skip_channel == Some(c) {
                continue;
            }
            let (_, idx) = levels.quantize(space.to_work[*p as usize] + off);
            *p = levels.encoded[idx];
        }
    }
}

/// Random (white-noise) dithering to a fixed palette.
#[allow(clippy::too_many_arguments)]
pub fn random_palette(
    buf: &mut [u8],
    channels: usize,
    palette: &Palette,
    space: &Space,
    strength: f32,
    seed: u64,
) {
    debug_assert!(channels >= 3);
    let mut rng = SplitMix64::new(seed);
    let amp = [
        palette.spread[0] * strength,
        palette.spread[1] * strength,
        palette.spread[2] * strength,
    ];
    for px in buf.chunks_exact_mut(channels) {
        let t = rng.uniform() - 0.5;
        let v = [
            space.to_work[px[0] as usize] + t * amp[0],
            space.to_work[px[1] as usize] + t * amp[1],
            space.to_work[px[2] as usize] + t * amp[2],
        ];
        let e = palette.srgb[palette.nearest(v)];
        px[0] = e[0];
        px[1] = e[1];
        px[2] = e[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dither_gray(plane: &mut [u8], w: usize, h: usize, size: usize, levels: u32) {
        let m = ThresholdMatrix::bayer(size);
        let s = Space::new(false);
        let l = Levels::new(levels, &s);
        ordered_levels(plane, w, h, 1, None, &m, &l, &s, 1.0);
    }

    fn mean(plane: &[u8]) -> f64 {
        plane.iter().map(|&v| v as f64).sum::<f64>() / plane.len() as f64
    }

    #[test]
    fn mid_gray_is_half_white_for_all_sizes() {
        for size in [2usize, 4, 8, 16] {
            let (w, h) = (64usize, 64usize);
            let mut plane = vec![128u8; w * h];
            dither_gray(&mut plane, w, h, size, 2);
            let white = plane.iter().filter(|&&v| v == 255).count() as f64 / (w * h) as f64;
            assert!(
                (white - 0.5).abs() <= 0.02,
                "{size}x{size}: mid gray gave {white} white, want ~0.5"
            );
        }
    }

    #[test]
    fn output_is_binary_for_two_levels() {
        let (w, h) = (37usize, 23usize); // deliberately not multiples of 8
        let mut plane: Vec<u8> = (0..w * h).map(|i| (i * 7 % 256) as u8).collect();
        dither_gray(&mut plane, w, h, 8, 2);
        assert!(plane.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn gradient_mean_is_roughly_preserved() {
        for size in [2usize, 4, 8, 16] {
            let (w, h) = (256usize, 64usize);
            let mut plane: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
            let before = mean(&plane);
            dither_gray(&mut plane, w, h, size, 2);
            let after = mean(&plane);
            assert!(
                (after - before).abs() < 4.0,
                "{size}x{size}: mean drifted {before} -> {after}"
            );
        }
    }

    #[test]
    fn black_and_white_are_fixed_points() {
        let mut black = vec![0u8; 64];
        dither_gray(&mut black, 8, 8, 4, 2);
        assert!(black.iter().all(|&v| v == 0));
        let mut white = vec![255u8; 64];
        dither_gray(&mut white, 8, 8, 4, 2);
        assert!(white.iter().all(|&v| v == 255));
    }

    #[test]
    fn multi_level_output_snaps_to_levels() {
        let (w, h) = (64usize, 64usize);
        let mut plane = vec![100u8; w * h];
        dither_gray(&mut plane, w, h, 8, 4);
        for &v in &plane {
            assert!([0u8, 85, 170, 255].contains(&v), "unexpected level {v}");
        }
        assert!((mean(&plane) - 100.0).abs() < 2.0);
    }

    #[test]
    fn strength_zero_is_pure_quantization() {
        let m = ThresholdMatrix::bayer(8);
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let (w, h) = (16usize, 16usize);
        let mut a = vec![100u8; w * h]; // below 127.5 -> all black
        ordered_levels(&mut a, w, h, 1, None, &m, &l, &s, 0.0);
        assert!(a.iter().all(|&v| v == 0));
        let mut b = vec![160u8; w * h];
        ordered_levels(&mut b, w, h, 1, None, &m, &l, &s, 0.0);
        assert!(b.iter().all(|&v| v == 255));
    }

    #[test]
    fn preserves_skip_channel() {
        let m = ThresholdMatrix::bayer(4);
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let (w, h, ch) = (16usize, 16usize, 4usize);
        let src: Vec<u8> = (0..w * h * ch).map(|i| (i * 31 % 256) as u8).collect();
        let mut out = src.clone();
        ordered_levels(&mut out, w, h, ch, Some(3), &m, &l, &s, 1.0);
        for i in 0..w * h {
            assert_eq!(
                out[i * ch + 3],
                src[i * ch + 3],
                "alpha changed at pixel {i}"
            );
            for c in 0..3 {
                let v = out[i * ch + c];
                assert!(v == 0 || v == 255);
            }
        }
    }

    #[test]
    fn linear_space_mid_gray_density_matches_linear_light() {
        // sRGB 128 is ~21.6% linear light, so gamma-correct binary dithering
        // should produce ~21.6% white, not 50%.
        let m = ThresholdMatrix::bayer(16);
        let s = Space::new(true);
        let l = Levels::new(2, &s);
        let (w, h) = (64usize, 64usize);
        let mut plane = vec![128u8; w * h];
        ordered_levels(&mut plane, w, h, 1, None, &m, &l, &s, 1.0);
        let white = plane.iter().filter(|&&v| v == 255).count() as f64 / (w * h) as f64;
        assert!(
            (white - 0.216).abs() < 0.02,
            "white fraction {white}, want ~0.216"
        );
    }

    #[test]
    fn palette_mode_bw_matches_levels_mode() {
        let m = ThresholdMatrix::bayer(8);
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let p = Palette::new(vec![[0, 0, 0], [255, 255, 255]], &s);
        let (w, h) = (32usize, 32usize);
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i / 3) * 5 % 256) as u8).collect();
        let mut a = src.clone();
        ordered_levels(&mut a, w, h, 3, None, &m, &l, &s, 1.0);
        let mut b = src;
        ordered_palette(&mut b, w, h, 3, &m, &p, &s, 1.0);
        // Not bit-identical in general (joint nearest vs per-channel), but for
        // a gray ramp replicated across channels they must agree.
        assert_eq!(a, b);
    }

    #[test]
    fn random_dither_is_seeded_and_mean_preserving() {
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        let (w, h) = (128usize, 128usize);
        let mut a = vec![128u8; w * h];
        random_levels(&mut a, 1, None, &l, &s, 1.0, 42);
        let mut b = vec![128u8; w * h];
        random_levels(&mut b, 1, None, &l, &s, 1.0, 42);
        assert_eq!(a, b);
        let mut c = vec![128u8; w * h];
        random_levels(&mut c, 1, None, &l, &s, 1.0, 43);
        assert_ne!(a, c);
        assert!((mean(&a) - 128.0).abs() < 4.0, "mean {}", mean(&a));
    }
}
