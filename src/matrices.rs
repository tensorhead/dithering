//! Threshold matrices for ordered dithering: generated Bayer matrices and
//! custom user-supplied matrices.

/// A tiled threshold matrix with normalized thresholds in [0, 1).
#[derive(Debug, Clone)]
pub struct ThresholdMatrix {
    pub width: usize,
    pub height: usize,
    /// Row-major normalized thresholds.
    pub t: Vec<f32>,
}

impl ThresholdMatrix {
    /// Bayer matrix of side `size` (a power of two), by the closed form of
    /// the recursion B(2n) = [[4B+0, 4B+2], [4B+3, 4B+1]]: each coordinate
    /// bit pair contributes the base-4 digit 2*(x_b ^ y_b) + y_b, with low
    /// coordinate bits landing in high digit positions. Thresholds use the
    /// mean-preserving (index + 0.5) / n^2 convention.
    pub fn bayer(size: usize) -> Self {
        debug_assert!(size.is_power_of_two());
        let bits = size.trailing_zeros();
        let n2 = (size * size) as f32;
        let mut t = vec![0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut v = 0u32;
                let xc = (x ^ y) as u32;
                let yc = y as u32;
                for bit in 0..bits {
                    v = (v << 1) | ((xc >> bit) & 1);
                    v = (v << 1) | ((yc >> bit) & 1);
                }
                t[y * size + x] = (v as f32 + 0.5) / n2;
            }
        }
        Self {
            width: size,
            height: size,
            t,
        }
    }

    /// Custom matrix from Bayer-style integer indices (0..N-1 convention).
    /// Values are normalized as (v + 0.5) / N with N = max(cells, max + 1).
    pub fn from_indices(height: usize, width: usize, vals: &[i64]) -> Result<Self, String> {
        debug_assert_eq!(vals.len(), width * height);
        if width == 0 || height == 0 {
            return Err("matrix must be non-empty".into());
        }
        let max = *vals.iter().max().unwrap();
        let min = *vals.iter().min().unwrap();
        if min < 0 {
            return Err(format!(
                "integer matrix values must be >= 0, got {min} (note: unsigned \
                 64-bit values above 2^63 overflow to negative)"
            ));
        }
        // Cap well below f32 integer precision so (v + 0.5) / (max + 1)
        // stays meaningful and max + 1 cannot overflow.
        if max >= 1 << 24 {
            return Err(format!(
                "integer matrix values must be below 2^24, got {max}"
            ));
        }
        let denom = ((width * height) as i64).max(max + 1) as f32;
        let t = vals.iter().map(|&v| (v as f32 + 0.5) / denom).collect();
        Ok(Self { width, height, t })
    }

    /// Custom matrix from float thresholds; values must lie in [0, 1].
    /// A threshold of exactly 1.0 (common after `m / m.max()` normalization)
    /// is clamped just below 1 so that pure black is preserved, mirroring
    /// how 0.0 preserves pure white.
    pub fn from_thresholds(height: usize, width: usize, vals: &[f64]) -> Result<Self, String> {
        debug_assert_eq!(vals.len(), width * height);
        if width == 0 || height == 0 {
            return Err("matrix must be non-empty".into());
        }
        if vals
            .iter()
            .any(|&v| !(0.0..=1.0).contains(&v) || v.is_nan())
        {
            return Err("float matrix thresholds must be in [0, 1]".into());
        }
        let t = vals
            .iter()
            .map(|&v| (v as f32).min(1.0 - f32::EPSILON))
            .collect();
        Ok(Self { width, height, t })
    }

    /// The threshold for pixel (x, y), tiling the matrix.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.t[(y % self.height) * self.width + (x % self.width)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(m: &ThresholdMatrix) -> Vec<u32> {
        let n2 = (m.width * m.height) as f32;
        m.t.iter().map(|&t| (t * n2 - 0.5).round() as u32).collect()
    }

    #[test]
    fn bayer_2x2_matches_reference() {
        assert_eq!(indices(&ThresholdMatrix::bayer(2)), vec![0, 2, 3, 1]);
    }

    #[test]
    fn bayer_4x4_matches_reference() {
        #[rustfmt::skip]
        let expected = vec![
             0,  8,  2, 10,
            12,  4, 14,  6,
             3, 11,  1,  9,
            15,  7, 13,  5,
        ];
        assert_eq!(indices(&ThresholdMatrix::bayer(4)), expected);
    }

    #[test]
    fn bayer_8x8_matches_reference() {
        #[rustfmt::skip]
        let expected = vec![
             0, 32,  8, 40,  2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44,  4, 36, 14, 46,  6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43,  1, 33,  9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47,  7, 39, 13, 45,  5, 37,
            63, 31, 55, 23, 61, 29, 53, 21,
        ];
        assert_eq!(indices(&ThresholdMatrix::bayer(8)), expected);
    }

    #[test]
    fn every_bayer_index_appears_exactly_once() {
        for size in [2usize, 4, 8, 16, 32] {
            let idx = indices(&ThresholdMatrix::bayer(size));
            let mut seen = vec![false; size * size];
            for i in idx {
                assert!(!seen[i as usize], "duplicate index {i} in {size}x{size}");
                seen[i as usize] = true;
            }
        }
    }

    #[test]
    fn thresholds_are_strictly_inside_unit_interval() {
        for t in &ThresholdMatrix::bayer(16).t {
            assert!(*t > 0.0 && *t < 1.0);
        }
    }

    #[test]
    fn from_indices_normalizes_bayer_style() {
        let m = ThresholdMatrix::from_indices(2, 2, &[0, 2, 3, 1]).unwrap();
        assert_eq!(m.t, ThresholdMatrix::bayer(2).t);
    }

    #[test]
    fn from_indices_handles_sparse_values() {
        // Max value 15 in a 2x2 matrix: denominator becomes 16.
        let m = ThresholdMatrix::from_indices(2, 2, &[0, 5, 10, 15]).unwrap();
        assert!((m.t[3] - 15.5 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn from_indices_rejects_negatives() {
        assert!(ThresholdMatrix::from_indices(1, 2, &[0, -1]).is_err());
    }

    #[test]
    fn from_indices_rejects_values_beyond_f32_precision() {
        assert!(ThresholdMatrix::from_indices(1, 2, &[0, 1 << 24]).is_err());
        assert!(ThresholdMatrix::from_indices(1, 2, &[0, i64::MAX]).is_err());
        assert!(ThresholdMatrix::from_indices(1, 2, &[0, (1 << 24) - 1]).is_ok());
    }

    #[test]
    fn from_thresholds_validates_range() {
        assert!(ThresholdMatrix::from_thresholds(1, 2, &[0.0, 1.0]).is_ok());
        assert!(ThresholdMatrix::from_thresholds(1, 2, &[0.5, 1.2]).is_err());
        assert!(ThresholdMatrix::from_thresholds(1, 2, &[-0.1, 0.5]).is_err());
    }

    #[test]
    fn from_thresholds_clamps_one_below_unity() {
        // t = 1.0 (e.g. from m / m.max()) must not flip black pixels white.
        let m = ThresholdMatrix::from_thresholds(1, 1, &[1.0]).unwrap();
        assert!(m.t[0] < 1.0);
        // The perturbation (t - 0.5) * 255 must round DOWN from the midpoint.
        assert!((m.t[0] - 0.5) * 255.0 < 127.5);
    }

    #[test]
    fn non_square_matrix_tiles_with_both_dimensions() {
        // 1 row, 4 cols: at() must tile x by width and y by height.
        let m = ThresholdMatrix::from_thresholds(1, 4, &[0.1, 0.3, 0.5, 0.7]).unwrap();
        assert_eq!(m.at(0, 0), m.at(0, 5));
        assert_eq!(m.at(1, 0), m.at(5, 3));
        assert!((m.at(2, 7) - 0.5).abs() < 1e-6);
    }
}
