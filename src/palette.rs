//! Fixed color palettes for dithering to arbitrary color sets.

use crate::space::Space;

pub struct Palette {
    /// Entries in working space (0..=255 scale per channel).
    pub work: Vec<[f32; 3]>,
    /// Original sRGB entries, written to the output verbatim.
    pub srgb: Vec<[u8; 3]>,
    /// Per-channel perturbation spread for ordered/random dithering, in
    /// working space: the channel range divided by (distinct values - 1),
    /// approximating the local quantization step. 0 for constant channels.
    pub spread: [f32; 3],
}

impl Palette {
    pub fn new(entries: Vec<[u8; 3]>, space: &Space) -> Self {
        let work: Vec<[f32; 3]> = entries
            .iter()
            .map(|e| {
                [
                    space.to_work[e[0] as usize],
                    space.to_work[e[1] as usize],
                    space.to_work[e[2] as usize],
                ]
            })
            .collect();
        let mut spread = [0f32; 3];
        for c in 0..3 {
            let mut vals: Vec<f32> = work.iter().map(|e| e[c]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();
            if vals.len() > 1 {
                spread[c] = (vals[vals.len() - 1] - vals[0]) / (vals.len() - 1) as f32;
            }
        }
        Self {
            work,
            srgb: entries,
            spread,
        }
    }

    /// Index of the nearest entry by Euclidean distance in working space.
    #[inline]
    pub fn nearest(&self, v: [f32; 3]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, e) in self.work.iter().enumerate() {
            let dr = v[0] - e[0];
            let dg = v[1] - e[1];
            let db = v[2] - e[2];
            let d = dr * dr + dg * dg + db * db;
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_picks_exact_matches() {
        let s = Space::new(false);
        let p = Palette::new(
            vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 255, 255]],
            &s,
        );
        assert_eq!(p.nearest([0.0, 0.0, 0.0]), 0);
        assert_eq!(p.nearest([250.0, 10.0, 10.0]), 1);
        assert_eq!(p.nearest([10.0, 250.0, 0.0]), 2);
        assert_eq!(p.nearest([255.0, 255.0, 255.0]), 3);
    }

    #[test]
    fn spread_matches_levels_for_bw() {
        let s = Space::new(false);
        let p = Palette::new(vec![[0, 0, 0], [255, 255, 255]], &s);
        assert_eq!(p.spread, [255.0, 255.0, 255.0]);
    }

    #[test]
    fn spread_zero_for_constant_channel() {
        let s = Space::new(false);
        // Blue channel constant.
        let p = Palette::new(vec![[0, 0, 7], [255, 128, 7]], &s);
        assert_eq!(p.spread[2], 0.0);
        assert_eq!(p.spread[0], 255.0);
        assert_eq!(p.spread[1], 128.0);
    }

    #[test]
    fn spread_uses_distinct_values() {
        let s = Space::new(false);
        // Red channel has distinct values {0, 128, 255} -> spread 127.5.
        let p = Palette::new(vec![[0, 0, 0], [128, 0, 0], [255, 0, 0], [128, 0, 0]], &s);
        assert!((p.spread[0] - 127.5).abs() < 1e-6);
    }
}
