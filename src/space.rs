//! Working color space for the dithering math: raw sRGB values (the common
//! convention) or linear light (gamma-correct; preserves perceived brightness
//! on real displays).

pub struct Space {
    /// u8 sRGB value -> working-space value on the 0..=255 scale.
    pub to_work: [f32; 256],
    linear: bool,
}

impl Space {
    pub fn new(linear: bool) -> Self {
        let mut to_work = [0f32; 256];
        for (i, w) in to_work.iter_mut().enumerate() {
            *w = if linear {
                srgb_to_linear(i as f32 / 255.0) * 255.0
            } else {
                i as f32
            };
        }
        Self { to_work, linear }
    }

    /// Encode a working-space value (0..=255 scale) back to an sRGB u8.
    /// Only called for the handful of quantized output values, never per pixel.
    pub fn encode(&self, v: f32) -> u8 {
        let v = v.clamp(0.0, 255.0);
        if self.linear {
            (linear_to_srgb(v / 255.0) * 255.0).round() as u8
        } else {
            v.round() as u8
        }
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// Evenly spaced quantization levels in the working space, with their
/// pre-encoded sRGB output values.
pub struct Levels {
    pub step: f32,
    inv_step: f32,
    max_idx: f32,
    /// level index -> output sRGB u8
    pub encoded: Vec<u8>,
    /// level index -> working-space value of that emitted byte. Error
    /// diffusion must account against what is actually written (the ideal
    /// level i*step is not always representable as a u8, e.g. 127.5 for
    /// levels=3), or the output mean drifts.
    pub work: Vec<f32>,
}

impl Levels {
    pub fn new(levels: u32, space: &Space) -> Self {
        let step = 255.0 / (levels - 1) as f32;
        let encoded: Vec<u8> = (0..levels).map(|i| space.encode(i as f32 * step)).collect();
        let work = encoded.iter().map(|&b| space.to_work[b as usize]).collect();
        Self {
            step,
            inv_step: 1.0 / step,
            max_idx: (levels - 1) as f32,
            encoded,
            work,
        }
    }

    /// Nearest level for a working-space value: (level value, level index).
    #[inline]
    pub fn quantize(&self, v: f32) -> (f32, usize) {
        let idx = (v * self.inv_step).round().clamp(0.0, self.max_idx);
        (idx * self.step, idx as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_space_round_trips_every_u8() {
        let s = Space::new(false);
        for i in 0..=255u8 {
            assert_eq!(s.to_work[i as usize], i as f32);
            assert_eq!(s.encode(i as f32), i);
        }
    }

    #[test]
    fn linear_space_round_trips_every_u8() {
        let s = Space::new(true);
        for i in 0..=255u8 {
            assert_eq!(
                s.encode(s.to_work[i as usize]),
                i,
                "round trip failed at {i}"
            );
        }
    }

    #[test]
    fn linear_space_maps_extremes_exactly() {
        let s = Space::new(true);
        assert_eq!(s.to_work[0], 0.0);
        assert!((s.to_work[255] - 255.0).abs() < 1e-3);
        // Mid sRGB gray is ~21.4% linear light.
        assert!((s.to_work[128] / 255.0 - 0.2158).abs() < 0.005);
    }

    #[test]
    fn levels_quantize_binary() {
        let s = Space::new(false);
        let l = Levels::new(2, &s);
        assert_eq!(l.quantize(0.0), (0.0, 0));
        assert_eq!(l.quantize(127.0), (0.0, 0));
        assert_eq!(l.quantize(128.0), (255.0, 1));
        assert_eq!(l.quantize(-40.0), (0.0, 0));
        assert_eq!(l.quantize(300.0), (255.0, 1));
        assert_eq!(l.encoded, vec![0, 255]);
    }

    #[test]
    fn levels_work_matches_emitted_bytes() {
        for linear in [false, true] {
            let s = Space::new(linear);
            let l = Levels::new(3, &s);
            for (i, &b) in l.encoded.iter().enumerate() {
                assert_eq!(l.work[i], s.to_work[b as usize], "linear={linear} idx={i}");
            }
        }
        // levels=3: the ideal mid level 127.5 is not a representable byte;
        // the emitted byte is 128 and `work` must reflect that.
        let s = Space::new(false);
        let l = Levels::new(3, &s);
        assert_eq!(l.encoded[1], 128);
        assert_eq!(l.work[1], 128.0);
    }

    #[test]
    fn levels_encoded_are_monotonic_in_linear_space() {
        let s = Space::new(true);
        let l = Levels::new(4, &s);
        assert_eq!(l.encoded[0], 0);
        assert_eq!(l.encoded[3], 255);
        assert!(l.encoded.windows(2).all(|w| w[0] < w[1]), "{:?}", l.encoded);
        // Linear 1/3 and 2/3 encode brighter than sRGB 85/170.
        assert!(l.encoded[1] > 85 && l.encoded[2] > 170, "{:?}", l.encoded);
    }
}
