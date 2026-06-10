//! # perception-encoder
//!
//! Converts a sensory "frame" (`FRAME` pixels) into an **embedding** of length `EMB` that feeds
//! into neural memory. Weights are trained on Host (`training/train_encoder.py`) so that similar
//! patterns cluster together, letting the robot recognize by similarity rather than exact match.
//! `no_std`, heap-free, `unsafe`-free.
//!
//! This bridges the "world" to a vector — the first step in the pipeline: perception → memory → policy.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod weights;

fn linear<const I: usize, const O: usize>(
    x: &[f32; I],
    w: &[[f32; I]; O],
    b: &[f32; O],
) -> [f32; O] {
    let mut out = [0.0f32; O];
    for o in 0..O {
        let row = &w[o];
        let mut acc = b[o];
        for i in 0..I {
            acc += row[i] * x[i];
        }
        out[o] = acc;
    }
    out
}

fn relu<const N: usize>(mut x: [f32; N]) -> [f32; N] {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    x
}

/// Encode a sensory frame into an embedding (`FRAME` → `EMB`, with ReLU).
pub fn encode(frame: &[f32; weights::FRAME]) -> [f32; weights::EMB] {
    relu(linear(frame, &weights::WE, &weights::BE))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..N {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        d / (na.sqrt() * nb.sqrt() + 1e-8)
    }

    #[test]
    fn matches_reference_within_epsilon() {
        // SIL: Rust inference matches numpy.
        for (k, refemb) in weights::REF_EMB.iter().enumerate() {
            let e = encode(&weights::CLEAN_FRAMES[k]);
            for (ev, rv) in e.iter().zip(refemb.iter()) {
                assert!((ev - rv).abs() < 1e-3, "case {}: {} vs {}", k, ev, rv);
            }
        }
    }

    #[test]
    fn noisy_query_recognized_as_correct_class() {
        // The noisy frame must be closest (cosine) to the prototype of its correct class.
        let eq = encode(&weights::QUERY_FRAME);
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for c in 0..weights::CLASSES {
            let ec = encode(&weights::CLEAN_FRAMES[c]);
            let s = cosine(&eq, &ec);
            if s > best_sim {
                best_sim = s;
                best = c;
            }
        }
        assert_eq!(
            best,
            weights::QUERY_CLASS,
            "recognized {} expected {}",
            best,
            weights::QUERY_CLASS
        );
        assert!(best_sim > 0.9, "weak recognition: {}", best_sim);
    }
}
