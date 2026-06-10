//! # policy-mlp
//!
//! Small MLP inference in `no_std`, heap-free, `unsafe`-free.
//! Weights are trained externally (on Host via `training/train_policy.py`) and exported
//! as `const`, then executed at the same precision inside seL4 (Software-in-the-Loop principle).
//!
//! Avoids the "tract/microflow no_std" risk by implementing inference manually — guaranteed on any target.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod weights;

/// Linear layer: `out[o] = b[o] + Σ_i w[o][i] * x[i]`.
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

/// In-place ReLU activation.
fn relu<const N: usize>(mut x: [f32; N]) -> [f32; N] {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    x
}

/// Network forward pass: input `IN` → hidden `HID` (ReLU) → output `OUT` (linear).
/// Bounded and deterministic: no allocation, no data-dependent branches in the hot path.
pub fn forward(x: &[f32; weights::IN]) -> [f32; weights::OUT] {
    let h = relu(linear(x, &weights::W1, &weights::B1));
    linear(&h, &weights::W2, &weights::B2)
}

/// Index of the highest value (action/class selection).
pub fn argmax<const N: usize>(x: &[f32; N]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_within_epsilon() {
        // SIL: Rust inference matches numpy on the same weights.
        for (k, (refin, refout)) in weights::REF_IN
            .iter()
            .zip(weights::REF_OUT.iter())
            .enumerate()
        {
            let y = forward(refin);
            for (yo, ro) in y.iter().zip(refout.iter()) {
                let diff = (yo - ro).abs();
                assert!(diff < 1e-3, "case {}: rust={} vs numpy={}", k, yo, ro);
            }
        }
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3, 0.2]), 1);
        assert_eq!(argmax(&[5.0, -1.0, 2.0]), 0);
    }
}
