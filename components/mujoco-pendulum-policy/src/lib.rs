//! # mujoco-pendulum-policy
//!
//! Neural policy (MLP 4→32→1) trained on **real MuJoCo physics** (InvertedPendulum)
//! via DAgger, executed `no_std` heap-free `unsafe`-free on seL4. Input: system state
//! `[x, theta, x_dot, theta_dot]`; output: normalised force command [-1,1] (clipped by the safety layer).
//!
//! Validates the pipeline: train on realistic physics (Host) → export → no_std inference on the verified kernel.

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

/// Normalised force command from system state (forward pass of the MuJoCo-trained network).
pub fn command(state: &[f32; weights::IN]) -> f32 {
    let h = relu(linear(state, &weights::W1, &weights::B1));
    linear(&h, &weights::W2, &weights::B2)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_within_epsilon() {
        // SIL: Rust inference matches numpy (trained on MuJoCo) on the same weights.
        for (refin, refout) in weights::REF_IN.iter().zip(weights::REF_OUT.iter()) {
            let y = command(refin);
            assert!(
                (y - refout[0]).abs() < 1e-3,
                "rust={} vs numpy={}",
                y,
                refout[0]
            );
        }
    }
}
