//! # nav-policy — Learned navigation policy (PPO) as a `no_std` MLP (Bridge B)
//!
//! Navigation brain trained and exported from PPO: `obs(11) → 64(tanh) → 64(tanh) → 2 = [steering, thrust]`.
//! Same export pattern as cartpole/walker → **deployable on seL4** as the brain domain, governed by the guard.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

mod weights;
use libm::tanhf;
use weights::{B0, B2, BA, W0, W2, WA};

/// Forward pass: returns [steering, thrust] ∈ [-1,1] (clipped to action space, as in deterministic PPO).
pub fn act(obs: &[f32; 11]) -> [f32; 2] {
    let mut h0 = [0.0f32; 64];
    for (i, h) in h0.iter_mut().enumerate() {
        let mut s = B0[i];
        for (w, o) in W0[i].iter().zip(obs.iter()) {
            s += w * o;
        }
        *h = tanhf(s);
    }
    let mut h1 = [0.0f32; 64];
    for (i, h) in h1.iter_mut().enumerate() {
        let mut s = B2[i];
        for (w, x) in W2[i].iter().zip(h0.iter()) {
            s += w * x;
        }
        *h = tanhf(s);
    }
    let mut a = [0.0f32; 2];
    for (i, ai) in a.iter_mut().enumerate() {
        let mut s = BA[i];
        for (w, x) in WA[i].iter().zip(h1.iter()) {
            s += w * x;
        }
        *ai = s.clamp(-1.0, 1.0);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::SAMPLES;

    // Matching: Rust no_std output == PPO/Python output (same brain).
    #[test]
    fn matches_python_policy() {
        for (obs, expected) in SAMPLES.iter() {
            let got = act(obs);
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 2e-3, "mismatch: got {g} expected {e}");
            }
        }
    }
}
