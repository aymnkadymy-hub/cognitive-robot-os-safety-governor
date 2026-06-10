//! # humanoid-policy
//!
//! **Humanoid** locomotion policy (MLP `IN`→`H1`→`H2`→`OUT` with tanh) trained via PPO on
//! **MuJoCo Humanoid**, executed `no_std` heap-free `unsafe`-free on seL4. Input: body state;
//! output: joint commands ∈ [-1,1] (governed by the guard before actuators).

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod weights;

fn layer<const I: usize, const O: usize>(
    x: &[f32; I],
    w: &[[f32; I]; O],
    b: &[f32; O],
) -> [f32; O] {
    let mut out = [0.0f32; O];
    for o in 0..O {
        let mut acc = b[o];
        for i in 0..I {
            acc += w[o][i] * x[i];
        }
        out[o] = acc;
    }
    out
}

fn tanh<const N: usize>(mut x: [f32; N]) -> [f32; N] {
    for v in x.iter_mut() {
        *v = libm::tanhf(*v);
    }
    x
}

/// Joint commands from body state (forward pass of the reinforcement-trained network).
pub fn command(state: &[f32; weights::IN]) -> [f32; weights::OUT] {
    let h1 = tanh(layer(state, &weights::W0, &weights::B0));
    let h2 = tanh(layer(&h1, &weights::W1, &weights::B1));
    let mut a = layer(&h2, &weights::WA, &weights::BA);
    for v in a.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_within_epsilon() {
        for (refin, refout) in weights::REF_IN.iter().zip(weights::REF_OUT.iter()) {
            let h1 = tanh(layer(refin, &weights::W0, &weights::B0));
            let h2 = tanh(layer(&h1, &weights::W1, &weights::B1));
            let y = layer(&h2, &weights::WA, &weights::BA);
            for (got, want) in y.iter().zip(refout.iter()) {
                assert!((got - want).abs() < 1e-2, "rust={} vs torch={}", got, want);
            }
        }
    }
}
