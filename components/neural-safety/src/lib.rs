//! # neural-safety — matched-compute *learned-safety* baselines
//!
//! Faithful, matched-compute reimplementations of the **mechanisms** of three learned-safety
//! families, for an honest comparison against the verification-preserving governor. As with the
//! CBF/Simplex baselines, these are reimplementations of the *mechanism* under identical sensing,
//! information, and training experience — **not** the original authors' published trained systems.
//!
//! All three share a small, genuinely-trained multilayer perceptron (`Mlp`) that learns a
//! **safe-speed boundary** `s(c) ∈ [0,1]` (a fraction of the pass speed) from the *same* incident
//! experience the governor's memory uses — same contexts, same sensing. They differ only in how the
//! learned estimate is *enforced*:
//!
//!  - **Neural CBF**: a learned barrier — cap the speed to the learned safe boundary (graded).
//!  - **Shielded RL**: a learned shield — let the policy run at full speed unless the learned
//!    classifier flags the context unsafe, then *project* the action down to the safe boundary.
//!  - **Neural Simplex**: a learned switch — run the untrusted controller unless the learned
//!    monitor flags the context unsafe, then *switch* to the verified slow backup.
//!
//! The MLP is deterministic (seeded init, `libm`-only `tanh`/`exp`, no heap, fixed-size), so the
//! comparison is reproducible and runs in the same `no_std` setting as the rest of the codebase.
#![no_std]
// Index loops are clearer than iterators for these small dense matrix-vector kernels.
#![allow(clippy::needless_range_loop)]

use libm::{expf, tanhf};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

/// A 2-layer MLP: `I` inputs → `H` tanh hidden units → 1 sigmoid output in `[0,1]`.
pub struct Mlp<const I: usize, const H: usize> {
    w1: [[f32; I]; H],
    b1: [f32; H],
    w2: [f32; H],
    b2: f32,
}

impl<const I: usize, const H: usize> Mlp<I, H> {
    /// Deterministic small-random initialization from `seed` (reproducible across runs/targets).
    pub fn new(seed: u64) -> Self {
        let mut st = seed | 1;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
        };
        let mut w1 = [[0.0f32; I]; H];
        let mut b1 = [0.0f32; H];
        for r in w1.iter_mut() {
            for x in r.iter_mut() {
                *x = rnd();
            }
        }
        for x in b1.iter_mut() {
            *x = rnd();
        }
        let mut w2 = [0.0f32; H];
        for x in w2.iter_mut() {
            *x = rnd();
        }
        Mlp {
            w1,
            b1,
            w2,
            b2: 0.0,
        }
    }

    fn forward(&self, x: &[f32; I]) -> ([f32; H], f32) {
        let mut h = [0.0f32; H];
        for j in 0..H {
            let mut s = self.b1[j];
            for i in 0..I {
                s += self.w1[j][i] * x[i];
            }
            h[j] = tanhf(s);
        }
        let mut o = self.b2;
        for j in 0..H {
            o += self.w2[j] * h[j];
        }
        (h, o)
    }

    /// Predicted safe-speed fraction in `[0,1]` for context `x`.
    pub fn predict(&self, x: &[f32; I]) -> f32 {
        sigmoid(self.forward(x).1)
    }

    /// One SGD step (MSE on the sigmoid output toward `target ∈ [0,1]`).
    pub fn train_step(&mut self, x: &[f32; I], target: f32, lr: f32) {
        let (h, o) = self.forward(x);
        let y = sigmoid(o);
        let dl = (y - target) * y * (1.0 - y); // dL/do
                                               // hidden layer first (uses the un-updated w2)
        for j in 0..H {
            let dh = dl * self.w2[j] * (1.0 - h[j] * h[j]); // tanh'
            for i in 0..I {
                self.w1[j][i] -= lr * dh * x[i];
            }
            self.b1[j] -= lr * dh;
        }
        // output layer
        for j in 0..H {
            self.w2[j] -= lr * dl * h[j];
        }
        self.b2 -= lr * dl;
    }

    /// Train the safe-speed boundary on incident experience: `hazard_ctx` get the low
    /// `hazard_target` safe-speed fraction, the sampled `safe_ctx` get `1.0`. For a fair
    /// comparison, `hazard_target` should encode the *same* safe speed the governor's warm memory
    /// stores at a hazard (so neither side is gratuitously more conservative there).
    pub fn train_boundary(
        &mut self,
        hazard_ctx: &[[f32; I]],
        safe_ctx: &[[f32; I]],
        hazard_target: f32,
        epochs: usize,
        lr: f32,
    ) {
        for _ in 0..epochs {
            for c in hazard_ctx {
                self.train_step(c, hazard_target, lr); // near a learned hazard: low safe speed
            }
            for c in safe_ctx {
                self.train_step(c, 1.0, lr); // elsewhere: full speed is safe
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_to_separate_hazard_from_safe() {
        // a hazard at context A=[1,0], safe everywhere else (sampled)
        let mut m = Mlp::<2, 8>::new(7);
        let haz = [[1.0f32, 0.0]];
        let safe = [[-1.0f32, 0.0], [0.0, 1.0], [0.0, -1.0]];
        m.train_boundary(&haz, &safe, 0.08, 400, 0.15);
        assert!(
            m.predict(&[1.0, 0.0]) < 0.4,
            "should flag the hazard context as low-safe-speed"
        );
        assert!(
            m.predict(&[-1.0, 0.0]) > 0.6,
            "should pass the safe context"
        );
    }

    #[test]
    fn predict_in_unit_range() {
        let m = Mlp::<2, 8>::new(1);
        let p = m.predict(&[0.3, -0.7]);
        assert!((0.0..=1.0).contains(&p));
    }
}
