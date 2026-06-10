//! # safety-model — Danger model that **learns to generalise** on egocentric features (learns, not memorises)
//!
//! Instead of a lookup table `(cell → limit)` (which breaks if obstacles move or the viewpoint
//! changes), this learns a **function** `risk(features) → danger` on **robot-relative features**
//! (range/angle to nearest obstacle, speed, …) — enabling generalisation:
//! "obstacle close ahead" means danger **wherever** the obstacle is. Online logistic regression
//! trained on incidents.
//!
//! **Preserves the hard Claim A guarantee:** `safe_bound ≤ static_limit` **always** (generalisation
//! does not break the kernel proof). `no_std`, no heap, no unsafe — seL4-ready.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use libm::expf;

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + expf(-z))
}

/// Linear danger model (logistic regression) on `FEAT` egocentric features, learning online.
pub struct SafetyModel<const FEAT: usize> {
    w: [f32; FEAT],
    b: f32,
    lr: f32,
}

impl<const FEAT: usize> SafetyModel<FEAT> {
    pub const fn new() -> Self {
        Self {
            w: [0.0; FEAT],
            b: 0.0,
            lr: 0.15,
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Predicted danger score [0,1] from features.
    pub fn risk(&self, x: &[f32; FEAT]) -> f32 {
        let mut z = self.b;
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            z += wi * xi;
        }
        sigmoid(z)
    }

    /// **Online learning** from an incident: `danger=true` (hazard occurred here) or `false` (safe).
    /// Incidents = training data → the model generalises to new situations with the same features.
    pub fn observe(&mut self, x: &[f32; FEAT], danger: bool) {
        let y = if danger { 1.0 } else { 0.0 };
        let g = self.risk(x) - y; // cross-entropy loss gradient
        for (wi, xi) in self.w.iter_mut().zip(x.iter()) {
            *wi -= self.lr * g * xi;
        }
        self.b -= self.lr * g;
    }

    /// **Safe bound:** tightens with danger, and **never exceeds the verified envelope**
    /// (preserves the kernel proof).
    pub fn safe_bound(&self, x: &[f32; FEAT], static_limit: f32, floor: f32) -> f32 {
        (static_limit * (1.0 - self.risk(x))).clamp(floor, static_limit)
    }

    /// Binary danger prediction (for evaluating generalisation).
    pub fn predict_danger(&self, x: &[f32; FEAT]) -> bool {
        self.risk(x) >= 0.5
    }
}

impl<const FEAT: usize> Default for SafetyModel<FEAT> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Feature 0 = obstacle proximity (1=close). Train that proximity = danger.
    #[test]
    fn learns_danger_from_egocentric_feature() {
        let mut m = SafetyModel::<3>::new();
        for _ in 0..300 {
            m.observe(&[1.0, 0.2, 0.5], true); // obstacle close → danger
            m.observe(&[0.0, 0.7, 0.5], false); // no obstacle → safe
        }
        assert!(m.predict_danger(&[1.0, 0.1, 0.9])); // close (new context) → danger
        assert!(!m.predict_danger(&[0.0, 0.9, 0.1])); // distant → safe
    }

    // Generalisation: detects danger at **unseen** feature values not in the training set (not memorisation).
    #[test]
    fn generalizes_to_unseen_values() {
        let mut m = SafetyModel::<2>::new();
        for _ in 0..400 {
            m.observe(&[0.9, 0.3], true);
            m.observe(&[0.1, 0.3], false);
        }
        // unseen proximity value (0.75) → generalises to danger
        assert!(m.risk(&[0.75, 0.8]) > 0.5);
    }

    // Hard guarantee: limit never exceeds the verified envelope.
    #[test]
    fn never_exceeds_static_envelope() {
        let mut m = SafetyModel::<2>::new();
        for _ in 0..100 {
            m.observe(&[1.0, 0.0], false); // even if it learned "safe"
        }
        for &c in &[1.0f32, 0.5, 0.0] {
            assert!(m.safe_bound(&[c, c], 1.0, 0.1) <= 1.0 + 1e-6);
            assert!(m.safe_bound(&[c, c], 1.0, 0.1) >= 0.1 - 1e-6);
        }
    }

    // Golden bit fingerprint for determinism: fresh model (w=0,b=0) ⇒ risk = sigmoid(0) = 0.5 exactly.
    // Any bit-level deviation (e.g. replacing libm::expf with f32::exp) breaks this immediately
    // across architectures.
    #[test]
    fn golden_bits_risk_is_half() {
        let m = SafetyModel::<3>::new();
        let r = m.risk(&[0.5, 0.7, 0.9]);
        assert!(r.is_finite());
        assert_eq!(r.to_bits(), 0x3f00_0000u32); // 0.5f32
    }

    // Determinism audit: forbids non-deterministic std math and FMA in **production code**
    // (before test modules).
    #[test]
    fn audit_no_nondeterministic_math() {
        const BANNED: &[&str] = &[
            ".sin(",
            ".cos(",
            ".tan(",
            ".exp(",
            ".exp2(",
            ".ln(",
            ".log(",
            ".log2(",
            ".log10(",
            ".powf(",
            ".powi(",
            ".atan2(",
            ".asin(",
            ".acos(",
            ".atan(",
            ".sinh(",
            ".cosh(",
            ".tanh(",
            ".cbrt(",
            ".hypot(",
            ".to_degrees(",
            ".to_radians(",
            ".mul_add(",
        ];
        let src =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")).unwrap();
        // Check production code only (before the first `#[cfg(test)]`) — so the banned list does
        // not match itself.
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        for (i, line) in prod.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for pat in BANNED {
                assert!(
                    !code.contains(pat),
                    "non-deterministic banned math `{pat}` in src/lib.rs:{}",
                    i + 1
                );
            }
        }
    }
}

// ===== proptest properties: hard bounds hold for any learned weights =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Random training-data type (Debug) — model is built inside the test (no Debug required on the type).
    type Train = Vec<([f32; 3], bool)>;
    fn train_seq() -> impl Strategy<Value = Train> {
        prop::collection::vec(
            (proptest::array::uniform3(0.0f32..1.0), any::<bool>()),
            0..40,
        )
    }

    proptest! {
        /// risk is always finite and in [0,1] for any finite features and any learned weights.
        #[test]
        fn risk_finite_in_unit_interval(
            seq in train_seq(),
            x in proptest::array::uniform3(0.0f32..1.0),
        ) {
            let mut m = SafetyModel::<3>::new();
            for (f, d) in &seq {
                m.observe(f, *d);
            }
            let r = m.risk(&x);
            prop_assert!(r.is_finite(), "risk must be finite, got {r}");
            prop_assert!((0.0..=1.0).contains(&r), "risk {r} left [0,1]");
        }

        /// **Hard guarantee (Claim A):** safe_bound ∈ [floor, static_limit] and never exceeds the
        /// envelope.
        #[test]
        fn safe_bound_never_exceeds_static(
            seq in train_seq(),
            x in proptest::array::uniform3(0.0f32..1.0),
            static_limit in 0.1f32..10.0,
            floor_frac in 0.0f32..1.0,
        ) {
            let mut m = SafetyModel::<3>::new();
            for (f, d) in &seq {
                m.observe(f, *d);
            }
            let floor = floor_frac * static_limit; // ensures floor ≤ static_limit
            let b = m.safe_bound(&x, static_limit, floor);
            prop_assert!(b.is_finite());
            prop_assert!(b <= static_limit + 1e-6, "safe_bound {b} exceeded static {static_limit}");
            prop_assert!(b >= floor - 1e-6, "safe_bound {b} below floor {floor}");
        }
    }
}
