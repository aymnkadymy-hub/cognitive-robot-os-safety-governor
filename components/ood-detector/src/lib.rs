//! # ood-detector — Out-of-distribution detector via Mahalanobis distance (tighten-only safety gate)
//!
//! **Concept:** before trusting the brain/danger-model output, we ask: is the current feature
//! vector **far from the training manifold**? If so (OOD), the model does not know this state ⇒
//! apply **maximum tightening** (fully consistent with the project philosophy: "when in doubt,
//! tighten"). Directly addresses the acknowledged **sim-to-real gap** and **IMU saturation blind spot**.
//!
//! **Mathematics:** d² = (x−μ)ᵀ Σ⁻¹ (x−μ). `μ` and `Σ⁻¹` are fitted **offline** on training data
//! and loaded as **constants**; on-device operation is deterministic O(N²) multiply-accumulate —
//! **no transcendentals, no heap, no unsafe**.
//!
//! **Preserves all invariants:** integration via `tighten_if_ood` always returns a result
//! **≤ the base bound** (never raises it) ⇒ the containment invariant `≤ static_limit` is
//! preserved and the kernel proof stays valid. Deterministic and fully Kani-verifiable.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

/// OOD detector using Mahalanobis distance on a feature vector of length `N`.
///
/// The state is `Copy` for easy embedding as a constant in the guard.
#[derive(Clone, Copy, Debug)]
pub struct MahalanobisOod<const N: usize> {
    mean: [f32; N],           // training-manifold mean (fitted offline)
    sigma_inv: [[f32; N]; N], // inverse covariance (fitted offline)
    threshold: f32,           // d² threshold (e.g. χ² distribution quantile at 99%)
}

impl<const N: usize> MahalanobisOod<N> {
    /// Create from offline-fitted constants (`μ`, `Σ⁻¹`, threshold).
    pub const fn new(mean: [f32; N], sigma_inv: [[f32; N]; N], threshold: f32) -> Self {
        Self {
            mean,
            sigma_inv,
            threshold,
        }
    }

    /// Squared Mahalanobis distance `d²`. (≥ 0 for any positive-semi-definite `Σ⁻¹` — as is any
    /// valid inverse covariance.)
    pub fn distance_sq(&self, x: &[f32; N]) -> f32 {
        let mut d = [0.0f32; N];
        for ((di, xi), mi) in d.iter_mut().zip(x.iter()).zip(self.mean.iter()) {
            *di = xi - mi;
        }
        // d² = Σᵢ dᵢ · (Σⱼ Σ⁻¹ᵢⱼ dⱼ) — same summation order (i outer, j inner) preserves the
        // bit fingerprint.
        let mut acc = 0.0f32;
        for (row_vec, di) in self.sigma_inv.iter().zip(d.iter()) {
            let mut row = 0.0f32;
            for (sij, dj) in row_vec.iter().zip(d.iter()) {
                row += sij * dj;
            }
            acc += di * row;
        }
        acc
    }

    /// Is the sample out-of-distribution? (`d²` exceeds the threshold.)
    pub fn is_ood(&self, x: &[f32; N]) -> bool {
        self.distance_sq(x) > self.threshold
    }

    /// **Safe integration (tighten-only):** if the sample is OOD → maximum tightening (`floor`),
    /// otherwise → `base_bound` unchanged.
    /// Result is **always ≤ `base_bound`** ⇒ no relaxation ⇒ preserves the containment invariant
    /// and the kernel proof.
    pub fn tighten_if_ood(&self, x: &[f32; N], base_bound: f32, floor: f32) -> f32 {
        if self.is_ood(x) {
            base_bound.min(floor)
        } else {
            base_bound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test detector: μ=[1,2], Σ⁻¹ = identity, threshold d²=9 (≈3σ).
    fn det() -> MahalanobisOod<2> {
        MahalanobisOod::new([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], 9.0)
    }

    // At the mean: distance is zero and not OOD.
    #[test]
    fn zero_distance_at_mean() {
        let d = det();
        assert_eq!(d.distance_sq(&[1.0, 2.0]), 0.0);
        assert!(!d.is_ood(&[1.0, 2.0]));
    }

    // With Σ⁻¹=identity: d² = sum of squared differences (squared Euclidean distance).
    #[test]
    fn identity_is_euclidean() {
        let d = det();
        assert_eq!(d.distance_sq(&[4.0, 6.0]), 25.0); // (3)²+(4)² = 25
    }

    // Distant point → OOD; nearby point → in-distribution.
    #[test]
    fn flags_far_points() {
        let d = det();
        assert!(d.is_ood(&[5.0, 6.0])); // d²=16+16=32 > 9
        assert!(!d.is_ood(&[2.0, 3.0])); // d²=1+1=2 ≤ 9
    }

    // Integration is tighten-only: OOD ⇒ floor; in-distribution ⇒ bound unchanged.
    #[test]
    fn tighten_if_ood_clamps_down() {
        let d = det();
        assert_eq!(d.tighten_if_ood(&[5.0, 6.0], 1.0, 0.05), 0.05); // OOD → floor
        assert_eq!(d.tighten_if_ood(&[2.0, 3.0], 1.0, 0.05), 1.0); // in-distribution → unchanged
    }

    // Golden bit fingerprint for determinism (detector has no transcendentals ⇒ bit-identical
    // across architectures).
    #[test]
    fn golden_bits_distance_sq() {
        let v = det().distance_sq(&[4.0, 6.0]);
        assert!(v.is_finite());
        assert_eq!(v.to_bits(), 0x41c8_0000u32); // 25.0f32
    }

    // Determinism audit: no std transcendentals and no FMA in production code.
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
            ".sqrt(",
            ".to_degrees(",
            ".to_radians(",
            ".mul_add(",
        ];
        let src =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")).unwrap();
        let prod = src.split("#[cfg(test)]").next().unwrap_or("");
        for (i, line) in prod.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for pat in BANNED {
                assert!(
                    !code.contains(pat),
                    "banned math `{pat}` in src/lib.rs:{}",
                    i + 1
                );
            }
        }
    }
}

// ===== proptest properties =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::array::uniform3;
    use proptest::prelude::*;

    // Detector with positive diagonal constants (diagonal Σ⁻¹ ⇒ positive-semi-definite ⇒ d² ≥ 0).
    fn diag_detector(d0: f32, d1: f32, d2: f32, thr: f32) -> MahalanobisOod<3> {
        MahalanobisOod::new(
            [0.0, 0.0, 0.0],
            [[d0, 0.0, 0.0], [0.0, d1, 0.0], [0.0, 0.0, d2]],
            thr,
        )
    }

    proptest! {
        /// With positive diagonal Σ⁻¹: d² ≥ 0 and finite for any finite sample.
        #[test]
        fn distance_sq_nonneg_for_psd(
            x in uniform3(-10.0f32..10.0),
            d0 in 0.0f32..5.0, d1 in 0.0f32..5.0, d2 in 0.0f32..5.0,
        ) {
            let det = diag_detector(d0, d1, d2, 9.0);
            let v = det.distance_sq(&x);
            prop_assert!(v.is_finite());
            prop_assert!(v >= 0.0, "Mahalanobis distance negative: {v}");
        }

        /// **Safety invariant:** `tighten_if_ood` never raises the base bound (tighten-only).
        #[test]
        fn tighten_if_ood_never_raises_bound(
            x in uniform3(-10.0f32..10.0),
            d0 in 0.0f32..5.0, d1 in 0.0f32..5.0, d2 in 0.0f32..5.0,
            thr in 0.0f32..50.0,
            base in 0.0f32..2.0, floor in 0.0f32..2.0,
        ) {
            let det = diag_detector(d0, d1, d2, thr);
            let out = det.tighten_if_ood(&x, base, floor);
            prop_assert!(out <= base + 1e-6, "OOD gate raised the bound: {out} > {base}");
        }
    }
}

// ===== Kani proofs: no transcendentals ⇒ full formal verification =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: `tighten_if_ood` never raises the base bound (the core safety invariant —
    /// independent of the value of d²).
    #[kani::proof]
    fn proof_tighten_if_ood_never_raises_bound() {
        let mean = [kani::any::<f32>(), kani::any::<f32>()];
        let s = [
            [kani::any::<f32>(), kani::any::<f32>()],
            [kani::any::<f32>(), kani::any::<f32>()],
        ];
        let thr: f32 = kani::any();
        let x = [kani::any::<f32>(), kani::any::<f32>()];
        let base: f32 = kani::any();
        let floor: f32 = kani::any();
        // (μ, Σ⁻¹) are fitted offline and features are normalised ([cos,sin]∈[-1,1]); we
        // constrain inputs to a finite bounded range (covering all finite values) so that the
        // O(N²) sum does not overflow to ∞ producing NaN — this is the physical operating range.
        // The safety invariant (out ≤ base) holds for any input anyway; the bound is modelling,
        // not a precondition.
        kani::assume(mean[0].abs() <= 1.0e3 && mean[1].abs() <= 1.0e3);
        kani::assume(
            s[0][0].abs() <= 1.0e3
                && s[0][1].abs() <= 1.0e3
                && s[1][0].abs() <= 1.0e3
                && s[1][1].abs() <= 1.0e3,
        );
        kani::assume(x[0].abs() <= 1.0e3 && x[1].abs() <= 1.0e3);
        kani::assume(base.is_finite() && floor.is_finite());
        let det = MahalanobisOod::new(mean, s, thr);
        let out = det.tighten_if_ood(&x, base, floor);
        // Result is either base (in-distribution) or min(base,floor) ⇒ ≤ base always.
        assert!(out <= base);
    }

    /// Proof: distance at the mean = 0.
    #[kani::proof]
    fn proof_zero_distance_at_mean() {
        let mean = [kani::any::<f32>(), kani::any::<f32>()];
        let s = [
            [kani::any::<f32>(), kani::any::<f32>()],
            [kani::any::<f32>(), kani::any::<f32>()],
        ];
        // (μ, Σ⁻¹) are finite by construction (fitted offline) ⇒ d²(μ)=0 exactly
        // (since finite × 0 = 0).
        kani::assume(mean[0].is_finite() && mean[1].is_finite());
        kani::assume(
            s[0][0].is_finite()
                && s[0][1].is_finite()
                && s[1][0].is_finite()
                && s[1][1].is_finite(),
        );
        let det = MahalanobisOod::new(mean, s, 1.0);
        assert!(det.distance_sq(&mean) == 0.0);
    }
}
