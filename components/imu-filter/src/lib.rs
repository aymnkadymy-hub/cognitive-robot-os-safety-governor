//! # imu-filter — Mahony-formulation complementary filter for the pitch axis
//! (fusion + online gyro-bias calibration)
//!
//! Fuses the pitch angle from **acceleration** (long-term reference, but noisy under
//! dynamic acceleration/vibration) with the **gyro** rate (smooth short-term, but drifts)
//! — with **online gyro-bias estimation** (the integral term I in Mahony's formulation)
//! = continuous calibration.
//!
//! **Problem it solves:** on real hardware (`drivers/pi4-body`) pitch was computed from
//! acceleration alone (`atan2`), making it susceptible to vibration/linear acceleration
//! ⇒ spurious `DANGER_THETA` alerts. This filter cleans the signal **before** it reaches
//! the guard, **without any change to the brain/guard/loop** — touching only the hardware
//! path (the simulation uses `MujocoTwinBody` and is unaffected).
//!
//! **Fully deterministic:** only `+,-,*` arithmetic (no transcendentals, no libm)
//! ⇒ bit-identical across architectures, fully amenable to formal verification with Kani.
//! `no_std`, no heap, no unsafe.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

/// Single-axis complementary filter using Mahony's formulation
/// (proportional-integral correction on pitch angle).
///
/// State is `Copy` for easy placement behind a `Cell` in a driver that lacks `&mut self` in `sense`.
#[derive(Clone, Copy, Debug)]
pub struct MahonyPitchFilter {
    pitch: f32, // fused pitch estimate (radians)
    bias: f32,  // estimated gyro bias (rad/s) — continuous calibration
    kp: f32,    // proportional correction gain (pulls toward accel reference)
    ki: f32,    // integral correction gain (learns the bias)
    initialized: bool,
}

impl MahonyPitchFilter {
    /// Create with given gains. **For stability:** require `0 < kp·dt < 2`; keep `ki` small (slow bias).
    pub const fn new(kp: f32, ki: f32) -> Self {
        Self {
            pitch: 0.0,
            bias: 0.0,
            kp,
            ki,
            initialized: false,
        }
    }

    /// Safe default gains for a ~50–200Hz loop (kp=1.0, ki=0.05).
    pub const fn default_gains() -> Self {
        Self::new(1.0, 0.05)
    }

    /// Set the gyro bias directly (boot calibration: average gyro readings while robot is stationary).
    pub fn set_gyro_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Current fused pitch (radians).
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Current estimated gyro bias (rad/s).
    pub fn gyro_bias(&self) -> f32 {
        self.bias
    }

    /// Single fusion step.
    /// - `accel_pitch`: pitch angle from acceleration (radians).
    /// - `gyro_rate`: raw gyro rate (rad/s).
    /// - `dt`: time step (s).
    ///
    /// Returns `(fused pitch, bias-corrected gyro rate)`.
    pub fn update(&mut self, accel_pitch: f32, gyro_rate: f32, dt: f32) -> (f32, f32) {
        if !self.initialized {
            // First sample: trust the accelerometer (avoids a long startup transient).
            self.pitch = accel_pitch;
            self.initialized = true;
            return (self.pitch, gyro_rate - self.bias);
        }
        // Innovation: how much does the accel reference differ from our current estimate.
        let error = accel_pitch - self.pitch;
        // Integral term (Mahony): estimates gyro bias. Sign is **negative**: a positive bias
        // shifts pitch upward ⇒ error is negative ⇒ bias must increase ⇒ `-ki*(negative)` is positive.
        self.bias -= self.ki * error * dt;
        // Corrected rate (bias-removed) = the smooth component.
        let rate = gyro_rate - self.bias;
        // Prediction (integrate corrected gyro) + proportional correction toward accel (kills drift).
        self.pitch += (rate + self.kp * error) * dt;
        (self.pitch, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // First sample trusts accelerometer: pitch = accel angle immediately.
    #[test]
    fn first_sample_trusts_accel() {
        let mut f = MahonyPitchFilter::default_gains();
        let (p, _r) = f.update(0.42, 1.0, 0.01);
        assert_eq!(p, 0.42);
    }

    // Fusion: start wrong (0) then feed a constant angle 0.3 with gyro 0 → converges to true angle.
    #[test]
    fn converges_to_accel_reference() {
        let mut f = MahonyPitchFilter::new(1.0, 0.05);
        f.update(0.0, 0.0, 0.01); // initialize at 0
        let mut pitch = 0.0;
        for _ in 0..5000 {
            pitch = f.update(0.3, 0.0, 0.01).0;
        }
        assert!(
            (pitch - 0.3).abs() < 1e-2,
            "should converge to 0.3, got {pitch}"
        );
    }

    // Calibration: true pitch is 0 but gyro is biased by 0.1 — must learn bias and keep pitch ≈ 0.
    #[test]
    fn rejects_and_estimates_gyro_bias() {
        let mut f = MahonyPitchFilter::new(2.0, 0.5);
        let beta = 0.1f32;
        f.update(0.0, beta, 0.01); // initialize at 0
        let mut pitch = 0.0;
        for _ in 0..6000 {
            pitch = f.update(0.0, beta, 0.01).0;
        }
        assert!(pitch.abs() < 2e-2, "should reject gyro bias, pitch={pitch}");
        assert!(
            (f.gyro_bias() - beta).abs() < 2e-2,
            "should estimate bias≈{beta}, got {}",
            f.gyro_bias()
        );
    }

    // Explicit calibration (boot): set_gyro_bias removes a fixed bias from the corrected rate immediately.
    #[test]
    fn explicit_calibration_removes_bias() {
        let mut f = MahonyPitchFilter::default_gains();
        f.set_gyro_bias(0.05);
        let (_p, rate) = f.update(0.0, 0.05, 0.01); // first sample: rate = gyro - bias = 0
        assert!(
            rate.abs() < 1e-6,
            "calibrated rate should be ~0, got {rate}"
        );
    }

    // Golden bit-pattern for determinism (filter has no transcendentals ⇒ bit-identical across architectures).
    #[test]
    fn golden_bits_update() {
        let mut f = MahonyPitchFilter::new(1.0, 0.05);
        f.update(0.0, 0.1, 0.01); // initialize at 0
        let (pitch, rate) = f.update(0.2, 0.1, 0.01);
        assert!(pitch.is_finite() && rate.is_finite());
        assert_eq!(pitch.to_bits(), 0x3b44_ac6d); // 0.003001f32
        assert_eq!(rate.to_bits(), 0x3dcd_013b); // 0.1001f32
    }

    // Determinism audit: no std transcendentals or FMA in production code (this filter has no libm at all).
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

// ===== proptest properties: filter never blows up for any finite measurement sequence =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Finite and bounded outputs for any reasonable measurement sequence and stable gains (kp·dt < 2).
        #[test]
        fn output_finite_and_bounded(
            steps in prop::collection::vec((-1.6f32..1.6, -10.0f32..10.0), 1..200),
            kp in 0.1f32..3.0,
            ki in 0.0f32..1.0,
        ) {
            let mut f = MahonyPitchFilter::new(kp, ki);
            for (accel, gyro) in &steps {
                let (p, r) = f.update(*accel, *gyro, 0.01);
                prop_assert!(p.is_finite() && r.is_finite(), "non-finite output");
                prop_assert!(p.abs() <= 100.0, "pitch blew up: {p}");
            }
        }
    }
}

// ===== Kani proofs: no transcendentals ⇒ full formal proof of no NaN/∞ =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: two steps (init + fusion) with finite bounded inputs never produce NaN/∞.
    #[kani::proof]
    fn proof_update_stays_finite() {
        let kp: f32 = kani::any();
        let ki: f32 = kani::any();
        kani::assume(kp.is_finite() && (0.0..=10.0).contains(&kp));
        kani::assume(ki.is_finite() && (0.0..=10.0).contains(&ki));
        let a0: f32 = kani::any();
        let g0: f32 = kani::any();
        let a1: f32 = kani::any();
        let g1: f32 = kani::any();
        let dt: f32 = kani::any();
        kani::assume(a0.is_finite() && a0.abs() <= 10.0);
        kani::assume(g0.is_finite() && g0.abs() <= 100.0);
        kani::assume(a1.is_finite() && a1.abs() <= 10.0);
        kani::assume(g1.is_finite() && g1.abs() <= 100.0);
        kani::assume(dt.is_finite() && (0.0..=0.1).contains(&dt));
        let mut f = MahonyPitchFilter::new(kp, ki);
        let _ = f.update(a0, g0, dt); // init
        let (p, r) = f.update(a1, g1, dt); // actual step
        assert!(p.is_finite());
        assert!(r.is_finite());
    }
}
