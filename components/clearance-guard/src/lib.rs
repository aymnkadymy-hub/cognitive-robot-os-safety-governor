//! # clearance-guard — Model-agnostic clearance barrier (enforced by the system, not learned by the policy)
//!
//! **Concept (Bridge A):** instead of the policy *learning* to maintain a safety margin (via reward),
//! the guard **enforces it as a verified property**: it caps the forward speed so the robot can
//! **always stop before violating the minimum clearance** `d_min` — regardless of the brain
//! (even if it commands maximum thrust toward an obstacle). This is the OS value:
//! **safety guaranteed for all**, not "a safe policy".
//!
//! **Mathematics (braking barrier / reachability):** at distance `d` to the nearest forward hazard,
//! the maximum safe speed is `v_safe(d) = sqrt( 2 · a_max · max(0, d − d_min) )`. If the robot
//! obeys `v ≤ v_safe` every cycle, its stopping distance `v²/(2·a_max) ≤ d − d_min` ⇒ **it never
//! violates `d_min`**. `no_std`, no unsafe, seL4-ready.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use libm::sqrtf;

/// Clearance guard. Parameters are physical (measurable on hardware).
#[derive(Clone, Copy)]
pub struct ClearanceGuard {
    /// Minimum clearance that must be maintained (metres, from robot centre to nearest hazard).
    pub d_min: f32,
    /// Maximum braking deceleration (m/s²) — measured braking capability.
    pub a_max: f32,
    /// Maximum forward speed (m/s).
    pub v_max: f32,
    /// Response lag (s): distance travelled before braking begins (control cycle) — makes the
    /// guarantee robust in discrete time.
    pub tau: f32,
}

impl ClearanceGuard {
    pub const fn new(d_min: f32, a_max: f32, v_max: f32, tau: f32) -> Self {
        Self {
            d_min,
            a_max,
            v_max,
            tau,
        }
    }

    /// Maximum safe speed at forward-hazard distance `d_front`: brakes to a stop before `d_min`
    /// **even with response lag τ**.
    /// Solves `v·τ + v²/(2·a) ≤ slack` ⇒ `v = a·(√(τ² + 2·slack/a) − τ)`.
    pub fn safe_speed(&self, d_front: f32) -> f32 {
        let slack = d_front - self.d_min;
        if slack <= 0.0 {
            0.0 // inside margin or closer → full stop
        } else {
            let v = self.a_max * (sqrtf(self.tau * self.tau + 2.0 * slack / self.a_max) - self.tau);
            v.min(self.v_max)
        }
    }

    /// **Govern the proposed throttle** (normalised [-1,1]): clamps **forward** throttle so that
    /// `v ≤ v_safe`; reverse throttle (backing away) is always permitted. Returns the safe throttle.
    pub fn govern(&self, d_front: f32, proposed_throttle: f32) -> f32 {
        let cap = self.safe_speed(d_front) / self.v_max; // forward throttle ceiling [0,1]
        if proposed_throttle <= 0.0 {
            proposed_throttle // reverse: unconstrained (safe exit)
        } else {
            proposed_throttle.min(cap)
        }
    }

    /// Is speed `v` safe at `d_front`? (For verification / final guard check.)
    pub fn is_safe(&self, d_front: f32, v: f32) -> bool {
        v <= self.safe_speed(d_front) + 1e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> ClearanceGuard {
        ClearanceGuard::new(0.6, 4.0, 3.0, 0.1) // d_min=0.6m, braking 4m/s², vmax 3m/s, τ=0.1s
    }

    #[test]
    fn zero_speed_at_or_inside_margin() {
        assert_eq!(g().safe_speed(0.6), 0.0);
        assert_eq!(g().safe_speed(0.3), 0.0);
    }

    #[test]
    fn monotone_increasing_with_distance() {
        let gd = g();
        let (mut prev, mut d) = (-1.0, 0.6);
        while d < 5.0 {
            let s = gd.safe_speed(d);
            assert!(s >= prev - 1e-6, "must not decrease as distance grows");
            prev = s;
            d += 0.1;
        }
    }

    // Core property: at v=v_safe(d), braking distance does not exceed (d − d_min) ⇒ no violation.
    #[test]
    fn braking_guarantee_never_violates() {
        let gd = g();
        for i in 1..60 {
            let d = 0.6 + i as f32 * 0.1;
            let v = gd.safe_speed(d);
            let stop_dist = v * gd.tau + v * v / (2.0 * gd.a_max); // reaction travel + braking
            assert!(
                stop_dist <= (d - gd.d_min) + 1e-4,
                "can always stop before d_min (with reaction)"
            );
        }
    }

    #[test]
    fn govern_caps_forward_allows_reverse() {
        let gd = g();
        // close obstacle (0.8m): low safe speed → forward throttle clamped
        let capped = gd.govern(0.8, 1.0);
        assert!((0.0..1.0).contains(&capped));
        // reverse is unconstrained (safe exit)
        assert_eq!(gd.govern(0.65, -1.0), -1.0);
        // clear path (4m): full throttle allowed
        assert!((gd.govern(4.0, 1.0) - 1.0).abs() < 1e-6);
    }

    // 1D simulation: "bad" brain commanding maximum thrust toward obstacle → guard prevents violation.
    #[test]
    fn bad_brain_kept_safe_in_sim() {
        let gd = g();
        let (mut x, mut v) = (0.0f32, 0.0f32);
        let obs = 5.0;
        let (dt, v_max) = (0.05, gd.v_max);
        let mut min_clear = f32::INFINITY;
        for _ in 0..400 {
            let d_front = obs - x;
            let throttle = gd.govern(d_front, 1.0); // brain: always max thrust
            let v_target = throttle * v_max;
            // dynamics: acceleration/deceleration bounded by a_max
            let dv = (v_target - v).clamp(-gd.a_max * dt, gd.a_max * dt);
            v += dv;
            x += v * dt;
            min_clear = min_clear.min(obs - x);
        }
        assert!(
            min_clear >= gd.d_min - 0.02,
            "guard kept clearance >= d_min (got {min_clear})"
        );
    }

    // Golden bit fingerprint for determinism (catches replacement of libm::sqrtf with another
    // provider across architectures).
    #[test]
    fn golden_bits_safe_speed() {
        let v = ClearanceGuard::new(0.6, 4.0, 3.0, 0.1).safe_speed(1.5);
        assert!(v.is_finite());
        assert_eq!(v.to_bits(), 0x4014_0713u32); // 2.31293178f32
    }

    // Determinism audit: forbids non-deterministic std math and FMA in production code
    // (libm::sqrtf is allowed).
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

// ===== proptest properties: barrier bounds hold for any valid physical parameters =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // valid parameters: a_max>0, v_max≥0, tau≥0, d_min≥0
    fn params() -> impl Strategy<Value = (f32, f32, f32, f32)> {
        (0.05f32..2.0, 0.5f32..10.0, 0.3f32..5.0, 0.0f32..0.4)
    }

    proptest! {
        /// safe_speed is finite, ≥ 0, and ≤ v_max for any distance.
        #[test]
        fn safe_speed_nonneg_and_capped((d_min, a_max, v_max, tau) in params(), d in -5.0f32..10.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            let s = g.safe_speed(d);
            prop_assert!(s.is_finite());
            prop_assert!(s >= 0.0, "safe_speed negative: {s}");
            prop_assert!(s <= v_max + 1e-6, "safe_speed {s} exceeded v_max {v_max}");
        }

        /// Within the margin (d ≤ d_min): full stop required.
        #[test]
        fn safe_speed_zero_within_margin((d_min, a_max, v_max, tau) in params(), frac in 0.0f32..1.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            prop_assert_eq!(g.safe_speed(d_min * frac), 0.0);
        }

        /// **Core guarantee:** at v=safe_speed(d), stopping distance (with response lag) ≤ slack ⇒
        /// no violation.
        #[test]
        fn braking_guarantee_never_violates((d_min, a_max, v_max, tau) in params(), slack in 0.01f32..4.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            let v = g.safe_speed(d_min + slack);
            let stop = v * tau + v * v / (2.0 * a_max);
            prop_assert!(stop <= slack + 1e-3 + 1e-3 * slack, "stop {stop} > slack {slack}");
        }

        /// Monotonically non-decreasing with distance.
        #[test]
        fn safe_speed_monotone((d_min, a_max, v_max, tau) in params(), d1 in 0.0f32..6.0, extra in 0.0f32..4.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            prop_assert!(g.safe_speed(d1) <= g.safe_speed(d1 + extra) + 1e-6);
        }

        /// govern: reverse throttle (≤ 0) is unconstrained (always a safe exit).
        #[test]
        fn govern_reverse_unrestricted((d_min, a_max, v_max, tau) in params(), d in -2.0f32..8.0, t in -1.0f32..=0.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            prop_assert_eq!(g.govern(d, t), t);
        }

        /// govern: forward throttle (> 0) never exceeds the proposed value (clamped).
        #[test]
        fn govern_forward_capped((d_min, a_max, v_max, tau) in params(), d in -2.0f32..8.0, t in 0.0f32..1.0) {
            let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
            prop_assert!(g.govern(d, t) <= t + 1e-6);
        }
    }
}

// ===== Kani proofs: the transcendental-free path (d ≤ d_min) is fully formally verifiable =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: within the margin (d ≤ d_min) → safe_speed = 0 (path does not invoke sqrtf,
    /// so no square-root axioms are needed).
    #[kani::proof]
    fn proof_safe_speed_zero_within_margin() {
        let d_min: f32 = kani::any();
        let a_max: f32 = kani::any();
        let v_max: f32 = kani::any();
        let tau: f32 = kani::any();
        kani::assume(a_max.is_finite() && a_max > 0.0);
        kani::assume(d_min.is_finite() && v_max.is_finite() && tau.is_finite());
        let d: f32 = kani::any();
        kani::assume(d.is_finite() && d <= d_min);
        let g = ClearanceGuard::new(d_min, a_max, v_max, tau);
        assert!(g.safe_speed(d) == 0.0);
    }
}
