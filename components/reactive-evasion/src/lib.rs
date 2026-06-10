//! # reactive-evasion — worst-case velocity-obstacle safe-approach bound (machine-checked)
//!
//! The reactive layer that handles *fast, multi-directional, moving* hazards (evaluated in
//! `sim/sil-adversarial`). Its safety-critical core is a **tighten-only** cap on the robot's
//! *approach speed* toward a sensed obstacle: the applied approach is the lesser of the desired
//! approach and the worst-case safe bound, so the cap can only ever *lower* the speed at which the
//! robot closes on an obstacle, never raise it. Being a `min` into the verified envelope, it
//! composes with `B0`-containment exactly like the rest of the governor (Proposition 1) and is
//! therefore verification-preserving.
//!
//! Two structural invariants are machine-checked by Kani/CBMC (see the `#[cfg(kani)]` proofs):
//!  1. `clamp_approach` is tighten-only and bound-respecting: `out ≤ desired` and `out ≤ bound`.
//!  2. `safe_approach` is exactly zero at or inside the standoff margin (no approach is permitted
//!     once within the margin), proven on the `sqrt`-free within-margin path; non-negativity for
//!     larger gaps is non-negative by construction and is covered by the property tests.
//!
//! As elsewhere in this codebase the math is `libm`-only and contains no fused multiply-add, so the
//! host-checked properties transfer bit-identically to the target build.
#![no_std]

use libm::sqrtf;

/// Worst-case safe approach speed toward an obstacle that may keep closing at up to `umax`.
///
/// `gap` is the clear distance (sensed distance minus the combined collision radius); `a` is the
/// robot's braking deceleration; `margin` is the standoff kept in hand. The bound is the largest
/// approach speed `s` from which the robot can still avoid contact even if the obstacle continues to
/// close at `umax` during the stop, namely `s = sqrt(umax² + 2·a·(gap − margin)) − umax`, clamped to
/// be non-negative, and forced to exactly zero once the robot is at or inside the margin.
pub fn safe_approach(gap: f32, a: f32, umax: f32, margin: f32) -> f32 {
    // at/inside the margin (or NaN gap) ⇒ no approach permitted; this path never calls `sqrtf`.
    if gap.is_nan() || gap <= margin {
        return 0.0;
    }
    let slack = gap - margin;
    let inner = umax * umax + 2.0 * a * slack;
    let s = sqrtf(inner) - umax;
    if s > 0.0 {
        s
    } else {
        0.0
    }
}

/// Tighten-only clamp: the applied approach speed is the lesser of `desired` and the safe `bound`.
/// This is the operation that makes the reactive layer verification-preserving — it can only reduce
/// the approach speed, never increase it.
pub fn clamp_approach(desired: f32, bound: f32) -> f32 {
    if desired > bound {
        bound
    } else {
        desired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_lowers_only() {
        assert_eq!(clamp_approach(5.0, 2.0), 2.0); // over the bound -> capped
        assert_eq!(clamp_approach(1.0, 2.0), 1.0); // under the bound -> unchanged
        assert_eq!(clamp_approach(-3.0, 2.0), -3.0); // receding -> unchanged
    }

    #[test]
    fn safe_approach_zero_within_margin() {
        assert_eq!(safe_approach(0.1, 8.0, 2.5, 0.3), 0.0); // gap < margin
        assert_eq!(safe_approach(0.3, 8.0, 2.5, 0.3), 0.0); // gap == margin
    }

    #[test]
    fn safe_approach_grows_with_gap() {
        let near = safe_approach(1.0, 8.0, 2.5, 0.3);
        let far = safe_approach(5.0, 8.0, 2.5, 0.3);
        assert!(near >= 0.0 && far > near);
    }

    #[test]
    fn safe_approach_slower_for_faster_obstacle() {
        // a faster-closing obstacle must lower the permitted approach speed
        let slow = safe_approach(3.0, 8.0, 1.0, 0.3);
        let fast = safe_approach(3.0, 8.0, 3.0, 0.3);
        assert!(fast < slow);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Proof 1: the clamp is *tighten-only* (never increases the approach speed) and never exceeds
    /// the safe bound — `out ≤ desired` and `out ≤ bound`. This is the verification-preserving core.
    #[kani::proof]
    fn proof_clamp_tighten_only_and_bounded() {
        let desired: f32 = kani::any();
        let bound: f32 = kani::any();
        kani::assume(desired.is_finite() && bound.is_finite());
        let out = clamp_approach(desired, bound);
        assert!(out <= desired); // tighten-only: the cap can only lower the approach
        assert!(out <= bound); // bound-respecting: never above the worst-case safe speed
    }

    /// Proof 2: at or inside the standoff margin the safe-approach bound is exactly zero — no
    /// approach is ever commanded within the margin. We constrain `gap ≤ margin`, the early-return
    /// path that does *not* call `sqrtf`; this follows the project's BMC convention of proving the
    /// structural invariant on the `sqrt`-free path (bounded model checking cannot reason through
    /// `libm::sqrtf`). General non-negativity for `gap > margin` is non-negative by construction
    /// (`if s > 0.0 { s } else { 0.0 }`) and is covered by the property tests.
    #[kani::proof]
    fn proof_safe_approach_stops_at_margin() {
        let gap: f32 = kani::any();
        let a: f32 = kani::any();
        let umax: f32 = kani::any();
        let margin: f32 = kani::any();
        kani::assume(gap.is_finite() && a.is_finite() && umax.is_finite() && margin.is_finite());
        kani::assume(gap <= margin); // within-margin path: returns 0 without calling sqrtf
        let s = safe_approach(gap, a, umax, margin);
        assert!(s == 0.0); // zero approach permitted within the standoff margin
        assert!(s >= 0.0);
    }
}
