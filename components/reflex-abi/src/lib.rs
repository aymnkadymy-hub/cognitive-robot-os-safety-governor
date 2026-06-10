//! # reflex-abi
//!
//! Shared contract between the two "reflex arc" layers:
//! - **Cognitive PD** (brain, lower priority): proposes a motion command.
//! - **Guard PD** (spinal cord, highest priority): enforces hard limits and pre-empts the intelligence.
//!
//! Contains the **shared memory layout** (offsets) and **pure safety logic** (`enforce_limit`)
//! fully decoupled from seL4 plumbing — enabling testing on Host (SIL principle).

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

/// Size of the shared memory region (one page).
pub const REGION_SIZE: usize = 0x1000;

/// Field offsets within the shared region (little-endian).
pub const OFF_STATE: usize = 0; // [f32; 4]: robot state (x, x_dot, theta, theta_dot)
pub const OFF_HEARTBEAT: usize = 16; // u64: heartbeat counter (incremented each healthy cycle)
pub const OFF_PROPOSED: usize = 24; // f32: motion command proposed by the intelligence
pub const OFF_APPROVED: usize = 28; // f32: command after guard review
pub const OFF_OVERRIDDEN: usize = 32; // u32: 1 if the guard intervened
pub const OFF_CYCLE: usize = 36; // u32: control cycle number

/// Maximum number of cycles in the unified loop display.
pub const MAX_CYCLES: u32 = 300;

/// Absolute maximum for the actuator command (abstract unit). Exceeding it = physical hazard.
pub const HARD_LIMIT: f32 = 1.0;

/// Liveness-based watchdog: if the heartbeat counter did not increment between two checks →
/// the brain has frozen. Pure deterministic logic, tested on Host.
/// (A fully independent periodic timer = future improvement via timer-driver PD.)
pub fn heartbeat_stalled(current: u64, last_seen: u64) -> bool {
    current == last_seen
}

/// Pure safety logic: clips the proposed command to the hard limits.
/// Returns `(approved, overridden)`.
///
/// This is the heart of the "reflex arc": deterministic, simple, formally verifiable — unlike intelligence.
pub fn enforce_limit(proposed: f32) -> (f32, bool) {
    if proposed.is_nan() {
        // Invalid value from intelligence → stop (0) and treat as intervention.
        (0.0, true)
    } else if proposed > HARD_LIMIT {
        (HARD_LIMIT, true)
    } else if proposed < -HARD_LIMIT {
        (-HARD_LIMIT, true)
    } else {
        (proposed, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_limits_passes_through() {
        assert_eq!(enforce_limit(0.5), (0.5, false));
        assert_eq!(enforce_limit(-0.9), (-0.9, false));
        assert_eq!(enforce_limit(0.0), (0.0, false));
    }

    #[test]
    fn over_limit_is_clamped_and_flagged() {
        assert_eq!(enforce_limit(2.5), (1.0, true));
        assert_eq!(enforce_limit(-3.0), (-1.0, true));
    }

    #[test]
    fn nan_is_stopped() {
        let (v, o) = enforce_limit(f32::NAN);
        assert_eq!(v, 0.0);
        assert!(o);
    }

    #[test]
    fn watchdog_detects_stalled_heartbeat() {
        assert!(!heartbeat_stalled(5, 4)); // counter advanced → brain alive
        assert!(heartbeat_stalled(4, 4)); // unchanged between two checks → frozen
    }
}

// ===== proptest properties: verified against thousands of random inputs (bug hunter) =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Strongest invariant: guard output is **always** within the hard envelope and never NaN — for any input (including NaN/∞).
        #[test]
        fn enforce_limit_output_always_within_envelope(p in proptest::num::f32::ANY) {
            let (v, _overridden) = enforce_limit(p);
            prop_assert!(v.is_finite(), "output must never be NaN/inf, got {v}");
            prop_assert!((-HARD_LIMIT..=HARD_LIMIT).contains(&v), "output {v} escaped envelope");
        }

        /// Inside the envelope: finite commands pass through unchanged without intervention.
        #[test]
        fn enforce_limit_passes_through_inside(p in -HARD_LIMIT..=HARD_LIMIT) {
            let (v, overridden) = enforce_limit(p);
            prop_assert_eq!(v, p);
            prop_assert!(!overridden);
        }

        /// Outside the envelope (finite): clipped to limit and flagged; NaN → stop (0) and flagged.
        #[test]
        fn enforce_limit_clamps_and_flags_outside(p in proptest::num::f32::ANY) {
            let (v, overridden) = enforce_limit(p);
            if p.is_nan() {
                prop_assert_eq!(v, 0.0);
                prop_assert!(overridden);
            } else if p > HARD_LIMIT {
                prop_assert_eq!(v, HARD_LIMIT);
                prop_assert!(overridden);
            } else if p < -HARD_LIMIT {
                prop_assert_eq!(v, -HARD_LIMIT);
                prop_assert!(overridden);
            } else {
                prop_assert!(!overridden);
            }
        }

        /// watchdog: frozen ⟺ counter did not advance.
        #[test]
        fn heartbeat_stalled_iff_unchanged(current in any::<u64>(), last in any::<u64>()) {
            prop_assert_eq!(heartbeat_stalled(current, last), current == last);
        }
    }
}

// ===== Kani proofs: formal verification for all inputs (run with `cargo kani`; excluded from normal builds) =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: hard envelope clipping never produces a value outside [-HARD_LIMIT, HARD_LIMIT] — for any symbolic f32.
    #[kani::proof]
    fn proof_enforce_limit_within_envelope() {
        let p: f32 = kani::any();
        let (v, _overridden) = enforce_limit(p);
        assert!(!v.is_nan());
        assert!(v >= -HARD_LIMIT && v <= HARD_LIMIT);
    }

    /// Proof: watchdog is correct for all counter pairs.
    #[kani::proof]
    fn proof_heartbeat_stalled_correct() {
        let current: u64 = kani::any();
        let last: u64 = kani::any();
        assert_eq!(heartbeat_stalled(current, last), current == last);
    }
}
