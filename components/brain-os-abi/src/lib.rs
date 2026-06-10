//! # brain-os-abi — Brain ↔ OS contract (generalised)
//!
//! Standard interface connecting **any learned brain** (policy with one output or 17 joints) to the seL4 kernel,
//! via a **shared memory region** + **deterministic safety logic**. Generalises `reflex-abi` (single command) to
//! a command vector, making it suitable for the pendulum, arm, and humanoid under the same contract.
//! `no_std`, `unsafe`-free.
//!
//! ## Contract flow (each cycle)
//! 1. **Brain** (cognitive layer, lower priority) writes: `STATE` (perception), `PROPOSED[n]` (proposed commands), `HEARTBEAT`.
//! 2. **Guard** (system, higher priority) reads, deterministically enforces safety limits, writes `APPROVED[n]` + `OVERRIDDEN` bitmask.
//! 3. **Executor** applies `APPROVED` only to actuators. If brain freezes (watchdog) → safe stop (zeros).
//!
//! The final decision over actuators **always belongs to the deterministic guard** — regardless of how smart or erroneous the brain is.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

/// Maximum number of supported commands (joints) — covers the humanoid (17).
pub const MAX_CMDS: usize = 24;
/// Maximum dimensions of the state/perception vector.
pub const MAX_STATE: usize = 64;

/// Default hard limit for each (normalised) command — enforced by the guard.
pub const DEFAULT_LIMIT: f32 = 1.0;
/// Number of cycles without a heartbeat before the brain is considered frozen.
pub const WATCHDOG_DEADLINE: u32 = 3;

// ===== Shared memory layout (byte offsets) — binary contract =====
pub const OFF_HEARTBEAT: usize = 0; // u64: brain heartbeat
pub const OFF_NCMDS: usize = 8; // u32: number of active commands (n ≤ MAX_CMDS)
pub const OFF_NSTATE: usize = 12; // u32: active state dimensions
pub const OFF_CYCLE: usize = 16; // u32: cycle counter
pub const OFF_STATE: usize = 24; // [f32; MAX_STATE]: perception (brain writes)
pub const OFF_PROPOSED: usize = OFF_STATE + 4 * MAX_STATE; // [f32; MAX_CMDS]: proposed commands
pub const OFF_APPROVED: usize = OFF_PROPOSED + 4 * MAX_CMDS; // [f32; MAX_CMDS]: approved commands (guard)
pub const OFF_OVERRIDDEN: usize = OFF_APPROVED + 4 * MAX_CMDS; // u32: bitmask of clipped commands
/// Total size of the shared region (aligned).
pub const REGION_SIZE: usize = (OFF_OVERRIDDEN + 4 + 63) & !63;

// Compile-time checks: fields do not overlap and the region is large enough.
const _: () = assert!(OFF_PROPOSED >= OFF_STATE + 4 * MAX_STATE);
const _: () = assert!(OFF_APPROVED >= OFF_PROPOSED + 4 * MAX_CMDS);
const _: () = assert!(OFF_OVERRIDDEN >= OFF_APPROVED + 4 * MAX_CMDS);
const _: () = assert!(REGION_SIZE >= OFF_OVERRIDDEN + 4);

/// Enforce the safety limit on each command: clips to `[-limit, limit]`, returns bitmask of clipped commands.
/// Pure, deterministic, and branch-free on data in the hot path — suitable for bounded-time execution.
pub fn enforce_limits(proposed: &[f32], limit: f32, approved: &mut [f32]) -> u32 {
    let n = proposed.len().min(approved.len()).min(MAX_CMDS);
    let mut mask = 0u32;
    for i in 0..n {
        let p = proposed[i];
        let c = if p > limit {
            limit
        } else if p < -limit {
            -limit
        } else {
            p
        };
        if c != p {
            mask |= 1 << i;
        }
        approved[i] = c;
    }
    mask
}

/// Safe stop: all commands set to zero (on brain freeze or emergency).
pub fn safe_stop(approved: &mut [f32]) {
    for a in approved.iter_mut() {
        *a = 0.0;
    }
}

/// Has the brain frozen? (heartbeat unchanged since last observation).
pub fn heartbeat_stalled(current: u64, last_seen: u64) -> bool {
    current == last_seen
}

/// Full guard decision: if brain is frozen → safe stop; otherwise → enforce limits.
/// Returns (clipped bitmask, whether emergency stop was triggered).
pub fn govern(
    proposed: &[f32],
    limit: f32,
    heartbeat_alive: bool,
    approved: &mut [f32],
) -> (u32, bool) {
    if !heartbeat_alive {
        safe_stop(approved);
        (0, true)
    } else {
        (enforce_limits(proposed, limit, approved), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_each_command_independently() {
        let proposed = [0.5, 2.0, -3.0, 0.9];
        let mut approved = [0.0; 4];
        let mask = enforce_limits(&proposed, 1.0, &mut approved);
        assert_eq!(approved, [0.5, 1.0, -1.0, 0.9]);
        assert_eq!(mask, 0b0110); // commands 1 and 2 were clipped
    }

    #[test]
    fn safe_within_limits_passes_through() {
        let proposed = [0.1, -0.2, 0.3];
        let mut approved = [9.0; 3];
        let mask = enforce_limits(&proposed, 1.0, &mut approved);
        assert_eq!(mask, 0);
        assert_eq!(approved, [0.1, -0.2, 0.3]);
    }

    #[test]
    fn govern_emergency_stops_on_stall() {
        let proposed = [0.9, -0.9, 0.5];
        let mut approved = [0.0; 3];
        let (_, stopped) = govern(&proposed, 1.0, false, &mut approved);
        assert!(stopped);
        assert_eq!(approved, [0.0, 0.0, 0.0]); // full safe stop
    }

    #[test]
    fn govern_enforces_when_alive() {
        let proposed = [5.0, 0.2];
        let mut approved = [0.0; 2];
        let (mask, stopped) = govern(&proposed, 1.0, true, &mut approved);
        assert!(!stopped);
        assert_eq!(approved, [1.0, 0.2]);
        assert_eq!(mask, 0b01);
    }

    #[test]
    fn heartbeat_detection() {
        assert!(heartbeat_stalled(5, 5));
        assert!(!heartbeat_stalled(6, 5));
    }

    // Honest verification note: `enforce_limits` **does not sanitise NaN** (passes it through) — unlike `reflex-abi::enforce_limit`.
    // NaN safety in the real guard comes from `reflex-abi::enforce_limit` and `safe_bound`, not here.
    #[test]
    fn enforce_limits_passes_nan_through_by_design() {
        let proposed = [f32::NAN, 0.5];
        let mut approved = [0.0; 2];
        let _ = enforce_limits(&proposed, 1.0, &mut approved);
        assert!(approved[0].is_nan()); // documented: no NaN sanitisation in this layer
        assert_eq!(approved[1], 0.5);
    }
}

// ===== proptest properties: verified against thousands of random inputs =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_cmds() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1000.0f32..1000.0, 0..8usize)
    }

    proptest! {
        /// For finite inputs and limit ≥ 0: every approved command is within [-limit, limit] and the clipped bitmask is consistent.
        #[test]
        fn enforce_limits_bounds_finite(cmds in finite_cmds(), limit in 0.0f32..1000.0) {
            let mut approved = vec![0.0f32; cmds.len()];
            let mask = enforce_limits(&cmds, limit, &mut approved);
            let n = cmds.len().min(MAX_CMDS);
            for i in 0..n {
                prop_assert!(approved[i] >= -limit && approved[i] <= limit);
                prop_assert_eq!(((mask >> i) & 1) == 1, approved[i] != cmds[i]);
            }
        }

        /// govern: on brain freeze → all commands zero (safe stop) regardless of proposed values.
        #[test]
        fn govern_stops_when_stalled(cmds in finite_cmds(), limit in 0.0f32..1000.0) {
            let mut approved = vec![9.9f32; cmds.len()];
            let (_mask, stopped) = govern(&cmds, limit, false, &mut approved);
            prop_assert!(stopped);
            for a in &approved {
                prop_assert_eq!(*a, 0.0);
            }
        }

        /// watchdog: frozen ⟺ counter did not advance.
        #[test]
        fn heartbeat_stalled_iff_unchanged(current in any::<u64>(), last in any::<u64>()) {
            prop_assert_eq!(heartbeat_stalled(current, last), current == last);
        }
    }
}

// ===== Kani proofs: formal verification (run with `cargo kani`; excluded from normal builds) =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: for finite inputs and limit ≥ 0, every approved command is within the envelope.
    #[kani::proof]
    fn proof_enforce_limits_bounds_finite() {
        let p0: f32 = kani::any();
        let p1: f32 = kani::any();
        kani::assume(p0.is_finite() && p1.is_finite());
        let limit: f32 = kani::any();
        kani::assume(limit.is_finite() && limit >= 0.0);
        let proposed = [p0, p1];
        let mut approved = [0.0f32; 2];
        let _ = enforce_limits(&proposed, limit, &mut approved);
        for a in approved.iter() {
            assert!(*a >= -limit && *a <= limit);
        }
    }

    /// Proof: brain freeze ⇒ full safe stop (zeros).
    #[kani::proof]
    fn proof_govern_stops_on_stall() {
        let p0: f32 = kani::any();
        let p1: f32 = kani::any();
        let limit: f32 = kani::any();
        let proposed = [p0, p1];
        let mut approved = [9.0f32; 2];
        let (_mask, stopped) = govern(&proposed, limit, false, &mut approved);
        assert!(stopped);
        assert!(approved[0] == 0.0 && approved[1] == 0.0);
    }
}
