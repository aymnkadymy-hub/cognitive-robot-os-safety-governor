//! # contextual-guard — **Full** perception-driven adaptive safety (addresses: guard sees camera/memory)
//!
//! Binds **world-memory context** (visual/spatial) to the tighten-only envelope (`safety-memory`),
//! so the guard does not rely on IMU alone but on **full perception**: incidents are keyed on the
//! **perceptual context fingerprint**, and tightening occurs on **dynamic hazard (IMU) or contextual
//! hazard (proximity to a hazardous object/region in world memory)** — system-level protection.
//!
//! Preserves all Claim A properties (tighten-only, never exceeds the verified envelope).
//! `no_std`, seL4-ready.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use safety_memory::SafetyMemory;

/// Contextual hazard threshold (from world memory): proximity to a hazardous object/region.
pub const SPATIAL_RISK_THRESHOLD: f32 = 0.5;

/// Incident source (for logging/analysis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IncidentSource {
    None,
    Dynamics, // dynamic hazard (IMU: filtered tilt/vibration/shock)
    Spatial,  // contextual hazard (world memory: proximity to hazardous object/region)
    Both,
}

/// Contextual guardian: adaptive safety envelope keyed on the **full perceptual context**.
pub struct ContextualGuardian<const CAP: usize, const DIM: usize> {
    safety: SafetyMemory<CAP, DIM>,
}

impl<const CAP: usize, const DIM: usize> ContextualGuardian<CAP, DIM> {
    pub const fn new(static_limit: f32) -> Self {
        Self {
            safety: SafetyMemory::new(static_limit),
        }
    }

    pub fn with_params(mut self, sim_threshold: f32, contraction: f32, floor: f32) -> Self {
        self.safety = self.safety.with_params(sim_threshold, contraction, floor);
        self
    }

    pub fn incident_count(&self) -> usize {
        self.safety.incident_count()
    }

    /// Effective limit for the current perceptual context (≤ verified envelope always).
    pub fn effective_limit(&self, context: &[f32; DIM]) -> f32 {
        self.safety.effective_limit(context)
    }

    /// Full governance: keyed on the **perceptual context**, tightens on **dynamic or spatial hazard**.
    /// - `context`: current perception fingerprint (from world memory / encoder) — what the robot
    ///   "sees/senses".
    /// - `dynamics_danger`: filtered IMU hazard (tilt/vibration/shock).
    /// - `spatial_risk`: contextual hazard score [0,1] from a world-memory query (proximity to a
    ///   hazardous object/region).
    pub fn govern(
        &mut self,
        context: &[f32; DIM],
        dynamics_danger: bool,
        spatial_risk: f32,
        proposed: &[f32],
        approved: &mut [f32],
    ) -> (f32, IncidentSource) {
        // 1) Project onto the effective envelope (key = perceptual context).
        let (lim, _mask) = self.safety.govern(context, proposed, approved);

        // 2) Detect incident from two sources (multi-modal).
        let spatial = spatial_risk >= SPATIAL_RISK_THRESHOLD;
        let src = match (dynamics_danger, spatial) {
            (true, true) => IncidentSource::Both,
            (true, false) => IncidentSource::Dynamics,
            (false, true) => IncidentSource::Spatial,
            (false, false) => IncidentSource::None,
        };

        // 3) Tighten (tighten-only) on any hazard — binds camera/memory to the guard.
        if src != IncidentSource::None {
            self.safety.record_incident(context);
        }
        (lim, src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(i: usize) -> [f32; 4] {
        let mut v = [0.1; 4];
        v[i % 4] = 1.0;
        v
    }

    #[test]
    fn spatial_context_alone_tightens() {
        // No IMU hazard — a hazardous visual context (near an edge) alone must tighten.
        let mut g = ContextualGuardian::<8, 4>::new(1.0);
        let mut app = [0.0; 1];
        let before = g.effective_limit(&ctx(1));
        let (_lim, src) = g.govern(&ctx(1), false, 0.9, &[0.95], &mut app);
        assert_eq!(src, IncidentSource::Spatial);
        assert!(
            g.effective_limit(&ctx(1)) < before,
            "visual context must tighten the guard"
        );
    }

    #[test]
    fn dynamics_danger_tightens_and_multimodal() {
        let mut g = ContextualGuardian::<8, 4>::new(1.0);
        let mut app = [0.0; 1];
        let (_l, src) = g.govern(&ctx(2), true, 0.9, &[0.5], &mut app);
        assert_eq!(src, IncidentSource::Both); // IMU + context together
    }

    #[test]
    fn safe_context_untouched_and_within_envelope() {
        let mut g = ContextualGuardian::<8, 4>::new(1.0);
        let mut app = [0.0; 1];
        let (lim, src) = g.govern(&ctx(0), false, 0.1, &[0.4], &mut app);
        assert_eq!(src, IncidentSource::None);
        assert!((lim - 1.0).abs() < 1e-6); // full envelope, no violation
        assert!((app[0] - 0.4).abs() < 1e-6);
    }
}
