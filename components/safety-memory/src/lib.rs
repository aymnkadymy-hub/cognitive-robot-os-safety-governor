//! # safety-memory — Adaptive safety envelope that **tightens only** (Claim A, core of the prospective patent)
//!
//! Fuses **memory and safety**: maintains an **incident memory** (state fingerprint + tightened limit),
//! and upon encountering a state that **resembles** a past incident, tightens the motion limits
//! **locally and automatically** — without modifying the kernel.
//!
//! ## Provable properties (the four novelty conditions):
//! 1. **The verified envelope is never exceeded:** `effective_limit(s) ≤ static_limit` for every state.
//! 2. **Tighten only:** recording an incident never increases any effective limit (non-increasing function).
//! 3. **No relaxation:** adding incidents never loosens limits (no deletion loosens).
//! 4. **Verification-preserving:** adaptation lives in **data** read by the guard; the kernel is unchanged
//!    → its proof stays valid (it only requires `|approved| ≤ static_limit`, guaranteed because
//!    `effective_limit ≤ static_limit`).
//!
//! "What happens after an incident?" We generate an **incident fingerprint** (state embedding) and bind
//! it to a region of state space; on similarity > threshold → tighten the limit there. `no_std`,
//! no heap, no unsafe — runs on seL4.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use brain_os_abi::enforce_limits;
use libm::sqrtf;

/// A recorded incident: the state fingerprint + the tightened limit associated with its region.
#[derive(Clone, Copy, Debug)]
struct Incident<const DIM: usize> {
    fingerprint: [f32; DIM],
    limit: f32,
    valid: bool,
    safe: u16, // accumulated safety evidence (for optional evidence-based forgetting)
}

impl<const DIM: usize> Incident<DIM> {
    const fn empty() -> Self {
        Self {
            fingerprint: [0.0; DIM],
            limit: 0.0,
            valid: false,
            safe: 0,
        }
    }
}

/// Adaptive tighten-only safety envelope driven by an incident memory.
pub struct SafetyMemory<const CAP: usize, const DIM: usize> {
    incidents: [Incident<DIM>; CAP],
    count: usize,
    static_limit: f32,  // the verified envelope — never exceeded
    sim_threshold: f32, // similarity sufficient to associate a state with an incident
    contraction: f32,   // tightening factor (< 1)
    floor: f32,         // minimum limit (never goes below this)
    evidence_k: u16, // evidence threshold for optional forgetting (safe passages before relaxing)
}

impl<const CAP: usize, const DIM: usize> SafetyMemory<CAP, DIM> {
    /// Create with a verified envelope `static_limit`. Adaptation never raises it.
    pub const fn new(static_limit: f32) -> Self {
        Self {
            incidents: [Incident::empty(); CAP],
            count: 0,
            static_limit,
            sim_threshold: 0.85,
            contraction: 0.6,
            floor: 0.05,
            evidence_k: 40,
        }
    }

    pub fn with_params(mut self, sim_threshold: f32, contraction: f32, floor: f32) -> Self {
        self.sim_threshold = sim_threshold;
        self.contraction = contraction.clamp(0.0, 1.0);
        self.floor = floor;
        self
    }

    pub fn incident_count(&self) -> usize {
        self.count
    }

    /// Effective limit for a state: the minimum of the verified envelope and the limits of
    /// similar-region incidents.
    /// **Invariant:** result ≤ `static_limit` always.
    pub fn effective_limit(&self, state: &[f32; DIM]) -> f32 {
        let mut lim = self.static_limit;
        for inc in &self.incidents {
            if inc.valid && cosine(state, &inc.fingerprint) >= self.sim_threshold {
                lim = lim.min(inc.limit); // tighten only
            }
        }
        lim
    }

    /// Index of the nearest incident (highest similarity) — used for merging when capacity is full
    /// (preserves monotonicity).
    fn nearest(&self, fp: &[f32; DIM]) -> usize {
        let mut best = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, inc) in self.incidents.iter().enumerate() {
            let s = cosine(fp, &inc.fingerprint);
            if s > best_sim {
                best_sim = s;
                best = i;
            }
        }
        best
    }

    /// Record an incident at this state → **tighten** the limit in its region (no deletion, no
    /// relaxation). Incident = guard clamp (near-miss) or external hazard signal — both call this.
    pub fn record_incident(&mut self, state: &[f32; DIM]) {
        let tightened = (self.effective_limit(state) * self.contraction).max(self.floor);
        if self.count < CAP {
            self.incidents[self.count] = Incident {
                fingerprint: *state,
                limit: tightened,
                valid: true,
                safe: 0,
            };
            self.count += 1;
        } else {
            // Full: merge into the nearest by taking the tighter limit (no deletion loosens) —
            // remains tighten-only.
            let best = self.nearest(state);
            self.incidents[best].limit = self.incidents[best].limit.min(tightened);
            self.incidents[best].safe = 0; // new incident resets safety evidence (fast re-tightening)
        }
    }

    /// Set the evidence threshold for optional forgetting (number of safe passages before one
    /// relaxation step).
    pub fn with_forgetting(mut self, evidence_k: u16) -> Self {
        self.evidence_k = evidence_k;
        self
    }

    /// **Evidence-based forgetting (optional — no effect if never called).** Records a safe
    /// passage through the current context; after `evidence_k` safe passages, relaxes the
    /// matching incident's limit **one step toward the verified envelope**
    /// (`limit/contraction`, clamped to `static_limit` ⇒ never exceeded ⇒ **verification-preserving**).
    /// On full relaxation the incident is forgotten (`valid=false`). Any new incident
    /// (`record_incident`) resets evidence and re-tightens.
    /// **Explicit trade-off:** monotonicity (Theorem 2) is replaced by a weaker property:
    /// no relaxation unless `evidence_k` **confirmed safe** passages have occurred (a hazard
    /// still present keeps the limit tight). The hard guarantee (containment ≤ L₀) **remains**.
    pub fn confirm_safe(&mut self, state: &[f32; DIM]) {
        let mut best: Option<usize> = None;
        let mut best_sim = self.sim_threshold;
        for (i, inc) in self.incidents.iter().enumerate() {
            if inc.valid {
                let s = cosine(state, &inc.fingerprint);
                if s >= best_sim {
                    best_sim = s;
                    best = Some(i);
                }
            }
        }
        if let Some(i) = best {
            let inc = &mut self.incidents[i];
            inc.safe = inc.safe.saturating_add(1);
            if inc.safe >= self.evidence_k {
                inc.safe = 0;
                let relaxed = if self.contraction > 0.0 {
                    (inc.limit / self.contraction).min(self.static_limit)
                } else {
                    self.static_limit
                };
                inc.limit = relaxed;
                if relaxed >= self.static_limit - 1e-6 {
                    inc.valid = false; // fully forgotten (no longer constraining)
                }
            }
        }
    }

    /// Adaptive governance: clamps commands to the effective limit (≤ verified envelope always).
    /// Returns (effective_limit, clamped_mask).
    pub fn govern(&self, state: &[f32; DIM], proposed: &[f32], approved: &mut [f32]) -> (f32, u32) {
        let lim = self.effective_limit(state);
        let mask = enforce_limits(proposed, lim, approved);
        (lim, mask)
    }

    /// **Load prior experience** (simulation → real robot): insert an incident with a known
    /// fingerprint and limit. Makes the robot "born cautious" in contexts it has not physically
    /// encountered. Limit is clamped to [floor, static].
    pub fn preload(&mut self, fingerprint: &[f32; DIM], limit: f32) {
        let lim = limit.clamp(self.floor, self.static_limit);
        if self.count < CAP {
            self.incidents[self.count] = Incident {
                fingerprint: *fingerprint,
                limit: lim,
                valid: true,
                safe: 0,
            };
            self.count += 1;
        } else {
            // Full: merge into the nearest by taking the minimum (preserves monotonicity).
            let best = self.nearest(fingerprint);
            self.incidents[best].limit = self.incidents[best].limit.min(lim);
        }
    }

    /// Number of bytes in an export (portable memory between simulation and hardware).
    pub const fn export_len() -> usize {
        4 + CAP * (DIM * 4 + 4)
    }

    /// **Export incident memory** as bytes (for transfer / persistent storage). Returns bytes written.
    pub fn export_bytes(&self, out: &mut [u8]) -> usize {
        let mut p = 0usize;
        out[p..p + 4].copy_from_slice(&(self.count as u32).to_le_bytes());
        p += 4;
        for inc in self.incidents.iter().take(self.count) {
            for v in inc.fingerprint.iter() {
                out[p..p + 4].copy_from_slice(&v.to_le_bytes());
                p += 4;
            }
            out[p..p + 4].copy_from_slice(&inc.limit.to_le_bytes());
            p += 4;
        }
        p
    }

    /// **Import incident memory** from bytes (device starts with simulation experience). Returns
    /// the number of incidents loaded.
    pub fn import_bytes(&mut self, data: &[u8]) -> usize {
        if data.len() < 4 {
            return 0;
        }
        let n = (u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize).min(CAP);
        let mut p = 4usize;
        let rec = DIM * 4 + 4;
        for k in 0..n {
            if p + rec > data.len() {
                break;
            }
            let mut fp = [0.0f32; DIM];
            for f in fp.iter_mut() {
                *f = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                p += 4;
            }
            let lim = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
            p += 4;
            self.incidents[k] = Incident {
                fingerprint: fp,
                limit: lim.clamp(self.floor, self.static_limit),
                valid: true,
                safe: 0,
            };
        }
        self.count = n;
        n
    }
}

#[cfg(not(kani))]
fn cosine<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..D {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = sqrtf(na) * sqrtf(nb);
    if d == 0.0 {
        0.0
    } else {
        dot / d
    }
}

/// Under `cargo kani` only: sound abstraction of similarity (any value). The containment invariant
/// `effective_limit ≤ static_limit` holds for any similarity value (because `effective_limit` is a
/// `min`-fold starting from `static_limit`), so it is fully verified without `sqrtf` axioms
/// (which CBMC lacks). No effect on normal builds or proptests.
#[cfg(kani)]
fn cosine<const D: usize>(_a: &[f32; D], _b: &[f32; D]) -> f32 {
    kani::any()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(i: usize) -> [f32; 4] {
        let mut v = [0.1; 4];
        v[i % 4] = 1.0;
        v
    }

    // Property 1: the verified envelope is never exceeded.
    #[test]
    fn never_exceeds_static_envelope() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0);
        for i in 0..20 {
            sm.record_incident(&st(i));
            for j in 0..4 {
                assert!(sm.effective_limit(&st(j)) <= 1.0 + 1e-6);
            }
        }
    }

    // Property 2 + 3: tighten only — recording an incident never increases any effective limit.
    #[test]
    fn monotone_tightening_only() {
        let mut sm = SafetyMemory::<16, 4>::new(1.0);
        let probes = [st(0), st(1), st(2), st(3)];
        let mut before = probes.map(|s| sm.effective_limit(&s));
        for i in 0..30 {
            sm.record_incident(&st(i));
            let after = probes.map(|s| sm.effective_limit(&s));
            for k in 0..4 {
                assert!(
                    after[k] <= before[k] + 1e-6,
                    "limit increased! relaxed safety"
                );
            }
            before = after;
        }
    }

    // An incident genuinely tightens its region.
    #[test]
    fn incident_tightens_its_region() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0);
        let s = st(2);
        let before = sm.effective_limit(&s);
        sm.record_incident(&s);
        assert!(
            sm.effective_limit(&s) < before,
            "should tighten near the incident"
        );
        // A distant state is unaffected (if capacity is not exceeded).
        assert_eq!(sm.effective_limit(&st(0)), 1.0);
    }

    // Governance clamps to the adaptive limit.
    #[test]
    fn govern_clamps_to_adaptive_limit() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0);
        let s = st(1);
        sm.record_incident(&s); // limit here is now 0.6
        let mut approved = [0.0; 2];
        let (lim, mask) = sm.govern(&s, &[0.9, -0.2], &mut approved);
        assert!((lim - 0.6).abs() < 1e-6);
        assert!((approved[0] - 0.6).abs() < 1e-6); // 0.9 clamped → 0.6
        assert_eq!(mask, 0b01);
    }

    // Optional forgetting: after sufficient safe passages the limit relaxes gradually then is
    // forgotten — **without ever exceeding the envelope**.
    #[test]
    fn evidence_based_forgetting_recovers_then_never_exceeds() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0).with_forgetting(5);
        let s = st(1);
        sm.record_incident(&s);
        let tight = sm.effective_limit(&s);
        assert!(tight < 1.0); // tightened after incident
                              // repeated safe passages → gradual relaxation toward the envelope
        let mut prev = tight;
        for _ in 0..200 {
            sm.confirm_safe(&s);
            let now = sm.effective_limit(&s);
            assert!(now >= prev - 1e-6, "forgetting must relax (non-decreasing)");
            assert!(now <= 1.0 + 1e-6, "MUST NEVER exceed the verified envelope");
            prev = now;
        }
        assert!(
            (sm.effective_limit(&s) - 1.0).abs() < 1e-6,
            "fully forgotten -> back to envelope"
        );
    }

    // Hazard still present: a new incident resets evidence and re-tightens (no premature forgetting).
    #[test]
    fn new_incident_resets_evidence() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0).with_forgetting(5);
        let s = st(1);
        sm.record_incident(&s);
        sm.confirm_safe(&s);
        sm.confirm_safe(&s); // partial evidence (2/5)
        sm.record_incident(&s); // hazard repeated → resets evidence and tightens
        let tight = sm.effective_limit(&s);
        sm.confirm_safe(&s);
        sm.confirm_safe(&s); // only 2/5 again
        assert!(
            sm.effective_limit(&s) <= tight + 1e-6,
            "must not relax before evidence threshold"
        );
    }

    // Incident-memory transfer: simulation → bytes → fresh device (born cautious).
    #[test]
    fn transfer_makes_fresh_memory_cautious() {
        let mut sim = SafetyMemory::<8, 4>::new(1.0);
        sim.record_incident(&st(1)); // simulation recorded an incident at st(1) → limit 0.6
        let mut blob = [0u8; SafetyMemory::<8, 4>::export_len()];
        let n = sim.export_bytes(&mut blob);

        // Fresh device with no experience → imports simulation memory.
        let mut fresh = SafetyMemory::<8, 4>::new(1.0);
        assert_eq!(fresh.effective_limit(&st(1)), 1.0); // before transfer: not cautious
        fresh.import_bytes(&blob[..n]);
        // after transfer: cautious at st(1) **without having experienced the incident**.
        assert!((fresh.effective_limit(&st(1)) - 0.6).abs() < 1e-6);
        assert_eq!(fresh.effective_limit(&st(0)), 1.0); // another context unaffected
    }

    // Golden bit fingerprint for determinism (passes through cosine/sqrtf): one incident ⇒
    // limit = 0.6 exactly.
    #[test]
    fn golden_bits_effective_limit() {
        let mut sm = SafetyMemory::<8, 4>::new(1.0);
        let s = [1.0, 0.1, 0.1, 0.1];
        sm.record_incident(&s);
        let e = sm.effective_limit(&s);
        assert!(e.is_finite());
        assert_eq!(e.to_bits(), 0x3f19_999au32); // 0.6f32
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

// ===== proptest properties: hard invariants hold over thousands of random incident sequences =====
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::array::uniform4;
    use proptest::prelude::*;

    proptest! {
        /// **Containment (the invariant that carries the kernel proof):** effective_limit ≤ static_limit
        /// for any incident sequence (sequences > CAP exercise the merge path).
        #[test]
        fn effective_limit_within_static(
            seq in prop::collection::vec(uniform4(-2.0f32..2.0), 0..24),
            probes in prop::collection::vec(uniform4(-2.0f32..2.0), 1..5),
        ) {
            let mut sm = SafetyMemory::<8, 4>::new(1.0);
            for s in &seq {
                sm.record_incident(s);
            }
            for p in &probes {
                prop_assert!(sm.effective_limit(p) <= 1.0 + 1e-6, "exceeded static envelope");
            }
        }

        /// **Tighten only:** recording an incident never raises any effective limit.
        #[test]
        fn record_incident_only_tightens(
            seq in prop::collection::vec(uniform4(-2.0f32..2.0), 1..20),
            probes in prop::collection::vec(uniform4(-2.0f32..2.0), 1..4),
        ) {
            let mut sm = SafetyMemory::<16, 4>::new(1.0);
            let mut before: Vec<f32> = probes.iter().map(|p| sm.effective_limit(p)).collect();
            for s in &seq {
                sm.record_incident(s);
                for (k, p) in probes.iter().enumerate() {
                    let now = sm.effective_limit(p);
                    prop_assert!(now <= before[k] + 1e-6, "limit increased -> relaxed safety");
                    before[k] = now;
                }
            }
        }

        /// preload clamps to the envelope: resulting limit ∈ [floor, static_limit].
        #[test]
        fn preload_clamps_to_envelope(fp in uniform4(-2.0f32..2.0), limit in -5.0f32..5.0) {
            let mut sm = SafetyMemory::<8, 4>::new(1.0);
            sm.preload(&fp, limit);
            let e = sm.effective_limit(&fp);
            prop_assert!(e.is_finite());
            prop_assert!((0.05 - 1e-6..=1.0 + 1e-6).contains(&e), "preload limit {e} left envelope");
        }

        /// Incident count never exceeds capacity.
        #[test]
        fn count_never_exceeds_cap(ops in prop::collection::vec(uniform4(-2.0f32..2.0), 0..50)) {
            let mut sm = SafetyMemory::<8, 4>::new(1.0);
            for s in &ops {
                sm.record_incident(s);
                prop_assert!(sm.incident_count() <= 8);
            }
        }

        /// Forgetting preserves the ceiling: effective_limit ≤ static_limit across all
        /// confirm_safe steps.
        #[test]
        fn forgetting_preserves_ceiling(
            s in uniform4(-2.0f32..2.0),
            k in 1u16..6,
            steps in 0usize..300,
        ) {
            let mut sm = SafetyMemory::<8, 4>::new(1.0).with_forgetting(k);
            sm.record_incident(&s);
            for _ in 0..steps {
                sm.confirm_safe(&s);
                prop_assert!(sm.effective_limit(&s) <= 1.0 + 1e-6, "forgetting exceeded envelope");
            }
        }
    }
}

// ===== Kani proofs: containment formally verified (cosine abstracted; holds for any similarity) =====
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: effective_limit ≤ static_limit for any state and any two symbolic incidents
    /// (abstraction makes this independent of sqrtf).
    #[kani::proof]
    #[kani::unwind(3)]
    fn proof_effective_limit_within_static() {
        let static_limit: f32 = kani::any();
        kani::assume(static_limit.is_finite());
        let mut sm = SafetyMemory::<2, 2>::new(static_limit);
        let s0 = [kani::any::<f32>(), kani::any::<f32>()];
        let s1 = [kani::any::<f32>(), kani::any::<f32>()];
        sm.record_incident(&s0);
        sm.record_incident(&s1);
        let probe = [kani::any::<f32>(), kani::any::<f32>()];
        assert!(sm.effective_limit(&probe) <= static_limit);
    }

    /// Proof: incident counter never exceeds capacity.
    #[kani::proof]
    #[kani::unwind(3)]
    fn proof_count_within_cap() {
        let mut sm = SafetyMemory::<2, 2>::new(1.0);
        let s0 = [kani::any::<f32>(), kani::any::<f32>()];
        let s1 = [kani::any::<f32>(), kani::any::<f32>()];
        let s2 = [kani::any::<f32>(), kani::any::<f32>()];
        sm.record_incident(&s0);
        sm.record_incident(&s1);
        sm.record_incident(&s2);
        assert!(sm.incident_count() <= 2);
    }
}
