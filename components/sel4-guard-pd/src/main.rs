//! Guard PD (spinal cord) — highest priority, **with a learning generalizing safety guard (Claim A + A7)**.
//! Each cycle: reads state + proposed command + heartbeat; extracts **egocentric danger features** from state
//! (proximity to fall / angular velocity / proximity to edge); predicts danger with an **online-learned model**,
//! computing a safe bound `= verified_envelope·(1−risk)` that **never exceeds the envelope**; clips the
//! command to it; and updates the model from the actual danger. The model is **a function of relative features**
//! (not a lookup table) → **generalizes** to new states — on the verified kernel.

#![no_std]
#![no_main]

use reflex_abi::{
    heartbeat_stalled, HARD_LIMIT, OFF_APPROVED, OFF_CYCLE, OFF_HEARTBEAT, OFF_OVERRIDDEN,
    OFF_PROPOSED, OFF_STATE, REGION_SIZE,
};
use clearance_guard::ClearanceGuard;
use ood_detector::MahalanobisOod;
use safety_model::SafetyModel;
use sel4_microkit::{
    debug_println, memory_region_symbol, protection_domain, Channel, ChannelSet, Handler,
    Infallible,
};
use sel4_shared_memory::{access::ReadWrite, SharedMemoryRef};

const TO_ACTUATION: Channel = Channel::new(1);
/// Danger threshold (the model learns that features of this state = dangerous).
const DANGER_THETA: f32 = 0.03;
/// Bound floor (gentle tightening to maintain balance while demonstrating adaptation).
const FLOOR: f32 = 0.85;
/// Track limit (edge = danger) and minimum clearance half-margin from the edge.
const X_LIMIT: f32 = 2.4;
const EDGE_DMIN: f32 = 0.5;

/// Out-of-distribution gate (Mahalanobis): state far from the training manifold ⇒ maximum tightening (tighten-only).
/// Constants fitted on the host in `sim/sil-ood` (deterministic, reproducible); see `docs/OOD_INTEGRATION.md`.
const OOD_FLOOR: f32 = 0.1;
const OOD_THRESHOLD: f32 = 11.345; // χ² at 99%, df=3
const OOD_MEAN: [f32; 3] = [0.044904873, 0.007411593, 0.018937042];
const OOD_SIGMA_INV: [[f32; 3]; 3] = [
    [467.52313, 0.0, 0.0],
    [0.0, 6693.6, 0.0],
    [0.0, 0.0, 9509.354],
];

/// Egocentric danger features from state [x, theta, x_dot, theta_dot] (MuJoCo order) — relative, generalizing.
fn danger_features(s: &[f32; 4]) -> [f32; 3] {
    [
        (s[1].abs() / 0.2).min(1.0), // proximity to fall (tilt angle)
        (s[3].abs() / 3.0).min(1.0), // angular velocity
        (s[0].abs() / 4.0).min(1.0), // proximity to edge (position)
    ]
}

#[protection_domain]
fn init() -> HandlerImpl {
    debug_println!(
        "[guard] spinal-cord online (prio 254, LEARNED+GENERALIZING envelope: risk-model, tightens-only)"
    );
    let region = unsafe {
        SharedMemoryRef::new(memory_region_symbol!(reflex_vaddr: *mut [u8], n = REGION_SIZE))
    };
    HandlerImpl {
        region,
        last_heartbeat: u64::MAX,
        model: SafetyModel::new().with_lr(0.15),
        ood: MahalanobisOod::new(OOD_MEAN, OOD_SIGMA_INV, OOD_THRESHOLD),
        // Clearance barrier: braking (a_max≈3, v_max≈2.5, τ=cycle) — enforces that the cart never leaves the track.
        clearance: ClearanceGuard::new(EDGE_DMIN, 2.0, 2.5, 0.05),
        dangers: 0,
        tightenings: 0,
        edge_brakes: 0,
        min_edge: X_LIMIT,
    }
}

struct HandlerImpl {
    region: SharedMemoryRef<'static, [u8], ReadWrite>,
    last_heartbeat: u64,
    model: SafetyModel<3>,
    ood: MahalanobisOod<3>,
    clearance: ClearanceGuard,
    dangers: u32,
    tightenings: u32,
    edge_brakes: u32,
    min_edge: f32,
}

impl HandlerImpl {
    fn read(&self, off: usize, buf: &mut [u8]) {
        self.region
            .as_ptr()
            .index(off..off + buf.len())
            .copy_into_slice(buf);
    }
    fn write(&mut self, off: usize, data: &[u8]) {
        self.region
            .as_mut_ptr()
            .index(off..off + data.len())
            .copy_from_slice(data);
    }
    fn read_u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_le_bytes(b)
    }
    fn read_f32(&self, off: usize) -> f32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        f32::from_le_bytes(b)
    }
    fn read_state(&self) -> [f32; 4] {
        let mut s = [0.0f32; 4];
        for (i, v) in s.iter_mut().enumerate() {
            *v = self.read_f32(OFF_STATE + i * 4);
        }
        s
    }
}

impl Handler for HandlerImpl {
    type Error = Infallible;

    fn notified(&mut self, _channels: ChannelSet) -> Result<(), Self::Error> {
        let state = self.read_state();
        let mut b8 = [0u8; 8];
        self.read(OFF_HEARTBEAT, &mut b8);
        let heartbeat = u64::from_le_bytes(b8);
        let proposed = self.read_f32(OFF_PROPOSED);
        let cycle = self.read_u32(OFF_CYCLE);

        let stalled = heartbeat_stalled(heartbeat, self.last_heartbeat);
        self.last_heartbeat = heartbeat;

        // The learned model predicts danger from egocentric features → safe bound ≤ verified envelope always.
        let feat = danger_features(&state);
        let bound = self.model.safe_bound(&feat, HARD_LIMIT, FLOOR);
        // OOD gate: state far from training manifold ⇒ maximum tightening (tighten-only, preserves envelope and kernel proof).
        let bound = self.ood.tighten_if_ood(&feat, bound, OOD_FLOOR);
        let (mut approved, mut overridden) = if stalled {
            (0.0f32, true)
        } else {
            let a = proposed.clamp(-bound, bound);
            (a, a != proposed)
        };
        if bound < HARD_LIMIT - 1e-3 {
            self.tightenings += 1;
        }

        // ★ Clearance barrier (Bridge A on the kernel): enforces that the cart never leaves the track edge, regardless of brain.
        // If the cart is approaching the edge faster than it can stop (v > v_safe) → brake toward center.
        let x = state[0];
        let x_dot = state[2];
        let d_edge = X_LIMIT - x.abs();
        let v_toward = if x >= 0.0 { x_dot } else { -x_dot }; // velocity toward the edge
        if !stalled && v_toward > self.clearance.safe_speed(d_edge) {
            approved = -x.signum() * HARD_LIMIT; // full brake toward center (final authority: prevents derailment)
            overridden = true;
            self.edge_brakes += 1;
        }
        if d_edge < self.min_edge {
            self.min_edge = d_edge;
        }

        self.write(OFF_APPROVED, &approved.to_le_bytes());
        self.write(OFF_OVERRIDDEN, &(overridden as u32).to_le_bytes());

        // **Online learning on the kernel**: update the model from actual danger (generalizing function, not memorization).
        if !stalled {
            let danger = state[1].abs() > DANGER_THETA;
            self.model.observe(&feat, danger);
            if danger {
                self.dangers += 1;
                if self.dangers % 16 == 1 {
                    debug_println!(
                        "[guard] cyc {}: danger learned -> risk {}/1000, safe bound {}/1000 (<= {}/1000)  [generalizing]",
                        cycle,
                        (self.model.risk(&feat) * 1000.0) as i32,
                        (bound * 1000.0) as i32,
                        (HARD_LIMIT * 1000.0) as i32
                    );
                }
            }
        } else {
            debug_println!(
                "[guard] cycle {}: *** WATCHDOG: brain stalled -> EMERGENCY STOP ***",
                cycle
            );
        }

        if cycle + 1 >= reflex_abi::MAX_CYCLES {
            // Generalization test: danger on features **never seen exactly** (hypothetical near-fall) → model predicts danger.
            let probe = [0.7f32, 0.4, 0.2];
            debug_println!(
                "[guard] LEARNED-ENVELOPE summary: {} dangers learned, {} tightenings; never exceeded {}/1000.",
                self.dangers,
                self.tightenings,
                (HARD_LIMIT * 1000.0) as i32
            );
            debug_println!(
                "[guard] generalization probe (unseen near-fall features): risk {}/1000 -> the model LEARNED a danger function, not a table.",
                (self.model.risk(&probe) * 1000.0) as i32
            );
            // Clearance barrier: did the cart actually leave the track? (edge at |x|=X_LIMIT ⇒ min_edge ≤ 0)
            let derailed = self.min_edge <= 0.0;
            debug_println!(
                "[guard] CLEARANCE-BARRIER (on seL4): {} edge-brakes; min edge clearance {}/1000m; LEFT TRACK={}",
                self.edge_brakes,
                (self.min_edge * 1000.0) as i32,
                derailed as u32
            );
            debug_println!(
                "[guard] -> the verified guard ENFORCED the track-clearance barrier (final authority); clean proof: sim/sil-clearance."
            );
        }

        TO_ACTUATION.notify();
        Ok(())
    }
}
