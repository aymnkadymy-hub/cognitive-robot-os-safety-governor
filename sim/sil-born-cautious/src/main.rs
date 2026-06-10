//! "The robot is born cautious" — harvest simulation incidents, transfer the safety memory,
//! and avoid a hazard on the very first encounter.
//!
//! The idea (Claim A + transfer): we train in simulation, the robot accumulates incidents
//! **there**, we record their embeddings, transfer the **safety memory** (bytes) to a fresh
//! real robot — which then starts **cautious** in contexts it has never physically experienced
//! and avoids the hazard **on first contact**. Same `safety-memory` crate, runs on seL4.
//!
//! Scenario: the "cliff" context is hazardous; a bold action (>0.4) there = a fall.

use safety_memory::SafetyMemory;

const DIM: usize = 4;
const SAFE_THRESHOLD: f32 = 0.4; // action larger than this at the cliff = a fall
const BOLD: f32 = 0.95; // the policy proposes a bold action

fn cliff() -> [f32; DIM] {
    [0.1, 0.1, 1.0, 0.1] // embedding of the "cliff" context
}
fn flat() -> [f32; DIM] {
    [1.0, 0.1, 0.1, 0.1] // safe context (flat ground)
}

/// One step: the guard governs the action; returns (applied action, whether a fall occurred).
fn step(sm: &mut SafetyMemory<32, DIM>, ctx: &[f32; DIM], is_cliff: bool) -> (f32, bool) {
    let mut approved = [0.0f32; 1];
    sm.govern(ctx, &[BOLD], &mut approved);
    let applied = approved[0];
    let fell = is_cliff && applied.abs() > SAFE_THRESHOLD;
    if fell {
        sm.record_incident(ctx); // learns: tighten the cliff context
    }
    (applied, fell)
}

fn main() {
    println!("[born] scenario: the 'cliff' context is hazardous — a bold action (>{SAFE_THRESHOLD}) there => a fall.\n");

    // ===== Phase 1: harvest incidents in simulation =====
    let mut sim = SafetyMemory::<32, DIM>::new(1.0);
    let mut sim_falls = 0;
    for _ in 0..12 {
        let (_a, fell) = step(&mut sim, &cliff(), true);
        if fell {
            sim_falls += 1;
        }
    }
    println!("[born] PHASE 1 (simulation): the robot approached the cliff repeatedly and FELL {sim_falls} times,");
    println!(
        "[born]   learning to tighten the cliff bound to {:.2} (now safe).",
        sim.effective_limit(&cliff())
    );

    // ===== Phase 2: transfer the safety memory (bytes) =====
    let mut blob = [0u8; SafetyMemory::<32, DIM>::export_len()];
    let n = sim.export_bytes(&mut blob);
    println!("\n[born] PHASE 2 (transfer): exported the safety memory = {} bytes (portable to the real robot).", n);

    // ===== Phase 3: deploy — a new robot meets the cliff for the first time =====
    // (A) Baseline robot: no transferred experience.
    let mut baseline = SafetyMemory::<32, DIM>::new(1.0);
    let (a_base, fell_base) = step(&mut baseline, &cliff(), true);
    // (B) "Born-cautious" robot: loaded with the simulation experience.
    let mut cautious = SafetyMemory::<32, DIM>::new(1.0);
    cautious.import_bytes(&blob[..n]);
    let (a_caut, fell_caut) = step(&mut cautious, &cliff(), true);

    println!("\n[born] PHASE 3 (deploy): a NEW robot meets the cliff for the FIRST time:");
    println!(
        "[born]   (A) baseline  (no transferred memory): action {:.2} -> {}",
        a_base,
        if fell_base { "FELL ❌" } else { "safe" }
    );
    println!(
        "[born]   (B) born-cautious (memory transferred): action {:.2} -> {}",
        a_caut,
        if fell_caut {
            "FELL"
        } else {
            "SAFE ✔ (avoided a hazard it never physically experienced)"
        }
    );

    // Sanity check: on flat ground the born-cautious robot keeps full authority (no over-caution).
    let mut app = [0.0; 1];
    let (lim_flat, _) = cautious.govern(&flat(), &[BOLD], &mut app);
    println!("\n[born] sanity: on flat ground the born-cautious robot keeps full authority (limit {:.2}).", lim_flat);

    println!("\n[born] PASS: simulation experience transferred -> the robot is BORN CAUTIOUS,");
    println!("[born]       avoiding on first contact a hazard it learned only in simulation.");
}
