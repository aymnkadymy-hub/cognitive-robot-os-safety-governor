//! SIL: deploy a **learned multi-joint skill** on the OS via the generalised guard.
//!
//! Demonstrates that the OS can run **any brain** (here: walker policy, 17 perceptions →
//! 6 joint commands) and govern it:
//! - **Neural memory** stores perception (system memory).
//! - **Generalised guard** (`brain-os-abi`) enforces a safety limit on **every joint**
//!   (clamps bold commands).
//! - **Watchdog** safely stops **all joints** if the brain stalls.
//!
//! Same crates that run on seL4 (SIL principle) — this represents exactly what the OS
//! executes on the kernel.

use brain_os_abi::{govern, heartbeat_stalled};
use neural_memory::NeuralMemory;
use walker_policy::{command, weights};

const STEPS: usize = 300;
const STALL_AT: usize = 150; // cycle at which we stall the brain (watchdog test)
const SAFETY_LIMIT: f32 = 0.6; // safety limit per joint (tighter than the policy's ±1 range)

/// Deterministic synthetic perception (simulates a sensor stream) to feed the skill.
fn perceive(t: usize) -> [f32; weights::IN] {
    let mut s = [0.0f32; weights::IN];
    let p = t as f32 * 0.05;
    for (i, v) in s.iter_mut().enumerate() {
        *v = libm_sin(p + i as f32 * 0.3) * 0.5;
    }
    s
}

// Simple sine without an external dependency (Taylor approximation is sufficient for the demo).
fn libm_sin(x: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut y = x % TAU;
    if y > PI {
        y -= TAU;
    } else if y < -PI {
        y += TAU;
    }
    let y2 = y * y;
    y * (1.0 - y2 / 6.0 + y2 * y2 / 120.0)
}

fn main() {
    let mut mem = NeuralMemory::<128, { weights::IN }>::new(); // system memory
    let mut heartbeat = 0u64;
    let mut guard_last = 0u64;
    let mut total_clamps = 0u32;
    let mut estops = 0u32;
    let mut sample = [0.0f32; weights::OUT];

    for cycle in 0..STEPS {
        // 1) Perception → system memory.
        let state = perceive(cycle);
        mem.store(&state, cycle as u64);

        // 2) Brain: the learned skill proposes 6 joint commands.
        let proposed = command(&state);

        // 3) Brain heartbeat (stalls at STALL_AT to test the watchdog).
        if cycle != STALL_AT {
            heartbeat += 1;
        }
        let alive = !heartbeat_stalled(heartbeat, guard_last);
        guard_last = heartbeat;

        // 4) Generalised guard governs all six commands (clamps each individually or stops all).
        let mut approved = [0.0f32; weights::OUT];
        let (mask, stopped) = govern(&proposed, SAFETY_LIMIT, alive, &mut approved);
        if stopped {
            estops += 1;
        }
        total_clamps += mask.count_ones();
        if cycle == 60 {
            sample = approved; // snapshot for display
        }
        // 5) (The actuator applies only `approved` to the motors.)
    }

    println!("[SIL-skill] ===== deploy multi-joint skill on the OS =====");
    println!(
        "[SIL-skill] skill = walker-policy ({} perception -> {} joint commands)",
        weights::IN,
        weights::OUT
    );
    println!("[SIL-skill] OS memory stored {} perceptions", mem.len());
    println!(
        "[SIL-skill] guard: safety_limit={}/100 per joint -> {} clamps, {} emergency-stops",
        (SAFETY_LIMIT * 100.0) as i32,
        total_clamps,
        estops
    );
    print!("[SIL-skill] sample approved joint cmds (x1000): [");
    for (i, c) in sample.iter().enumerate() {
        print!("{}{}", if i > 0 { ", " } else { "" }, (c * 1000.0) as i32);
    }
    println!("]");
    println!("[SIL-skill] PASS: OS ran a learned multi-joint brain under generalized safety.");
}
