//! Closed loop on **real sensor data** — the system's first contact with the real world.
//!
//! Reads laptop-camera features (from `drivers/camera_sensor.py` via stdin), completing the
//! full pipeline:
//! ```
//! world → camera → driver → perception → world-memory + FSM(decision) → guard(safety) → action
//! ```
//! All components (FSM + world-memory + guard) are `no_std` and run on seL4 — here they are
//! fed real sensor data.

use std::io::BufRead;

use behavior_fsm::{BehaviorFsm, Intent, Percept, State};
use brain_os_abi::govern;
use world_memory::WorldMemory;

const DIM: usize = 8;
const SAFETY_LIMIT: f32 = 0.6;

/// Intent → normalised motion command (proposed by the brain; governed by the guard).
fn intent_cmd(i: Intent) -> f32 {
    match i {
        Intent::Hold => 0.0,
        Intent::Scan => 0.3,
        Intent::MoveToward => 0.9, // deliberately bold to demonstrate guard clamping
        Intent::Manipulate => 0.5,
        Intent::BackOff => -0.8,
    }
}

fn state_name(s: State) -> &'static str {
    match s {
        State::Idle => "Idle",
        State::Search => "Search",
        State::Approach => "Approach",
        State::Act => "Act",
        State::Recover => "Recover",
    }
}

fn main() {
    let mut fsm = BehaviorFsm::new();
    let mut world = WorldMemory::<128, DIM>::new();
    let (mut frame, mut clamps, mut transitions) = (0u32, 0u32, 0u32);
    let mut prev_state = fsm.state();

    println!(
        "[real] closed loop on REAL camera data (world -> sensor -> memory+FSM -> guard -> act)"
    );
    let stdin = std::io::stdin();
    for line in stdin.lock().lines().map_while(Result::ok) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 3 + DIM {
            continue;
        }
        let vis = t[0] == "1";
        let near = t[1] == "1";
        let danger = t[2] == "1";
        let mut emb = [0.0f32; DIM];
        for (k, e) in emb.iter_mut().enumerate() {
            *e = t[3 + k].parse().unwrap_or(0.0);
        }

        // Perception → world-memory (stores what it sees, tagged with behaviour state and time).
        let p = Percept {
            target_visible: vis,
            target_near: near,
            unsafe_situation: danger,
        };
        let state = fsm.step(p);
        world.observe(&emb, [frame as f32, 0.0, 0.0], state as u32, frame);

        // FSM decision → proposed command → guard governs it (safety).
        let proposed = intent_cmd(fsm.intent());
        let mut approved = [0.0f32; 1];
        let (mask, _) = govern(&[proposed], SAFETY_LIMIT, true, &mut approved);
        clamps += mask.count_ones();

        if state != prev_state {
            transitions += 1;
            println!(
                "[real] frame {:>3}: vis={} near={} danger={} -> STATE {} ({:?}) cmd={:.2}->{:.2}",
                frame,
                vis as u8,
                near as u8,
                danger as u8,
                state_name(state),
                fsm.intent(),
                proposed,
                approved[0]
            );
        }
        prev_state = state;
        frame += 1;
    }

    println!("\n[real] ===== summary =====");
    println!("[real] processed {frame} REAL camera frames");
    println!("[real] world-memory holds {} perceptions", world.len());
    println!("[real] FSM state transitions: {transitions} · guard clamps: {clamps}");
    println!(
        "[real] PASS: the OS received REAL world data, decided (FSM), and acted under the guard."
    );
}
