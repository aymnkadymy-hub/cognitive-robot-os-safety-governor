//! # Software-in-the-Loop + viewer feed (cartpole)
//!
//! Closes the loop (cartpole-sim physics ↔ trained policy ↔ safety layer) using the same
//! seL4 crates, and writes the trajectory to CSV for visualization by `sim/view_cartpole.py`.
//!
//! The task is configurable via command-line arguments:
//!   sil-cartpole [target_x] [initial_theta] [out.csv]
//! Example: `sil-cartpole 0.6 0.05`  =  "drive right to x=0.6 and balance there".

use cartpole_policy::command;
use cartpole_sim::{step, State, FALL_ANGLE, FORCE_MAG};
use neural_memory::NeuralMemory;
use reflex_abi::enforce_limit;
use std::io::Write;

const STEPS: usize = 500;

fn main() {
    // Task parameters (with defaults).
    let args: Vec<String> = std::env::args().collect();
    let target_x: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let theta0: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{}/cartpole_trajectory.csv", env!("CARGO_MANIFEST_DIR")));

    let mut s = State::upright(theta0);
    let mut mem = Box::new(NeuralMemory::<512, 4>::new());
    let mut traj: Vec<(f32, f32, f32, u8)> = Vec::with_capacity(STEPS);

    let mut clamps = 0usize;
    let mut max_theta = 0.0f32;
    let mut fell_at: Option<usize> = None;

    println!(
        "[SIL] task: balance at target_x={:.2}, initial_theta={:.2}, {} steps",
        target_x, theta0, STEPS
    );
    for k in 0..STEPS {
        if k == 100 || k == 250 || k == 400 {
            s.theta_dot += 1.8;
            println!("[SIL] step {}: external disturbance (+1.8 rad/s)", k);
        }
        mem.store(&[s.x, s.x_dot, s.theta, s.theta_dot], k as u64);

        // Task: shift position toward target_x; the policy regulates the cart there.
        let obs = [s.x - target_x, s.x_dot, s.theta, s.theta_dot];
        let proposed = command(&obs);

        // The same safety layer that runs on seL4.
        let (approved, overridden) = enforce_limit(proposed);
        if overridden {
            clamps += 1;
        }

        traj.push((s.x, s.theta, approved, overridden as u8));
        s = step(s, approved * FORCE_MAG);

        max_theta = max_theta.max(s.theta.abs());
        if s.theta.abs() > FALL_ANGLE && fell_at.is_none() {
            fell_at = Some(k);
        }
    }

    // Write the trajectory for the visualizer.
    write_csv(&out_path, target_x, &traj);

    println!("\n[SIL] ===== results =====");
    println!(
        "[SIL] balanced={} max|theta|={}/1000 (limit {}/1000) clamps={} memory={}",
        if fell_at.is_none() { "YES" } else { "NO" },
        (max_theta * 1000.0) as i32,
        (FALL_ANGLE * 1000.0) as i32,
        clamps,
        mem.len()
    );
    println!("[SIL] trajectory written: {}", out_path);
    println!("[SIL] view it:  python3 sim/view_cartpole.py {}", out_path);
    if fell_at.is_some() {
        std::process::exit(1);
    }
}

fn write_csv(path: &str, target_x: f32, traj: &[(f32, f32, f32, u8)]) {
    let mut f = std::fs::File::create(path).expect("create csv");
    writeln!(f, "# target_x={}", target_x).unwrap();
    writeln!(f, "x,theta,motor,clamped").unwrap();
    for (x, th, m, c) in traj {
        writeln!(f, "{},{},{},{}", x, th, m, c).unwrap();
    }
}
