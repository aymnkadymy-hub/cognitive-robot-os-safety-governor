//! Real-time benchmark of a single guard decision (same seL4 crates, no_std) — replaces
//! architectural hand-waving with actual numbers.
//!
//! The decision is exactly what the seL4 guard does every cycle: danger features → learned
//! safety bound + clearance barrier + incident memory + clamp. We measure mean/p50/p99/max
//! and jitter on the host (the same code runs on the target device).

use std::hint::black_box;
use std::time::Instant;

use clearance_guard::ClearanceGuard;
use safety_memory::SafetyMemory;
use safety_model::SafetyModel;

fn danger_features(s: &[f32; 4]) -> [f32; 3] {
    [
        (s[1].abs() / 0.2).min(1.0),
        (s[3].abs() / 3.0).min(1.0),
        (s[0].abs() / 4.0).min(1.0),
    ]
}

fn main() {
    let mut model = SafetyModel::<3>::new();
    let mem = SafetyMemory::<32, 4>::new(1.0);
    let cbf = ClearanceGuard::new(0.5, 3.0, 2.5, 0.05);
    // Warm-start the model with a few observations (realistic initial state).
    for _ in 0..200 {
        model.observe(&[0.5, 0.3, 0.2], true);
        model.observe(&[0.05, 0.0, 0.1], false);
    }

    const N: usize = 2_000_000;
    let mut times = Vec::with_capacity(N);
    let mut sink = 0.0f32;
    // Varying inputs (prevents the compiler from over-optimizing the loop away).
    for i in 0..N {
        let t = i as f32 * 1e-4;
        let state = [0.3 * t.sin(), 0.1 * t.cos(), 0.2 * t.sin(), 0.05 * t.cos()];
        let d_front = 1.0 + (t.cos().abs()) * 3.0;
        let proposed = 0.9f32;

        let t0 = Instant::now();
        // ===== Full guard decision =====
        let feat = danger_features(black_box(&state));
        let bound = model.safe_bound(&feat, 1.0, 0.85); // generalized learned bound
        let eff = mem.effective_limit(black_box(&state)); // incident memory
        let v_safe = cbf.safe_speed(black_box(d_front)); // clearance barrier
        let limit = bound.min(eff).min(1.0);
        let approved = proposed.clamp(-limit, limit).min(v_safe);
        // ==============================
        times.push(t0.elapsed().as_nanos() as u64);
        sink += black_box(approved);
    }
    black_box(sink);

    times.sort_unstable();
    let mean = times.iter().sum::<u64>() as f64 / N as f64;
    let p50 = times[N / 2];
    let p99 = times[N * 99 / 100];
    let p999 = times[N * 999 / 1000];
    let max = *times.last().unwrap();
    println!("[bench] one full guard decision (learned bound + incident memory + clearance barrier + clamp)");
    println!("[bench] N={N} on host (x86-64; same no_std code runs on seL4/aarch64)");
    println!("[bench]   mean = {mean:.0} ns   p50 = {p50} ns   p99 = {p99} ns   p99.9 = {p999} ns   max = {max} ns");
    println!(
        "[bench]   jitter (p99 - p50) = {} ns",
        p99.saturating_sub(p50)
    );
    println!("[bench] => the guard decision is O(1), branch-light, allocation-free (no_std), bounded compute.");
    println!("[bench] NOTE: host timing (cache/OS noise inflates max). On-target WCET on the Pi4 + seL4's");
    println!("[bench]       verified highest-priority scheduling gives the end-to-end real-time bound (future).");
}
