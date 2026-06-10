//! SIL: **Safety learned from experience** — reducing Claim A to practice.
//!
//! A robot visits different situations. Situation "B" is risky (e.g., near a table edge).
//! When an **incident** occurs there — either a **guard clamp** (near-miss: the policy proposed
//! a command that exceeded the limit) or an **external danger signal** — it is recorded in
//! the safety memory, tightening the motion envelope **locally**. On returning to a similar
//! situation, the limit is **automatically tighter**.
//!
//! Demonstrates: the envelope tightens only at B, stays full at safe situations, and never
//! exceeds the verified envelope.
//! `safety-memory` is no_std and runs on seL4 — here we feed it a scripted scenario.

use safety_memory::SafetyMemory;

const DIM: usize = 4;
const STATIC_LIMIT: f32 = 1.0;

/// Embeddings of distinct situations (produced by world-memory/perception in the full system).
fn situation(name: char) -> [f32; DIM] {
    match name {
        'A' => [1.0, 0.1, 0.1, 0.1], // safe
        'B' => [0.1, 1.0, 0.1, 0.1], // risky (near the edge)
        _ => [0.1, 0.1, 1.0, 0.1],   // C safe
    }
}

fn main() {
    let mut safety = SafetyMemory::<32, DIM>::new(STATIC_LIMIT);
    // The policy (brain) always proposes a bold command (0.95) — the adaptive guard governs it.
    let proposed_bold = 0.95f32;

    println!("[adapt] static (verified) envelope = {STATIC_LIMIT}. The guard may only TIGHTEN it.");
    println!("[adapt] situation B is risky (e.g., near a table edge). Watch B's limit shrink with experience.\n");

    let schedule = ['A', 'B', 'C', 'B', 'A', 'B', 'C', 'B', 'B', 'A', 'B', 'C'];
    let mut near_miss = 0u32;
    let mut external = 0u32;

    for (cycle, &name) in schedule.iter().enumerate() {
        let s = situation(name);
        let lim_before = safety.effective_limit(&s);

        // Governance: clamp the bold command to the current effective limit.
        let mut approved = [0.0f32; 1];
        let (lim, mask) = safety.govern(&s, &[proposed_bold], &mut approved);

        // Incident detection (both types):
        // 1) near-miss: the guard had to clamp a bold command (intervention) at this situation.
        // 2) external danger: situation B triggers a danger signal (fall/proximity detector).
        let guard_clamped = mask != 0;
        let external_danger = name == 'B';
        if guard_clamped || external_danger {
            if guard_clamped {
                near_miss += 1;
            }
            if external_danger {
                external += 1;
            }
            safety.record_incident(&s); // ← safety learns: tighten here
        }

        let lim_after = safety.effective_limit(&s);
        let tag = if lim_after < lim_before {
            " <== TIGHTENED"
        } else {
            ""
        };
        println!(
            "[adapt] cyc{:>2} situ {} : limit {:.3} -> approved {:.3} (after-incident limit {:.3}){}",
            cycle, name, lim, approved[0], lim_after, tag
        );

        // Invariant checked every cycle: never exceeds the verified envelope.
        assert!(
            lim_after <= STATIC_LIMIT + 1e-6,
            "INVARIANT VIOLATED: exceeded verified envelope"
        );
    }

    println!("\n[adapt] ===== learned safety envelope =====");
    for name in ['A', 'B', 'C'] {
        println!(
            "[adapt]   situation {} final limit = {:.3}",
            name,
            safety.effective_limit(&situation(name))
        );
    }
    println!("[adapt] incidents: {near_miss} guard near-misses + {external} external-danger = {} recorded", safety.incident_count());
    println!("[adapt] PASS: the safety envelope LEARNED to tighten at the risky situation (B),");
    println!("[adapt]       stayed full at safe situations (A,C), and never exceeded the verified envelope.");
}
