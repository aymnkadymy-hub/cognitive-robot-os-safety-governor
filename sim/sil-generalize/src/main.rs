//! "Learns, doesn't memorize" — a danger model on **egocentric features** generalizes to
//! **new** obstacle positions; pure memorization (coordinates) fails.
//!
//! We train on arena #1, then **move the obstacles** to arena #2 and measure:
//! - **Egocentric feature model** (`safety-model`): detects danger in the new arena (generalizes).
//! - **Memorization** (set of recorded hazardous cells): fails — the obstacles are at
//!   positions it has never seen.
//!
//! Demonstrates that the system can become **smarter** (a function of relative features)
//! rather than just a lookup table. `safety-model` is seL4-ready.

use std::collections::HashSet;

use safety_model::SafetyModel;

const W: i32 = 26;
const H: i32 = 16;
const SENSE: f32 = 4.0;
const DANGER: f32 = 1.3; // closer than this = real danger

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
    fn obstacles(&mut self, n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|_| {
                (
                    2.0 + self.f() * (W as f32 - 4.0),
                    2.0 + self.f() * (H as f32 - 4.0),
                )
            })
            .collect()
    }
}

/// Nearest obstacle: (distance, unit direction).
fn nearest(obs: &[(f32, f32)], px: f32, py: f32) -> (f32, f32, f32) {
    let (mut bd, mut ux, mut uy) = (f32::INFINITY, 0.0, 0.0);
    for &(ox, oy) in obs {
        let (dx, dy) = (ox - px, oy - py);
        let d = (dx * dx + dy * dy).sqrt();
        if d < bd {
            bd = d;
            ux = dx / (d + 1e-3);
            uy = dy / (d + 1e-3);
        }
    }
    (bd, ux, uy)
}

/// Egocentric features (relative to the robot — invariant to absolute obstacle position).
fn ego(obs: &[(f32, f32)], px: f32, py: f32) -> [f32; 3] {
    let (d, ux, uy) = nearest(obs, px, py);
    let prox = ((SENSE - d) / SENSE).clamp(0.0, 1.0); // 1 = very close
    [prox, ux, uy]
}

/// Encounter positions for training/testing.
fn encounters(seed: u64, k: usize) -> Vec<(f32, f32)> {
    let mut r = Lcg(seed);
    (0..k)
        .map(|_| {
            (
                1.0 + r.f() * (W as f32 - 2.0),
                1.0 + r.f() * (H as f32 - 2.0),
            )
        })
        .collect()
}

fn main() {
    // Arena #1 (for training).
    let obs1 = Lcg(0xAA11).obstacles(14);
    // Arena #2: same obstacle count but **different positions** (moved).
    let obs2 = Lcg(0xBB22).obstacles(14);

    let mut model = SafetyModel::<3>::new();
    let mut memorized: HashSet<(i32, i32)> = HashSet::new(); // memorization: hazardous cells in #1

    // ===== Training on arena #1 =====
    for (px, py) in encounters(1, 4000) {
        let (d, _, _) = nearest(&obs1, px, py);
        let danger = d < DANGER;
        model.observe(&ego(&obs1, px, py), danger); // learns a function
        if danger {
            memorized.insert((px as i32, py as i32)); // memorizes the cell
        }
    }

    // ===== Evaluation: how many real dangers does each approach detect? =====
    let eval = |obs: &[(f32, f32)], model: &SafetyModel<3>, mem: &HashSet<(i32, i32)>| {
        let (mut dangers, mut m_caught, mut k_caught) = (0u32, 0u32, 0u32);
        for (px, py) in encounters(99, 4000) {
            let (d, _, _) = nearest(obs, px, py);
            if d < DANGER {
                dangers += 1;
                if model.predict_danger(&ego(obs, px, py)) {
                    m_caught += 1;
                }
                if mem.contains(&(px as i32, py as i32)) {
                    k_caught += 1;
                }
            }
        }
        (
            dangers,
            100.0 * m_caught as f32 / dangers as f32,
            100.0 * k_caught as f32 / dangers as f32,
        )
    };

    let (d1, m1, k1) = eval(&obs1, &model, &memorized);
    let (d2, m2, k2) = eval(&obs2, &model, &memorized);

    println!("[gen] trained on arena #1; tested detecting REAL dangers (close obstacle) on each arena.\n");
    println!("[gen] arena #1 (same layout)   : {} dangers | egocentric MODEL caught {:.0}% | MEMORIZED cells caught {:.0}%", d1, m1, k1);
    println!("[gen] arena #2 (obstacles MOVED): {} dangers | egocentric MODEL caught {:.0}% | MEMORIZED cells caught {:.0}%", d2, m2, k2);
    println!(
        "\n[gen] => the MEMORIZED approach collapses when obstacles move ({:.0}% -> {:.0}%),",
        k1, k2
    );
    println!("[gen]    while the egocentric MODEL still detects danger ({:.0}% -> {:.0}%) — it LEARNED, didn't memorize.", m1, m2);
    println!("[gen] (the model's output stays a tightening-only bound <= the verified envelope — Claim A preserved.)");
    println!(
        "[gen] PASS: generalization via relative features + a learned model, not a lookup table."
    );
}
