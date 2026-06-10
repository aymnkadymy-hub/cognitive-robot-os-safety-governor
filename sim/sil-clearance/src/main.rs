//! "Safety from the OS, not the brain" — the clearance guard keeps a **bad** brain safe (Bridge A).
//!
//! The brain here has **zero obstacle awareness**: it steers toward the target and always
//! commands maximum throttle. Without a guard it collides; with `clearance-guard` (a
//! model-agnostic braking barrier) it **never breaches the minimum clearance**. Same
//! seL4-ready crate.

use clearance_guard::ClearanceGuard;

const DMIN: f32 = 0.6; // minimum clearance to obstacle centre
const AMAX: f32 = 4.0;
const VMAX: f32 = 3.0;
const TAU: f32 = 0.1;
const DT: f32 = 0.05;
const AX: f32 = 5.0;
const AY: f32 = 3.0;

fn obstacles() -> [(f32, f32); 5] {
    [
        (-1.5, 0.5),
        (0.5, -0.8),
        (2.0, 1.0),
        (-2.5, -1.2),
        (3.0, -0.5),
    ]
}

/// Distance to the nearest hazard (obstacle or wall) in the forward direction (front ray).
fn front_distance(x: f32, y: f32, th: f32, obs: &[(f32, f32)]) -> f32 {
    let (dx, dy) = (libm::cosf(th), libm::sinf(th));
    let mut best = 100.0f32;
    for &(ox, oy) in obs {
        let (fx, fy) = (x - ox, y - oy);
        let b = 2.0 * (fx * dx + fy * dy);
        let c = fx * fx + fy * fy - DMIN * DMIN; // obstacle centre within DMIN = hazard
        let disc = b * b - 4.0 * c;
        if disc >= 0.0 {
            let t = (-b - libm::sqrtf(disc)) * 0.5;
            if t > 0.0 && t < best {
                best = t;
            }
        }
    }
    // Walls
    for t in [
        if dx > 1e-6 {
            (AX - x) / dx
        } else if dx < -1e-6 {
            (-AX - x) / dx
        } else {
            1e9
        },
        if dy > 1e-6 {
            (AY - y) / dy
        } else if dy < -1e-6 {
            (-AY - y) / dy
        } else {
            1e9
        },
    ] {
        if t > 0.0 && t < best {
            best = t;
        }
    }
    best
}

/// Actual minimum clearance (robot centre to nearest hazard) — used to measure violations.
fn clearance(x: f32, y: f32, obs: &[(f32, f32)]) -> f32 {
    let mut m = (AX - x.abs()).min(AY - y.abs()) + DMIN; // wall on the DMIN scale
    for &(ox, oy) in obs {
        m = m.min(libm::sqrtf((ox - x) * (ox - x) + (oy - y) * (oy - y)));
    }
    m
}

/// Run: bad brain (maximum throttle toward target, ignores obstacles). Returns (min clearance, violation count).
fn run(guard: Option<ClearanceGuard>, obs: &[(f32, f32)]) -> (f32, u32) {
    let targets = [(4.0, 2.0), (-4.0, 2.0), (4.0, -2.0), (-3.5, -2.0)];
    let (mut x, mut y, mut th, mut v) = (-4.5f32, -2.5f32, 0.0f32, 0.0f32);
    let (mut ti, mut min_clear, mut viol) = (0usize, f32::INFINITY, 0u32);
    let mut was_in = false;
    for _ in 0..4000 {
        let (tx, ty) = targets[ti];
        // Bad brain: steer toward target, maximum throttle — zero avoidance logic.
        let desired = libm::atan2f(ty - y, tx - x);
        let err = libm::atan2f(libm::sinf(desired - th), libm::cosf(desired - th));
        let steer = err.clamp(-0.6, 0.6);
        let mut throttle = 1.0f32;
        if let Some(g) = guard {
            let d_front = front_distance(x, y, th, obs);
            throttle = g.govern(d_front, throttle); // ← guard clamps the throttle
        }
        let dv = (throttle * VMAX - v).clamp(-AMAX * DT, AMAX * DT);
        v += dv;
        x += v * libm::cosf(th) * DT;
        y += v * libm::sinf(th) * DT;
        th += (v / 0.42) * libm::tanf(steer) * DT;
        x = x.clamp(-AX, AX);
        y = y.clamp(-AY, AY);

        let c = clearance(x, y, obs);
        min_clear = min_clear.min(c);
        let inside = c < DMIN;
        if inside && !was_in {
            viol += 1;
        }
        was_in = inside;
        if libm::sqrtf((tx - x) * (tx - x) + (ty - y) * (ty - y)) < 0.5 {
            ti = (ti + 1) % targets.len();
        }
    }
    (min_clear, viol)
}

fn main() {
    let obs = obstacles();
    println!(
        "[clr] a BAD brain (zero obstacle awareness: full throttle toward target, no avoidance)."
    );
    println!("[clr] d_min = {DMIN}m (minimum clearance the OS must guarantee).\n");

    let (mc_a, v_a) = run(None, &obs);
    let guard = ClearanceGuard::new(DMIN, AMAX, VMAX, TAU);
    let (mc_b, v_b) = run(Some(guard), &obs);

    println!(
        "[clr] (A) brain alone         : min clearance {:.2}m | violations {}",
        mc_a, v_a
    );
    println!(
        "[clr] (B) brain + CLEARANCE GUARD: min clearance {:.2}m | violations {}",
        mc_b, v_b
    );
    println!();
    if v_b == 0 && mc_b >= DMIN - 0.02 {
        println!("[clr] PASS: the GUARD kept clearance >= {DMIN}m for a brain with NO obstacle awareness —");
        println!("[clr]       safety is GUARANTEED BY THE OS (verified barrier), not learned by the policy.");
    } else {
        println!(
            "[clr] FAIL: clearance violated (got {:.2}m, {} violations)",
            mc_b, v_b
        );
    }
}
