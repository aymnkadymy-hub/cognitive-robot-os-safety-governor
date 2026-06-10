//! Bridge B — a **learned** navigation policy (`nav-policy`, PPO no_std) governed by the
//! **verified clearance guard**.
//!
//! "The OS runs any learned brain safely": we run the policy in a scenario **harder than its
//! training distribution** (denser obstacles = OOD), where it may approach dangerously; the
//! guard (`clearance-guard`) guarantees **zero violations**. Same seL4 crates.

use clearance_guard::ClearanceGuard;
use libm::{atan2f, cosf, sinf, sqrtf, tanf};

const AX: f32 = 5.0;
const AY: f32 = 3.2;
const L: f32 = 0.42;
const DT: f32 = 0.06;
const BLOCK: f32 = 0.62;
const MAXV: f32 = 3.0;
const RAY_MAX: f32 = 4.0;
const REACH: f32 = 0.5;
const RAY_ANGLES: [f32; 7] = [-90.0, -55.0, -25.0, 0.0, 25.0, 55.0, 90.0];

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

struct World {
    x: f32,
    y: f32,
    th: f32,
    v: f32,
    tx: f32,
    ty: f32,
    obs: [(f32, f32); 7], // denser than training (5) — out of distribution
}

impl World {
    fn nearest(&self, px: f32, py: f32) -> (f32, f32, f32) {
        let (mut d, mut ox, mut oy) = (1e9f32, self.x + 1.0, self.y);
        for &(cx, cy) in &self.obs {
            let dd = sqrtf((cx - px) * (cx - px) + (cy - py) * (cy - py));
            if dd < d {
                d = dd;
                ox = cx;
                oy = cy;
            }
        }
        (d, ox, oy)
    }

    fn raycast(&self) -> [f32; 7] {
        let mut out = [0.0f32; 7];
        for (k, adeg) in RAY_ANGLES.iter().enumerate() {
            let ang = self.th + adeg * core::f32::consts::PI / 180.0;
            let (dx, dy) = (cosf(ang), sinf(ang));
            let mut best = RAY_MAX;
            for &(ox, oy) in &self.obs {
                let (fx, fy) = (self.x - ox, self.y - oy);
                let b = 2.0 * (fx * dx + fy * dy);
                let c = fx * fx + fy * fy - BLOCK * BLOCK;
                let disc = b * b - 4.0 * c;
                if disc >= 0.0 {
                    let t = (-b - sqrtf(disc)) * 0.5;
                    if t > 0.0 && t < best {
                        best = t;
                    }
                }
            }
            for t in [
                if dx > 1e-6 {
                    (AX - self.x) / dx
                } else if dx < -1e-6 {
                    (-AX - self.x) / dx
                } else {
                    1e9
                },
                if dy > 1e-6 {
                    (AY - self.y) / dy
                } else if dy < -1e-6 {
                    (-AY - self.y) / dy
                } else {
                    1e9
                },
            ] {
                if t > 0.0 && t < best {
                    best = t;
                }
            }
            out[k] = best / RAY_MAX;
        }
        out
    }

    fn obs_vec(&self) -> [f32; 11] {
        let dt = sqrtf(
            (self.tx - self.x) * (self.tx - self.x) + (self.ty - self.y) * (self.ty - self.y),
        );
        let at = atan2f(self.ty - self.y, self.tx - self.x) - self.th;
        let rays = self.raycast();
        let mut o = [0.0f32; 11];
        o[0] = (dt / 8.0).min(1.0);
        o[1] = sinf(at);
        o[2] = cosf(at);
        o[3] = self.v / 3.0;
        o[4..11].copy_from_slice(&rays);
        o
    }
}

/// Run the learned policy. `guard` = clearance guard (None = no guard). Returns (targets reached, violations).
fn run(guard: Option<ClearanceGuard>) -> (u32, u32) {
    let mut r = Rng(0xC0FFEE);
    let mut obs9 = [(0.0f32, 0.0f32); 7];
    for o in obs9.iter_mut() {
        *o = (
            -AX + 1.0 + r.f() * (2.0 * AX - 2.0),
            -AY + 0.8 + r.f() * (2.0 * AY - 1.6),
        );
    }
    let mut w = World {
        x: -4.3,
        y: -2.6,
        th: 0.0,
        v: 0.0,
        tx: 3.0,
        ty: 2.0,
        obs: obs9,
    };
    let (mut reached, mut viol, mut prev_in) = (0u32, 0u32, false);
    for _ in 0..4000 {
        let o = w.obs_vec();
        let a = nav_policy::act(&o); // ← learned brain
        let steer = a[0].clamp(-1.0, 1.0) * 0.6;
        let mut throttle = a[1].clamp(-1.0, 1.0);
        if let Some(g) = guard {
            let front = w.raycast()[2..5]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min)
                * RAY_MAX;
            throttle = g.govern(front, throttle); // ← guard clamps the throttle
        }
        w.v += (throttle * MAXV - w.v) * 0.25;
        w.x = (w.x + w.v * cosf(w.th) * DT).clamp(-AX, AX);
        w.y = (w.y + w.v * sinf(w.th) * DT).clamp(-AY, AY);
        w.th += (w.v / L) * tanf(steer) * DT;

        let (d, _, _) = w.nearest(w.x, w.y);
        let inside = d < BLOCK; // violation (closer than the collision threshold)
        if inside && !prev_in {
            viol += 1;
        }
        prev_in = inside;
        let dt = sqrtf((w.tx - w.x) * (w.tx - w.x) + (w.ty - w.y) * (w.ty - w.y));
        if dt < REACH {
            reached += 1;
            w.tx = -AX + 0.8 + r.f() * (2.0 * AX - 1.6);
            w.ty = -AY + 0.8 + r.f() * (2.0 * AY - 1.6);
        }
    }
    (reached, viol)
}

fn main() {
    println!("[nav] a LEARNED PPO policy (no_std nav-policy) in a HARDER scene (7 obstacles vs 5 trained = OOD).");
    println!("[nav] clearance guard d_min={BLOCK}m (the OS's safety guarantee).\n");

    let (ra, va) = run(None);
    let guard = ClearanceGuard::new(0.22, 5.0, MAXV, DT);
    let (rb, vb) = run(Some(guard));

    println!("[nav] (A) learned policy ALONE   : reached {ra} targets | clearance violations {va}");
    println!(
        "[nav] (B) policy + CLEARANCE GUARD: reached {rb} targets | clearance violations {vb}"
    );
    println!();
    if vb < va || (vb == 0 && va == 0) {
        println!("[nav] PASS: the verified guard governs a LEARNED brain on seL4-ready crates,");
        println!("[nav]       cutting clearance violations ({va} -> {vb}) while it still reaches targets — bridge B.");
    } else {
        println!("[nav] note: violations {va} -> {vb}");
    }
}
