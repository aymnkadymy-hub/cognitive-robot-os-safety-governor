//! # sil-adversarial — 2-D arena, random fast multi-directional adversarial obstacles
//!
//! Stress-tests the hardest case the paper flags as a weakness: hazards that attack from **any
//! direction** with **fast, random** patterns. The robot patrols toward a goal while N moving
//! obstacles (max speed `umax`) re-pick a random heading every `REPICK` steps. Two adversaries:
//! `Random` (non-homing) and `Homing` (each obstacle biases 60% toward the robot — an *adaptive*
//! adversary). Acceleration is bounded (`A_MAX`), so stopping is not instantaneous and fast
//! obstacles genuinely threaten — reaction quality matters.
//!
//! Safety filters on the goal-seeking command (all share the same dynamics and obstacles):
//!  - **Nominal** — no avoidance (upper bound on collisions).
//!  - **CBF** — distance-only control-barrier filter: caps approach speed to `α·(d−R)` (common form).
//!  - **VO-CBF** — *velocity-aware* worst-case cap, omnidirectional but **passive** (no fleeing). A
//!    strong baseline and the `cap-only` ablation of Ours: it isolates "velocity awareness" alone.
//!  - **Simplex** — binary switch: actively retreat from the nearest obstacle within `D_SWITCH`.
//!  - **FrontOnly** — our *current* design's analogue: brakes only in the heading cone (blind to side/rear).
//!  - **Evade-only** — active evasion **without** the worst-case cap (the `evasion-only` ablation).
//!  - **Ours** — **velocity-aware active evasion + worst-case tighten-only cap**: flee threats
//!    weighted by closeness and closing speed, then cap the component toward every obstacle by the
//!    braking speed that stays safe even if it keeps closing. Each cap is a `min` into the verified
//!    envelope ⇒ verification-preserving (Proposition 1) — the robot both *flees* and *never over-commits*.
//!
//! The two ablations attribute the win: VO-CBF (cap, no flee) and Evade-only (flee, no cap) both
//! underperform Ours, so neither ingredient alone suffices. Deterministic LCG, 40 paired seeds.

use libm::{atan2f, cosf, erff, sinf, sqrtf};
use reactive_evasion::{clamp_approach, safe_approach};
use std::f32::consts::{PI, TAU};

const ARENA: f32 = 16.0;
const DT: f32 = 0.02;
const STEPS: usize = 4000;
const V_MAX: f32 = 3.0;
const A_MAX: f32 = 8.0;
const R_ROBOT: f32 = 0.35;
const R_OBS: f32 = 0.35;
const R_COLL: f32 = R_ROBOT + R_OBS;
const MARGIN: f32 = 0.30;
const SENSE: f32 = 9.0;
const N_OBS: usize = 8;
const U_MAX: f32 = 2.8;
const REPICK: usize = 18;
const SEEDS: usize = 40;
const ALPHA: f32 = 2.5;
const D_SWITCH: f32 = 2.4;
const FRONT_COS: f32 = 0.5;
const DANGER: f32 = 5.0;
const REP_GAIN: f32 = 1.6;
const OMEGA_MAX: f32 = 4.0; // unicycle max turn rate (rad/s)

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn raw(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.raw() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn span(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    fn dir(&mut self) -> [f32; 2] {
        let a = self.span(0.0, 6.283_185_5);
        [cosf(a), sinf(a)]
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Nominal,
    Cbf,
    LearnedCbf,
    CbfVel,
    Simplex,
    FrontOnly,
    EvadeOnly,
    Ours,
}
#[derive(Clone, Copy, PartialEq)]
enum Adv {
    Random,
    Homing,
}
/// Robot dynamics model — a second "simulator" to test cross-dynamics robustness.
#[derive(Clone, Copy, PartialEq)]
enum Dyn {
    Holonomic, // omnidirectional point mass (bounded acceleration)
    Unicycle,  // non-holonomic differential drive (must turn to change direction)
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm(a: [f32; 2]) -> f32 {
    sqrtf(a[0] * a[0] + a[1] * a[1])
}
fn unit_to(v: [f32; 2], speed: f32) -> [f32; 2] {
    let n = norm(v);
    if n > 1e-4 {
        [v[0] / n * speed, v[1] / n * speed]
    } else {
        [0.0, 0.0]
    }
}
/// Tighten-only cap: lower the component of `v` toward `rhat` to the machine-checked
/// `clamp_approach` bound (`reactive-evasion`); the tangential component is preserved.
fn cap(v: [f32; 2], rhat: [f32; 2], s_max: f32) -> [f32; 2] {
    let t = dot(v, rhat);
    let capped = clamp_approach(t, s_max); // proven: capped <= t and capped <= s_max
    [v[0] + (capped - t) * rhat[0], v[1] + (capped - t) * rhat[1]]
}

/// Ours-family filter, with toggles to form the two ablations:
/// `evade` = active velocity-aware repulsion; `capit` = worst-case tighten-only cap.
fn ours_family(
    p: [f32; 2],
    vdes: [f32; 2],
    op: &[[f32; 2]; N_OBS],
    ov: &[[f32; 2]; N_OBS],
    umax: f32,
    evade: bool,
    capit: bool,
) -> [f32; 2] {
    let mut vc = vdes;
    if evade {
        let mut rep = [0.0f32, 0.0];
        for i in 0..N_OBS {
            let r = [op[i][0] - p[0], op[i][1] - p[1]];
            let d = norm(r);
            if d > SENSE {
                continue;
            }
            let rhat = [r[0] / d.max(1e-3), r[1] / d.max(1e-3)];
            let gap = (d - R_COLL).max(0.0);
            if gap < DANGER {
                let closing = (-dot(ov[i], rhat)).max(0.0);
                let w = ((DANGER - gap) / DANGER) * (1.0 + closing / umax);
                rep[0] -= rhat[0] * w;
                rep[1] -= rhat[1] * w;
            }
        }
        let blended = [
            vdes[0] + rep[0] * REP_GAIN * V_MAX,
            vdes[1] + rep[1] * REP_GAIN * V_MAX,
        ];
        vc = unit_to(blended, V_MAX);
    }
    if capit {
        for _pass in 0..5 {
            for i in 0..N_OBS {
                let r = [op[i][0] - p[0], op[i][1] - p[1]];
                let d = norm(r);
                if d > SENSE {
                    continue;
                }
                let rhat = [r[0] / d.max(1e-3), r[1] / d.max(1e-3)];
                let closing = (-dot(ov[i], rhat)).max(0.0).min(umax);
                // worst-case safe approach toward this obstacle (machine-checked bound)
                vc = cap(vc, rhat, safe_approach(d - R_COLL, A_MAX, closing, MARGIN));
            }
        }
    }
    vc
}

/// One world (same seed → same obstacle world for every method). Returns (collisions, distance).
fn run(method: Method, adv: Adv, seed: u64, umax: f32, dynamics: Dyn) -> (u32, f32) {
    let mut rng = Rng::new(seed);
    let mut p = [-ARENA * 0.8, 0.0f32];
    let mut v = [0.0f32, 0.0];
    let mut theta = 0.0f32; // robot heading (unicycle dynamics), starts toward the +x goal
    let mut goal = [ARENA * 0.8, 0.0f32];
    let mut op = [[0.0f32; 2]; N_OBS];
    let mut ov = [[0.0f32; 2]; N_OBS];
    let mut timer = [0usize; N_OBS];
    for i in 0..N_OBS {
        op[i] = [rng.span(-ARENA, ARENA), rng.span(-ARENA, ARENA)];
        let d = rng.dir();
        let s = rng.span(umax * 0.6, umax);
        ov[i] = [d[0] * s, d[1] * s];
        timer[i] = (rng.raw() as usize) % REPICK;
    }
    let (mut coll, mut dist) = (0u32, 0.0f32);
    let mut contact = [false; N_OBS];
    // LearnedCbf state: a control-barrier margin LEARNED online from collision feedback.
    let mut learned_margin = 0.5f32;
    let mut safe_streak = 0u32;

    for _ in 0..STEPS {
        let g = [goal[0] - p[0], goal[1] - p[1]];
        let vdes = unit_to(g, V_MAX);
        let mut vcmd = vdes;

        match method {
            Method::Nominal => {}
            Method::Simplex => {
                let (mut best, mut bi) = (f32::MAX, 0usize);
                for (i, o) in op.iter().enumerate() {
                    let d = norm([o[0] - p[0], o[1] - p[1]]);
                    if d < best {
                        best = d;
                        bi = i;
                    }
                }
                if best - R_COLL < D_SWITCH {
                    vcmd = unit_to([p[0] - op[bi][0], p[1] - op[bi][1]], V_MAX);
                    // retreat
                }
            }
            Method::Cbf => {
                for _pass in 0..5 {
                    for o in op.iter() {
                        let r = [o[0] - p[0], o[1] - p[1]];
                        let d = norm(r);
                        if d > SENSE {
                            continue;
                        }
                        let rhat = [r[0] / d.max(1e-3), r[1] / d.max(1e-3)];
                        vcmd = cap(vcmd, rhat, ALPHA * (d - R_COLL)); // distance-only
                    }
                }
            }
            Method::LearnedCbf => {
                // adaptive/learning CBF: a distance-only barrier whose safety margin is LEARNED
                // online from collision feedback. Represents the learning/adaptive-CBF class
                // [12-15]; it tightens after a near-miss but *relaxes* after sustained safe travel
                // (the standard "grow confidence" direction, opposite to our tighten-only rule).
                for _pass in 0..5 {
                    for o in op.iter() {
                        let r = [o[0] - p[0], o[1] - p[1]];
                        let d = norm(r);
                        if d > SENSE {
                            continue;
                        }
                        let rhat = [r[0] / d.max(1e-3), r[1] / d.max(1e-3)];
                        vcmd = cap(vcmd, rhat, ALPHA * (d - R_COLL - learned_margin));
                    }
                }
            }
            Method::FrontOnly => {
                for _pass in 0..5 {
                    for o in op.iter() {
                        let r = [o[0] - p[0], o[1] - p[1]];
                        let d = norm(r);
                        if d > SENSE {
                            continue;
                        }
                        let rhat = [r[0] / d.max(1e-3), r[1] / d.max(1e-3)];
                        let vn = norm(vcmd);
                        if vn > 1e-3 && dot([vcmd[0] / vn, vcmd[1] / vn], rhat) > FRONT_COS {
                            vcmd = cap(vcmd, rhat, safe_approach(d - R_COLL, A_MAX, umax, MARGIN));
                        }
                    }
                }
            }
            Method::CbfVel => vcmd = ours_family(p, vdes, &op, &ov, umax, false, true),
            Method::EvadeOnly => vcmd = ours_family(p, vdes, &op, &ov, umax, true, false),
            Method::Ours => vcmd = ours_family(p, vdes, &op, &ov, umax, true, true),
        }

        // integrate under the selected dynamics model (the same filters drive both)
        match dynamics {
            Dyn::Holonomic => {
                // omnidirectional, acceleration-limited tracking + speed clamp
                let dv = [vcmd[0] - v[0], vcmd[1] - v[1]];
                let dn = norm(dv);
                let amax = A_MAX * DT;
                if dn > amax {
                    v = [v[0] + dv[0] / dn * amax, v[1] + dv[1] / dn * amax];
                } else {
                    v = vcmd;
                }
                let sp = norm(v);
                if sp > V_MAX {
                    v = [v[0] / sp * V_MAX, v[1] / sp * V_MAX];
                }
            }
            Dyn::Unicycle => {
                // non-holonomic: steer toward the commanded direction, drive forward when aligned
                let speed = norm(vcmd).min(V_MAX);
                let want = if norm(vcmd) > 1e-3 {
                    atan2f(vcmd[1], vcmd[0])
                } else {
                    theta
                };
                let mut herr = want - theta;
                while herr > PI {
                    herr -= TAU;
                }
                while herr < -PI {
                    herr += TAU;
                }
                theta += herr.clamp(-OMEGA_MAX * DT, OMEGA_MAX * DT);
                let fwd = speed * cosf(herr).max(0.0); // less forward speed while turning
                v = [cosf(theta) * fwd, sinf(theta) * fwd];
            }
        }
        p = [p[0] + v[0] * DT, p[1] + v[1] * DT];
        dist += norm(v) * DT;
        if norm([goal[0] - p[0], goal[1] - p[1]]) < 1.0 {
            goal = [-goal[0], 0.0];
        }

        // adversary moves
        for i in 0..N_OBS {
            if timer[i] == 0 {
                let mut d = rng.dir();
                if adv == Adv::Homing {
                    let r = [p[0] - op[i][0], p[1] - op[i][1]];
                    let rn = norm(r).max(1e-3);
                    d = [0.6 * r[0] / rn + 0.4 * d[0], 0.6 * r[1] / rn + 0.4 * d[1]];
                    d = unit_to(d, 1.0);
                }
                let s = rng.span(umax * 0.6, umax);
                ov[i] = [d[0] * s, d[1] * s];
                timer[i] = REPICK;
            } else {
                timer[i] -= 1;
            }
            op[i] = [op[i][0] + ov[i][0] * DT, op[i][1] + ov[i][1] * DT];
            for k in 0..2 {
                if op[i][k] > ARENA {
                    op[i][k] = ARENA;
                    ov[i][k] = -ov[i][k].abs();
                }
                if op[i][k] < -ARENA {
                    op[i][k] = -ARENA;
                    ov[i][k] = ov[i][k].abs();
                }
            }
        }
        let mut near_miss = false;
        for i in 0..N_OBS {
            let dd = norm([op[i][0] - p[0], op[i][1] - p[1]]);
            let c = dd < R_COLL;
            if c && !contact[i] {
                coll += 1;
            }
            contact[i] = c;
            if dd < R_COLL + 0.6 {
                near_miss = true;
            }
        }
        // LearnedCbf online adaptation: tighten the learned margin on a near-miss; relax it after a
        // sustained safe run (confidence-based relaxation, the learning-CBF direction).
        if method == Method::LearnedCbf {
            if near_miss {
                learned_margin = (learned_margin + 0.06).min(3.0);
                safe_streak = 0;
            } else {
                safe_streak += 1;
                if safe_streak > 100 {
                    learned_margin = (learned_margin - 0.02).max(0.0);
                    safe_streak = 0;
                }
            }
        }
    }
    (coll, dist)
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Per-seed collisions and distance for one (method, adversary, umax, dynamics).
fn sweep_perseed(
    method: Method,
    adv: Adv,
    umax: f32,
    dynamics: Dyn,
) -> ([f32; SEEDS], [f32; SEEDS]) {
    let mut c = [0.0f32; SEEDS];
    let mut d = [0.0f32; SEEDS];
    for s in 0..SEEDS {
        let (cc, dd) = run(method, adv, s as u64, umax, dynamics);
        c[s] = cc as f32;
        d[s] = dd;
    }
    (c, d)
}
fn sweep(method: Method, adv: Adv, umax: f32, dynamics: Dyn) -> (f32, f32) {
    let (c, d) = sweep_perseed(method, adv, umax, dynamics);
    (mean(&c), mean(&d))
}

// ---------- statistics ----------
fn normal_cdf(z: f32) -> f32 {
    0.5 * (1.0 + erff(z / std::f32::consts::SQRT_2))
}
/// Mean and 95% CI (normal approximation).
fn mean_ci(x: &[f32]) -> (f32, f32, f32) {
    let n = x.len() as f32;
    let m = mean(x);
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / (n - 1.0).max(1.0);
    let se = sqrtf(var / n);
    (m, m - 1.96 * se, m + 1.96 * se)
}
/// Two-sided Wilcoxon signed-rank p (normal approximation, continuity correction). Pairs `a`,`b`.
fn wilcoxon_p(a: &[f32], b: &[f32]) -> f32 {
    let diffs: Vec<f32> = a
        .iter()
        .zip(b)
        .map(|(x, y)| y - x)
        .filter(|d| d.abs() > 1e-9)
        .collect();
    let n = diffs.len();
    if n == 0 {
        return 1.0;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| diffs[i].abs().partial_cmp(&diffs[j].abs()).unwrap());
    let mut ranks = vec![0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (diffs[idx[j + 1]].abs() - diffs[idx[i]].abs()).abs() < 1e-6 {
            j += 1;
        }
        let avg = ((i + 1 + j + 1) as f32) / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let w_plus: f32 = (0..n).filter(|&k| diffs[k] > 0.0).map(|k| ranks[k]).sum();
    let nn = n as f32;
    let mean_w = nn * (nn + 1.0) / 4.0;
    let sd_w = sqrtf(nn * (nn + 1.0) * (2.0 * nn + 1.0) / 24.0);
    let z = ((w_plus - mean_w).abs() - 0.5).max(0.0) / sd_w;
    (2.0 * (1.0 - normal_cdf(z))).max(1e-12)
}
/// Cliff's delta in [-1,1]; +1 ⇒ `a` is always less than `b` (here: Ours always fewer collisions).
fn cliffs_delta(a: &[f32], b: &[f32]) -> f32 {
    let (mut less, mut greater) = (0i64, 0i64);
    for &x in a {
        for &y in b {
            if x < y {
                less += 1;
            } else if x > y {
                greater += 1;
            }
        }
    }
    (less - greater) as f32 / (a.len() * b.len()) as f32
}
fn safer_in(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x < y).count()
}

const N_ROBOTS: usize = 4; // multi-agent: robots cross to opposite sides (mutual avoidance)
const K_MA_OBS: usize = 4; // + homing obstacles (3 other robots + 4 obstacles = 7 <= N_OBS)

/// Multi-agent world: `N_ROBOTS` robots each run `method`, crossing to the opposite side while
/// avoiding each other AND `K_MA_OBS` homing obstacles. Returns (total collisions, mean distance).
fn run_multiagent(method: Method, seed: u64) -> (u32, f32) {
    let mut rng = Rng::new(seed ^ 0x4D41);
    let starts = [
        [-ARENA * 0.8, 0.0f32],
        [ARENA * 0.8, 0.0],
        [0.0, -ARENA * 0.8],
        [0.0, ARENA * 0.8],
    ];
    let mut rp = starts;
    let mut rv = [[0.0f32; 2]; N_ROBOTS];
    let mut rg = [[0.0f32; 2]; N_ROBOTS];
    let mut rmargin = [0.5f32; N_ROBOTS];
    for i in 0..N_ROBOTS {
        rg[i] = [-starts[i][0], -starts[i][1]];
    }
    let mut op = [[0.0f32; 2]; K_MA_OBS];
    let mut ov = [[0.0f32; 2]; K_MA_OBS];
    let mut timer = [0usize; K_MA_OBS];
    for i in 0..K_MA_OBS {
        op[i] = [rng.span(-ARENA, ARENA), rng.span(-ARENA, ARENA)];
        let d = rng.dir();
        let s = rng.span(U_MAX * 0.6, U_MAX);
        ov[i] = [d[0] * s, d[1] * s];
        timer[i] = (rng.raw() as usize) % REPICK;
    }
    let mut coll = 0u32;
    let mut dist = 0.0f32;
    let mut crr = [[false; N_ROBOTS]; N_ROBOTS];
    let mut cro = [[false; K_MA_OBS]; N_ROBOTS];

    for _ in 0..STEPS {
        let mut newv = [[0.0f32; 2]; N_ROBOTS];
        for r in 0..N_ROBOTS {
            // combined hazard array: other robots + obstacles (unused slots stay far away)
            let mut hop = [[1.0e6f32; 2]; N_OBS];
            let mut hov = [[0.0f32; 2]; N_OBS];
            let mut k = 0;
            for (j, rpj) in rp.iter().enumerate() {
                if j != r {
                    hop[k] = *rpj;
                    hov[k] = rv[j];
                    k += 1;
                }
            }
            for j in 0..K_MA_OBS {
                hop[k] = op[j];
                hov[k] = ov[j];
                k += 1;
            }
            let g = [rg[r][0] - rp[r][0], rg[r][1] - rp[r][1]];
            let vdes = unit_to(g, V_MAX);
            let vcmd = match method {
                Method::Ours => ours_family(rp[r], vdes, &hop, &hov, U_MAX, true, true),
                Method::Simplex => {
                    let (mut best, mut bi) = (f32::MAX, 0usize);
                    for (i, o) in hop.iter().enumerate() {
                        let d = norm([o[0] - rp[r][0], o[1] - rp[r][1]]);
                        if d < best {
                            best = d;
                            bi = i;
                        }
                    }
                    if best - R_COLL < D_SWITCH {
                        unit_to([rp[r][0] - hop[bi][0], rp[r][1] - hop[bi][1]], V_MAX)
                    } else {
                        vdes
                    }
                }
                _ => {
                    // CBF / LearnedCBF: distance-only barrier (LearnedCBF adds the learned margin)
                    let m = if method == Method::LearnedCbf {
                        rmargin[r]
                    } else {
                        0.0
                    };
                    let mut vc = vdes;
                    for _pass in 0..5 {
                        for o in hop.iter() {
                            let rr = [o[0] - rp[r][0], o[1] - rp[r][1]];
                            let d = norm(rr);
                            if d > SENSE {
                                continue;
                            }
                            let rhat = [rr[0] / d.max(1e-3), rr[1] / d.max(1e-3)];
                            vc = cap(vc, rhat, ALPHA * (d - R_COLL - m));
                        }
                    }
                    vc
                }
            };
            let dv = [vcmd[0] - rv[r][0], vcmd[1] - rv[r][1]];
            let dn = norm(dv);
            let am = A_MAX * DT;
            let mut nv = if dn > am {
                [rv[r][0] + dv[0] / dn * am, rv[r][1] + dv[1] / dn * am]
            } else {
                vcmd
            };
            let sp = norm(nv);
            if sp > V_MAX {
                nv = [nv[0] / sp * V_MAX, nv[1] / sp * V_MAX];
            }
            newv[r] = nv;
        }
        for r in 0..N_ROBOTS {
            rv[r] = newv[r];
            rp[r] = [rp[r][0] + rv[r][0] * DT, rp[r][1] + rv[r][1] * DT];
            dist += norm(rv[r]) * DT;
            if norm([rg[r][0] - rp[r][0], rg[r][1] - rp[r][1]]) < 1.0 {
                rg[r] = [-rg[r][0], -rg[r][1]];
            }
            if method == Method::LearnedCbf {
                let mut nm = false;
                for j in 0..N_ROBOTS {
                    if j != r && norm([rp[j][0] - rp[r][0], rp[j][1] - rp[r][1]]) < R_COLL + 0.6 {
                        nm = true;
                    }
                }
                for opj in &op {
                    if norm([opj[0] - rp[r][0], opj[1] - rp[r][1]]) < R_COLL + 0.6 {
                        nm = true;
                    }
                }
                rmargin[r] = if nm {
                    (rmargin[r] + 0.06).min(3.0)
                } else {
                    (rmargin[r] - 0.005).max(0.0)
                };
            }
        }
        // obstacles home toward the nearest robot
        for i in 0..K_MA_OBS {
            if timer[i] == 0 {
                let (mut best, mut bp) = (f32::MAX, rp[0]);
                for rpj in rp.iter() {
                    let d = norm([rpj[0] - op[i][0], rpj[1] - op[i][1]]);
                    if d < best {
                        best = d;
                        bp = *rpj;
                    }
                }
                let toward = unit_to([bp[0] - op[i][0], bp[1] - op[i][1]], 1.0);
                let rd = rng.dir();
                let d = unit_to(
                    [0.6 * toward[0] + 0.4 * rd[0], 0.6 * toward[1] + 0.4 * rd[1]],
                    1.0,
                );
                let s = rng.span(U_MAX * 0.6, U_MAX);
                ov[i] = [d[0] * s, d[1] * s];
                timer[i] = REPICK;
            } else {
                timer[i] -= 1;
            }
            op[i] = [op[i][0] + ov[i][0] * DT, op[i][1] + ov[i][1] * DT];
            for kk in 0..2 {
                if op[i][kk].abs() > ARENA {
                    op[i][kk] = op[i][kk].clamp(-ARENA, ARENA);
                    ov[i][kk] = -ov[i][kk];
                }
            }
        }
        // collisions: robot-robot (each pair once) + robot-obstacle, rising edge
        for r in 0..N_ROBOTS {
            for j in (r + 1)..N_ROBOTS {
                let c = norm([rp[j][0] - rp[r][0], rp[j][1] - rp[r][1]]) < R_COLL;
                if c && !crr[r][j] {
                    coll += 1;
                }
                crr[r][j] = c;
            }
            for j in 0..K_MA_OBS {
                let c = norm([op[j][0] - rp[r][0], op[j][1] - rp[r][1]]) < R_COLL;
                if c && !cro[r][j] {
                    coll += 1;
                }
                cro[r][j] = c;
            }
        }
    }
    (coll, dist / N_ROBOTS as f32)
}

const METHODS: [(&str, Method); 8] = [
    ("Nominal", Method::Nominal),
    ("CBF", Method::Cbf),
    ("LearnedCBF", Method::LearnedCbf),
    ("VO-CBF", Method::CbfVel),
    ("Simplex", Method::Simplex),
    ("FrontOnly", Method::FrontOnly),
    ("Evade-only", Method::EvadeOnly),
    ("Ours", Method::Ours),
];

fn main() {
    println!(
        "# sil-adversarial — 2-D arena, {N_OBS} fast random multi-directional obstacles (V_MAX={V_MAX}, A_MAX={A_MAX})"
    );
    println!("# collisions = mean over {SEEDS} paired seeds (lower=better); distance = liveness");

    for (an, adv) in [
        ("RANDOM (non-homing)", Adv::Random),
        ("HOMING (adaptive adversary)", Adv::Homing),
    ] {
        println!("\n## Adversary: {an}  [obstacle U_MAX={U_MAX}]");
        println!(
            "   {:<12} {:>11} {:>10}",
            "method", "collisions", "distance"
        );
        for (mn, m) in METHODS {
            let (c, d) = sweep(m, adv, U_MAX, Dyn::Holonomic);
            println!("   {mn:<12} {c:>11.2} {d:>10.0}");
        }
    }

    println!(
        "\n## Robustness vs obstacle speed — HOMING adversary (collisions, mean/{SEEDS} seeds)"
    );
    print!("   {:<8}", "U_MAX");
    for (mn, _) in METHODS {
        print!(" {mn:>10}");
    }
    println!();
    for &umax in &[2.0f32, 2.4, 2.8, 3.2, 3.6] {
        let faster = if umax > V_MAX { "  (>robot)" } else { "" };
        print!("   {umax:<8.1}");
        for (_, m) in METHODS {
            let (c, _) = sweep(m, Adv::Homing, umax, Dyn::Holonomic);
            print!(" {c:>10.2}");
        }
        println!("{faster}");
    }

    // Statistical analysis: Ours vs each baseline on the headline scenario (Homing, U_MAX=2.8).
    println!("\n## Statistics — HOMING, U_MAX={U_MAX}: Ours vs each baseline (collisions, {SEEDS} paired seeds)");
    let (ours_c, ours_d) = sweep_perseed(Method::Ours, Adv::Homing, U_MAX, Dyn::Holonomic);
    let (om, olo, ohi) = mean_ci(&ours_c);
    println!(
        "   Ours collisions: mean {om:.2} (95% CI {:.2}..{ohi:.2})",
        olo.max(0.0)
    );
    println!(
        "   {:<12} {:>14} {:>10} {:>10} {:>10}",
        "baseline", "mean(95%CI)", "safer/40", "Cliff δ", "Wilcoxon p"
    );
    for (mn, m) in [
        ("CBF", Method::Cbf),
        ("LearnedCBF", Method::LearnedCbf),
        ("VO-CBF", Method::CbfVel),
        ("Simplex", Method::Simplex),
        ("FrontOnly", Method::FrontOnly),
        ("Evade-only", Method::EvadeOnly),
    ] {
        let (bc, _) = sweep_perseed(m, Adv::Homing, U_MAX, Dyn::Holonomic);
        let (bm, blo, bhi) = mean_ci(&bc);
        let s = safer_in(&ours_c, &bc);
        let delta = cliffs_delta(&ours_c, &bc);
        let p = wilcoxon_p(&ours_c, &bc);
        println!(
            "   {mn:<12} {:>14} {s:>8}/40 {delta:>10.3} {p:>10.1e}",
            format!("{bm:.2}({:.1}-{bhi:.1})", blo.max(0.0))
        );
    }
    // Liveness: Ours vs the only safe baseline (Simplex).
    let (simp_c, simp_d) = sweep_perseed(Method::Simplex, Adv::Homing, U_MAX, Dyn::Holonomic);
    let (odm, odlo, odhi) = mean_ci(&ours_d);
    let (sdm, sdlo, sdhi) = mean_ci(&simp_d);
    let _ = simp_c;
    println!(
        "   Liveness (distance): Ours {odm:.0} ({odlo:.0}..{odhi:.0}) vs Simplex {sdm:.0} ({sdlo:.0}..{sdhi:.0}) — Ours travels {:.0}% farther",
        (odm / sdm - 1.0) * 100.0
    );

    // CROSS-SIMULATOR: replicate the headline comparison under a *different dynamics model* —
    // a non-holonomic unicycle (the robot must turn to change direction). Same filters, same worlds.
    println!("\n## Cross-simulator — UNICYCLE (non-holonomic) dynamics, HOMING adversary [U_MAX={U_MAX}]");
    println!(
        "   {:<12} {:>11} {:>10}",
        "method", "collisions", "distance"
    );
    for (mn, m) in METHODS {
        let (c, d) = sweep(m, Adv::Homing, U_MAX, Dyn::Unicycle);
        println!("   {mn:<12} {c:>11.2} {d:>10.0}");
    }
    let (uo_c, _) = sweep_perseed(Method::Ours, Adv::Homing, U_MAX, Dyn::Unicycle);
    for (mn, m) in [
        ("CBF", Method::Cbf),
        ("LearnedCBF", Method::LearnedCbf),
        ("Simplex", Method::Simplex),
    ] {
        let (b, _) = sweep_perseed(m, Adv::Homing, U_MAX, Dyn::Unicycle);
        println!(
            "   [unicycle] Ours vs {mn:<10}: safer {:>2}/40, Cliff δ {:.3}, Wilcoxon p {:.1e}",
            safer_in(&uo_c, &b),
            cliffs_delta(&uo_c, &b),
            wilcoxon_p(&uo_c, &b)
        );
    }

    // MULTI-AGENT: N robots cross to opposite sides (mutual reciprocal avoidance) + homing obstacles.
    println!("\n## Multi-agent — {N_ROBOTS} robots crossing + {K_MA_OBS} homing obstacles (mean/{SEEDS} seeds)");
    println!(
        "   {:<12} {:>12} {:>10}",
        "method", "total-coll", "distance"
    );
    for (mn, m) in [
        ("CBF", Method::Cbf),
        ("LearnedCBF", Method::LearnedCbf),
        ("Simplex", Method::Simplex),
        ("Ours", Method::Ours),
    ] {
        let mut c = [0.0f32; SEEDS];
        let mut d = [0.0f32; SEEDS];
        for s in 0..SEEDS {
            let (cc, dd) = run_multiagent(m, s as u64);
            c[s] = cc as f32;
            d[s] = dd;
        }
        println!("   {mn:<12} {:>12.2} {:>10.0}", mean(&c), mean(&d));
    }

    println!("\n[adversarial] done.");
}
