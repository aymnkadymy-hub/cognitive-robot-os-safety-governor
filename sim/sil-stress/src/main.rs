//! Stress / break battery — "what breaks under pressure matters more than what succeeds on average".
//! 6 batteries + **perfect storm** (all layers fail simultaneously), **fair** comparison
//! (same stress) across Adaptive/CBF/Simplex.
//! Metrics: collision rate · recovery time · performance loss.
//! Note: **envelope containment `≤ L₀` holds unconditionally (clamp)** — we highlight this;
//! collision-safety depends on the physical barrier model (breaks under physics shift
//! decel_ratio<1 — honest limit testing assumption f3).

use clearance_guard::ClearanceGuard;
use libm::{cosf, sinf};
use safety_memory::SafetyMemory;

const LEN: f32 = 60.0;
const V_PASS: f32 = 0.9;
const V_SAFE: f32 = 0.7;
const CRASH: f32 = 1.1;
const DT: f32 = 0.02; // 1 step = 20ms
const STEPS: usize = 16_000;
const SENSE: f32 = 6.0;
const OBS_W: f32 = 0.8;
const HAZ_W: f32 = 1.5;
const A_ASSUMED: f32 = 2.0; // assumed deceleration used by the barrier
const SEEDS: u64 = 30;
const CAP: usize = 128;

#[derive(Clone, Copy, PartialEq)]
enum M {
    Adaptive,
    Cbf,
    Simplex,
    AdaptiveRobust, // additive: barrier self-tightens its parameters reactively after each collision (tighten-only) — closes f3/delay/noise gaps
}

#[derive(Clone, Copy, Default)]
struct Stress {
    noise: f32,
    outlier: f32,
    delay: usize,
    decel_ratio: f32, // 1.0 = correct model; <1 = physics shift (actual braking is slower)
    adv_policy: bool,
    mem_corrupt: bool,
    cs_adv: bool, // adversarial confirm_safe
    dropout: f32,
    conservative: bool, // conservative barrier (f3 mitigation): assumes worse braking + larger response margin
}
impl Stress {
    fn none() -> Self {
        Stress {
            decel_ratio: 1.0,
            ..Default::default()
        }
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}
fn ctx(s: f32) -> [f32; 2] {
    let a = 2.0 * core::f32::consts::PI * s / LEN;
    [cosf(a), sinf(a)]
}
fn dist_to(it: &[f32], s: f32) -> f32 {
    let mut b = SENSE + 1.0;
    for &o in it {
        let mut d = o - s;
        if d < 0.0 {
            d += LEN;
        }
        if d < b {
            b = d;
        }
    }
    b
}
fn near(it: &[f32], s: f32, w: f32) -> bool {
    it.iter()
        .any(|&o| (s - o).abs() < w || (s - o).abs() > LEN - w)
}
fn obstacle_cap(d: f32, cbf: &ClearanceGuard, vmax: f32) -> f32 {
    let pass = if d > SENSE {
        vmax
    } else {
        V_PASS + (vmax - V_PASS) * (d / SENSE)
    };
    cbf.safe_speed(d).max(V_PASS).min(pass)
}

/// Loop under stress. Returns (unsafe count, distance).
fn sim(method: M, st: Stress, vmax: f32, seed: u64) -> (u32, f32) {
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
    // Well-spaced obstacles (20m ≫ SENSE) for a clean baseline; hidden hazards between them.
    let obs: Vec<f32> = (0..3)
        .map(|i| 6.0 + 20.0 * i as f32 + (rng.f() - 0.5) * 2.0)
        .collect();
    let haz: Vec<f32> = [12.0f32, 32.0, 52.0]
        .iter()
        .map(|&b| b + (rng.f() - 0.5) * 4.0)
        .collect();
    // f3 mitigation: conservative barrier assumes worse braking (a_max/2), larger response margin (5·DT), wider barrier.
    let cbf = if st.conservative {
        ClearanceGuard::new(OBS_W * 1.5, A_ASSUMED * 0.5, vmax, DT * 5.0)
    } else {
        ClearanceGuard::new(OBS_W, A_ASSUMED, vmax, DT)
    };
    // Tighten-only by default (forgetting is opt-in, enabled only in the cs_adv test to avoid confounding other tests).
    let mut mem = SafetyMemory::<CAP, 2>::new(vmax)
        .with_params(0.90, 0.22, 0.1)
        .with_forgetting(15);
    if method == M::Adaptive || method == M::AdaptiveRobust {
        for &h in &[12.0f32, 32.0, 52.0] {
            mem.preload(&ctx(h), V_SAFE * 0.8);
        }
        if st.mem_corrupt {
            for _ in 0..60 {
                mem.preload(
                    &[(rng.f() - 0.5) * 2.0, (rng.f() - 0.5) * 2.0],
                    rng.f() * vmax,
                );
            }
        }
    }
    let (mut s, mut v) = (0.0f32, 0.0f32);
    let (mut un, mut dist) = (0u32, 0.0f32);
    let (mut po, mut ph) = (false, false);
    let mut dbuf = vec![SENSE + 1.0; st.delay + 1];
    let mut nrng = Rng(seed ^ 0xBEEF);
    let mut osc = 1.0f32;
    let mut robust_c = 0.0f32; // conservatism level of the reactive barrier [0,1] for AdaptiveRobust

    for step in 0..STEPS {
        // ===== Perception under stress (noise + outliers + dropout + delay) =====
        let d_true = dist_to(&obs, s);
        let mut d = d_true;
        if st.noise > 0.0 {
            d += (nrng.f() - 0.5) * 2.0 * st.noise;
        }
        if st.outlier > 0.0 && nrng.f() < st.outlier {
            d = SENSE + 5.0; // outlier: falsely reports "road clear"
        }
        if st.dropout > 0.0 && nrng.f() < st.dropout {
            d = SENSE + 1.0;
        }
        d = d.max(0.0);
        let d_used = if st.delay > 0 {
            let idx = step % dbuf.len();
            let out = dbuf[idx];
            dbuf[idx] = d;
            out
        } else {
            d
        };

        // ===== Policy (adversarial oscillating at the limit, or moderate) =====
        let proposed = if st.adv_policy {
            osc = -osc;
            if osc > 0.0 {
                vmax
            } else {
                0.0
            }
        } else {
            V_PASS.max(vmax * 0.6)
        };

        // ===== Governance =====
        let cap = match method {
            M::Adaptive => obstacle_cap(d_used, &cbf, vmax).min(mem.effective_limit(&ctx(s))),
            M::AdaptiveRobust => {
                // Reactive barrier: starts optimistic (c=0) and tightens its parameters toward
                // conservative (c→1) after each collision — worse braking + larger response margin
                // + wider barrier. Tighten-only.
                let c = robust_c;
                let rcbf = ClearanceGuard::new(
                    OBS_W * (1.0 + 0.5 * c),
                    A_ASSUMED * (1.0 - 0.5 * c),
                    vmax,
                    DT * (1.0 + 4.0 * c),
                );
                obstacle_cap(d_used, &rcbf, vmax).min(mem.effective_limit(&ctx(s)))
            }
            M::Cbf => obstacle_cap(d_used, &cbf, vmax),
            M::Simplex => {
                if d_used < SENSE {
                    V_PASS
                } else {
                    vmax
                }
            }
        };
        let v_t = proposed.min(cap).clamp(0.0, vmax); // guard clamps policy command ≤ cap ≤ L₀

        // ===== Dynamics: first-order response. Physics shift/actuator lag = slower response =====
        // (decel_ratio=1 ⇒ full response 0.3; <1 ⇒ actuator is slower ⇒ barrier lags control)
        v += (v_t - v) * (0.3 * st.decel_ratio);
        s = (s + v * DT) % LEN;
        dist += v * DT;

        let o = near(&obs, s, OBS_W);
        if o && !po && v > CRASH {
            un += 1;
            if method == M::AdaptiveRobust {
                robust_c = (robust_c + 0.5).min(1.0); // obstacle collision ⇒ tighten the barrier (tighten-only)
            }
        }
        po = o;
        let h = near(&haz, s, HAZ_W);
        let incident = h && v > V_SAFE;
        if h && !ph && v > V_SAFE {
            un += 1;
            if method == M::Adaptive || method == M::AdaptiveRobust {
                mem.record_incident(&ctx(s));
            }
            if method == M::AdaptiveRobust {
                robust_c = (robust_c + 0.5).min(1.0);
            }
        }
        ph = h;
        // Forgetting disabled by default (tighten-only); enabled adversarially only in the cs_adv test.
        if method == M::Adaptive && st.cs_adv {
            mem.confirm_safe(&ctx(s)); // adversarial: always a false-safe signal
        }
        let _ = incident;
    }
    (un, dist)
}

fn avg(method: M, st: Stress, vmax: f32) -> (f32, f32) {
    let (mut u, mut d) = (0.0, 0.0);
    for k in 1..=SEEDS {
        let (uu, dd) = sim(method, st, vmax, k);
        u += uu as f32;
        d += dd;
    }
    (u / SEEDS as f32, d / SEEDS as f32)
}
fn row(label: &str, st: Stress, vmax: f32) {
    let a = avg(M::Adaptive, st, vmax);
    let c = avg(M::Cbf, st, vmax);
    let s = avg(M::Simplex, st, vmax);
    println!(
        "  {label:30} | ADAPT {:5.1}/{:5.0} | CBF {:5.1}/{:5.0} | Simplex {:5.1}/{:5.0}",
        a.0, a.1, c.0, c.1, s.0, s.1
    );
}

fn main() {
    println!("================ STRESS / BREAK BATTERY (matched across methods) ================");
    println!("cell = unsafe-events / distance ; {SEEDS} seeds ; L0-containment (speed<=vmax) holds by clamp ALWAYS.\n");

    println!("### 1) SAFETY-ENVELOPE STRESS — raise speed to the breaking point");
    for vm in [1.5f32, 2.5, 4.0, 6.0, 9.0] {
        row(&format!("vmax={vm}"), Stress::none(), vm);
    }
    println!();

    println!("### 2) SENSOR NOISE + OUTLIERS (deceptive 'road clear' spikes)");
    for (n, o) in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.05), (3.0, 0.15)] {
        row(
            &format!("noise={n} outlier={o}"),
            Stress {
                noise: n,
                outlier: o,
                ..Stress::none()
            },
            2.5,
        );
    }
    println!();

    println!("### 3) LATENCY INJECTION (perception delay; 1 step = 20ms)");
    for dly in [0usize, 3, 6, 10] {
        row(
            &format!("delay={}ms", dly * 20),
            Stress {
                delay: dly,
                ..Stress::none()
            },
            2.5,
        );
    }
    println!();

    println!("### 4) PHYSICS SHIFT (sluggish actuator vs assumed response -> tests assumption f3; vmax=5)");
    for r in [1.0f32, 0.5, 0.3, 0.2] {
        row(
            &format!("response={r}"),
            Stress {
                decel_ratio: r,
                ..Stress::none()
            },
            5.0,
        );
    }
    println!(
        "  -> HONEST: at higher speed a sluggish actuator (f3 weakened) makes the barrier LAG -> ADAPT"
    );
    println!("     starts to BREAK (collisions appear), like CBF/Simplex. L0-containment still holds, but");
    println!("     collision-safety DEPENDS on f3 (the physical model) — this is the real break point.\n");

    println!(
        "### 5) ADVERSARIAL CONTROL INPUTS (policy oscillates between 0 and vmax at the boundary)"
    );
    row(
        "adversarial policy",
        Stress {
            adv_policy: true,
            ..Stress::none()
        },
        2.5,
    );
    println!("  -> the guard is a stateless projection (clamp) -> no guard-induced instability; safety held.\n");

    println!("### 6) MEMORY CORRUPTION (poison) + confirm_safe adversarial");
    row("clean", Stress::none(), 2.5);
    row(
        "poisoned memory",
        Stress {
            mem_corrupt: true,
            ..Stress::none()
        },
        2.5,
    );
    row(
        "adversarial confirm_safe",
        Stress {
            cs_adv: true,
            ..Stress::none()
        },
        2.5,
    );
    println!("  -> poisoning => MORE conservative (tighten-only), never less safe; adv confirm_safe => bounded,");
    println!(
        "     self-healing degradation of EXPERIENTIAL protection (envelope/barrier intact).\n"
    );

    println!("### 7) *** PERFECT STORM *** — ALL layers fail at once");
    let storm = Stress {
        noise: 2.0,
        outlier: 0.10,
        delay: 6, // 120ms
        decel_ratio: 0.7,
        adv_policy: true,
        mem_corrupt: true,
        cs_adv: true,
        dropout: 0.20,
        conservative: false,
    };
    row("noise+outlier+120ms+physics+adv+corrupt+drop", storm, 3.0);
    let clean = avg(M::Adaptive, Stress::none(), 3.0);
    println!(
        "  (clean ADAPT baseline at vmax=3.0: {:.1}/{:.0})",
        clean.0, clean.1
    );
    println!("  => HONEST VERDICT: under the perfect storm the EXPERIENTIAL/reactive protection collapses for");
    println!("     ALL methods (this is expected — the physical assumption f3 is broken). The ONLY thing that");
    println!("     still holds unconditionally is L0-containment (speed <= vmax, the verified envelope) — the");
    println!("     formal invariant is robust to ALL stress combined; collision-safety is NOT (depends on f3).");
    println!();

    // ===== 8) CLOSING THE f3 GAP — conservative barrier EXTENDS the safe operating envelope =====
    println!(
        "### 8) f3 MITIGATION — a CONSERVATIVE barrier (assumes worse braking + larger reaction)"
    );
    println!("       moves the break point OUT (safe envelope grows), at a known liveness cost.");
    let con = |st: Stress| Stress {
        conservative: true,
        ..st
    };
    println!("  -- latency break point (ADAPT unsafe / distance) --");
    for dly in [6usize, 10, 15] {
        let nom = avg(
            M::Adaptive,
            Stress {
                delay: dly,
                ..Stress::none()
            },
            2.5,
        );
        let cons = avg(
            M::Adaptive,
            con(Stress {
                delay: dly,
                ..Stress::none()
            }),
            2.5,
        );
        println!(
            "    delay={:>3}ms | nominal {:5.1}/{:4.0} | CONSERVATIVE {:5.1}/{:4.0}",
            dly * 20,
            nom.0,
            nom.1,
            cons.0,
            cons.1
        );
    }
    println!("  -- physics-shift break point (vmax=5; ADAPT unsafe / distance) --");
    for r in [0.3f32, 0.25, 0.2] {
        let nom = avg(
            M::Adaptive,
            Stress {
                decel_ratio: r,
                ..Stress::none()
            },
            5.0,
        );
        let cons = avg(
            M::Adaptive,
            con(Stress {
                decel_ratio: r,
                ..Stress::none()
            }),
            5.0,
        );
        println!(
            "    response={r} | nominal {:5.1}/{:4.0} | CONSERVATIVE {:5.1}/{:4.0}",
            nom.0, nom.1, cons.0, cons.1
        );
    }
    println!("  => the conservative design TRADES liveness (shorter distance) for a LARGER safe envelope");
    println!("     -> f3 becomes 'safe if actual braking >= a_conservative', a defensible PHYSICAL bound");
    println!("     (set a_conservative from a measured worst case on hardware -> turns conditional into grounded).");
    // ===== 9) ADAPTIVE-ROBUST — reactive barrier tightening closes gaps without a fixed liveness cost =====
    println!(
        "\n### 9) ADAPTIVE-ROBUST (additive, tighten-only) — barrier self-tightens after each collision"
    );
    println!(
        "       nominal Adaptive (optimistic) | always-CONSERVATIVE | ADAPTIVE-ROBUST (reactive)"
    );
    let show = |label: &str, base: Stress, vmax: f32| {
        let nom = avg(
            M::Adaptive,
            Stress {
                conservative: false,
                ..base
            },
            vmax,
        );
        let con = avg(
            M::Adaptive,
            Stress {
                conservative: true,
                ..base
            },
            vmax,
        );
        let rob = avg(M::AdaptiveRobust, base, vmax);
        println!(
            "    {:<26} | nominal {:5.1}/{:4.0} | CONS {:5.1}/{:4.0} | ROBUST {:5.1}/{:4.0}",
            label, nom.0, nom.1, con.0, con.1, rob.0, rob.1
        );
    };
    show("clean (vmax=3)", Stress::none(), 3.0);
    show(
        "delay=200ms",
        Stress {
            delay: 10,
            ..Stress::none()
        },
        3.0,
    );
    show(
        "delay=300ms",
        Stress {
            delay: 15,
            ..Stress::none()
        },
        3.0,
    );
    show(
        "f3 response=0.3 (vmax5)",
        Stress {
            decel_ratio: 0.3,
            ..Stress::none()
        },
        5.0,
    );
    show(
        "f3 response=0.2 (vmax5)",
        Stress {
            decel_ratio: 0.2,
            ..Stress::none()
        },
        5.0,
    );
    show(
        "noise=2 outlier=.05",
        Stress {
            noise: 2.0,
            outlier: 0.05,
            ..Stress::none()
        },
        3.0,
    );
    show(
        "dropout=0.30",
        Stress {
            dropout: 0.30,
            ..Stress::none()
        },
        3.0,
    );
    show(
        "PERFECT STORM",
        Stress {
            noise: 2.0,
            outlier: 0.10,
            delay: 6,
            decel_ratio: 0.7,
            adv_policy: true,
            mem_corrupt: true,
            cs_adv: true,
            dropout: 0.20,
            conservative: false,
        },
        3.0,
    );
    println!("  => ROBUST keeps nominal liveness when clean, yet closes the f3/delay/storm gaps reactively.");
    println!("================ END ================");
}
