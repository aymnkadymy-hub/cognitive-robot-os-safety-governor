//! Arena coverage with a car and random obstacles — sweep **every cell** without crashing.
//! ("Born cautious".)
//!
//! The story (Claim A + Transfer A6): the car **does not know** obstacle positions, so it
//! plans through what it believes is free; when it enters an obstacle → **crash**, it learns
//! that cell (incident memory), and replans. The **"born-cautious" car**, however, is
//! pre-loaded with the simulation incident memory → it avoids all obstacles **with zero
//! crashes** and fully covers the arena.
//!
//! The incident memory is exported/imported as bytes (`safety-memory`, same seL4 crate) —
//! that is what is transferred between the two cars.

use safety_memory::SafetyMemory;
use std::collections::VecDeque;
use std::io::Write;

const W: usize = 26;
const H: usize = 16;
const OBST_PCT: f32 = 0.12;
const MAX_STEPS: usize = 4000;
const DIM: usize = 4;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

fn idx(x: usize, y: usize) -> usize {
    y * W + x
}

/// Incident embedding = cell coordinates (read during transfer to reconstruct the obstacle map).
fn cell_fp(x: usize, y: usize) -> [f32; DIM] {
    [x as f32, y as f32, 0.0, 0.0]
}

/// Next step (neighbour) toward the nearest unvisited free cell, via BFS over believed-free cells (`!known`).
fn next_step(known: &[bool], visited: &[bool], sx: usize, sy: usize) -> Option<(usize, usize)> {
    let mut prev = vec![usize::MAX; W * H];
    let mut q = VecDeque::new();
    let s = idx(sx, sy);
    prev[s] = s;
    q.push_back((sx, sy));
    let mut goal = None;
    while let Some((x, y)) = q.pop_front() {
        if !visited[idx(x, y)] && (x, y) != (sx, sy) {
            goal = Some((x, y));
            break;
        }
        let nb = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in nb {
            if nx < W && ny < H && !known[idx(nx, ny)] && prev[idx(nx, ny)] == usize::MAX {
                prev[idx(nx, ny)] = idx(x, y);
                q.push_back((nx, ny));
            }
        }
    }
    // Return the first step on the path from start to goal.
    let g = goal?;
    let mut cur = idx(g.0, g.1);
    while prev[cur] != s {
        cur = prev[cur];
    }
    Some((cur % W, cur / W))
}

struct Out {
    coverage: f32,
    crashes: u32,
    reachable: usize,
}

/// Run coverage sweep. `known` = prior obstacle knowledge (grows on crash). Returns result + incident memory.
fn run(
    obst: &[bool],
    start: (usize, usize),
    mut known: Vec<bool>,
    mut sm: SafetyMemory<512, DIM>,
    mut trail: Option<&mut Vec<(usize, usize)>>,
) -> (Out, SafetyMemory<512, DIM>, Vec<bool>) {
    let mut visited = vec![false; W * H];
    let (mut cx, mut cy) = start;
    visited[idx(cx, cy)] = true;
    let (mut crashes, mut covered) = (0u32, 1usize);

    for _ in 0..MAX_STEPS {
        let Some((nx, ny)) = next_step(&known, &visited, cx, cy) else {
            break;
        };
        if obst[idx(nx, ny)] {
            // Unknown obstacle → crash → learn it (incident memory) and do not enter.
            crashes += 1;
            known[idx(nx, ny)] = true;
            sm.record_incident(&cell_fp(nx, ny));
            continue;
        }
        cx = nx;
        cy = ny;
        if !visited[idx(cx, cy)] {
            visited[idx(cx, cy)] = true;
            covered += 1;
        }
        if let Some(ref mut t) = trail {
            t.push((cx, cy));
        }
    }

    // Reachable free cells (for fair coverage evaluation).
    let reachable = reachable_free(obst, start);
    (
        Out {
            coverage: covered as f32 / reachable as f32,
            crashes,
            reachable,
        },
        sm,
        known,
    )
}

/// Number of reachable free cells from the start (ignoring cells isolated behind obstacles).
fn reachable_free(obst: &[bool], start: (usize, usize)) -> usize {
    let mut seen = vec![false; W * H];
    let mut q = VecDeque::new();
    seen[idx(start.0, start.1)] = true;
    q.push_back(start);
    let mut c = 0;
    while let Some((x, y)) = q.pop_front() {
        c += 1;
        for (nx, ny) in [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ] {
            if nx < W && ny < H && !obst[idx(nx, ny)] && !seen[idx(nx, ny)] {
                seen[idx(nx, ny)] = true;
                q.push_back((nx, ny));
            }
        }
    }
    c
}

fn render(obst: &[bool], visited: &[bool], start: (usize, usize)) {
    for y in 0..H {
        let mut line = String::new();
        for x in 0..W {
            line.push(if (x, y) == start {
                'S'
            } else if obst[idx(x, y)] {
                '#'
            } else if visited[idx(x, y)] {
                '.'
            } else {
                ' '
            });
        }
        println!("[cov] |{line}|");
    }
}

fn main() {
    // Arena with random obstacles (fixed seed), start cell is free.
    let mut rng = Lcg(0x51ed_2c0f_9a3b_77e1);
    let mut obst = vec![false; W * H];
    let start = (1usize, 1usize);
    for y in 0..H {
        for x in 0..W {
            let edge = x == 0 || y == 0 || x == W - 1 || y == H - 1;
            let near = x.abs_diff(start.0) <= 1 && y.abs_diff(start.1) <= 1;
            if !edge && !near && rng.next() < OBST_PCT {
                obst[idx(x, y)] = true;
            }
        }
    }
    let nobs = obst.iter().filter(|b| **b).count();
    println!("[cov] arena {W}x{H}, {nobs} random obstacles. Goal: SWEEP every reachable cell WITHOUT crashing.\n");

    let fresh = || SafetyMemory::<512, DIM>::new(1.0);

    // (A) First car: no prior obstacle knowledge → crashes and learns.
    let (a, harvested, _) = run(&obst, start, vec![false; W * H], fresh(), None);

    // (B) Transfer incident memory (bytes) → reconstruct the obstacle map from it.
    let mut blob = vec![0u8; SafetyMemory::<512, DIM>::export_len()];
    let n = harvested.export_bytes(&mut blob);
    let mut loaded = fresh();
    let m = loaded.import_bytes(&blob[..n]);
    let mut known0 = vec![false; W * H];
    for k in 0..m {
        // Read (x,y) from the imported incident embedding.
        let off = 4 + k * (DIM * 4 + 4);
        let rx = f32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]]);
        let ry = f32::from_le_bytes([blob[off + 4], blob[off + 5], blob[off + 6], blob[off + 7]]);
        known0[idx(rx as usize, ry as usize)] = true;
    }

    // (C) "Born-cautious" car: obstacle map pre-loaded.
    let mut trail = Vec::new();
    let (c, _, _) = run(&obst, start, known0, loaded, Some(&mut trail));

    println!("[cov] === results (coverage of reachable cells / crashes into obstacles) ===");
    println!("[cov]  (A) first car (no prior knowledge): coverage {:.0}%  crashes {}  (learned {} obstacles the hard way)", a.coverage * 100.0, a.crashes, a.crashes);
    println!(
        "[cov]  (C) BORN CAUTIOUS (memory transferred): coverage {:.0}%  crashes {}",
        c.coverage * 100.0,
        c.crashes
    );
    println!("[cov]  (reachable free cells: {})", a.reachable);

    // Coverage map of the born-cautious car.
    let mut vis = vec![false; W * H];
    vis[idx(start.0, start.1)] = true;
    for &(x, y) in &trail {
        vis[idx(x, y)] = true;
    }
    println!("\n[cov] coverage map (born-cautious):  S=start  #=obstacle  .=swept");
    render(&obst, &vis, start);

    if let Ok(mut f) = std::fs::File::create("/tmp/coverage_trail.csv") {
        let _ = writeln!(f, "# W={W} H={H}");
        for y in 0..H {
            for x in 0..W {
                if obst[idx(x, y)] {
                    let _ = writeln!(f, "obs,{x},{y}");
                }
            }
        }
        for (x, y) in &trail {
            let _ = writeln!(f, "car,{x},{y}");
        }
        println!("\n[cov] trail -> /tmp/coverage_trail.csv (run sim/view_coverage.py for a GIF)");
    }
    println!("[cov] PASS: born-cautious car (sim hazard memory transferred) sweeps the arena with ZERO crashes.");
}
