//! Dynamic analysis example (valgrind / heaptrack).
//! Allocates memory once (Box = simulating static in no_std), then performs
//! thousands of store/search operations. Expected: allocations do not grow with
//! operations (no-heap inside the loop).

use neural_memory::NeuralMemory;

const CAP: usize = 5000;
const DIM: usize = 256;

fn main() {
    // Single large allocation (in no_std this would be a static in .bss, zero heap).
    let mut mem = Box::new(NeuralMemory::<CAP, DIM>::new());

    // Store 5000 vectors.
    let mut v = [0.0f32; DIM];
    for k in 0..CAP as u64 {
        let idx = (k as usize) % DIM;
        v[idx] = 1.0 + (k % 97) as f32;
        mem.store(&v, k);
        v[idx] = 0.0;
    }

    // 2000 searches — must not allocate anything + measure search latency.
    let mut q = [0.0f32; DIM];
    let mut checksum = 0u64;
    const N: u64 = 2000;
    let t0 = std::time::Instant::now();
    for k in 0..N {
        let idx = (k as usize) % DIM;
        q[idx] = 1.0;
        if let Some(m) = mem.search(&q) {
            checksum = checksum.wrapping_add(m.meta);
        }
        q[idx] = 0.0;
    }
    let elapsed = t0.elapsed();
    let per_search_us = elapsed.as_secs_f64() * 1e6 / N as f64;

    // top-k (k=5) latency + exercise under dynamic analysis
    let mut topsum = 0u64;
    let t1 = std::time::Instant::now();
    for k in 0..N {
        let idx = (k as usize) % DIM;
        q[idx] = 1.0;
        let top = mem.search_top_k::<5>(&q);
        topsum = topsum.wrapping_add(top.len() as u64);
        q[idx] = 0.0;
    }
    let per_topk_us = t1.elapsed().as_secs_f64() * 1e6 / N as f64;

    println!("stored={} searches={} checksum={}", mem.len(), N, checksum);
    println!(
        "search    latency: {:.2} us/search over {} entries (target < 5000 us)",
        per_search_us, CAP
    );
    println!(
        "top-k(5)  latency: {:.2} us/search over {} entries (topsum={})",
        per_topk_us, CAP, topsum
    );
}
