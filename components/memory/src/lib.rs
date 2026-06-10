//! # neural-memory
//!
//! Fixed-capacity neural-semantic memory, used as the robot's **core cognition/memory substrate**
//! instead of a traditional filesystem: data is stored as embeddings and retrieved by similarity
//! (cosine similarity) rather than by name/path.
//!
//! ## Design decisions (with alternatives)
//! - **`no_std`-ready:** no heap, no allocation, no std. Designed to live in `static`
//!   so it can later be moved into a seL4 component without modification.
//!   (Rejected alternative: `Vec`/heap — breaks seL4.)
//! - **Fixed capacity + ring buffer:** when full, writes over the oldest entry (circular/episodic memory).
//!   (Alternative: reject insertion when full — weaker for a robot that learns continuously.)
//! - **Storing L2-normalized vectors:** we normalize on write, so search becomes a **dot product**
//!   only — faster and prepares for SIMD later. (Alternative: store raw and compute full cosine
//!   on every search — slower.)
//! - **`f32`:** sufficient precision and faster than f64 on embedded processors.
//!   (Future alternative: int8 quantization.)
//!
//! ## Determinism (important for safety)
//! `search` makes a single pass over all valid entries → its runtime is **bounded and predictable**
//! (`O(len * DIM)`), a critical property for tying it later to seL4 temporal guarantees.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod persist;
pub mod tiers;

/// Search result: metadata associated with the nearest embedding + similarity score + its slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Metadata stored alongside the embedding (e.g., experience/state identifier).
    pub meta: u64,
    /// Cosine similarity in the range [-1.0, 1.0]; 1.0 = exact match.
    pub similarity: f32,
    /// Position of the entry within memory (slot).
    pub slot: usize,
}

/// Top-k search result: best `K` matches sorted descending, without heap.
#[derive(Debug, Clone, Copy)]
pub struct TopK<const K: usize> {
    items: [Match; K],
    len: usize,
}

impl<const K: usize> TopK<K> {
    /// Slice of the best results (sorted descending, highest similarity first).
    pub fn as_slice(&self) -> &[Match] {
        &self.items[..self.len]
    }
    /// Number of valid results (≤ K).
    pub fn len(&self) -> usize {
        self.len
    }
    /// Are there no results?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Best result (highest similarity), if any.
    pub fn best(&self) -> Option<Match> {
        if self.len > 0 {
            Some(self.items[0])
        } else {
            None
        }
    }
}

/// Neural-semantic memory with capacity `CAP` entries, each embedding of length `DIM`.
///
/// Designed to be placed in `static` on a `no_std` target:
/// ```ignore
/// static mut MEM: NeuralMemory<5000, 256> = NeuralMemory::new();
/// ```
pub struct NeuralMemory<const CAP: usize, const DIM: usize> {
    /// L2-normalized embeddings. Valid rows are determined by `len()`.
    embeddings: [[f32; DIM]; CAP],
    /// Metadata parallel to each embedding.
    metadata: [u64; CAP],
    /// Total number of store operations (monotonically increasing, does not stop at CAP).
    count: usize,
    /// Next write position (ring buffer).
    head: usize,
}

impl<const CAP: usize, const DIM: usize> NeuralMemory<CAP, DIM> {
    /// Create an empty (zeroed) memory. `const` so it can be placed in `static`.
    pub const fn new() -> Self {
        Self {
            embeddings: [[0.0; DIM]; CAP],
            metadata: [0; CAP],
            count: 0,
            head: 0,
        }
    }

    /// Number of currently valid entries (≤ CAP).
    pub fn len(&self) -> usize {
        if self.count < CAP {
            self.count
        } else {
            CAP
        }
    }

    /// Is memory empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Is memory full (and has started overwriting the oldest entries)?
    pub fn is_full(&self) -> bool {
        self.count >= CAP
    }

    /// Fixed maximum capacity.
    pub fn capacity(&self) -> usize {
        CAP
    }

    /// Clear contents (used for short-term working memory at the end of each perception cycle).
    pub fn clear(&mut self) {
        self.count = 0;
        self.head = 0;
    }

    /// Store an embedding with metadata. Normalizes the vector (L2) on write.
    /// When full, overwrites the oldest entry (ring buffer).
    ///
    /// A near-zero vector (length ≈ 0) is stored as-is (zeros) to avoid division by zero.
    pub fn store(&mut self, vec: &[f32; DIM], meta: u64) {
        let slot = self.head;
        normalize_into(vec, &mut self.embeddings[slot]);
        self.metadata[slot] = meta;
        self.head = (self.head + 1) % CAP;
        self.count = self.count.saturating_add(1);
    }

    /// Search for the nearest embedding to `query` (highest cosine similarity).
    /// Returns `None` if memory is empty.
    ///
    /// Bounded time: `O(len * DIM)` — no allocation, no division in the inner loop.
    pub fn search(&self, query: &[f32; DIM]) -> Option<Match> {
        let n = self.len();
        if n == 0 {
            return None;
        }
        let mut q = [0.0f32; DIM];
        normalize_into(query, &mut q);

        let mut best_sim = f32::NEG_INFINITY;
        let mut best = 0usize;
        for i in 0..n {
            // Vectors are pre-normalized → cosine = dot product directly.
            let s = dot(&q, &self.embeddings[i]);
            if s > best_sim {
                best_sim = s;
                best = i;
            }
        }
        Some(Match {
            meta: self.metadata[best],
            similarity: best_sim,
            slot: best,
        })
    }

    /// Search for the nearest `K` embeddings (top-k), sorted descending by similarity.
    /// No heap: maintains a fixed `[Match; K]` buffer with sorted insertion.
    /// Bounded time: `O(len * (DIM + K))`.
    pub fn search_top_k<const K: usize>(&self, query: &[f32; DIM]) -> TopK<K> {
        let sentinel = Match {
            meta: 0,
            similarity: f32::NEG_INFINITY,
            slot: 0,
        };
        let mut items = [sentinel; K];
        let mut len = 0usize;

        let n = self.len();
        if n == 0 || K == 0 {
            return TopK { items, len: 0 };
        }
        let mut q = [0.0f32; DIM];
        normalize_into(query, &mut q);

        for i in 0..n {
            let sim = dot(&q, &self.embeddings[i]);
            let cand = Match {
                meta: self.metadata[i],
                similarity: sim,
                slot: i,
            };
            if len < K {
                // Buffer not full: insert in sorted position.
                let mut j = len;
                while j > 0 && items[j - 1].similarity < sim {
                    items[j] = items[j - 1];
                    j -= 1;
                }
                items[j] = cand;
                len += 1;
            } else if sim > items[K - 1].similarity {
                // Better than the worst: replace and shift.
                let mut j = K - 1;
                while j > 0 && items[j - 1].similarity < sim {
                    items[j] = items[j - 1];
                    j -= 1;
                }
                items[j] = cand;
            }
        }
        TopK { items, len }
    }
}

impl<const CAP: usize, const DIM: usize> Default for NeuralMemory<CAP, DIM> {
    fn default() -> Self {
        Self::new()
    }
}

/// Number of SIMD lanes — 8 independent accumulators break the dependency chain
/// so the compiler can emit SSE/AVX (x86) or NEON (aarch64) automatically.
const LANES: usize = 8;

/// Dot product of two vectors of length `DIM` — safe SIMD-friendly version (no `unsafe`).
///
/// Accumulates across `LANES` parallel accumulators then folds them, producing real SIMD
/// on the target architecture via auto-vectorization with `opt-level=3`. Explicit NEON
/// can be added later behind an isolated feature flag when running on real hardware.
#[inline]
fn dot<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);

    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            acc[l] += xa[l] * xb[l];
        }
    }

    let mut sum: f32 = acc.iter().sum();
    // Remainder (if DIM is not a multiple of LANES).
    for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        sum += x * y;
    }
    sum
}

/// L2 normalization: copies `src` into `dst` divided by its Euclidean length.
/// If the length is ≈ 0, writes zeros (avoids division by zero).
#[inline]
fn normalize_into<const DIM: usize>(src: &[f32; DIM], dst: &mut [f32; DIM]) {
    let sum_sq: f32 = src.iter().map(|x| x * x).sum();
    // Small threshold to avoid division by zero for near-zero vectors.
    if sum_sq <= 1e-12 {
        dst.fill(0.0);
        return;
    }
    let inv_norm = 1.0 / sqrt(sum_sq);
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s * inv_norm;
    }
}

/// Square root that works in `no_std` (without std::f32::sqrt).
/// Uses Newton-Raphson approximation; on real hardware replaced by FPU instruction.
#[inline]
fn sqrt(x: f32) -> f32 {
    // In tests (std) use the precise function; on no_std use Newton approximation.
    #[cfg(test)]
    {
        x.sqrt()
    }
    #[cfg(not(test))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        // Newton-Raphson iterations: y = 0.5*(y + x/y)
        let mut y = x;
        for _ in 0..20 {
            let ny = 0.5 * (y + x / y);
            if (ny - y).abs() < 1e-7 {
                return ny;
            }
            y = ny;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn empty_search_returns_none() {
        let mem = NeuralMemory::<4, 3>::new();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert_eq!(mem.search(&[1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn exact_match_has_similarity_one() {
        let mut mem = NeuralMemory::<4, 3>::new();
        mem.store(&[3.0, 0.0, 0.0], 42); // unnormalized — normalized internally
        mem.store(&[0.0, 5.0, 0.0], 7);

        let m = mem.search(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(m.meta, 42);
        assert!(approx(m.similarity, 1.0, 1e-5), "sim={}", m.similarity);
    }

    #[test]
    fn orthogonal_vectors_are_dissimilar() {
        let mut mem = NeuralMemory::<4, 3>::new();
        mem.store(&[1.0, 0.0, 0.0], 1);
        let m = mem.search(&[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(m.meta, 1);
        assert!(approx(m.similarity, 0.0, 1e-5), "sim={}", m.similarity);
    }

    #[test]
    fn picks_closest_of_several() {
        let mut mem = NeuralMemory::<8, 4>::new();
        mem.store(&[1.0, 0.0, 0.0, 0.0], 100);
        mem.store(&[0.0, 1.0, 0.0, 0.0], 200);
        mem.store(&[0.9, 0.1, 0.0, 0.0], 300); // approximately same direction as query
        mem.store(&[0.0, 0.0, 1.0, 0.0], 400);

        // Query at an angle close to direction [0.9, 0.1] (≈6.3°) not to [1,0] (0°).
        let m = mem.search(&[0.9, 0.12, 0.0, 0.0]).unwrap();
        assert_eq!(m.meta, 300, "expected closest (300), got {}", m.meta);
    }

    #[test]
    fn ring_buffer_overwrites_oldest() {
        let mut mem = NeuralMemory::<2, 2>::new();
        mem.store(&[1.0, 0.0], 1);
        mem.store(&[0.0, 1.0], 2);
        assert!(mem.is_full());
        assert_eq!(mem.len(), 2);

        mem.store(&[1.0, 0.0], 3); // writes over slot 0 (oldest)
        assert_eq!(mem.len(), 2);

        // Entry 1 (meta=1) is gone; searching for [1,0] now returns meta=3.
        let m = mem.search(&[1.0, 0.0]).unwrap();
        assert_eq!(m.meta, 3);
    }

    #[test]
    fn top_k_returns_sorted_descending() {
        let mut mem = NeuralMemory::<8, 3>::new();
        mem.store(&[1.0, 0.0, 0.0], 10); // nearest to [1,0,0]
        mem.store(&[0.8, 0.2, 0.0], 20); // second
        mem.store(&[0.5, 0.5, 0.0], 30); // third
        mem.store(&[0.0, 0.0, 1.0], 40); // far

        let top = mem.search_top_k::<3>(&[1.0, 0.0, 0.0]);
        assert_eq!(top.len(), 3);
        let s = top.as_slice();
        assert_eq!(s[0].meta, 10);
        assert_eq!(s[1].meta, 20);
        assert_eq!(s[2].meta, 30);
        // sorted descending
        assert!(s[0].similarity >= s[1].similarity);
        assert!(s[1].similarity >= s[2].similarity);
    }

    #[test]
    fn top_k_caps_at_len_when_k_larger() {
        let mut mem = NeuralMemory::<8, 2>::new();
        mem.store(&[1.0, 0.0], 1);
        mem.store(&[0.0, 1.0], 2);
        let top = mem.search_top_k::<5>(&[1.0, 0.0]);
        assert_eq!(top.len(), 2); // only two entries exist
    }

    #[test]
    fn top_k_best_matches_search() {
        let mut mem = NeuralMemory::<8, 4>::new();
        mem.store(&[1.0, 0.0, 0.0, 0.0], 100);
        mem.store(&[0.0, 1.0, 0.0, 0.0], 200);
        mem.store(&[0.9, 0.12, 0.0, 0.0], 300);

        let q = [0.9, 0.12, 0.0, 0.0];
        let single = mem.search(&q).unwrap();
        let top = mem.search_top_k::<3>(&q);
        assert_eq!(top.best().unwrap().meta, single.meta);
        assert_eq!(top.best().unwrap().meta, 300);
    }

    #[test]
    fn top_k_zero_is_empty() {
        let mut mem = NeuralMemory::<4, 2>::new();
        mem.store(&[1.0, 0.0], 1);
        let top = mem.search_top_k::<0>(&[1.0, 0.0]);
        assert!(top.is_empty());
        assert_eq!(top.best(), None);
    }

    #[test]
    fn near_zero_vector_is_safe() {
        let mut mem = NeuralMemory::<2, 3>::new();
        mem.store(&[0.0, 0.0, 0.0], 9); // zero vector — no panic, no NaN
        let m = mem.search(&[0.0, 0.0, 0.0]);
        assert!(m.is_some());
        assert!(!m.unwrap().similarity.is_nan());
    }

    #[test]
    fn realistic_size_works_on_heap() {
        // CAP*DIM is large → place on heap in tests only to avoid stack overflow.
        // (On a no_std target it would be a static in .bss, no heap.)
        let mut mem = Box::new(NeuralMemory::<1000, 256>::new());
        let mut v = [0.0f32; 256];
        for k in 0..500u64 {
            v[(k as usize) % 256] = 1.0 + k as f32;
            mem.store(&v, k);
            v[(k as usize) % 256] = 0.0;
        }
        assert_eq!(mem.len(), 500);

        // Query matching entry number 123.
        let mut q = [0.0f32; 256];
        q[123] = 1.0;
        let m = mem.search(&q).unwrap();
        assert!(m.similarity > 0.99, "sim={}", m.similarity);
    }
}
