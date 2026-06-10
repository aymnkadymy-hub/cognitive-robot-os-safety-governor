//! # Three-Tier Memory System
//!
//! Emulates the structure of human memory on top of the core vector memory:
//! - **Working:** short-term, cleared every perception cycle.
//! - **Episodic:** robot experiences (persisted via append-only log).
//! - **Semantic:** pre-trained knowledge (loaded at boot, mostly read-only).
//!
//! `recall` searches across tiers and returns the nearest match together with its
//! **tier identity** — so the robot knows whether the recalled information is general
//! knowledge, a personal experience, or an instantaneous state. `no_std`, zero heap.

use crate::{Match, NeuralMemory};

/// Memory tier from which a match originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Working,
    Episodic,
    Semantic,
}

/// Cross-tier retrieval result: the match + its tier.
#[derive(Debug, Clone, Copy)]
pub struct Recall {
    pub hit: Match,
    pub tier: Tier,
}

/// Three-tier memory system with capacities: working `W`, episodic `E`, semantic `S`, dimension `DIM`.
pub struct MemorySystem<const W: usize, const E: usize, const S: usize, const DIM: usize> {
    working: NeuralMemory<W, DIM>,
    episodic: NeuralMemory<E, DIM>,
    semantic: NeuralMemory<S, DIM>,
}

impl<const W: usize, const E: usize, const S: usize, const DIM: usize> MemorySystem<W, E, S, DIM> {
    /// Create an empty memory system.
    pub const fn new() -> Self {
        Self {
            working: NeuralMemory::new(),
            episodic: NeuralMemory::new(),
            semantic: NeuralMemory::new(),
        }
    }

    /// Load semantic knowledge (reference model/concept) — typically called at boot.
    pub fn load_semantic(&mut self, vec: &[f32; DIM], meta: u64) {
        self.semantic.store(vec, meta);
    }

    /// Perceive an experience: enters both working memory (instantaneous) and episodic (persistent).
    pub fn perceive(&mut self, vec: &[f32; DIM], meta: u64) {
        self.working.store(vec, meta);
        self.episodic.store(vec, meta);
    }

    /// Clear working memory (end of perception cycle).
    pub fn clear_working(&mut self) {
        self.working.clear();
    }

    /// Retrieve across all tiers; returns the highest-similarity match with its tier identity.
    pub fn recall(&self, query: &[f32; DIM]) -> Option<Recall> {
        let candidates = [
            (Tier::Working, self.working.search(query)),
            (Tier::Episodic, self.episodic.search(query)),
            (Tier::Semantic, self.semantic.search(query)),
        ];
        let mut best: Option<Recall> = None;
        for (tier, found) in candidates {
            if let Some(hit) = found {
                let better = best
                    .as_ref()
                    .is_none_or(|b| hit.similarity > b.hit.similarity);
                if better {
                    best = Some(Recall { hit, tier });
                }
            }
        }
        best
    }

    /// Retrieve restricted to a single tier (e.g., semantic knowledge only).
    pub fn recall_in(&self, tier: Tier, query: &[f32; DIM]) -> Option<Match> {
        match tier {
            Tier::Working => self.working.search(query),
            Tier::Episodic => self.episodic.search(query),
            Tier::Semantic => self.semantic.search(query),
        }
    }

    /// Entry counts for each tier (working, episodic, semantic).
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.working.len(), self.episodic.len(), self.semantic.len())
    }
}

impl<const W: usize, const E: usize, const S: usize, const DIM: usize> Default
    for MemorySystem<W, E, S, DIM>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onehot<const N: usize>(i: usize) -> [f32; N] {
        let mut v = [0.0f32; N];
        v[i % N] = 1.0;
        v
    }

    #[test]
    fn recall_picks_correct_tier() {
        let mut m = MemorySystem::<4, 8, 8, 6>::new();
        // Semantic knowledge: concept at position 5.
        m.load_semantic(&onehot::<6>(5), 500);
        // Episodic experience: at position 2.
        m.perceive(&onehot::<6>(2), 200);

        // Query near semantic → comes from Semantic.
        let r = m.recall(&onehot::<6>(5)).unwrap();
        assert_eq!(r.tier, Tier::Semantic);
        assert_eq!(r.hit.meta, 500);

        // Query near experience → from Working or Episodic (same vector in both).
        let r2 = m.recall(&onehot::<6>(2)).unwrap();
        assert_eq!(r2.hit.meta, 200);
        assert!(matches!(r2.tier, Tier::Working | Tier::Episodic));
    }

    #[test]
    fn clear_working_keeps_episodic_and_semantic() {
        let mut m = MemorySystem::<4, 8, 8, 4>::new();
        m.load_semantic(&onehot::<4>(0), 1);
        m.perceive(&onehot::<4>(1), 2);
        assert_eq!(m.counts(), (1, 1, 1)); // working, episodic, semantic

        m.clear_working();
        assert_eq!(m.counts(), (0, 1, 1)); // only working was cleared

        // Experience is still in episodic.
        let hit = m.recall_in(Tier::Episodic, &onehot::<4>(1)).unwrap();
        assert_eq!(hit.meta, 2);
        // Semantic is still present.
        assert!(m.recall_in(Tier::Semantic, &onehot::<4>(0)).is_some());
        // Working is empty.
        assert!(m.recall_in(Tier::Working, &onehot::<4>(1)).is_none());
    }

    #[test]
    fn recall_empty_is_none() {
        let m = MemorySystem::<2, 2, 2, 3>::new();
        assert!(m.recall(&onehot::<3>(0)).is_none());
    }
}
