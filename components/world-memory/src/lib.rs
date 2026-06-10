//! # world-memory — spatial-semantic world model (perception layer 3)
//!
//! Extends the neural memory concept from "isolated vectors" to a **world map**: every
//! perceived object is stored as (semantic `embedding` vector + **pose** + **class** + **time**).
//! The robot can thus **know what is around it** via queries:
//! - **By similarity** (`recall_similar`): "what is this object?"
//! - **By location** (`nearest`, `count_within`): "what is near me right now?"
//! - **By class** (`nearest_of_class`): "where is the nearest chair?"
//!
//! `no_std`, zero heap, no `unsafe`, fixed capacity (ring buffer for continuous perception) —
//! runs on seL4. This is the system's unique role in perception:
//! **a queryable world memory.**

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use libm::sqrtf;

/// A single entry in the world map.
#[derive(Clone, Copy, Debug)]
pub struct Entry<const DIM: usize> {
    pub emb: [f32; DIM], // semantic vector (from the encoder)
    pub pose: [f32; 3],  // world position: x, y, theta
    pub class: u32,      // semantic class
    pub time: u32,       // perception timestamp (cycle)
    pub valid: bool,
}

impl<const DIM: usize> Entry<DIM> {
    const fn empty() -> Self {
        Self {
            emb: [0.0; DIM],
            pose: [0.0; 3],
            class: 0,
            time: 0,
            valid: false,
        }
    }
}

/// Query result: entry index + score (similarity, or negative squared distance for spatial queries).
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub index: usize,
    pub score: f32,
}

/// World map with capacity `CAP` and vector dimension `DIM`.
pub struct WorldMemory<const CAP: usize, const DIM: usize> {
    entries: [Entry<DIM>; CAP],
    next: usize,
    count: usize,
}

impl<const CAP: usize, const DIM: usize> WorldMemory<CAP, DIM> {
    pub const fn new() -> Self {
        Self {
            entries: [Entry::empty(); CAP],
            next: 0,
            count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn get(&self, i: usize) -> Option<&Entry<DIM>> {
        self.entries.get(i).filter(|e| e.valid)
    }

    /// Store a perceived object (ring buffer — overwrites the oldest entry when full,
    /// for continuous perception).
    pub fn observe(&mut self, emb: &[f32; DIM], pose: [f32; 3], class: u32, time: u32) {
        self.entries[self.next] = Entry {
            emb: *emb,
            pose,
            class,
            time,
            valid: true,
        };
        self.next = (self.next + 1) % CAP;
        if self.count < CAP {
            self.count += 1;
        }
    }

    /// "What is this?" — nearest entry by cosine similarity.
    pub fn recall_similar(&self, query: &[f32; DIM]) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !e.valid {
                continue;
            }
            let s = cosine(query, &e.emb);
            if best.is_none_or(|b| s > b.score) {
                best = Some(Hit { index: i, score: s });
            }
        }
        best
    }

    /// "What is closest to me?" — nearest entry by spatial distance (xy).
    pub fn nearest(&self, x: f32, y: f32) -> Option<Hit> {
        self.nearest_filtered(x, y, |_| true)
    }

    /// "Where is the nearest object of a given class?"
    pub fn nearest_of_class(&self, class: u32, x: f32, y: f32) -> Option<Hit> {
        self.nearest_filtered(x, y, |e| e.class == class)
    }

    fn nearest_filtered<F: Fn(&Entry<DIM>) -> bool>(&self, x: f32, y: f32, pred: F) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !e.valid || !pred(e) {
                continue;
            }
            let (dx, dy) = (e.pose[0] - x, e.pose[1] - y);
            let score = -(dx * dx + dy * dy); // larger = closer
            if best.is_none_or(|b| score > b.score) {
                best = Some(Hit { index: i, score });
            }
        }
        best
    }

    /// "How many objects are within radius r of me?"
    pub fn count_within(&self, x: f32, y: f32, r: f32) -> usize {
        let r2 = r * r;
        self.entries
            .iter()
            .filter(|e| e.valid)
            .filter(|e| {
                let (dx, dy) = (e.pose[0] - x, e.pose[1] - y);
                dx * dx + dy * dy <= r2
            })
            .count()
    }
}

impl<const CAP: usize, const DIM: usize> Default for WorldMemory<CAP, DIM> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors.
fn cosine<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..D {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = sqrtf(na) * sqrtf(nb);
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(i: usize) -> [f32; 4] {
        let mut v = [0.0; 4];
        v[i % 4] = 1.0;
        v
    }

    #[test]
    fn knows_whats_around() {
        let mut w = WorldMemory::<16, 4>::new();
        w.observe(&emb(0), [2.0, 0.0, 0.0], 10, 1); // object 2m away
        w.observe(&emb(1), [0.5, 0.0, 0.0], 11, 2); // nearby object
        w.observe(&emb(2), [5.0, 5.0, 0.0], 12, 3); // far away

        // Objects near the origin within 1.5m = one.
        assert_eq!(w.count_within(0.0, 0.0, 1.5), 1);
        // Nearest to origin is the object at 0.5m (class 11).
        assert_eq!(w.get(w.nearest(0.0, 0.0).unwrap().index).unwrap().class, 11);
    }

    #[test]
    fn recall_by_meaning_and_class() {
        let mut w = WorldMemory::<16, 4>::new();
        w.observe(&emb(0), [1.0, 0.0, 0.0], 100, 1); // class 100
        w.observe(&emb(2), [3.0, 0.0, 0.0], 200, 2); // class 200

        // "What is this?" with a query resembling the first entry.
        let hit = w.recall_similar(&emb(0)).unwrap();
        assert_eq!(w.get(hit.index).unwrap().class, 100);
        assert!(hit.score > 0.9);

        // "Where is the nearest class 200?"
        let h = w.nearest_of_class(200, 0.0, 0.0).unwrap();
        assert_eq!(w.get(h.index).unwrap().class, 200);
    }

    #[test]
    fn ring_buffer_keeps_capacity() {
        let mut w = WorldMemory::<2, 4>::new();
        for t in 0..5 {
            w.observe(&emb(t), [t as f32, 0.0, 0.0], t as u32, t as u32);
        }
        assert_eq!(w.len(), 2); // does not exceed capacity
    }
}
