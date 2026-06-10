//! Persistence demo: a file represents a "flash region". Each run = one "boot":
//! recovers previous memories, recalls one by similarity, appends a new memory, then "powers off".
//! Running it multiple times shows memories surviving and growing across "power interruptions".
//!
//! On seL4: exactly the same code, but `buf` = the physical flash region (no file).

use neural_memory::persist::{AppendLog, SliceStore};
use neural_memory::NeuralMemory;
use std::fs;

const DIM: usize = 16;
const CAP_RECORDS: usize = 64;
const PATH: &str = "/tmp/crob_flash.bin";

fn onehot(i: usize) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    v[i % DIM] = 1.0;
    v
}

fn main() {
    let rec = AppendLog::<SliceStore, DIM>::REC_SIZE;
    let size = rec * CAP_RECORDS + rec;

    // Load the "flash" from disk (or zero it on first run).
    let mut buf = match fs::read(PATH) {
        Ok(b) if b.len() == size => b,
        _ => vec![0u8; size],
    };

    let mut mem = NeuralMemory::<128, DIM>::new();
    let total;
    {
        // Recover all memories from flash into NeuralMemory.
        let store = SliceStore::new(&mut buf);
        let (mut log, stats) =
            AppendLog::<_, DIM>::recover(store, |meta, emb| mem.store(emb, meta));
        println!(
            "[boot ] recovered {} memories (stop={:?})",
            stats.recovered, stats.reason
        );

        // Recall the first memory (onehot(0), meta=1000) by similarity — not by name/path.
        if let Some(m) = mem.search(&onehot(0)) {
            println!(
                "[recall] query=onehot(0) -> meta={} sim={:.3}",
                m.meta, m.similarity
            );
        } else {
            println!("[recall] (no memories yet)");
        }

        // Append a new memory for this run.
        let n = log.record_count();
        let meta = 1000 + n;
        match log.append(&onehot(n as usize), meta) {
            Ok(seq) => println!("[write] appended new memory seq={} meta={}", seq, meta),
            Err(e) => println!("[write] FULL: {:?}", e),
        }
        total = log.record_count();
    } // drop log → releases borrow of buf

    // "Flash retains bytes" — write the buffer to disk (not needed on seL4).
    fs::write(PATH, &buf).unwrap();
    println!("[flush] total memories now = {}\n", total);
}
