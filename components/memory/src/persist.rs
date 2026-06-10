//! # Persistence layer — append-only + journal
//!
//! Embodies the "abolish the filesystem" idea: no directories, no filenames, no tree.
//! Just an **append-only sequential log** of vectors written directly to raw storage,
//! with a **CRC32 per record** to detect torn writes after sudden power loss,
//! and **recovery** that rebuilds memory from intact records at boot time.
//!
//! ## Why this matters (thesis + patent)
//! - Traditional filesystems need complex journaling (like NTFS) to avoid corruption.
//! - Here: sequential writes only (protects flash cells from wear) + CRC for determinism.
//! - Semantic recovery after "sudden death" = a physical artifact amenable to patenting.
//!
//! ## Abstraction (SIL)
//! `BlockStore` decouples logic from hardware:
//! - On host: `SliceStore` over `&mut [u8]` (backed by Vec in tests).
//! - On seL4: same `SliceStore` but the slice = the physically mapped flash region.

/// Marker for the start of a valid record ("NMEM" in little-endian).
const MAGIC: u32 = 0x4E4D_454D;

/// Raw block-storage device interface (no filesystem).
pub trait BlockStore {
    /// Total capacity in bytes.
    fn capacity(&self) -> usize;
    /// Read `buf.len()` bytes from `offset`.
    fn read(&self, offset: usize, buf: &mut [u8]);
    /// Write `data` at `offset`.
    fn write(&mut self, offset: usize, data: &[u8]);
    /// Sync to hardware (no-op for RAM; on real flash flushes the cache).
    fn flush(&mut self) {}
}

/// Storage backed by a byte slice — works in `no_std`.
/// On seL4 the slice is the physical flash region itself.
pub struct SliceStore<'a> {
    buf: &'a mut [u8],
}

impl<'a> SliceStore<'a> {
    /// Create storage over a byte slice.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf }
    }
}

impl BlockStore for SliceStore<'_> {
    fn capacity(&self) -> usize {
        self.buf.len()
    }
    fn read(&self, offset: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.buf[offset..offset + buf.len()]);
    }
    fn write(&mut self, offset: usize, data: &[u8]) {
        self.buf[offset..offset + data.len()].copy_from_slice(data);
    }
}

/// Log errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogError {
    /// Storage is full — no room for a new record.
    Full,
}

/// Reason recovery stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached an unwritten region (MAGIC absent) — normal end of log.
    EndOfLog,
    /// CRC mismatch — torn record (crash/power-loss boundary).
    CorruptRecord,
    /// Not enough space remaining for a complete record.
    Truncated,
}

/// Recovery statistics.
#[derive(Debug, Clone, Copy)]
pub struct RecoverStats {
    /// Number of intact records recovered.
    pub recovered: u64,
    /// Next write position (after the last intact record).
    pub write_head: usize,
    /// Reason recovery stopped.
    pub reason: StopReason,
}

/// Append-only sequential log of vectors of length `DIM`, over storage `S`.
pub struct AppendLog<S: BlockStore, const DIM: usize> {
    store: S,
    write_head: usize,
    next_seq: u64,
}

impl<S: BlockStore, const DIM: usize> AppendLog<S, DIM> {
    /// Record size in bytes: MAGIC(4) + seq(8) + meta(8) + embedding(DIM*4) + crc(4).
    pub const REC_SIZE: usize = 4 + 8 + 8 + DIM * 4 + 4;

    /// Create a new empty log (starts from zero).
    pub fn create(store: S) -> Self {
        Self {
            store,
            write_head: 0,
            next_seq: 0,
        }
    }

    /// Recover an existing log: scans intact records and calls `on_record` for each,
    /// then returns the log positioned after the last intact record.
    pub fn recover<F>(store: S, mut on_record: F) -> (Self, RecoverStats)
    where
        F: FnMut(u64, &[f32; DIM]),
    {
        let cap = store.capacity();
        let mut offset = 0usize;
        let mut recovered = 0u64;
        let mut next_seq = 0u64;
        let reason;

        loop {
            if offset + Self::REC_SIZE > cap {
                reason = StopReason::Truncated;
                break;
            }
            // Read MAGIC.
            let mut w4 = [0u8; 4];
            store.read(offset, &mut w4);
            if u32::from_le_bytes(w4) != MAGIC {
                reason = StopReason::EndOfLog;
                break;
            }
            // Read the rest of the record and verify CRC.
            let mut emb = [0.0f32; DIM];
            let (seq, meta, stored_crc) = read_record_body::<S, DIM>(&store, offset, &mut emb);
            let crc = compute_record_crc::<DIM>(seq, meta, &emb);
            if crc != stored_crc {
                reason = StopReason::CorruptRecord;
                break;
            }
            // Intact record.
            on_record(meta, &emb);
            recovered += 1;
            next_seq = seq + 1;
            offset += Self::REC_SIZE;
        }

        let log = Self {
            store,
            write_head: offset,
            next_seq,
        };
        (
            log,
            RecoverStats {
                recovered,
                write_head: offset,
                reason,
            },
        )
    }

    /// Append a new record (vector + metadata). Returns the sequence number (seq).
    pub fn append(&mut self, emb: &[f32; DIM], meta: u64) -> Result<u64, LogError> {
        if self.write_head + Self::REC_SIZE > self.store.capacity() {
            return Err(LogError::Full);
        }
        let seq = self.next_seq;
        let off = self.write_head;
        let crc = compute_record_crc::<DIM>(seq, meta, emb);

        self.store.write(off, &MAGIC.to_le_bytes());
        self.store.write(off + 4, &seq.to_le_bytes());
        self.store.write(off + 12, &meta.to_le_bytes());
        let mut p = off + 20;
        for &f in emb.iter() {
            self.store.write(p, &f.to_le_bytes());
            p += 4;
        }
        self.store.write(p, &crc.to_le_bytes());
        self.store.flush();

        self.write_head += Self::REC_SIZE;
        self.next_seq += 1;
        Ok(seq)
    }

    /// Number of records written (= next seq number).
    pub fn record_count(&self) -> u64 {
        self.next_seq
    }

    /// Current write position (bytes).
    pub fn write_head(&self) -> usize {
        self.write_head
    }
}

/// Read a record body (after MAGIC is verified): fills `emb` and returns (seq, meta, crc).
fn read_record_body<S: BlockStore, const DIM: usize>(
    store: &S,
    offset: usize,
    emb: &mut [f32; DIM],
) -> (u64, u64, u32) {
    let mut w8 = [0u8; 8];
    store.read(offset + 4, &mut w8);
    let seq = u64::from_le_bytes(w8);
    store.read(offset + 12, &mut w8);
    let meta = u64::from_le_bytes(w8);

    let mut p = offset + 20;
    let mut w4 = [0u8; 4];
    for slot in emb.iter_mut() {
        store.read(p, &mut w4);
        *slot = f32::from_le_bytes(w4);
        p += 4;
    }
    store.read(p, &mut w4);
    let crc = u32::from_le_bytes(w4);
    (seq, meta, crc)
}

/// Compute CRC32 over the record payload (MAGIC + seq + meta + embedding).
fn compute_record_crc<const DIM: usize>(seq: u64, meta: u64, emb: &[f32; DIM]) -> u32 {
    let mut c = Crc32::new();
    c.update(&MAGIC.to_le_bytes());
    c.update(&seq.to_le_bytes());
    c.update(&meta.to_le_bytes());
    for &f in emb.iter() {
        c.update(&f.to_le_bytes());
    }
    c.finalize()
}

/// Incremental CRC32 (reversed IEEE), table-free, no dependencies — works in `no_std`.
struct Crc32 {
    state: u32,
}

impl Crc32 {
    fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }
    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.state ^= b as u32;
            for _ in 0..8 {
                let mask = (self.state & 1).wrapping_neg();
                self.state = (self.state >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
    }
    fn finalize(self) -> u32 {
        !self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeuralMemory;

    const DIM: usize = 8;

    // Storage region larger than the data by one empty record (simulates real flash:
    // the tail is zeros → MAGIC absent → EndOfLog).
    fn vec_buf(records: usize) -> Vec<u8> {
        let rec = AppendLog::<SliceStore, DIM>::REC_SIZE;
        vec![0u8; rec * records + rec]
    }

    #[test]
    fn append_then_recover_all() {
        let mut buf = vec_buf(5);
        {
            let store = SliceStore::new(&mut buf);
            let mut log = AppendLog::<_, DIM>::create(store);
            for k in 0..5u64 {
                let mut e = [0.0f32; DIM];
                e[(k as usize) % DIM] = 1.0 + k as f32;
                log.append(&e, 1000 + k).unwrap();
            }
            assert_eq!(log.record_count(), 5);
        }
        // Fresh "boot": recover from the same bytes.
        let store = SliceStore::new(&mut buf);
        let mut recovered = Vec::new();
        let (_log, stats) = AppendLog::<_, DIM>::recover(store, |meta, _emb| {
            recovered.push(meta);
        });
        assert_eq!(stats.recovered, 5);
        assert_eq!(stats.reason, StopReason::EndOfLog);
        assert_eq!(recovered, vec![1000, 1001, 1002, 1003, 1004]);
    }

    #[test]
    fn torn_last_record_is_rejected_prior_survive() {
        let mut buf = vec_buf(5);
        {
            let store = SliceStore::new(&mut buf);
            let mut log = AppendLog::<_, DIM>::create(store);
            for k in 0..5u64 {
                let e = [k as f32 + 1.0; DIM];
                log.append(&e, 2000 + k).unwrap();
            }
        }
        // Simulate power loss: corrupt a byte in the last record (flip the CRC).
        let rec = AppendLog::<SliceStore, DIM>::REC_SIZE;
        let last_crc_pos = rec * 5 - 1;
        buf[last_crc_pos] ^= 0xFF;

        let store = SliceStore::new(&mut buf);
        let mut recovered = Vec::new();
        let (log, stats) = AppendLog::<_, DIM>::recover(store, |meta, _e| recovered.push(meta));
        // First 4 intact, fifth rejected.
        assert_eq!(stats.recovered, 4);
        assert_eq!(stats.reason, StopReason::CorruptRecord);
        assert_eq!(recovered, vec![2000, 2001, 2002, 2003]);
        // write_head is at the start of the corrupt record → next write overwrites it (self-healing).
        assert_eq!(log.write_head(), rec * 4);
    }

    #[test]
    fn partial_write_stops_recovery() {
        let mut buf = vec_buf(5);
        {
            let store = SliceStore::new(&mut buf);
            let mut log = AppendLog::<_, DIM>::create(store);
            for k in 0..3u64 {
                let e = [1.0f32; DIM];
                log.append(&e, 3000 + k).unwrap();
            }
        }
        // Partial write: only MAGIC for the fourth record (interrupted mid-write).
        let rec = AppendLog::<SliceStore, DIM>::REC_SIZE;
        buf[rec * 3..rec * 3 + 4].copy_from_slice(&MAGIC.to_le_bytes());

        let store = SliceStore::new(&mut buf);
        let mut n = 0u64;
        let (_log, stats) = AppendLog::<_, DIM>::recover(store, |_m, _e| n += 1);
        // CRC of partial record (zeros) will not match → rejected.
        assert_eq!(stats.recovered, 3);
        assert_eq!(n, 3);
        assert_eq!(stats.reason, StopReason::CorruptRecord);
    }

    #[test]
    fn full_storage_errors() {
        // Tight buffer: exactly two records worth of space (no surplus).
        let mut buf = vec![0u8; AppendLog::<SliceStore, DIM>::REC_SIZE * 2];
        let store = SliceStore::new(&mut buf);
        let mut log = AppendLog::<_, DIM>::create(store);
        let e = [1.0f32; DIM];
        assert!(log.append(&e, 1).is_ok());
        assert!(log.append(&e, 2).is_ok());
        assert_eq!(log.append(&e, 3), Err(LogError::Full));
    }

    #[test]
    fn memories_survive_reboot_into_neural_memory() {
        // Thesis: memories persist across reboots and are retrieved by similarity.
        let mut buf = vec_buf(4);
        {
            let store = SliceStore::new(&mut buf);
            let mut log = AppendLog::<_, DIM>::create(store);
            let mut e = [0.0f32; DIM];
            e[3] = 1.0;
            log.append(&e, 777).unwrap(); // distinctive memory
            e[3] = 0.0;
            e[0] = 1.0;
            log.append(&e, 111).unwrap();
        }
        // "Boot": rebuild NeuralMemory from the log.
        let mut mem = NeuralMemory::<16, DIM>::new();
        let store = SliceStore::new(&mut buf);
        let (_log, stats) = AppendLog::<_, DIM>::recover(store, |meta, emb| mem.store(emb, meta));
        assert_eq!(stats.recovered, 2);

        // Searching for the distinctive memory returns it — it survived the "power loss".
        let mut q = [0.0f32; DIM];
        q[3] = 1.0;
        let m = mem.search(&q).unwrap();
        assert_eq!(m.meta, 777);
        assert!(m.similarity > 0.99);
    }
}
