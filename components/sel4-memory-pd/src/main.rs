//! Cognitive/Brain PD (Phase D0) — the brain running on seL4.
//! Each cycle: reads the body state, stores it in neural memory (perception + memory), runs
//! the trained neural policy → proposed command, then notifies the guard. Carries a heartbeat for the watchdog.

#![no_std]
#![no_main]

use neural_memory::NeuralMemory;
use reflex_abi::{OFF_CYCLE, OFF_HEARTBEAT, OFF_PROPOSED, OFF_STATE, REGION_SIZE};
use sel4_microkit::{
    debug_println, memory_region_symbol, protection_domain, Channel, ChannelSet, Handler,
    Infallible,
};
use sel4_shared_memory::{access::ReadWrite, SharedMemoryRef};

/// Guard channel (id=1 in the system description).
const TO_GUARD: Channel = Channel::new(1);
/// Single cycle in which the brain "stalls" (watchdog demonstration inside the real loop).
const STALL_CYCLE: u32 = 150;
/// Task: balance the cart at this target position (0.0 = center).
/// Change it (e.g. to 0.6) to give the robot a different task: "move right and balance there".
const TARGET_X: f32 = 0.0;

#[protection_domain]
fn init() -> HandlerImpl {
    debug_println!("[brain] cognitive PD online (state-perception -> memory -> trained policy)");
    let region = unsafe {
        SharedMemoryRef::new(memory_region_symbol!(reflex_vaddr: *mut [u8], n = REGION_SIZE))
    };
    HandlerImpl {
        region,
        memory: NeuralMemory::new(),
        heartbeat: 0,
    }
}

struct HandlerImpl {
    region: SharedMemoryRef<'static, [u8], ReadWrite>,
    memory: NeuralMemory<64, 4>,
    heartbeat: u64,
}

impl HandlerImpl {
    fn read(&self, off: usize, buf: &mut [u8]) {
        self.region
            .as_ptr()
            .index(off..off + buf.len())
            .copy_into_slice(buf);
    }
    fn write(&mut self, off: usize, data: &[u8]) {
        self.region
            .as_mut_ptr()
            .index(off..off + data.len())
            .copy_from_slice(data);
    }
    fn read_u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_le_bytes(b)
    }
    fn read_state(&self) -> [f32; 4] {
        let mut buf = [0u8; 16];
        self.read(OFF_STATE, &mut buf);
        let mut s = [0.0f32; 4];
        for (i, v) in s.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&buf[i * 4..i * 4 + 4]);
            *v = f32::from_le_bytes(b);
        }
        s
    }
}

impl Handler for HandlerImpl {
    type Error = Infallible;

    fn notified(&mut self, _channels: ChannelSet) -> Result<(), Self::Error> {
        let cycle = self.read_u32(OFF_CYCLE);
        let state = self.read_state();

        // Perception + memory: store the state as an embedding (experience).
        self.memory.store(&state, cycle as u64);

        // The body is a MuJoCo twin and publishes state in MuJoCo order [x, θ, ẋ, θ̇]; apply position target to x.
        let obs = [state[0] - TARGET_X, state[1], state[2], state[3]];
        // Trained MuJoCo policy → normalized command (the body converts it to force via gear).
        let proposed = mujoco_pendulum_policy::command(&obs);

        // Heartbeat (held constant for one cycle to simulate a stall → watchdog test).
        if cycle == STALL_CYCLE {
            debug_println!(
                "[brain] cycle {}: (simulated STALL — heartbeat held)",
                cycle
            );
        } else {
            self.heartbeat += 1;
        }

        self.write(OFF_PROPOSED, &proposed.to_le_bytes());
        self.write(OFF_HEARTBEAT, &self.heartbeat.to_le_bytes());
        TO_GUARD.notify();
        Ok(())
    }
}
