//! Actuation/Body PD (Phase D0/D2) — the simulated robot body running on seL4.
//! Owns the cartpole physics, applies the **guard-approved command** (actuation), steps
//! the physics, and publishes the new state to the brain — closing the full loop on the verified kernel.

#![no_std]
#![no_main]

use cartpole_sim::mujoco_twin::FALL_ANGLE;
use hal::{MujocoTwinBody, RobotBody};
use reflex_abi::{MAX_CYCLES, OFF_APPROVED, OFF_CYCLE, OFF_OVERRIDDEN, OFF_STATE, REGION_SIZE};
use sel4_microkit::{
    debug_println, memory_region_symbol, protection_domain, Channel, ChannelSet, Handler,
    Infallible,
};
use sel4_shared_memory::{access::ReadWrite, SharedMemoryRef};

/// Brain channel (id=0 in the system description): notifies the brain of the new state.
const TO_COGNITIVE: Channel = Channel::new(0);

#[protection_domain]
fn init() -> HandlerImpl {
    debug_println!("[body] MuJoCo-twin physics PD online (the simulated robot body)");
    let region = unsafe {
        SharedMemoryRef::new(memory_region_symbol!(reflex_vaddr: *mut [u8], n = REGION_SIZE))
    };
    let mut h = HandlerImpl {
        region,
        body: MujocoTwinBody::new(0.05),
        max_theta: 0.05,
        fell: false,
        clamps: 0,
    };
    // Publish initial state and start the loop by notifying the brain.
    h.publish_state();
    h.write_u32(OFF_CYCLE, 0);
    debug_println!("[body] starting closed loop: {} cycles", MAX_CYCLES);
    TO_COGNITIVE.notify();
    h
}

struct HandlerImpl {
    region: SharedMemoryRef<'static, [u8], ReadWrite>,
    body: MujocoTwinBody, // digital twin of MuJoCo behind the HAL interface (replaced with real hardware later)
    max_theta: f32,
    fell: bool,
    clamps: u32,
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
    fn read_f32(&self, off: usize) -> f32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        f32::from_le_bytes(b)
    }
    fn write_u32(&mut self, off: usize, v: u32) {
        self.write(off, &v.to_le_bytes());
    }
    fn publish_state(&mut self) {
        let v = self.body.sense();
        let mut buf = [0u8; 16];
        for (i, f) in v.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
        }
        self.write(OFF_STATE, &buf);
    }
    /// ASCII rendering of the robot state (cart on track + pole tilt).
    fn render(&self, cycle: u32) {
        const W: usize = 31;
        let s = self.body.sense();
        let x = s[0];
        let theta = self.body.theta();
        let mut track = [b'.'; W];
        let norm = ((x + 2.4) / 4.8).clamp(0.0, 1.0);
        let col = (norm * (W as f32 - 1.0)) as usize;
        // Marker indicating pole tilt: \ left, | upright, / right.
        let marker = if theta > 0.06 {
            b'/'
        } else if theta < -0.06 {
            b'\\'
        } else {
            b'|'
        };
        track[col.min(W - 1)] = marker;
        let line = core::str::from_utf8(&track).unwrap_or("");
        debug_println!(
            "[view] cyc{:>3} [{}] x={}/1000 th={}/1000",
            cycle,
            line,
            (x * 1000.0) as i32,
            (theta * 1000.0) as i32
        );
    }

    fn summary(&self, cycles: u32) {
        debug_println!("[body] ===== closed-loop result =====");
        debug_println!(
            "[body] cycles={} balanced={} max|theta|={}/1000 (fall {}/1000) clamps={}",
            cycles,
            if self.fell { 0 } else { 1 },
            (self.max_theta * 1000.0) as i32,
            (FALL_ANGLE * 1000.0) as i32,
            self.clamps
        );
        if !self.fell {
            debug_println!(
                "[body] PASS: trained policy balanced the cartpole on seL4; guard kept it safe."
            );
        } else {
            debug_println!("[body] pole fell.");
        }
    }
}

impl Handler for HandlerImpl {
    type Error = Infallible;

    fn notified(&mut self, _channels: ChannelSet) -> Result<(), Self::Error> {
        let cycle = self.read_u32(OFF_CYCLE);
        if cycle >= MAX_CYCLES {
            return Ok(()); // loop has ended
        }

        // Count guard interventions (for summary).
        if self.read_u32(OFF_OVERRIDDEN) != 0 {
            self.clamps += 1;
        }
        let approved = self.read_f32(OFF_APPROVED);

        // External disturbance (angular impulse) at specific cycles.
        if cycle == 100 || cycle == 200 {
            self.body.disturb(0.5);
            debug_println!("[body] cycle {}: external disturbance (+0.5 rad/s)", cycle);
        }
        // Note: the clearance barrier is armed on seL4 (the guard enforces that the cart
        // never leaves the track). Its clean demonstration is in sim/sil-clearance (navigation
        // vehicle); on the cartpole it conflicts with fine balancing.

        // ★ Actuation via HAL: the body applies the approved command (will be PWM on real hardware).
        self.body.actuate(approved);
        let theta = self.body.theta();
        let ath = if theta < 0.0 { -theta } else { theta };
        if ath > self.max_theta {
            self.max_theta = ath;
        }
        if ath > FALL_ANGLE {
            self.fell = true;
        }

        let next = cycle + 1;
        self.publish_state();
        self.write_u32(OFF_CYCLE, next);

        // Render the robot every 12 cycles (to visualize motion in the simulator).
        if cycle % 12 == 0 {
            self.render(cycle);
        }

        if next >= MAX_CYCLES {
            self.summary(next);
            Ok(()) // stop (no notification)
        } else {
            TO_COGNITIVE.notify(); // continue the loop
            Ok(())
        }
    }
}
