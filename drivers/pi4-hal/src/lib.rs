//! # pi4-hal — BCM2711 (Raspberry Pi 4) drivers for seL4: GPIO · PWM · I²C (BSC)
//!
//! Hardware access layer (MMIO) for driving the real robot:
//! - **GPIO**: pin function selection + set/clear (motor direction + STBY for TB6612).
//! - **PWM**: motor speed (mark-space) — two channels.
//! - **I²C (BSC)**: IMU read (MPU-6050) — polled write/read transactions.
//!
//! The logic (bit/register composition) is **testable on the host** via a mock `Bus`. On seL4 the
//! mapped device memory addresses are passed in from the system description. This layer **requires
//! unsafe (MMIO)** — it is the sole trusted layer; the rest of the system (guard/memory) remains `forbid(unsafe_code)`.

#![cfg_attr(not(test), no_std)]

// ===== BCM2711 addresses (Pi4, low-peripheral mode) =====
pub const PERIPH_BASE: usize = 0xFE00_0000;
pub const GPIO_BASE: usize = PERIPH_BASE + 0x20_0000; // 0xFE20_0000
pub const PWM0_BASE: usize = PERIPH_BASE + 0x20_C000; // 0xFE20_C000
pub const BSC1_BASE: usize = PERIPH_BASE + 0x80_4000; // 0xFE80_4000 (I²C on GPIO2/3)
pub const CM_PWM_BASE: usize = PERIPH_BASE + 0x10_10A0; // PWM clock manager

/// 32-bit access bus (real MMIO on seL4, or mock in tests).
pub trait Bus {
    fn read32(&self, off: usize) -> u32;
    fn write32(&self, off: usize, val: u32);
}

/// Real MMIO access for a base address mapped by seL4. **unsafe**: assumes `base` is a valid mapped device address.
#[derive(Clone, Copy)]
pub struct Mmio {
    base: usize,
}
impl Mmio {
    /// # Safety
    /// `base` must be a valid MMIO address mapped into this PD's address space (from the system description).
    pub const unsafe fn new(base: usize) -> Self {
        Self { base }
    }
}
impl Bus for Mmio {
    fn read32(&self, off: usize) -> u32 {
        // SAFETY: base is mapped device memory; off is aligned within the device range.
        unsafe { core::ptr::read_volatile((self.base + off) as *const u32) }
    }
    fn write32(&self, off: usize, val: u32) {
        // SAFETY: same as above.
        unsafe { core::ptr::write_volatile((self.base + off) as *mut u32, val) }
    }
}

// ===================== GPIO =====================
pub mod gpio {
    use super::Bus;
    pub const GPFSEL0: usize = 0x00;
    pub const GPSET0: usize = 0x1C;
    pub const GPCLR0: usize = 0x28;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Func {
        Input = 0b000,
        Output = 0b001,
        Alt0 = 0b100, // alternate function (PWM/I²C on specific pins)
        Alt5 = 0b010,
    }

    pub struct Gpio<'a, B: Bus>(pub &'a B);

    impl<B: Bus> Gpio<'_, B> {
        /// Select pin function (3 bits per pin in GPFSELn).
        pub fn set_function(&self, pin: u32, f: Func) {
            let reg = GPFSEL0 + (pin as usize / 10) * 4;
            let shift = (pin % 10) * 3;
            let mut v = self.0.read32(reg);
            v &= !(0b111 << shift);
            v |= (f as u32) << shift;
            self.0.write32(reg, v);
        }
        /// Drive pin high (assumes pin < 32).
        pub fn set_high(&self, pin: u32) {
            self.0.write32(GPSET0, 1 << pin);
        }
        /// Drive pin low.
        pub fn set_low(&self, pin: u32) {
            self.0.write32(GPCLR0, 1 << pin);
        }
        pub fn write(&self, pin: u32, high: bool) {
            if high {
                self.set_high(pin)
            } else {
                self.set_low(pin)
            }
        }
    }
}

// ===================== PWM =====================
pub mod pwm {
    use super::Bus;
    pub const CTL: usize = 0x00;
    pub const RNG1: usize = 0x10;
    pub const DAT1: usize = 0x14;
    pub const RNG2: usize = 0x20;
    pub const DAT2: usize = 0x24;
    // CTL bits
    pub const PWEN1: u32 = 1 << 0;
    pub const MSEN1: u32 = 1 << 7; // mark-space mode (motor speed)
    pub const PWEN2: u32 = 1 << 8;
    pub const MSEN2: u32 = 1 << 15;

    pub struct Pwm<'a, B: Bus>(pub &'a B);

    impl<B: Bus> Pwm<'_, B> {
        /// Initialize both channels in mark-space mode with a shared range (range = cycle resolution).
        pub fn init(&self, range: u32) {
            self.0.write32(RNG1, range);
            self.0.write32(RNG2, range);
            self.0.write32(CTL, PWEN1 | MSEN1 | PWEN2 | MSEN2);
        }
        /// Duty cycle for channel 1 [0..range] (e.g. left motor speed).
        pub fn set_duty1(&self, duty: u32) {
            self.0.write32(DAT1, duty);
        }
        pub fn set_duty2(&self, duty: u32) {
            self.0.write32(DAT2, duty);
        }
    }
}

// ===================== I²C (BSC) =====================
pub mod i2c {
    use super::Bus;
    pub const C: usize = 0x00;
    pub const S: usize = 0x04;
    pub const DLEN: usize = 0x08;
    pub const A: usize = 0x0C;
    pub const FIFO: usize = 0x10;
    pub const DIV: usize = 0x14;
    // C bits
    pub const I2CEN: u32 = 1 << 15;
    pub const ST: u32 = 1 << 7;
    pub const CLEAR: u32 = 1 << 4;
    pub const READ: u32 = 1 << 0;
    // S bits
    pub const DONE: u32 = 1 << 1;
    pub const TXD: u32 = 1 << 4;
    pub const RXD: u32 = 1 << 5;
    pub const ERR: u32 = 1 << 8;
    pub const CLKT: u32 = 1 << 9;

    const TIMEOUT: u32 = 100_000; // polling limit (avoids hanging on hardware error)

    /// I²C transaction error (useful for hardware diagnostics at boot).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Error {
        Timeout,  // transaction did not complete within the polling limit
        BusError, // ERR/CLKT (no device acknowledgement or clock stretch)
    }

    pub struct I2c<'a, B: Bus>(pub &'a B);

    impl<B: Bus> I2c<'_, B> {
        /// Clock divider to set the frequency (e.g. ~100kHz from 150MHz core clock → DIV≈1500).
        pub fn init(&self, divider: u16) {
            self.0.write32(DIV, divider as u32);
        }

        fn start(&self, slave: u8, len: u32, read: bool) {
            self.0.write32(A, slave as u32);
            self.0.write32(DLEN, len);
            let mut c = I2CEN | ST | CLEAR;
            if read {
                c |= READ;
            }
            self.0.write32(C, c);
        }

        fn wait_done(&self) -> Result<(), Error> {
            for _ in 0..TIMEOUT {
                let s = self.0.read32(S);
                if s & (ERR | CLKT) != 0 {
                    return Err(Error::BusError);
                }
                if s & DONE != 0 {
                    self.0.write32(S, DONE);
                    return Ok(());
                }
            }
            Err(Error::Timeout)
        }

        /// Write bytes to a slave device.
        pub fn write(&self, slave: u8, data: &[u8]) -> Result<(), Error> {
            self.start(slave, data.len() as u32, false);
            for &b in data {
                let mut spun = 0;
                while self.0.read32(S) & TXD == 0 {
                    spun += 1;
                    if spun > TIMEOUT {
                        return Err(Error::Timeout);
                    }
                }
                self.0.write32(FIFO, b as u32);
            }
            self.wait_done()
        }

        /// Read `buf.len()` bytes from a slave device.
        pub fn read(&self, slave: u8, buf: &mut [u8]) -> Result<(), Error> {
            self.start(slave, buf.len() as u32, true);
            for byte in buf.iter_mut() {
                let mut spun = 0;
                while self.0.read32(S) & RXD == 0 {
                    spun += 1;
                    if spun > TIMEOUT {
                        return Err(Error::Timeout);
                    }
                }
                *byte = (self.0.read32(FIFO) & 0xFF) as u8;
            }
            self.wait_done()
        }

        /// Read a register: write register address then read (standard pattern for most I²C sensors like MPU-6050).
        pub fn read_reg(&self, slave: u8, reg: u8, buf: &mut [u8]) -> Result<(), Error> {
            self.write(slave, &[reg])?;
            self.read(slave, buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;

    /// Mock bus: register memory + last written value log (for bit-logic testing).
    struct MockBus {
        regs: RefCell<[u32; 64]>,
    }
    impl MockBus {
        fn new() -> Self {
            Self {
                regs: RefCell::new([0; 64]),
            }
        }
    }
    impl Bus for MockBus {
        fn read32(&self, off: usize) -> u32 {
            self.regs.borrow()[off / 4]
        }
        fn write32(&self, off: usize, val: u32) {
            self.regs.borrow_mut()[off / 4] = val;
        }
    }

    #[test]
    fn gpio_function_select_sets_right_bits() {
        let bus = MockBus::new();
        let g = gpio::Gpio(&bus);
        g.set_function(18, gpio::Func::Output); // GPFSEL1, shift (18%10)*3=24
        assert_eq!(bus.read32(gpio::GPFSEL0 + 4), 0b001 << 24);
        g.set_function(2, gpio::Func::Alt0); // I²C SDA on GPFSEL0 shift 6
        assert_eq!(bus.read32(gpio::GPFSEL0) & (0b111 << 6), 0b100 << 6);
    }

    #[test]
    fn gpio_set_clear() {
        let bus = MockBus::new();
        let g = gpio::Gpio(&bus);
        g.set_high(23);
        assert_eq!(bus.read32(gpio::GPSET0), 1 << 23);
        g.set_low(23);
        assert_eq!(bus.read32(gpio::GPCLR0), 1 << 23);
    }

    #[test]
    fn pwm_init_and_duty() {
        let bus = MockBus::new();
        let p = pwm::Pwm(&bus);
        p.init(1024);
        assert_eq!(bus.read32(pwm::RNG1), 1024);
        assert!(bus.read32(pwm::CTL) & pwm::MSEN1 != 0); // mark-space mode
        assert!(bus.read32(pwm::CTL) & pwm::PWEN1 != 0);
        p.set_duty1(512);
        assert_eq!(bus.read32(pwm::DAT1), 512); // 50% duty
    }

    #[test]
    fn i2c_write_sets_slave_len_and_start() {
        let bus = MockBus::new();
        // Set DONE and TXD always high so the transaction completes in the test.
        bus.write32(i2c::S, i2c::DONE | i2c::TXD);
        let dev = i2c::I2c(&bus);
        // Simulate: every read of S returns DONE|TXD (kept high).
        let r = dev.write(0x68, &[0x6B, 0x00]); // MPU-6050: PWR_MGMT_1=0 (wake)
        assert!(r.is_ok());
        assert_eq!(bus.read32(i2c::A), 0x68);
        assert!(bus.read32(i2c::C) & i2c::ST != 0);
        assert!(bus.read32(i2c::C) & i2c::READ == 0); // write
    }

    #[test]
    fn i2c_read_sets_read_bit() {
        let bus = MockBus::new();
        bus.write32(i2c::S, i2c::DONE | i2c::RXD);
        let dev = i2c::I2c(&bus);
        let mut buf = [0u8; 6];
        let r = dev.read(0x68, &mut buf);
        assert!(r.is_ok());
        assert!(bus.read32(i2c::C) & i2c::READ != 0); // read
        assert_eq!(bus.read32(i2c::DLEN), 6);
    }
}
