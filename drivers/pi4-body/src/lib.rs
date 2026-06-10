//! # pi4-body — `RealBody`: the real robot body on Pi4 (same interface as `hal::RobotBody`)
//!
//! - **`sense()`**: reads MPU-6050 (GY-521) over I²C → computes tilt and tilt rate → state vector [f32;4].
//! - **`actuate(cmd)`**: `cmd∈[-1,1]` → direction (TB6612 pins) + speed (PWM) for both motors.
//!
//! Replaces `MujocoTwinBody` in the actuation PD on hardware **with no changes to brain/guard/loop**.
//! Wiring in `docs/HARDWARE_DRIVERS.md`. Conversion logic (i16/tilt/cycle) tested on the host.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)] // all unsafe is confined to pi4-hal (the trusted MMIO layer)

use core::cell::Cell;
use hal::RobotBody;
use imu_filter::MahonyPitchFilter;
use pi4_hal::{gpio, i2c, pwm, Mmio};

// MPU-6050
const MPU_ADDR: u8 = 0x68;
const REG_PWR_MGMT_1: u8 = 0x6B;
const REG_ACCEL_XOUT_H: u8 = 0x3B;

/// Filter time step (s) — **must match the actual control loop period** (~100Hz by default).
const SENSE_DT: f32 = 0.01;

// GPIO pins (BCM numbering) — see docs/HARDWARE_DRIVERS.md
const AIN1: u32 = 5;
const AIN2: u32 = 6;
const BIN1: u32 = 20;
const BIN2: u32 = 21;
const STBY: u32 = 26;
const PWM_A: u32 = 12; // ALT0 = PWM channel 0
const PWM_B: u32 = 13; // ALT0 = PWM channel 1
const SDA: u32 = 2;
const SCL: u32 = 3;

const PWM_RANGE: u32 = 1024;

/// Convert two big-endian bytes to i16 (MPU-6050 register format).
pub fn be_i16(hi: u8, lo: u8) -> i16 {
    (((hi as u16) << 8) | lo as u16) as i16
}

/// Pitch angle (radians) from axis accelerations (atan2).
pub fn accel_to_pitch(ax: f32, ay: f32, az: f32) -> f32 {
    libm::atan2f(ax, libm::sqrtf(ay * ay + az * az))
}

/// Normalized command [-1,1] → PWM duty cycle within [0, range].
pub fn cmd_to_duty(cmd: f32, range: u32) -> u32 {
    let mag = cmd.abs().min(1.0);
    (mag * range as f32) as u32
}

/// Real Pi4 body. Owns the hardware addresses (passed in from seL4-mapped vaddrs).
pub struct RealBody {
    gpio: Mmio,
    pwm: Mmio,
    i2c: Mmio,
    range: u32,
    /// Mahony pitch fusion filter — behind `Cell` because `sense` takes `&self`. Single-threaded scope.
    filter: Cell<MahonyPitchFilter>,
}

impl RealBody {
    /// Takes pre-built MMIO objects (constructed by the PD via `Mmio::new` unsafe from seL4 vaddrs) —
    /// keeping `pi4-body` fully safe, with unsafe confined to `pi4-hal` and the PD.
    pub fn new(gpio: Mmio, pwm: Mmio, i2c: Mmio) -> Self {
        Self {
            gpio,
            pwm,
            i2c,
            range: PWM_RANGE,
            filter: Cell::new(MahonyPitchFilter::default_gains()),
        }
    }

    /// **Gyro calibration at boot:** averages `n` readings while the robot is stationary → initial bias subtracted from rate.
    /// Called once after `init` with the robot still. (The filter's integral term refines it further online.)
    pub fn calibrate_gyro(&self, n: u32) {
        let dev = i2c::I2c(&self.i2c);
        let mut sum = 0.0f32;
        let mut got = 0u32;
        for _ in 0..n {
            let mut b = [0u8; 14];
            if dev.read_reg(MPU_ADDR, REG_ACCEL_XOUT_H, &mut b).is_ok() {
                let gy = be_i16(b[10], b[11]) as f32 / 131.0;
                sum += gy * core::f32::consts::PI / 180.0;
                got += 1;
            }
        }
        if got > 0 {
            let mut filter = self.filter.get();
            filter.set_gyro_bias(sum / got as f32);
            self.filter.set(filter);
        }
    }

    /// Configure pins + PWM + I²C, and wake the IMU. Called once at boot.
    pub fn init(&self) -> Result<(), i2c::Error> {
        let g = gpio::Gpio(&self.gpio);
        // Motor direction pins + STBY as outputs.
        for p in [AIN1, AIN2, BIN1, BIN2, STBY] {
            g.set_function(p, gpio::Func::Output);
        }
        g.set_high(STBY); // enable motor driver
                          // PWM and I²C pins: alternate function 0.
        for p in [PWM_A, PWM_B, SDA, SCL] {
            g.set_function(p, gpio::Func::Alt0);
        }
        pwm::Pwm(&self.pwm).init(self.range);
        let dev = i2c::I2c(&self.i2c);
        dev.init(1500); // ~100kHz
        dev.write(MPU_ADDR, &[REG_PWR_MGMT_1, 0x00]) // wake MPU-6050
    }
}

impl RobotBody for RealBody {
    fn sense(&self) -> [f32; 4] {
        let dev = i2c::I2c(&self.i2c);
        let mut b = [0u8; 14]; // accel xyz, temp, gyro xyz
        if dev.read_reg(MPU_ADDR, REG_ACCEL_XOUT_H, &mut b).is_err() {
            return [0.0; 4]; // read failure → neutral state (guard handles safety)
        }
        let ax = be_i16(b[0], b[1]) as f32 / 16384.0; // ±2g
        let ay = be_i16(b[2], b[3]) as f32 / 16384.0;
        let az = be_i16(b[4], b[5]) as f32 / 16384.0;
        let gy = be_i16(b[10], b[11]) as f32 / 131.0; // ±250°/s (pitch axis)
        let accel_pitch = accel_to_pitch(ax, ay, az);
        let gyro_rate = gy * core::f32::consts::PI / 180.0; // rad/s (raw)
                                                            // Mahony fusion: drift-free, low-noise pitch + online gyro bias removal —
                                                            // eliminates the main source of false DANGER_THETA alarms on real hardware.
        let mut filter = self.filter.get();
        let (pitch, pitch_rate) = filter.update(accel_pitch, gyro_rate, SENSE_DT);
        self.filter.set(filter);
        // [pitch, pitch_rate, (reserved), (reserved)] — state semantics per the robot.
        [pitch, pitch_rate, 0.0, 0.0]
    }

    fn actuate(&mut self, command: f32) {
        let g = gpio::Gpio(&self.gpio);
        let forward = command >= 0.0;
        // Motor direction (TB6612: IN1/IN2 are complementary).
        g.write(AIN1, forward);
        g.write(AIN2, !forward);
        g.write(BIN1, forward);
        g.write(BIN2, !forward);
        // Speed (PWM duty cycle).
        let duty = cmd_to_duty(command, self.range);
        let p = pwm::Pwm(&self.pwm);
        p.set_duty1(duty);
        p.set_duty2(duty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn be_i16_signed() {
        assert_eq!(be_i16(0x00, 0x00), 0);
        assert_eq!(be_i16(0x40, 0x00), 16384); // +1g on ±2g
        assert_eq!(be_i16(0xFF, 0xFF), -1);
        assert_eq!(be_i16(0xC0, 0x00), -16384);
    }

    #[test]
    fn pitch_upright_is_zero() {
        // Level: az=1g, ax=0 → pitch ≈ 0.
        assert!(accel_to_pitch(0.0, 0.0, 1.0).abs() < 1e-3);
        // Tilted forward: ax=1g, az=0 → ≈ +90°.
        assert!((accel_to_pitch(1.0, 0.0, 0.0) - core::f32::consts::FRAC_PI_2).abs() < 1e-3);
    }

    #[test]
    fn duty_maps_magnitude() {
        assert_eq!(cmd_to_duty(0.0, 1024), 0);
        assert_eq!(cmd_to_duty(0.5, 1024), 512);
        assert_eq!(cmd_to_duty(-1.0, 1024), 1024); // magnitude only (direction via pins)
        assert_eq!(cmd_to_duty(2.0, 1024), 1024); // clamped
    }
}
