//! # cartpole-sim
//!
//! Cart-pole physics (simulated robot body) in `no_std` — runs on seL4 and on Host.
//! Used as the "body" inside a PD on the verified kernel (Phase D0/D2) and in the SIL loop on Host.
//! Uses `libm` for sin/cos (not available in core).

#![no_std]
#![forbid(unsafe_code)]

pub mod mujoco_twin;

pub const G: f32 = 9.8;
pub const M_CART: f32 = 1.0;
pub const M_POLE: f32 = 0.1;
pub const LEN: f32 = 0.5; // half pole length
pub const FORCE_MAG: f32 = 10.0; // actuator force at ±1 command
pub const DT: f32 = 0.02;
pub const FALL_ANGLE: f32 = 0.40; // radians (~23°): exceeding this = fallen

/// System state: cart position and velocity, pole angle and angular velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub x: f32,
    pub x_dot: f32,
    pub theta: f32,
    pub theta_dot: f32,
}

impl State {
    /// Initial state with pole tilt `theta0`.
    pub const fn upright(theta0: f32) -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: theta0,
            theta_dot: 0.0,
        }
    }

    /// Represent state as a vector (for neural memory and policy).
    pub fn as_vec(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

/// Single physics step with force `force` (Newtons). Semi-implicit Euler integration.
pub fn step(s: State, force: f32) -> State {
    let total_mass = M_CART + M_POLE;
    let pml = M_POLE * LEN;
    let sin = libm::sinf(s.theta);
    let cos = libm::cosf(s.theta);
    let temp = (force + pml * s.theta_dot * s.theta_dot * sin) / total_mass;
    let theta_acc = (G * sin - cos * temp) / (LEN * (4.0 / 3.0 - M_POLE * cos * cos / total_mass));
    let x_acc = temp - pml * theta_acc * cos / total_mass;
    State {
        x: s.x + DT * s.x_dot,
        x_dot: s.x_dot + DT * x_acc,
        theta: s.theta + DT * s.theta_dot,
        theta_dot: s.theta_dot + DT * theta_acc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_force_pole_falls() {
        // Without force, the tilted pole falls (angle grows).
        let mut s = State::upright(0.1);
        for _ in 0..100 {
            s = step(s, 0.0);
        }
        assert!(s.theta.abs() > 0.1, "pole should diverge without control");
    }

    #[test]
    fn as_vec_roundtrip() {
        let s = State {
            x: 1.0,
            x_dot: 2.0,
            theta: 0.3,
            theta_dot: -0.4,
        };
        assert_eq!(s.as_vec(), [1.0, 2.0, 0.3, -0.4]);
    }
}
