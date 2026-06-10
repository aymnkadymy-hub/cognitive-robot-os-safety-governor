//! # hal — robot body hardware abstraction layer
//!
//! Unified interface for sensing and actuation. **Today** it is implemented by a simulated
//! body (`SimBody`); **later on real hardware** it will be implemented by a real body
//! (IMU over I2C + motors over PWM) behind the same interface — so the body can be swapped
//! without any change to the brain/guard/loop. This prepares the project for hardware
//! integration without friction.

#![no_std]
#![forbid(unsafe_code)]

use cartpole_sim::{State, FORCE_MAG};

/// Robot body interface: sense current state + apply a normalized motion command.
pub trait RobotBody {
    /// Sense the current state as a vector (e.g., from IMU/encoder sensors).
    fn sense(&self) -> [f32; 4];
    /// Apply a normalized motion command in the range [-1, 1] (e.g., PWM to motors).
    fn actuate(&mut self, command: f32);
}

/// Simulated body implementation: cart-pole. (Replaced by `RealBody` on hardware.)
pub struct SimBody {
    state: State,
}

impl SimBody {
    /// Simulated body with an initial pendulum tilt.
    pub fn new(theta0: f32) -> Self {
        Self {
            state: State::upright(theta0),
        }
    }
    /// Inject an external disturbance (impulse on angular velocity) — for testing.
    pub fn disturb(&mut self, delta_theta_dot: f32) {
        self.state.theta_dot += delta_theta_dot;
    }
    /// Current pendulum angle (for monitoring/summarization).
    pub fn theta(&self) -> f32 {
        self.state.theta
    }
}

impl RobotBody for SimBody {
    fn sense(&self) -> [f32; 4] {
        self.state.as_vec()
    }
    fn actuate(&mut self, command: f32) {
        // Normalized command → physical force (on hardware: → PWM duty cycle).
        self.state = cartpole_sim::step(self.state, command * FORCE_MAG);
    }
}

/// Digital twin body for MuJoCo InvertedPendulum — the MuJoCo-trained policy runs directly on it.
pub struct MujocoTwinBody {
    state: cartpole_sim::mujoco_twin::MjState,
}

impl MujocoTwinBody {
    pub fn new(theta0: f32) -> Self {
        Self {
            state: cartpole_sim::mujoco_twin::MjState::upright(theta0),
        }
    }
    pub fn disturb(&mut self, delta_theta_dot: f32) {
        self.state.theta_dot += delta_theta_dot;
    }
    /// Impulse on the cart (velocity) — pushes it toward the track edge (to test the clearance barrier).
    pub fn push_cart(&mut self, delta_x_dot: f32) {
        self.state.x_dot += delta_x_dot;
    }
    pub fn theta(&self) -> f32 {
        self.state.theta
    }
}

impl RobotBody for MujocoTwinBody {
    fn sense(&self) -> [f32; 4] {
        self.state.as_vec() // MuJoCo order [x, theta, x_dot, theta_dot]
    }
    fn actuate(&mut self, command: f32) {
        self.state = cartpole_sim::mujoco_twin::step(self.state, command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sense_actuate_via_trait() {
        let mut body = SimBody::new(0.1);
        let s0 = body.sense();
        assert_eq!(s0[2], 0.1); // theta
        body.actuate(0.0); // no force → pendulum starts to fall
        let s1 = body.sense();
        assert!(s1 != s0); // state changed after physics step
    }

    #[test]
    fn disturb_changes_velocity() {
        let mut body = SimBody::new(0.0);
        body.disturb(1.5);
        assert_eq!(body.sense()[3], 1.5); // theta_dot
    }
}
