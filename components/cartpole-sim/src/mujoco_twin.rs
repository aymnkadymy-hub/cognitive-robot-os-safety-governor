//! no_std digital twin of MuJoCo InvertedPendulum-v5 physics — actual parameters.
//! Allows a policy trained in MuJoCo to run and balance on seL4 (same physics = successful transfer).
//! State in MuJoCo order: [x, theta, x_dot, theta_dot].

use libm::{cosf, sinf};

pub const MC: f32 = 10.47; // cart mass (MuJoCo)
pub const MP: f32 = 5.02; // pole mass
pub const L: f32 = 0.3; // half length
pub const G: f32 = 9.81;
pub const B: f32 = 1.0; // joint damping
pub const DT: f32 = 0.02;
pub const SUBSTEPS: usize = 2; // frameskip
pub const GEAR: f32 = 100.0; // force = gear × ctrl
pub const CTRL_MAX: f32 = 3.0; // ctrl range
pub const FALL_ANGLE: f32 = 0.2; // fall threshold in MuJoCo

/// State in MuJoCo order: [x, theta, x_dot, theta_dot].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MjState {
    pub x: f32,
    pub theta: f32,
    pub x_dot: f32,
    pub theta_dot: f32,
}

impl MjState {
    pub const fn upright(theta0: f32) -> Self {
        Self {
            x: 0.0,
            theta: theta0,
            x_dot: 0.0,
            theta_dot: 0.0,
        }
    }
    /// Vector in MuJoCo order (for the trained policy).
    pub fn as_vec(&self) -> [f32; 4] {
        [self.x, self.theta, self.x_dot, self.theta_dot]
    }
}

fn substep(s: MjState, force: f32) -> MjState {
    let (sin, cos) = (sinf(s.theta), cosf(s.theta));
    let total = MC + MP;
    let f = force - B * s.x_dot; // cart damping
    let temp = (f + MP * L * s.theta_dot * s.theta_dot * sin) / total;
    let theta_acc = (G * sin - cos * temp - B * s.theta_dot / (MP * L))
        / (L * (4.0 / 3.0 - MP * cos * cos / total));
    let x_acc = temp - MP * L * theta_acc * cos / total;
    MjState {
        x: s.x + DT * s.x_dot,
        theta: s.theta + DT * s.theta_dot,
        x_dot: s.x_dot + DT * x_acc,
        theta_dot: s.theta_dot + DT * theta_acc,
    }
}

/// Single control step with normalised `command` ∈ [-1,1] (converted to force via gear).
pub fn step(mut s: MjState, command: f32) -> MjState {
    let force = command * CTRL_MAX * GEAR;
    for _ in 0..SUBSTEPS {
        s = substep(s, force);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_force_falls() {
        let mut s = MjState::upright(0.1);
        for _ in 0..200 {
            s = step(s, 0.0);
        }
        assert!(s.theta.abs() > 0.1);
    }
}
