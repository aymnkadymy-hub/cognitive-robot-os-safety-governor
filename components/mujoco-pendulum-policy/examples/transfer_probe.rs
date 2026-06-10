//! Transfer probe with physics matching MuJoCo InvertedPendulum (digital twin).
//! MuJoCo parameters: mc=10.47, mp=5.02, half_length=0.3, g=9.81, damping=1, gear=100, dt=0.02 ×2.
use mujoco_pendulum_policy::command;

const MC: f32 = 10.47;
const MP: f32 = 5.02;
const L: f32 = 0.3;
const G: f32 = 9.81;
const B: f32 = 1.0; // joint damping
const DT: f32 = 0.02;
const SUBSTEPS: usize = 2;
const GEAR: f32 = 100.0;

fn substep(s: [f32; 4], force: f32) -> [f32; 4] {
    let [x, th, xd, thd] = s;
    let (sin, cos) = (th.sin(), th.cos());
    let total = MC + MP;
    // Effective force with cart damping and pole damping torque.
    let f = force - B * xd;
    let temp = (f + MP * L * thd * thd * sin) / total;
    let thacc =
        (G * sin - cos * temp - B * thd / (MP * L)) / (L * (4.0 / 3.0 - MP * cos * cos / total));
    let xacc = temp - MP * L * thacc * cos / total;
    [x + DT * xd, th + DT * thd, xd + DT * xacc, thd + DT * thacc]
}

fn run() -> i32 {
    // State [x, theta, x_dot, theta_dot] (MuJoCo order).
    let mut s = [0.0f32, 0.02, 0.0, 0.0];
    for t in 0..1000 {
        let cmd = command(&s); // [-1,1]
        let force = cmd * 3.0 * GEAR; // ctrl=cmd*3, force=gear*ctrl
        for _ in 0..SUBSTEPS {
            s = substep(s, force);
        }
        if s[1].abs() > 0.2 {
            return t;
        }
    }
    1000
}

fn main() {
    let len = run();
    println!(
        "MuJoCo-twin physics: balanced {len}/1000  ({})",
        if len >= 990 {
            "TRANSFER OK ✓"
        } else {
            "still a gap"
        }
    );
}
