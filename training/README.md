# training/ — host-side RL training (optional; provided for transparency)

**You do not need anything in this directory to reproduce any safety result.** The trained
weights are already committed inside the policy crates as Rust constants
(`components/*-policy/src/weights.rs`, `components/perception/src/weights.rs`). See
[`docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md) §4.

## Scripts

| Script | Produces |
|---|---|
| `train_policy.py` | `components/policy/src/weights.rs` |
| `train_cartpole.py` | `components/cartpole-policy/src/weights.rs` |
| `train_mujoco_pendulum.py` | `components/mujoco-pendulum-policy/src/weights.rs` |
| `train_walker.py` | `components/walker-policy/src/weights.rs` |
| `train_humanoid.py` | `components/humanoid-policy/src/weights.rs` |
| `train_encoder.py` | `components/perception/src/weights.rs` |
| `train_car.py` | `car_ppo.zip` (PPO checkpoint, see below) |
| `export_car_policy.py` | Rust weights exported from `car_ppo.zip` |

Training uses Python (PyTorch / Stable-Baselines3-style PPO/SAC); environments live under
[`sim/`](../sim/) (`car_env.py`, `minecraft_env.py`, MuJoCo XMLs).

## Provenance of the committed checkpoint `car_ppo.zip`

`car_ppo.zip` is a Stable-Baselines3 PPO checkpoint for the car environment
(`sim/car_env.py`), produced by `train_car.py` (which resumes from this file when present and
overwrites it each round). It is committed so that `export_car_policy.py` and the live viewer
(`sim/car_rl_watch.py`) work without retraining. Evaluation inside `train_car.py` uses fixed
seeds (`seed0=9000`, episodes paired across governed/ungoverned runs); the *safety* results in
the paper do not depend on this checkpoint's exact weights — the governor treats every policy
as an untrusted black box.
