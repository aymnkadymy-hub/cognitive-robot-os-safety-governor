# Repository Map — concepts → paths

This repository keeps the **buildable Cargo crate tree** so that everything compiles and the
results reproduce exactly. A reorganization into top-level `governor/`, `memory/`, … directories
would break Cargo's relative path-dependencies between crates. Instead, this document provides the
conceptual grouping and maps each concept to the concrete source paths.

## Conceptual grouping

### `governor/` — the verified safety guard (the core invention)
| Crate | Role |
|---|---|
| [`components/safety-memory/`](../components/safety-memory/) | **Claim A core.** Adaptive, tighten-only safety envelope: incident memory + similarity retrieval + effective-limit `Λ(c) = min(L0, …) ≤ L0`; optional evidence-based bounded relaxation (`confirm_safe`). |
| [`components/clearance-guard/`](../components/clearance-guard/) | Clearance braking barrier `v_safe(d)` (Theorem 4) for **sensed** hazards. |
| [`components/safety-model/`](../components/safety-model/) | Learned per-context risk bound (logistic), clamped to the verified envelope. |
| [`components/contextual-guard/`](../components/contextual-guard/) | Context → effective-limit selection / arbitration. |
| [`components/ood-detector/`](../components/ood-detector/) | Mahalanobis out-of-distribution gate (caution on novel contexts). |
| [`components/reactive-evasion/`](../components/reactive-evasion/) | Worst-case evasion cap (tighten-only, machine-checked). |
| [`components/neural-safety/`](../components/neural-safety/), [`components/safety/`](../components/safety/) | Supporting safety glue. |

### `memory/` — neural-semantic memory
| Crate | Role |
|---|---|
| [`components/memory/`](../components/memory/) | Neural-semantic memory: meaning-vector store, similarity recall, persistence across power loss (`persist.rs`), memory tiers (`tiers.rs`). |
| [`components/world-memory/`](../components/world-memory/) | World-level memory abstraction. |
| (`components/safety-memory/` also lives here conceptually — it *is* memory used for safety.) |

### `verification/` — machine-checked guarantees
Verification lives **inside the crates it verifies** (Rust `#[test]`, `proptest`, and
`#[cfg(kani)]` proof harnesses). See [`VERIFICATION.md`](VERIFICATION.md) for the theorem ↔ test
ledger. Crates carrying proof harnesses: `safety-memory`, `clearance-guard`, `safety-model`,
`ood-detector`, `imu-filter`, `reactive-evasion`, `reflex-abi`, `brain-os-abi`, `sel4-guard-pd`.

### `experiments/` — software-in-the-loop campaigns
All under [`sim/`](../sim/). Each crate is a standalone binary that prints a result table.
See [`REPRODUCE.md`](REPRODUCE.md) for which experiment maps to which paper table.

### `os/` — seL4 / Microkit system
| Path | Role |
|---|---|
| [`components/sel4-guard-pd/`](../components/sel4-guard-pd/) | Highest-priority Guard protection domain (the "spinal cord"). |
| [`components/sel4-memory-pd/`](../components/sel4-memory-pd/) | Cognitive/memory protection domain (the "brain"). |
| [`components/sel4-actuation-pd/`](../components/sel4-actuation-pd/) | Actuation protection domain (reads `OFF_APPROVED` only). |
| [`components/reflex-abi/`](../components/reflex-abi/), [`components/brain-os-abi/`](../components/brain-os-abi/) | Shared-memory ABIs between PDs. |
| [`kernel/reflex.system`](../kernel/reflex.system), [`kernel/reflex-hw.system`](../kernel/reflex-hw.system) | Microkit system descriptions (QEMU and hardware). |
| [`components/hal/`](../components/hal/), [`drivers/`](../drivers/) | HAL + Raspberry Pi 4 drivers (I2C/UART/GPIO, IMU, camera). |

### `policies/` — the untrusted "brains" (RL/AI) and perception
`components/policy`, `nav-policy`, `cartpole-policy`, `humanoid-policy`, `walker-policy`,
`mujoco-pendulum-policy`, `perception`, `behavior-fsm`, `cartpole-sim`, `imu-filter`.
Trained host-side under [`training/`](../training/) (PPO/SAC → ONNX → Rust `weights.rs`).

### `scripts/`, `results/`, `docs/`
[`scripts/reproduce.sh`](../scripts/reproduce.sh) (one-command repro),
[`scripts/build.sh`](../scripts/build.sh) (full build + static/dynamic analysis),
[`results/reference/`](../results/reference/) (committed deterministic outputs),
and this `docs/` tree.

## Why per-crate (no workspace root)
Each crate has its own `Cargo.toml`/`Cargo.lock` and is built independently, exactly as
`scripts/reproduce.sh` invokes them. This keeps `no_std` safety crates isolated from `std`
experiment harnesses and lets each be verified/benchmarked on its own.
