# Cognitive Robot OS — Verification-Preserving Adaptive Safety Governor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20645786.svg)](https://doi.org/10.5281/zenodo.20645786)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)

> 📄 **Preprint (full paper, open access):** https://doi.org/10.5281/zenodo.20645786

> **Reproducibility artifact** for the paper *Verification-Preserving Adaptive Safety for Untrusted
> Robot Policies on a Formally Verified Microkernel*. Everything needed for an independent
> reader to **rebuild the system and reproduce every experimental result** is in this repository.

> **Open source.** Code is released under the MIT licence; the paper, figures, and prose
> documentation under CC BY 4.0 (see [`LICENSE`](LICENSE)). To cite, see [`CITATION.cff`](CITATION.cff).

---

## 1. What this is (precise positioning)

This is **not** a replacement for ROS 2 and not a general-purpose robot OS. The core contribution
is a **model-agnostic, formally-bounded safety-governance architecture for learned policies**:

> A small, mathematically-verified **Guard** has final authority over the actuators. It keeps
> **any** untrusted "brain" (an RL/AI policy, treated as a black box) inside a **proven safety
> envelope** that (a) adapts from an **incident memory** (tighten-only, with optional
> bounded relaxation), and (b) enforces a **clearance braking barrier** — **without ever
> re-verifying the microkernel**. Learning changes *data the guard reads*, never verified code.

The system is implemented as isolated **seL4 / Microkit protection domains** (a deterministic
Guard "spinal cord" at highest priority, and a Rust + neural Cognitive "brain"), plus a
**neural-semantic memory** that stores *meaning vectors* and retrieves by similarity.

### Project status (honest)
Research / early proof-of-concept. Validated on **seL4 + QEMU** and in **software-in-the-loop
(SiL)** simulation. **Not yet on physical hardware** — that is the single remaining gap, stated
explicitly throughout. The formal theorems are **machine-checked by property tests** (thousands
of inputs) and Kani proof harnesses, *not yet* by an interactive prover (Coq/Isabelle) — also
stated explicitly.

---

## 2. The headline results (all reproducible — see §4)

All experimental numbers are **deterministic** (fixed-seed LCG; `no_std`, no heap, no global
state): anyone who runs the code gets the **same numbers**. Reference outputs are committed under
[`results/reference/`](results/reference/).

- **Beats established safety techniques, not naive baselines.** Across **6 environments × 8
  methods × 40 seeds** (95% CI), the adaptive governor is **5–10× safer than reactive CBF and
  Simplex** and Pareto-dominates static envelopes in familiar/delayed conditions.
  ([`docs/EVALUATION.md`](docs/EVALUATION.md), [`results/reference/B-sil-eval.txt`](results/reference/B-sil-eval.txt))
- **Mechanistic evidence (why it works).** Ablation: removing **incident memory** or
  **similarity generalization** raises unsafe events from **0 → 29**. The safety source is the
  memory + similarity, nothing else.
  ([`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), [`results/reference/C-sil-experiments.txt`](results/reference/C-sil-experiments.txt))
- **Pareto frontier.** As safe as `static-slow` (0 unsafe) with **~1.8× the liveness**, and safe
  where reactive methods are blind to hidden hazards.
  ([`results/reference/D-sil-ablation.txt`](results/reference/D-sil-ablation.txt))
- **Robust to poisoning.** Because tightening is monotone, a poisoned memory can only make the
  robot **more conservative**, never less safe (worst case = over-caution).
- **Bounded, cheap compute.** One full guard decision ≈ **38 ns** (host; `O(1)`, allocation-free);
  `SafetyMemory<128,2>` ≈ **2 KB** static RAM.
  ([`results/reference/E-bench-guard.txt`](results/reference/E-bench-guard.txt))
- **Honest failure map.** Under broken physical assumptions (sluggish actuator, 200 ms latency,
  "perfect storm") collision-safety degrades for **all** methods; only the **L0-containment**
  invariant (speed ≤ vmax) holds unconditionally. Reported, not hidden.
  ([`results/reference/F-sil-stress.txt`](results/reference/F-sil-stress.txt))

---

## 3. Repository layout

This repository keeps the **buildable Cargo crate tree** (so it compiles and the results
reproduce). The conceptual grouping requested for a safety-governance artifact
(*governor / memory / verification / experiments / results / scripts / docs*) is provided by
[`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md), which maps each concept to concrete paths.

```
Repository/
├── README.md                  ← you are here
├── LICENSE, CITATION.cff
├── components/                ← Rust no_std crates (the system)
│   ├── safety-memory/         ← adaptive tighten-only envelope  (Claim A — core invention)
│   ├── clearance-guard/       ← clearance braking barrier (Thm 4)
│   ├── safety-model/          ← learned per-context risk bound
│   ├── contextual-guard/      ← context → effective-limit selection
│   ├── ood-detector/          ← Mahalanobis OOD gate
│   ├── memory/, world-memory/ ← neural-semantic memory + persistence
│   ├── sel4-guard-pd/ …-memory-pd/ …-actuation-pd/   ← seL4/Microkit protection domains
│   └── …policies, perception, ABIs, HAL…
├── sim/                       ← software-in-the-loop EXPERIMENTS (each crate prints a result table)
│   ├── sil-eval/              ← B) multi-environment evaluation + 95% CI
│   ├── sil-experiments/       ← C) 12-test reviewer battery
│   ├── sil-ablation/          ← D) Pareto / mechanistic ablation
│   ├── sil-stress/            ← F) stress / break battery
│   ├── bench-guard/           ← E) guard-decision timing
│   └── …clearance, ood, generalize, campaign, adversarial…
├── kernel/                    ← seL4 system descriptions (reflex.system)
├── drivers/                   ← Raspberry Pi 4 HAL + body drivers (I2C/UART/GPIO)
├── training/                  ← host-side RL training (PPO/SAC) → ONNX → Rust weights
├── scripts/                   ← reproduce.sh (one command) + build.sh + md2pdf.py
├── results/reference/         ← committed deterministic reference outputs (the ground truth)
└── docs/                      ← all documentation (see below)
```

### Documentation index
| Document | Purpose |
|---|---|
| [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) | Concept → path map, crate-by-crate index |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Exact toolchain + optional seL4/QEMU container setup |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | One-command + per-experiment reproduction, expected outputs |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Multi-environment evaluation (fairness protocol + tables) |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | The 12-test reviewer battery, with result tables |
| [`docs/FORMAL_GUARANTEES.md`](docs/FORMAL_GUARANTEES.md) | Assumptions, invariants, theorems, proof sketches, limits |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | Verification ledger: each theorem ↔ its machine-checked test/proof |
| [`results/README.md`](results/README.md) | Provenance of every committed reference output |

---

## 4. Quick start (reproduce everything)

**Requirement:** a Rust toolchain (tested with `rustc 1.96.0`). No GPU, network, or hardware is
needed for the SiL evaluation. See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for details and the
optional seL4/QEMU path.

```sh
# One command — deterministic; prints the experimental rows used in the paper:
scripts/reproduce.sh | tee my-results.txt

# Then confirm you got the published numbers:
diff <(grep -E '^\[eval\]|^=== environment' my-results.txt) \
     <(grep -E '^\[eval\]|^=== environment' results/reference/B-sil-eval.txt)
```

Per-experiment runs (each compiles in seconds and prints its own table):

```sh
(cd sim/sil-eval        && cargo run --release)   # multi-environment evaluation + CI
(cd sim/sil-experiments && cargo run --release)   # 12-test reviewer battery
(cd sim/sil-ablation    && cargo run --release)   # Pareto / mechanistic ablation
(cd sim/sil-stress      && cargo run --release)   # stress / break battery
(cd sim/bench-guard     && cargo run --release)   # guard-decision timing

# The theorems, machine-checked on thousands of inputs:
(cd components/safety-memory  && cargo test --release)   # 14 tests
(cd components/clearance-guard && cargo test --release)  # 13 tests
(cd components/safety-model    && cargo test --release)  # 7 tests
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full mapping of *paper table → command →
reference file*, and the (optional) seL4-on-QEMU reflex-arc demo.

---

## 5. Authorship & citation

Author: **Ayman Kazem Yousef** (Undergraduate Student, Department of Artificial Intelligence Engineering, AlSafwa University). See [`CITATION.cff`](CITATION.cff).

## 6. License

Code is released under the **MIT** licence; the paper, figures, and prose documentation under
**CC BY 4.0**. See [`LICENSE`](LICENSE).
