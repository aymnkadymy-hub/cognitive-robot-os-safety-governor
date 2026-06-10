# Reproduction Guide

Goal: let an independent reader **rebuild the system and obtain the exact numbers** reported in the
paper. All experimental results are deterministic (see [`ENVIRONMENT.md`](ENVIRONMENT.md) §2).

## 0. One command

```sh
scripts/reproduce.sh | tee my-results.txt
```

This prints, in order:
- **(A)** property tests = the theorems, machine-checked on many inputs;
- **(B)** rigorous evaluation: 6 environments × methods × 40 seeds, with 95% CI;
- **(C)** the full 12-test reviewer battery;
- **(D)** the Pareto ablation;
- **(E)** guard timing (host-dependent absolute numbers);
- **(F)** the stress / break battery.

## 1. Paper table → command → reference file

| Paper artifact | Command (from repo root) | Reference output |
|---|---|---|
| Multi-environment evaluation + 95% CI | `(cd sim/sil-eval && cargo run --release)` | [`results/reference/B-sil-eval.txt`](../results/reference/B-sil-eval.txt) |
| 12-test reviewer battery (ablation, memory, adversarial, forgetting, transfer, stress, compute, failure, confirm_safe, scalability, FP/FN, long-horizon) | `(cd sim/sil-experiments && cargo run --release)` | [`results/reference/C-sil-experiments.txt`](../results/reference/C-sil-experiments.txt) |
| Pareto / mechanistic ablation | `(cd sim/sil-ablation && cargo run --release)` | [`results/reference/D-sil-ablation.txt`](../results/reference/D-sil-ablation.txt) |
| Guard-decision timing | `(cd sim/bench-guard && cargo run --release)` | [`results/reference/E-bench-guard.txt`](../results/reference/E-bench-guard.txt) |
| Stress / break battery | `(cd sim/sil-stress && cargo run --release)` | [`results/reference/F-sil-stress.txt`](../results/reference/F-sil-stress.txt) |
| Theorem 1–3 (envelope, tightening, verification-preservation) | `(cd components/safety-memory && cargo test --release)` | [`results/reference/A-proptest-safety-memory.txt`](../results/reference/A-proptest-safety-memory.txt) |
| Theorem 4 (clearance braking barrier) | `(cd components/clearance-guard && cargo test --release)` | [`results/reference/A-proptest-clearance-guard.txt`](../results/reference/A-proptest-clearance-guard.txt) |
| Learned-risk bound ≤ envelope | `(cd components/safety-model && cargo test --release)` | [`results/reference/A-proptest-safety-model.txt`](../results/reference/A-proptest-safety-model.txt) |

Additional SiL campaigns (supporting figures): `sim/sil-clearance` (67→0 hidden-hazard demo),
`sim/sil-ood` (OOD detection ROC/recall/precision), `sim/sil-generalize` (similarity-generalization
curve), `sim/sil-campaign` (E1–E12 evaluation campaign), `sim/sil-adversarial`
(multi-agent reciprocal-avoidance, cross-simulator unicycle dynamics, LearnedCBF baseline).

## 2. Verifying you reproduced the numbers

The experiment binaries print human-readable tables prefixed by tags (`[eval]`, `[abl]`,
`[bench]`, etc.). To confirm an exact match against the committed reference, compare the
program-output lines (ignoring cargo's compile messages on stderr):

```sh
(cd sim/sil-eval && cargo run --release 2>/dev/null) > /tmp/mine.txt
diff <(grep -E '^\[eval\]|^=== environment' /tmp/mine.txt) \
     <(grep -E '^\[eval\]|^=== environment' results/reference/B-sil-eval.txt) \
  && echo "EXACT MATCH"
```

A clean `EXACT MATCH` means your build reproduced the published evaluation bit-for-bit.
(Timing in `E-bench-guard.txt` will differ in absolute ns but not in ratios/ordering — expected.)

## 3. Optional: seL4-on-QEMU reflex arc

Requires the optional container + Microkit SDK ([`ENVIRONMENT.md`](ENVIRONMENT.md) §3):

```sh
scripts/build.sh sel4     # build + boot seL4 "hello" on QEMU (sanity)
scripts/build.sh reflex   # build + boot Guard + Cognitive + Actuation PDs; the Guard preempts
                          # the (deliberately adversarial) brain on the verified microkernel
```
Expected: QEMU log shows the guard clamping/preempting the brain, the watchdog firing an
emergency stop on brain-stall, and the actuation PD applying only `OFF_APPROVED` commands.

## 4. Optional: full build + analysis pipeline

```sh
scripts/build.sh host      # cargo test + clippy -D warnings + fmt --check, every crate
scripts/build.sh dynamic   # valgrind (no leaks) + latency bench
scripts/build.sh demo      # memories survive 3 simulated power-loss reboots
scripts/build.sh all       # everything in sequence
```

## 5. Expected wall-clock

On a typical laptop, each SiL experiment compiles in a few seconds and runs in well under a
minute. The full `scripts/reproduce.sh` completes in a couple of minutes from a clean checkout.
