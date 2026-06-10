# Reference Results — provenance

This directory holds the **committed reference outputs**: the deterministic ground truth an
independent reader compares their own runs against. Every file is the verbatim stdout of a command
in this repository (see the mapping below). The experimental rows are deterministic
(fixed-seed LCG); only absolute timing (`E-bench-guard.txt`) is host-dependent.

## How these were produced
- Toolchain: `rustc 1.96.0` (recorded in [`reference/00-toolchain.txt`](reference/00-toolchain.txt)).
- Each file = `cargo run --release` (experiments) or `cargo test --release` (theorems), stdout only.
- These outputs were verified **identical** before and after the English translation of the source
  comments — i.e. translating comments did not change a single number (the program output is
  byte-for-byte unchanged).

## File → command provenance

| File | Produced by | Documented in |
|---|---|---|
| `reference/00-toolchain.txt` | `rustc --version` | — |
| `reference/A-proptest-safety-memory.txt` | `cd components/safety-memory && cargo test --release` | [`../docs/VERIFICATION.md`](../docs/VERIFICATION.md) |
| `reference/A-proptest-clearance-guard.txt` | `cd components/clearance-guard && cargo test --release` | VERIFICATION.md |
| `reference/A-proptest-safety-model.txt` | `cd components/safety-model && cargo test --release` | VERIFICATION.md |
| `reference/B-sil-eval.txt` | `cd sim/sil-eval && cargo run --release` | [`../docs/EVALUATION.md`](../docs/EVALUATION.md) |
| `reference/C-sil-experiments.txt` | `cd sim/sil-experiments && cargo run --release` | [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) |
| `reference/D-sil-ablation.txt` | `cd sim/sil-ablation && cargo run --release` | EXPERIMENTS.md §1, EVALUATION.md |
| `reference/E-bench-guard.txt` | `cd sim/bench-guard && cargo run --release` | EXPERIMENTS.md §7 |
| `reference/F-sil-stress.txt` | `cd sim/sil-stress && cargo run --release` | EXPERIMENTS.md §6, FORMAL_GUARANTEES.md §10 |

## Reproduce and compare
See [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md). The quick check:
```sh
(cd sim/sil-eval && cargo run --release 2>/dev/null) > /tmp/mine.txt
diff <(grep -E '^\[eval\]|^=== environment' /tmp/mine.txt) \
     <(grep -E '^\[eval\]|^=== environment' results/reference/B-sil-eval.txt) \
  && echo "EXACT MATCH"
```
