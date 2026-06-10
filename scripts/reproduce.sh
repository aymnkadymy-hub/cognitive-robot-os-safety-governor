#!/usr/bin/env bash
#
# Reproduce the safety-governance evaluation with a single command (deterministic; fixed seeds).
# Usage:  scripts/reproduce.sh   [> results.txt]
#
# Experimental results (eval/ablation/battery) are **deterministic** (LCG with fixed seeds)
# => anyone obtains the same numbers. Timing numbers (bench) depend on the host but ratios are stable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================ REPRODUCIBILITY RUN ================"
echo "date(UTC-agnostic): deterministic outputs below"
echo "=== toolchain ==="
rustc --version || true
cargo --version || true

echo ""
echo "=== A) property tests = the theorems, machine-checked on many inputs ==="
for c in safety-memory clearance-guard safety-model; do
  printf '  %-16s ' "$c:"
  (cd "$ROOT/components/$c" && cargo test --release 2>&1 | grep "test result" | head -1)
done

echo ""
echo "=== B) rigorous evaluation: 6 environments x 5 methods x 40 seeds, 95% CI ==="
(cd "$ROOT/sim/sil-eval" && cargo run --release 2>/dev/null)

echo ""
echo "=== C) full experiment battery (ablation/memory/adversarial/forgetting/transfer/stress/compute/failure) ==="
(cd "$ROOT/sim/sil-experiments" && cargo run --release 2>/dev/null)

echo ""
echo "=== D) Pareto ablation (safety-liveness frontier) ==="
(cd "$ROOT/sim/sil-ablation" && cargo run --release 2>/dev/null | grep -E "abl\]")

echo ""
echo "=== E) guard timing (machine-dependent absolute numbers) ==="
(cd "$ROOT/sim/bench-guard" && cargo run --release 2>/dev/null | grep -E "bench\]")

echo ""
echo "=== F) stress/break battery (noise, latency, physics shift, adversarial, perfect storm) ==="
(cd "$ROOT/sim/sil-stress" && cargo run --release 2>/dev/null)

echo ""
echo "================ END — experimental rows above are deterministic ================"
