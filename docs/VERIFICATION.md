# Verification Ledger — each theorem ↔ its machine-checked evidence

Two complementary, runnable forms of evidence back the formal claims:

1. **Property tests** (`proptest`): each invariant is checked on thousands of randomized inputs.
   Run with stock `cargo test` — no extra tooling.
2. **Kani proof harnesses** (`#[kani::proof]`): bounded model-checking proofs over symbolic inputs.
   Require the [Kani](https://model-checking.github.io/kani/) verifier (`cargo kani`).

> **Honest scope:** these are *machine-checked* (model checking + property-based), **not** an
> interactive-prover proof (Coq/Isabelle). Porting the theorems — especially Theorem 3 — to an
> interactive prover is stated as valuable future work in
> [`FORMAL_GUARANTEES.md`](FORMAL_GUARANTEES.md) §7.4.

## Theorem ↔ evidence

| Theorem (see FORMAL_GUARANTEES.md) | Property tests | Kani proof harness |
|---|---|---|
| **Thm 1** — envelope containment `a* ∈ B0` | `never_exceeds_static_envelope`, `proptests::effective_limit_within_static`, `proptests::preload_clamps_to_envelope` (safety-memory) | `safety-memory::proof_effective_limit_within_static`; `reflex-abi::proof_enforce_limit_within_envelope`; `brain-os-abi::proof_enforce_limits_bounds_finite` |
| **Thm 2** — monotone tightening | `incident_tightens_its_region`, `monotone_tightening_only`, `proptests::record_incident_only_tightens`, `transfer_makes_fresh_memory_cautious` (safety-memory) | (covered by `proof_effective_limit_within_static` + the OOD/evasion tighten-only proofs below) |
| **Thm 3** — verification preservation | `golden_bits_effective_limit`, `audit_no_nondeterministic_math` (safety-memory) — the audit enforces deterministic, bounded math so the kernel proof stays input-independent | — (the invariant it rests on, `Λ ≤ L0`, is proven by `proof_effective_limit_within_static`) |
| **Thm 4** — clearance braking barrier | `braking_guarantee_never_violates`, `bad_brain_kept_safe_in_sim`, `zero_speed_at_or_inside_margin`, `proptests::safe_speed_*`, `proptests::braking_guarantee_never_violates` (clearance-guard) | `clearance-guard::proof_safe_speed_zero_within_margin`; `reactive-evasion::proof_safe_approach_stops_at_margin` |
| **Thm 5′** — bounded relaxation floor `L_floor ≤ Λ ≤ L0` | `evidence_based_forgetting_recovers_then_never_exceeds`, `new_incident_resets_evidence`, `proptests::forgetting_preserves_ceiling` (safety-memory) | `safety-memory::proof_effective_limit_within_static` (cap at `L0` holds under relaxation) |
| Memory bound `|M| ≤ CAP` | `proptests::count_never_exceeds_cap` (safety-memory) | `safety-memory::proof_count_within_cap` |
| Learned risk ≤ envelope | `never_exceeds_static_envelope`, `proptests::safe_bound_never_exceeds_static`, `proptests::risk_finite_in_unit_interval`, `generalizes_to_unseen_values` (safety-model) | — |
| OOD gate tightens only | (safety-memory tighten-only tests apply) | `ood-detector::proof_tighten_if_ood_never_raises_bound`, `ood-detector::proof_zero_distance_at_mean` |
| Brain-stall → stop / watchdog | (sim `sil-experiments §8`; seL4 `sel4-guard-pd`) | `brain-os-abi::proof_govern_stops_on_stall`, `reflex-abi::proof_heartbeat_stalled_correct` |
| IMU filter stays finite | (imu-filter property tests) | `imu-filter::proof_update_stays_finite` |

## Test inventory (per verified crate)

| Crate | `#[test]` | proptest blocks | Kani proofs |
|---|---|---|---|
| `safety-memory` | 14 | 3 | 2 |
| `clearance-guard` | 13 | 7 | 1 |
| `safety-model` | 7 | 1 | 0 |
| `ood-detector` | 8 | 1 | 2 |
| `imu-filter` | 7 | 1 | 1 |
| `reactive-evasion` | 4 | 0 | 2 |
| `reflex-abi` | 8 | 5 | 2 |
| `brain-os-abi` | 9 | 4 | 2 |

## How to run

```sh
# Property tests (the theorems on thousands of inputs) — no extra tooling:
(cd components/safety-memory   && cargo test --release)   # → 14 passed
(cd components/clearance-guard && cargo test --release)   # → 13 passed
(cd components/safety-model    && cargo test --release)   # →  7 passed
# …and ood-detector, imu-filter, reactive-evasion, reflex-abi, brain-os-abi

# Kani proofs (optional; requires `cargo install --locked kani-verifier && cargo kani setup`):
(cd components/safety-memory   && cargo kani)
(cd components/clearance-guard && cargo kani)
(cd components/ood-detector    && cargo kani)
```

Reference test logs are committed under
[`results/reference/A-proptest-*.txt`](../results/reference/).

## The "no non-deterministic math" audit
Several crates include an `audit_no_nondeterministic_math` test that scans the source to forbid
constructs that would make outputs platform- or order-dependent (e.g. unordered float reductions).
This is what makes the experimental results bit-for-bit reproducible and keeps the kernel-side
proof (Thm 3) independent of memory contents.
