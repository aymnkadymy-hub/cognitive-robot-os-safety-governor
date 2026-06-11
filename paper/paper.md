---
title: 'Cognitive Robot OS: A reproducible, verification-preserving adaptive safety governor for untrusted learned policies'
tags:
  - Rust
  - robotics
  - safety-critical systems
  - runtime assurance
  - formal methods
  - seL4
  - reinforcement learning
  - control barrier functions
  - reproducibility
authors:
  - name: Ayman Kazem Yousef
    orcid: 0009-0006-7409-9367
    affiliation: 1
affiliations:
  - name: Independent Researcher, Iraq
    index: 1
date: 11 June 2026
bibliography: paper.bib
---

# Summary

`Cognitive Robot OS` is a reproducible software artifact implementing a *verification-preserving
adaptive safety governor*: a small, machine-checked guard that holds final authority over a robot's
actuators and keeps an **untrusted** learned policy — a reinforcement-learning agent, a neural
controller, or any other black box — inside a proven safety envelope. The guard projects every
proposed action into a verified envelope with static limit `L0`, and tightens a per-context
effective limit `Λ(x) ≤ L0` using an *incident memory* that records hazards encountered at
deployment. Adaptation happens in the **data the guard reads**, never in verified code, so the
existing functional-correctness proof of the underlying seL4 microkernel [@klein2009; @klein2014]
remains valid without re-verification. The artifact is the companion software for the research paper
[@yousef2026governor] and is published so that every reported number can be independently
regenerated with a single command.

The system is written in Rust as a tree of `no_std`, allocation-free crates (`#![forbid(unsafe_code)]`):
an adaptive *tighten-only* safety-memory envelope, a clearance-braking barrier guard, a learned
per-context risk bound, and a Mahalanobis out-of-distribution (OOD) gate. These are wired into
isolated seL4/Microkit protection domains — a highest-priority deterministic Guard "spinal cord", a
Cognitive/memory "brain", and an actuation domain that applies only approved commands. The
repository also ships software-in-the-loop (SiL) evaluation, ablation, and stress harnesses, plus
Kani/CBMC bounded-model-checking proof harnesses that machine-check the structural safety
invariants.

# Statement of need

Robots increasingly act on commands from untrusted learned policies whose failures can drive
actuators into unsafe states [@garcia2015]. Established runtime-safety filters — control-barrier-function
filters [@ames2017], Simplex-style runtime assurance [@sha2001; @hobbs2023], and shielding
[@alshiekh2018] — require an *a priori* model of the unsafe set, switch on a binary basis, and cannot
tighten against hazards first encountered at deployment; learned variants that adapt typically
*relax* bounds and do not preserve a separate kernel verification. There is a need for a safety layer
that (i) adapts to deployment experience while only ever becoming *more* conservative, and (ii) does
so without invalidating an existing formal proof of the platform underneath it.

This artifact fills that gap and, equally important, makes the claim *checkable*. Safety research is
frequently hard to reproduce; here every experimental number is deterministic (fixed-seed LCG,
`no_std`, no heap, no global state), so an independent reader who runs the code obtains the published
numbers bit-for-bit. It is useful to two audiences: researchers in safe reinforcement learning and
runtime assurance who want a reproducible, head-to-head baseline against CBF, Simplex, shielding, and
learned-safety methods; and robotics/systems engineers who want a deployable, capability-isolated
safety guard that runs atop a formally verified microkernel — or, today, atop an existing robot stack
such as ROS 2 or a conventional RTOS.

# Functionality / Key features

- **Adaptive, tighten-only envelope (`safety-memory`).** Incident memory with similarity retrieval
  yields an effective limit `Λ(x) ≤ L0`; tightening is monotone, so even a poisoned memory can only
  make the robot more cautious, never less safe. Optional evidence-based bounded relaxation never
  exceeds `L0`.
- **Clearance braking barrier (`clearance-guard`)** for sensed hazards, and an **OOD gate
  (`ood-detector`)** that extends the tighten-only principle to distributional novelty.
- **Capability-isolated deployment.** seL4/Microkit protection domains separate the deterministic
  Guard, the cognitive/memory brain, and actuation; the guard preempts a deliberately adversarial
  brain on the verified microkernel.
- **Machine-checked invariants.** Each theorem is backed by property tests over thousands of inputs
  and by Kani/CBMC bounded-model-checking proof harnesses; a "no non-deterministic math" audit keeps
  results reproducible and the kernel-side proof input-independent.
- **Evaluation, ablation, and stress harnesses** (`sim/`): a multi-environment evaluation with 95%
  confidence intervals (6 environments × 40 paired seeds), a 12-test reviewer battery, a Pareto
  ablation, guard-decision timing (`O(1)`, ≈38 ns/decision, ~2 KB static RAM), and a stress battery
  reporting honest failure modes.

**Scope (honest).** All results are simulation / software-in-the-loop on seL4-QEMU; there is **no
physical-hardware validation yet** — the single stated remaining gap. The theorems are
machine-checked by bounded model checking and property tests, **not** by an interactive prover
(Coq/Isabelle), which is identified as future work.

# Reproducibility

The repository (`https://github.com/aymnkadymy-hub/cognitive-robot-os-safety-governor`) requires only a Rust toolchain — no GPU, network, or hardware for the
SiL evaluation; the seL4/QEMU reflex-arc demo is optional. A single command,
`scripts/reproduce.sh`, runs the property tests (the theorems), the multi-environment evaluation, the
12-test battery, the Pareto ablation, the guard timing, and the stress battery, and prints the rows
used in the paper. Committed deterministic reference outputs under `results/reference/` let a reader
confirm an exact, bit-for-bit match. `docs/REPRODUCE.md` maps each paper table to its command and
reference file; `docs/VERIFICATION.md` is the theorem-to-evidence ledger.

# Acknowledgements

We acknowledge the seL4 Foundation and the Kani/CBMC developers, whose tools underpin the verified
microkernel substrate and the bounded-model-checking proof harnesses used here.

# References
