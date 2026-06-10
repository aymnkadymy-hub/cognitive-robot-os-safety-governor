# Formal Guarantees — assumptions, invariants, theorems, proof sketches, and honest limits

A precise statement of what the system guarantees **and what it does not** (scope and limits,
honestly). Matches the code in `safety-memory`, `safety-model`, `clearance-guard`, `sel4-guard-pd`.

> **Precise framing (terminology):** the core invariant is `Λ ≤ L0` (*verification-preserving*).
> The system has two modes: *tighten-only (default)* or *bounded, evidence-based relaxation
> (optional, §8)*. The exact description is an **invariant-preserving adaptive safety envelope with
> bounded relaxation** — and the hard safety is independent of the adaptation policy (Theorem 6).

## 0. Model
- State `x ∈ ℝⁿ`; a proposed command `a_p ∈ ℝᵐ` from an **untrusted** brain (RL/AI, black box).
- Verified envelope (fixed, enforced by the kernel): `B0 = { a : |a_j| ≤ L0 ∀j }`.
- Incident memory `M = {(cᵢ, ℓᵢ)}`, similarity `s(c,c′)=cos`, threshold `τ`.
- Effective limit: `Λ(c) = min(L0, min_{i: s(c,cᵢ)≥τ} ℓᵢ)`; applied action `a* = clamp(a_p, ±Λ(c))`.
- Clearance barrier: at sensed frontal hazard distance `d`, the safe speed is
  `v_safe(d) = a·(√(τ_r² + 2·max(0, d−d_min)/a) − τ_r)`.

## 1. Assumptions (validity conditions — explicit)
- **(A1) Guard priority:** the Guard is a highest-priority protection domain on the *verified*
  seL4 scheduler, so it runs before any command reaches the actuators; its response time ≤ one
  control cycle `T`. *(Guaranteed by the verified seL4 architecture.)*
- **(A2) Final authority on the path:** every command reaching the actuators passes through the
  Guard (no side path). *(Enforced by the system description: actuation reads only `OFF_APPROVED`.)*
- **(A3) Braking capability:** the robot's real deceleration ≥ the `a` used in `v_safe`
  (measured/conservative). *(Needs hardware calibration — the most important not-yet-physically-
  verified assumption.)*
- **(A4) Sensing honesty:** the sensed `d_front` ≤ the true distance to the nearest **sensed**
  hazard (no under-reporting). *(For unsensed hazards the reactive barrier guarantees nothing —
  these are handled by the incident memory, §4.)*
- **(A5) Kernel correctness:** seL4's proof assumptions (no hardware/compiler bugs outside the
  verified model).

## 2. Invariants (maintained every cycle)
- **(I1) Envelope containment:** `a* ∈ B0` always (because `Λ(c) ≤ L0`).
- **(I2) Per-context monotone tightening:** for any `c`, the sequence `Λ_t(c)` is non-increasing.
- **(I3) Data/code separation:** adaptation modifies `M` (data) only; kernel and Guard code are
  fixed.
- **(I4) Floor:** `Λ(c) ≥ L_floor > 0` (no tightening to total paralysis).

## 3. Theorems (with proof sketches)

**Theorem 1 (envelope safety — containment).** Under (A1)(A2): `∀t, a*_t ∈ B0`.
*Proof.* `Λ(c) = min(L0, …) ≤ L0`; and `a* = clamp(a_p, ±Λ)` ⇒ `|a*_j| ≤ Λ(c) ≤ L0` ⇒ `a* ∈ B0`.
By (A2) the applied command equals `a*`. ∎ *(Machine-checked: `never_exceeds_static_envelope` +
the seL4 property.)*

**Theorem 2 (monotone tightening).** The update `record_incident` preserves (I2).
*Proof.* Adding `(c, ℓ_new)` cannot increase the `min`; the merge-when-full `ℓ_k ← min(ℓ_k, ℓ_new)`
does not increase it; no deletion raises the `min`. ⇒ `Λ_t(c)` non-increasing. ∎
*(`record_tightens` + `merge_by_min`.)*

**Theorem 3 (verification preservation).** The kernel's safety proof, which depends on `B0`,
remains valid under adaptation **without re-verification**. *Proof.* The proof depends on the
invariant `|a*_j| ≤ L0` (Theorem 1), which is independent of `M` (I3). Changing `M` does not touch
the kernel's code/proof. ∎ *(This is the core value: learning does not break verification.)*

**Theorem 4 (clearance safety — frontal invariant).** Under (A1)(A3)(A4), if the robot obeys
`v ≤ v_safe(d)` every cycle, it never penetrates `d_min` in front of a sensed hazard.
*Proof.* `v_safe` solves `v·τ_r + v²/(2a) ≤ d − d_min` (reaction travel + braking). From any state,
braking at `a` stops the robot within `d − d_min` even with reaction delay `τ_r` ⇒ `d_min` is never
reached. Induction over cycles preserves the invariant. ∎ *(Machine-checked:
`braking_guarantee_never_violates` + `bad_brain_kept_safe_in_sim`; empirically `sil-clearance`: 67 → 0.)*

## 4. Beyond reactive: the value of adaptation (proven empirically)
The reactive barrier (Theorem 4) only guarantees safety for **sensed** hazards (A4). **Unsensed**
hazards (contextual: slip, tilt, a learned danger) are guaranteed by no reactive barrier and no
reasonable fixed limit. The **incident memory** learns them from experience and tightens `Λ(c)`
there (I2).
- **Utility proof (`sil-ablation`, Pareto frontier):** the adaptive method = static-slow safety
  (zero unsafe) at **~1.8× the liveness**, and is safe where CBF/Simplex are blind. ⇒ no
  fixed/manual envelope achieves (safe ∧ fast) together without prior knowledge of the hazards.

## 5. Limits of the guarantees (honest — correct scope)
- **(L1) Physically unverified:** (A3) braking capability needs calibration on real hardware;
  noise/latency/slip can break the assumptions → today's guarantee is **in simulation/QEMU**, not
  on a physical robot.
- **(L2) Unsensed, unexperienced hazards:** anything neither sensed nor learned (a first incident,
  a sudden unseen hazard) **has no guarantee** — learning needs experience (or a born-cautious
  prior transfer).
- **(L3) WCET not measured on target:** decision time ~40–76 ns on the host (`bench-guard`); WCET
  on the Pi4 under seL4 scheduling is future work ((A1) is architectural, not yet measured).
- **(L4) Validity of `B0`:** the guarantee is relative to the correctness of the verified envelope
  `L0` (if `L0` itself is physically unsafe, the Guard cannot save you) — choosing `L0` is an
  engineering responsibility outside the model.

## 6. Formal summary
The system guarantees **mathematically** (Theorems 1–4) that any untrusted brain stays inside a
verified envelope + clearance barrier, and that learning does not break verification — **under
explicit assumptions (§1) and within explicit limits (§5)**. The unique value: provable,
model-agnostic safety that adapts in data, with a safety–liveness frontier that dominates fixed
alternatives.

---
## 7. Precise mathematical formulation

### 7.1 Definitions
- Discrete-time system: state `x_t ∈ X ⊆ ℝⁿ`, dynamics `x_{t+1} = f(x_t, a_t)` (unknown/uncertain).
- **Untrusted** policy `π: X → ℝᵐ`, proposing `a_p = π(x)`.
- **Verified envelope:** `B0 = { a ∈ ℝᵐ : ‖a‖_∞ ≤ L0 }`. **Kernel safety property:** `φ ≡ (∀t, a_t ∈ B0)`.
- **Incident memory:** finite `M ⊆ X × ℝ₊`, `|M| ≤ CAP`; context map `c: X → ℝ^d`, similarity
  `sim(·,·) ∈ [−1,1]`, threshold `τ`.
- **Effective limit:** `Λ_M(x) = min( L0, min{ ℓ : (c′,ℓ) ∈ M, sim(c(x),c′) ≥ τ } )` (min over ∅ = `L0`).
- **Guard map:** `G_M(x, a_p) = Π_{B(x)}(a_p)`, where `B(x) = { a : ‖a‖_∞ ≤ Λ_M(x) }` (per-channel clamp).
- **Tightening operator (incident at `x`):** `U_inc: M ↦ M ∪ {(c(x), max(L_floor, γ·Λ_M(x)))}`,
  `γ ∈ (0,1)`; when full: `ℓ_k ← min(ℓ_k, ℓ_new)` for the nearest (merge-by-min, no deletion).

### 7.2 Definition of "verification preservation" (precise)
An adaptation operator `U: M ↦ M′` **preserves verification** for property `φ` if the proof of `φ`
depends only on facts **independent of `M`**; formally: `(⊢ φ under M) ⟹ (⊢ φ under M′) by the same
proof, without re-derivation.`

### 7.3 Theorems (full proofs)
**Theorem 1 (containment).** `∀M, ∀x, ∀a_p: G_M(x,a_p) ∈ B0`.
*Proof.* By definition `Λ_M(x) = min(L0, …) ≤ L0`, so `B(x) ⊆ B0`, and `Π_{B(x)}` returns a point in
`B(x)`. Hence `G_M(x,a_p) ∈ B(x) ⊆ B0`. ∎ *(`never_exceeds_static_envelope`.)*

**Theorem 2 (monotonicity).** `∀x′: Λ_{U_inc(M)}(x′) ≤ Λ_M(x′)`.
*Proof.* Adding `(c(x), ℓ_new)` adds a term to the `min` ⇒ cannot increase it. The merge
`ℓ_k ← min(ℓ_k, ℓ_new) ≤ ℓ_k` does not increase any term. Nothing is deleted or raised. ∎
*(`record_tightens`, `merge_by_min`, `transfer_makes_fresh_memory_cautious`.)*

**Theorem 3 (verification preservation).** `U_inc` preserves verification for `φ ≡ (∀t, a_t ∈ B0)`.
*Proof.* By Theorem 1, `φ` holds for **any** `M`. The proof of `φ` uses only `Λ_M ≤ L0` — a
definitional fact true for every `M` including `M′ = U_inc(M)`. Hence `φ` holds under `M′` by the
same proof, without re-derivation. ⇒ adaptation (modifying `M`) does not touch the kernel proof. ∎

**Theorem 4 (clearance frontal invariant).** Under (A1)(A3)(A4): if `∀t, v_t ≤ v_safe(d_t)` with
`v_safe(d) = a(√(τ_r² + 2·max(0,d−d_min)/a) − τ_r)`, then `∀t, clearance ≥ d_min` in front of a
sensed hazard.
*Proof.* `v_safe` solves `v·τ_r + v²/(2a) = d − d_min`. At any `t` with `v_t ≤ v_safe(d_t)`, braking
at `a` (after delay `τ_r`) stops within `d_t − d_min`. Re-enforced every cycle ⇒ invariant by
induction. ∎ *(`braking_guarantee_never_violates`, `bad_brain_kept_safe_in_sim`; empirically
`sil-clearance` 67 → 0, and `sil-experiments §8` adversarial brain → 0.)*

### 7.4 Machine-checked invariants
Each theorem is tied to a property test that checks the invariant on thousands of inputs
(property-based) — an **empirical certificate** complementing the paper proof. It is **not** an
interactive-prover proof (Coq/Isabelle) yet — an explicit limit; porting the theorems (especially
Theorem 3) to a prover is valuable future work. See [`VERIFICATION.md`](VERIFICATION.md).

---
## 8. Bounded, invariant-preserving relaxation (optional)
Adding forgetting moves the system from the *tighten-only (monotone)* class to a **non-monotone but
invariant-preserving bounded** safety-adaptation class — a **change of theoretical class**, stated
explicitly. Paper term: **"invariant-preserving adaptive safety envelope with bounded relaxation."**

**Mechanism (`confirm_safe`, optional):** each confirmed-safe pass adds evidence; after
`evidence_k` passes the incident limit relaxes `ℓ ← min(L0, ℓ/γ)` (capped at `L0`); a new incident
resets the evidence and re-tightens.

### 8.1 Exactly what is preserved vs. replaced
- **Preserved:** Theorem 1 (containment `Λ ≤ L0`) — because relaxation is capped at `L0`. Hence
  **Theorem 3 (verification preservation)** still holds (it depends only on `Λ ≤ L0`, not on
  monotonicity). Theorem 4 (clearance barrier) is **independent of `M`** ⇒ preserved.
- **Replaced:** Theorem 2 (strict monotonicity) by a weaker explicit property:
  - **(I2′) bounded, fenced relaxation:** `L_floor ≤ Λ_M(x) ≤ L0` always; `Λ` rises only after
    `evidence_k` consecutive **confirmed-safe** passes, and any incident drops it immediately.

### 8.2 Strict A/B layering — bounding the claim
- **(A) Hard safety (formal invariant):** `Λ ≤ L0` (containment) + clearance barrier for **sensed**
  hazards. **Proven, independent of `confirm_safe`.**
- **(B) Risk shaping (adaptive heuristic layer):** memory protection for **experiential/unsensed**
  hazards. `confirm_safe` modifies **B only**. **B is heuristic, not proven-safe.**

**Explicit assumption (A7):** the reactive barrier captures only **sensed** hazards within its
model. **Unsensed / late-perceived / hidden-dynamics** hazards are not captured by the barrier —
they are protected by the **memory layer (B) alone**.

**Theorem 5′ (degradation floor — what is actually proven).** For any `confirm_safe` sequence (even
adversarial): `L_floor ≤ Λ_M(x) ≤ L0 ∀x`, so the verified envelope is never exceeded (A is sound),
and **the worst it can do is return the added protection (B) toward the memoryless baseline
(`Λ = L0`)** — not below. *Proof.* `min(L0, ℓ/γ) ≤ L0` for any number of calls. ∎

### 8.3 What is **not** proven (the honest limit)
Theorem 5′ proves **no envelope violation**, but does **not** prove the absence of an
**unjustified approach to a real hazard before the barrier engages** for **unsensed** hazards: at
such hazards there is no reactive barrier (A7), so memory (B) is the only protection; a wrong
`confirm_safe` relaxes it toward `L0` ⇒ the robot may approach a *learned experiential* hazard
**before** re-tightening. **The first incident is not prevented** — it is handled only
**reactively** afterward (`record_incident`). What we have there is **empirical**:
`sil-experiments §9` shows bounded, self-healing degradation (0 → 28, envelope intact), but that is
**bounded empirical behavior, not a proof** for all sensing uncertainty.

### 8.4 The honest statement (no overclaiming)
> "A formally constrained safety envelope with **adaptive risk shaping** that **preserves hard
> invariants** (`Λ ≤ L0` + sensed-hazard barrier) but does **not** guarantee bounded
> **pre-barrier** behavior under all sensing uncertainties (unsensed/delayed/hidden-dynamics
> hazards)."

The positioning is precise: **"verified hard floor + adaptive risk-shaping heuristic,"** not a
"fully formally-verified adaptive safety system." *(Default: without `confirm_safe`, behavior is
tighten-only and Theorem 2 holds in full. Tests: `evidence_based_forgetting_recovers_then_never_
exceeds`, `new_incident_resets_evidence`.)*

---
## 9. Separation theorem (safety–performance)
**Theorem 6 (separation).** The hard safety invariant `φ ≡ (Λ ≤ L0) ∧ (sensed-hazard clearance
safety)` is **completely independent of** the memory `M` and the `confirm_safe` policy; whereas the
**experiential protection and the liveness** are functions of them: `Safety = g(L0, barrier) ⊥ M`
and `Performance = h(M, confirm_safe, env)`. *Proof.* `φ` follows from Theorems 1, 4, 5′, all of
which depend only on `L0` and the barrier (independent of `M`). ∎ This cleanly separates "what is
guaranteed" (safety) from "what improves adaptively" (performance).

## 10. Assumption-break set `A_break` and closing (A3)
Let `Θ` be the operating-condition space (speed, delay, braking ratio, noise). The safe envelope
`S = { θ ∈ Θ : assumptions A1–A4 hold at θ }`, and `A_break = Θ \ S`. Concretely: break A1 (response
time) `{ latency > τ_r·margin }`; break A3 (braking) `{ a_actual < a_assumed }`; break A4 (sensing)
spoofing beyond the noise margin.
**Theorem 7 (safety on the complement).** On `S = Θ\A_break`, collision-safety (Theorem 4) holds;
outside it there is no guarantee — characterized empirically (`sil-stress`): nominally
`latency > 120 ms` or `response < 0.5`; conservatively `A_break` shrinks.
**Theorem 8 (conservative design widens S).** `v_safe` is decreasing in `a` and increasing in `τ_r`;
choosing `a_conservative ≤ a_nominal` and `τ_conservative ≥ τ_nominal` lowers `Λ`, widening the
margin ⇒ shrinking `A_break`. *Empirically (`sil-stress §8`):* the latency breaking point moves
`120 ms → > 300 ms` and physics `0.5 → 0.2`, at small distance cost. ⇒ **closing (A3):** the
guarantee becomes "safe if `a_actual ≥ a_conservative`," with `a_conservative` set from the
**worst-case measured on hardware** — turning a conditional into a grounded physical bound.

## 11. Structural WCET bound (analytic, not just empirical)
The guard decision is **free of unbounded loops, allocation, and recursion**: `effective_limit`
iterates over `≤ CAP` incidents, each a cosine of `DIM` multiplies. Hence **WCET ≤ CAP·c_cos +
c_const** — a constant analytic upper bound independent of the data (no hidden worst case). For
`CAP=128, DIM=2` ⇒ ~`128·(~5 ns)` ≈ **0.7 µs** ceiling; the measured mean is ~40–76 ns
(`bench-guard`). *(A final certified WCET needs the target compiler + cache analysis — future work.)*
