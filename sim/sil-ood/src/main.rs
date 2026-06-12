//! # sil-ood — Software-in-the-Loop demo of the OOD gate (Mahalanobis, tighten-only)
//!
//! Closes **the same guard loop** (MujocoTwin body + trained pendulum policy + `safe_bound`)
//! using the same seL4 crates, and adds an **out-of-distribution gate**: a state far from the
//! operating manifold ⇒ **maximum tightening** (when in doubt, tighten).
//!
//! Proved on the host (valid on the target by the SiL principle):
//! 1. **Tighten-only:** the gate never **raises** the base bound (0 raises).
//! 2. **Envelope preserved:** final bound ≤ `HARD_LIMIT` **always** (0 violations) ⇒ kernel proof intact.
//! 3. **Actually fires:** rare on normal operation, triggered on injected extreme states.
//!
//! Also prints the fitted constants (μ, diagonal Σ⁻¹, threshold) for uploading to
//! `sel4-guard-pd` (hardware integration, future work).

use hal::{MujocoTwinBody, RobotBody};
use mujoco_pendulum_policy::command;
use ood_detector::MahalanobisOod;
use safety_model::SafetyModel;

const HARD_LIMIT: f32 = 1.0;
const FLOOR: f32 = 0.85; // learned-model floor (gentle tightening)
const OOD_FLOOR: f32 = 0.1; // maximum tightening under uncertainty (much tighter than FLOOR)
const DANGER_THETA: f32 = 0.03;
const TARGET_X: f32 = 0.0;
const CHI2_99_DF3: f32 = 11.345; // χ² threshold at 99%, 3 degrees of freedom
const STEPS: usize = 2000;

/// **Same** danger features computed by the guard on seL4 (sel4-guard-pd).
fn danger_features(s: &[f32; 4]) -> [f32; 3] {
    [
        (s[1].abs() / 0.2).min(1.0), // proximity to fall (tilt)
        (s[3].abs() / 3.0).min(1.0), // angular velocity
        (s[0].abs() / 4.0).min(1.0), // proximity to edge (position)
    ]
}

fn main() {
    // ===== Phase 1: calibration — collect normal-operation features and fit a diagonal Gaussian =====
    let mut body = MujocoTwinBody::new(0.05);
    let mut model = SafetyModel::<3>::new().with_lr(0.15);
    let mut feats: Vec<[f32; 3]> = Vec::with_capacity(STEPS);
    for k in 0..STEPS {
        if k == 500 || k == 1000 || k == 1500 {
            body.disturb(0.5);
        }
        let s = body.sense();
        let feat = danger_features(&s);
        feats.push(feat);
        let obs = [s[0] - TARGET_X, s[1], s[2], s[3]];
        let proposed = command(&obs);
        let bound = model.safe_bound(&feat, HARD_LIMIT, FLOOR);
        let approved = proposed.clamp(-bound, bound);
        body.actuate(approved);
        model.observe(&feat, s[1].abs() > DANGER_THETA);
    }

    // Diagonal Gaussian: mean and variance per dimension (diagonal Σ⁻¹ ⇒ positive-definite ⇒ d² ≥ 0).
    let n = feats.len() as f32;
    let mut mean = [0.0f32; 3];
    for f in &feats {
        for i in 0..3 {
            mean[i] += f[i];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = [0.0f32; 3];
    for f in &feats {
        for i in 0..3 {
            let d = f[i] - mean[i];
            var[i] += d * d;
        }
    }
    for v in &mut var {
        *v = (*v / n).max(1e-6); // floor to avoid division by zero
    }
    let sigma_inv = [
        [1.0 / var[0], 0.0, 0.0],
        [0.0, 1.0 / var[1], 0.0],
        [0.0, 0.0, 1.0 / var[2]],
    ];
    let ood = MahalanobisOod::new(mean, sigma_inv, CHI2_99_DF3);

    println!("[ood] Fitted constants (upload to sel4-guard-pd):");
    println!("[ood]   mean    = {mean:?}");
    println!("[ood]   var     = {var:?}");
    println!(
        "[ood]   sigma_inv = diag({:?})",
        [1.0 / var[0], 1.0 / var[1], 1.0 / var[2]]
    );
    println!("[ood]   threshold = {CHI2_99_DF3}  (χ² 99%, df=3)");

    // ===== Phase 2: normal operation with the gate — verify safety invariants =====
    let mut body = MujocoTwinBody::new(0.05);
    let (mut raises, mut violations, mut ood_fires, mut total) = (0usize, 0usize, 0usize, 0usize);
    let mut max_bound = 0.0f32;
    for k in 0..STEPS {
        if k == 500 || k == 1000 || k == 1500 {
            body.disturb(0.5);
        }
        let s = body.sense();
        let feat = danger_features(&s);
        let obs = [s[0] - TARGET_X, s[1], s[2], s[3]];
        let proposed = command(&obs);
        let base_bound = model.safe_bound(&feat, HARD_LIMIT, FLOOR);
        let final_bound = ood.tighten_if_ood(&feat, base_bound, OOD_FLOOR);

        if final_bound > base_bound + 1e-6 {
            raises += 1; // gate raised the bound (must never happen)
        }
        if final_bound > HARD_LIMIT + 1e-6 {
            violations += 1; // envelope violated (must never happen)
        }
        if ood.is_ood(&feat) {
            ood_fires += 1;
        }
        max_bound = max_bound.max(final_bound);
        total += 1;

        let approved = proposed.clamp(-final_bound, final_bound);
        body.actuate(approved);
        model.observe(&feat, s[1].abs() > DANGER_THETA);
    }

    // ===== Phase 3: inject extreme states (explicit OOD) — gate must fire and tighten =====
    let extremes: [[f32; 3]; 3] = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let mut detected = 0usize;
    let mut all_tightened = true;
    for e in &extremes {
        let base = model.safe_bound(e, HARD_LIMIT, FLOOR);
        let gated = ood.tighten_if_ood(e, base, OOD_FLOOR);
        if ood.is_ood(e) {
            detected += 1;
        }
        if gated > base + 1e-6 {
            all_tightened = false;
        }
    }

    // ===== Results =====
    let ood_rate = 100.0 * ood_fires as f32 / total as f32;
    println!("\n[ood] ===== Results (SiL) =====");
    println!("[ood] bound raises   (must be 0): {raises}");
    println!("[ood] envelope violations (must be 0): {violations}");
    println!("[ood] max final bound:            {max_bound} (≤ {HARD_LIMIT})");
    println!("[ood] OOD fires on normal operation: {ood_fires}/{total} ({ood_rate:.1}%)");
    println!(
        "[ood] extreme states detected:    {detected}/{} (all tightened: {all_tightened})",
        extremes.len()
    );

    // ===== Phase 4: detection quality (ROC / recall / precision / AUC) =====
    // Ground truth: confirmed extreme state (high tilt or high angular velocity) = anomaly;
    // confirmed normal state = inlier.
    // We measure d² separation between the two classes across thresholds to obtain ROC and
    // values at the operating threshold χ²₉₉.
    let mut in_d: Vec<f32> = Vec::new();
    let mut ood_d: Vec<f32> = Vec::new();
    let mut bt = MujocoTwinBody::new(0.05); // same calibration step to let the policy settle
    for k in 0..6000 {
        if k % 120 == 0 {
            bt.disturb(if k < 3000 { 0.4 } else { 2.5 }); // gentle then strong ⇒ spectrum of states
        }
        let s = bt.sense();
        let f = danger_features(&s);
        let d2 = ood.distance_sq(&f);
        if f[0] >= 0.6 || f[1] >= 0.6 {
            ood_d.push(d2); // confirmed extreme state (outside operating manifold)
        } else if f[0] < 0.25 && f[1] < 0.25 && f[2] < 0.5 {
            in_d.push(d2); // confirmed normal state
        }
        let obs = [s[0] - TARGET_X, s[1], s[2], s[3]];
        let p = command(&obs);
        let bd = model.safe_bound(&f, HARD_LIMIT, FLOOR);
        let g = ood.tighten_if_ood(&f, bd, OOD_FLOOR);
        bt.actuate(p.clamp(-g, g));
    }
    println!(
        "\n[ood] ===== Detection quality (ROC): {} inlier samples, {} outlier samples =====",
        in_d.len(),
        ood_d.len()
    );
    println!("[ood]   d²-thresh  recall    FPR     precision");
    let thrs = [1.0f32, 2.0, 4.0, 7.0, CHI2_99_DF3, 16.0, 25.0, 50.0, 100.0];
    let (mut auc, mut pf, mut pt) = (0.0f32, 1.0f32, 1.0f32);
    for thr in thrs {
        let tp = ood_d.iter().filter(|&&d| d > thr).count() as f32;
        let fp = in_d.iter().filter(|&&d| d > thr).count() as f32;
        let tpr = tp / ood_d.len().max(1) as f32;
        let fpr = fp / in_d.len().max(1) as f32;
        let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
        let mark = if (thr - CHI2_99_DF3).abs() < 0.01 {
            "  ← operating threshold"
        } else {
            ""
        };
        println!("[ood]   {thr:7.2}   {tpr:6.3}   {fpr:6.3}   {prec:6.3}{mark}");
        auc += (pf - fpr) * (tpr + pt) / 2.0;
        pf = fpr;
        pt = tpr;
    }
    auc += pf * pt / 2.0;
    println!("[ood]   AUC ≈ {auc:.3}");

    // Core safety invariants are non-negotiable.
    assert_eq!(
        raises, 0,
        "OOD gate raised the bound — violated the tighten-only property"
    );
    assert_eq!(violations, 0, "envelope violated — broke the kernel proof");
    assert!(all_tightened, "an extreme state was not tightened");
    assert_eq!(detected, extremes.len(), "OOD gate missed extreme states");
    assert!(
        ood_rate < 50.0,
        "OOD firing rate too high on normal operation ({ood_rate:.1}%) — poor calibration"
    );

    println!("\n[ood] PASS: OOD gate is tighten-only, preserves the envelope, and detects anomalies — on the same seL4 crates.");
}
