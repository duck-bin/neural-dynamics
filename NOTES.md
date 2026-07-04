# NOTES — math, design rationale, and rejected alternatives

Learning artifact (mandatory). For each stage: the math, **why this choice**, and
**what I rejected**. Comments in code explain WHY, not WHAT. Treat jPCA
preprocessing, fixed-point vs slow-point, and Jacobian stability classification in
particular depth.

---

## 0 · Conceptual framing — descriptive vs mechanistic

- jPCA is **descriptive**: one global skew-symmetric linear fit to trajectory
  geometry; needs only observed states, so it runs on brain AND RNN.
- Fixed-point analysis is **mechanistic**: needs the evaluable/differentiable
  vector field F(x) = dx/dt; we have it for the RNN, not the brain.
- Fixed points GENERATE the rotation jPCA DESCRIBES — cause vs description, not
  two things compared. Viewer asymmetry: brain = trajectories + jPCA plane only;
  RNN = trajectories + jPCA + fixed points + flow field.

---

## M0 · jPCA hand implementation
_(math of soft-norm, cross-condition-mean subtraction, PCA, skew-symmetric
constrained least squares, eigendecomposition; why each; rejected alternatives;
differential-test result vs Antin `jPCA`.)_

## M1 · Flip-flop calibration (known answer)
_(3-bit flip-flop task; fixed-point finder; the 8-corner gate; differential test
vs pytorch-fixed-point-analysis; numbers.)_

## M2 · Brain data — PCA + jPCA (Stage 1)
_(MC_Maze preprocessing choices; fit R²; rotation-plane variance; adversarial
check: rotation with vs without cross-condition-mean subtraction — before/after.)_

## M3 · Reaching RNN + jPCA (Stage 2 part 1)
_(train on TASK not spikes; metabolic reg; velocity R² task gate; does the RNN
rotate — variance fraction.)_

## M4 · Reaching RNN fixed points + flow field (Stage 2 part 2)
_(q(x) objective; IC sampling rule; tolerance from q-distribution; clustering;
Jacobian stability classes; adversarial check: FP stability to IC re-sampling +
hand-vs-reference Hungarian alignment; numbers.)_

## M5 · Unified viewer + deploy
_(space asymmetry handled honestly; matched visual scale only; deploy notes.)_

---

## Differential-test log
_(each hand-vs-reference comparison, identical input, tolerance, result.)_

## Proposed deviations (I propose, the user decides)
_(Any change to the FIXED commitments — jPCA steps, fixed-point objective + IC
rule, differential-test tolerances, flip-flop 8-corner gate, train-on-task-not-
spikes, scope boundary — goes here with rationale. Do NOT silently change these.)_

## Honesty citations (in scope to mention, out of scope to run)
- Lebedev et al. (2019) — critique of the cross-condition-mean subtraction step.
- Elsayed & Cunningham (2017) — low-D/rotation may be a byproduct.
- Maheswaranathan et al. (2019) — geometric match is cheap (universality).

## Next iteration (noted, deliberately NOT built now)
_(shuffle/TME null models, DSA (Ostrow et al. 2023), cross-day/cross-task
alignment, LFADS/GPFA, disease data.)_

## References
See README "Key references" for exact citations.
