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

**Status: PASS.** Implemented in `python/pca_jpca.py`, validated in
`tests/test_m0_jpca.py` against Benjamin Antin's `jPCA` (the oracle).

### The math, step by step (README M0 steps 1–7)

**Convention.** States are rows; the model is `Ẋ = X Mᵀ`, equivalently
`dx/dt = M x`, with `M` skew-symmetric (`M = −Mᵀ`). This matches Antin, so our
`M` and his are directly comparable.

1. **Condition-averaged rates** `X_c(t)`, shape `(C, T, N)`.
2. **Soft-normalize** each neuron: divide by `(range + 5)`, range taken over all
   conditions and times. *Why:* pure range-normalization amplifies low-modulation
   units (tiny denominator turns noise into apparent signal); the `+5` softens it.
   **Then subtract the cross-condition mean** at each timepoint. *Why:* removes
   the condition-independent component so jPCA sees only condition-*dependent*
   structure. **Contested (Lebedev et al. 2019):** this step can manufacture
   apparent rotation. Kept for faithful reproduction; tested adversarially at M2.
3. **PCA to k = 6.** *Why:* denoise + make the skew fit well-posed (`M` is 6×6,
   15 free params). sklearn PCA on the identical preprocessed matrix is
   deterministic, so our reduced space equals Antin's exactly.
4. **Ẋ by forward first difference, within each condition** (never across the seam
   between conditions), `dt = 1 bin`. Drop each condition's last sample from `X`.
5. **Skew-symmetric constrained least squares — the crux.** Minimize
   `‖X Mᵀ − Ẋ‖²` over skew `M`. We write `M = Σ θ_p E_p` for the orthogonal basis
   of skew matrices `{E_p}` (`E_p` has +1 at (a,b), −1 at (b,a)); the model is then
   *linear* in `θ` because `X Mᵀ = −Σ θ_p (X E_p)`. One `lstsq` solves it in closed
   form — the "vectorize + closed form" the spec asks for. Cross-checked internally
   against the Sylvester normal equations `C M + M C = Dᵀ − D` (`C = XᵀX`,
   `D = XᵀẊ`); the two closed forms agree to 9e-16.
6. **Eigendecompose `M`.** A real skew matrix has purely imaginary eigenvalues in
   conjugate pairs `±iω`. The pair with largest `|ω|` spans the top jPC plane;
   `Re(v), Im(v)` of its eigenvector, QR-orthonormalized, are the plane axes.
7. **Project** trajectories onto the plane → rotations (verified visually).

### Why closed form (not Antin's iterative CG)
The spec (step 5) demands a closed-form constrained solve, and it makes the
differential test *stronger*: our closed-form vectorized LS and Antin's iterative
CG are genuinely different algorithms for the same convex problem, so their
agreement (below) is independent corroboration, not a tautology.

### Rejected alternatives
- **Unconstrained `M` (plain lstsq):** fits better (R² 0.99 vs 0.86 on synthetic)
  but its eigenvalues have real parts → decay/expansion, not pure rotation. The
  whole point of jPCA is to *ask how rotational* the data is; the skew constraint
  is the question, not a limitation. The gap (0.99→0.86) *is* the non-rotational
  structure, and reporting it is informative.
- **Central difference for Ẋ:** would cut the frequency bias (see below) but
  diverges from Churchland/Antin's forward difference; rejected for faithful
  reproduction. Recorded as an available, not-taken option.

### Forward-difference frequency bias (understood, not a bug)
Forward difference on a rotation of `ω` rad/bin gives `X(t+1) = R(ω) X(t)`, so
`Ẋ = (R(ω) − I) X`. The skew part of `R(ω) − I` is `[[0,−sinω],[sinω,0]]`, whose
eigenvalues are `±i·sinω`. So the fit recovers **sin(ω), not ω** — a known
`≈ω²/6` bias (1.5% at 0.3 rad/bin). The test confirms recovered ω = sin(ω) to
0.0000%. The oracle shares the bias exactly. At M2 the 20 ms bin converts ω to Hz.

### Reported numbers (synthetic known-answer case)
- fit R² of `Ẋ = X Mᵀ` (skew): **0.855** (unconstrained ceiling 0.989).
- top rotation-plane variance fraction: **0.902**.
- differential test vs Antin: principal angle **0.0008°** (« 5°), frequency match
  **0.000%** (« 5%); `‖M_hand − M_Antin‖/‖M_Antin‖ = 3.8e-5`.
- ground-truth (clean rotation): recovered ω = sin(ω) to **0.0000%**.
(These are on synthetic data used ONLY to unit-test the algorithm against the
oracle with a known answer — NOT a stand-in for brain data. The same hand-vs-Antin
test re-runs on MC_Maze at M2.)

## M1 · Flip-flop calibration (known answer)

**Status: PASS.** RNN + task in `python/rnn.py`, finder in `python/fixedpoints.py`,
gate + differential test in `tests/test_m1_flipflop.py`.

### The task and why it is the calibration
3-bit flip-flop (Sussillo & Barak 2013): 3 input channels emit sparse +/-1
pulses; each output must hold the sign of its channel's last pulse. The network
is therefore a 3-bit memory with 2^3 = 8 states, and a correct solution stores
them as **8 stable fixed points at the corners of a cube** in readout space. The
answer is known in advance, so this falsifies the finder before we trust it on
reaching data: wrong count or location => STOP and debug.

### Two bugs found and fixed (recorded because they were instructive)
1. **Unknowable initial bit.** The first task version set a random initial memory
   and asked the network to output it *before any pulse arrived* — information it
   cannot have. This capped accuracy and pinned MSE at ~0.18. Fix: memory starts
   at 0 and the target is 0 until the first pulse. MSE then dropped to 0.008.
2. **Contractive recurrent init.** Orthogonal init with gain 0.9 contracts and
   forgets across pulse gaps. Raised to gain 1.3 (memory needs the recurrent map
   to preserve state). The mandatory metabolic penalty is an M3/reaching
   requirement; here a large one smears the attractors, so it is kept tiny (1e-5).

### Finder design (fixed commitments)
- Velocity = autonomous map residual `F(h) = tanh(W_hh h + b) - h` (input held 0).
- Minimize `q = 1/2 ||F||^2` with Adam (+ short L-BFGS polish).
- **ICs sampled only from visited hidden states + noise**, never uniformly.
- **Speed cutoff from the q-distribution**, not a hard-coded absolute: cut at the
  largest log-gap if one exists; if unimodal (all ICs converge, as here) keep the
  whole low-speed cluster. (An early bug: the gap heuristic sliced *inside* the
  converged cluster and kept nothing — fixed by the unimodal branch.)
- **De-duplicate** nearby minima by clustering.
- Stability from Jacobian eigenvalues (continuous-surrogate: `Re(lambda) < 0`),
  which for a discrete stable attractor (`|eig(J_map)| < 1`) agrees.

### Reported numbers
- flip-flop task MSE: **0.0083** (solved; threshold 0.02).
- fixed points found: **27 unique = 8 stable + 19 saddles** — the exact Sussillo &
  Barak topology (8 corner attractors, saddles on the connecting edges/faces).
- 8-CORNER GATE: exactly **8 stable**, at **8 distinct cube corners**, readouts all
  within 0.05 of +/-1. **PASS.**
- differential test vs `pytorch-fixed-point-analysis`: the reference finder,
  seeded in each basin, converges to OUR corners. Metric is direction-aware
  because the reference's plain gradient descent is slow near an attractor and may
  not return from every seed within the epoch budget. Result:
  **precision = 0.0005** (worst reference point → nearest of our 8 corners; tol
  0.15 — i.e. every reference fixed point lands essentially exactly on one of
  ours) and **coverage = 6/8** (reference converged from 12/16 seeds in ~100 s).
  Coverage is reference-tool-limited, not a disagreement: our own finder recovers
  all 8, and where the reference converges it agrees to 5e-4.

## M2 · Brain data — PCA + jPCA (Stage 1)

**Status: PASS.** Loader in `python/data.py`, analysis in `tests/test_m2_brain.py`,
web export in `python/export.py`.

### Dataset: MC_Maze blocked → Churchland 2012 (see "Proposed deviations")
MC_Maze needs the DANDI API (`api.dandiarchive.org`), which the egress policy
blocks (403 on CONNECT; the DANDI *S3 bucket* is reachable, but enumerating a
dandiset's assets requires the API). HuggingFace and Zenodo are also blocked;
`raw.githubusercontent.com` is reachable. Per the README ("report it; use an
alternative PUBLIC reaching dataset; NOT synthetic") we use the **Churchland et
al. 2012** dataset itself — 218 neurons × 108 maze/reach conditions of
condition-averaged smoothed rates, aligned to movement onset (−50..550 ms, 10 ms
bins). This is the *actual Stage-1 reproduction target*, and Antin's reference
loader reads it, so the differential test runs on identical real input.

### Preprocessing note
Churchland's published rates are already trial-averaged and smoothed, so the
README's raw-spike steps (20 ms bins, Gaussian σ=40 ms) are already baked in by
the source; we apply the jPCA-specific preprocessing (soft-normalize → CCM
subtraction → PCA) on top, over the classic −50..150 ms rotation window.

### Reported numbers (window −50..150 ms, k=6)
- fit R² of `Ẋ = X Mᵀ` (skew, CCM subtracted): **0.524**.
- top rotation-plane variance fraction: **0.464** (one plane holds ~46% of the
  variance — strong rotational structure).
- rotation frequency: **1.37 Hz** (period ~730 ms; the reach sweeps a partial arc
  in the ~200 ms window, as in the paper).
- differential test vs Antin on REAL data: principal angle **0.0001°**, freq match
  **0.000%** — hand matches the oracle exactly on brain input.

### Adversarial check (Lebedev et al. 2019): rotation without CCM subtraction?
Running jPCA WITHOUT the cross-condition-mean subtraction: fit R² **0.556**, plane
variance **0.558**, frequency **2.49 Hz**. So **the rotation survives** — it is NOT
manufactured by the CCM step (contra a strong reading of the Lebedev critique).
BUT the frequency and plane change: without CCM, jPCA characterizes the large
condition-*independent* movement signal; with CCM it isolates the condition-
*dependent* rotation that is Churchland's actual claim. Both are real rotations of
different components — the preprocessing choice decides *which* you measure. The
before/after figure is saved with the analysis.

### Viewer / export
`export_brain` writes `data/brain.json`: 108 trajectories in a 3D basis
(jPC1, jPC2, top-orthogonal PC), `jpca_plane` = the z=0 plane, and **no fixed
points / no flow field** — the brain has no equations, and the file encodes that
asymmetry directly.

## M3 · Reaching RNN + jPCA (Stage 2 part 1)

**Status: PASS.** RNN in `python/rnn.py` (`ReachingRNN`, `train_reaching`),
analysis in `tests/test_m3_reaching.py`.

### The key concept (the one the README says not to get wrong)
The RNN is trained on the **behavioral task**, never on neural data. Inputs =
target direction cue `[cosθ, sinθ]` + a go signal; output = hand velocity (a
bell-shaped speed bump in direction θ, zero during the delay). We then ask whether
the **emergent** hidden dynamics rotate like M1. Fitting the RNN *to spikes* would
make "the RNN reproduces the brain" circular — the whole claim is that rotation
falls out of the *task demands*, so the training signal must be behavior only.

### RNN spec
256-unit continuous-time tanh RNN, Euler-integrated:
`dx/dt = −x + W_rec φ(x) + W_in u + b`, `x_{t+1} = x_t + (dt/τ) dx/dt`, `z = W_out x`,
τ=10. The closed-form velocity `dx/dt` is exposed for the M4 fixed-point finder.

### Reported numbers (movement epoch, k=6)
- velocity **R² = 0.9989** (task gate > 0.90) — task solved, so dynamics analysis
  is meaningful.
- jPCA on condition-averaged hidden states: skew fit **R² = 0.884**, top rotation-
  plane variance fraction **0.149**. **The RNN rotates** — same signature as the
  brain, with no spikes ever shown (Sussillo et al. 2015 reproduced).
- Why plane-variance is "only" 15% while fit R² is high: the largest-variance
  hidden direction is the *static target-tuning* axis; the rotation is a real but
  lower-variance component. jPCA fits the **derivative** structure, which is
  strongly rotational, hence high fit R² despite modest plane variance.

### Adversarial / honesty: the metabolic-regularization rationale does NOT reproduce
The README makes the metabolic penalty (L2 on firing rates) mandatory because
*without* it the network supposedly finds a non-biological **high-dimensional**
solution. Ablation in this simplified 16-direction task (participation ratio PR =
effective dimensionality):

| metabolic λ | velocity R² | jPCA fit R² | plane var | PR | mean\|rate\| |
|---|---|---|---|---|---|
| 0     | 0.998 | 0.884 | 0.137 | 2.52 | 0.392 |
| 1e-3  | 0.998 | 0.905 | 0.151 | 2.56 | 0.368 |
| 1e-2  | 0.997 | 0.912 | 0.164 | 2.69 | 0.292 |
| 1e-1  | 0.985 | 0.942 | 0.102 | 3.53 | 0.120 |

Two things stand out. (1) The penalty **does** do mechanical work: `mean|rate|`
falls monotonically as λ grows (0.39 → 0.12), which is exactly what an L2-on-rates
term should do. (2) But **effective dimensionality moves the wrong way** — PR
*rises* with λ (2.52 → 3.53), the **opposite** of the README's "without it →
high-dimensional" rationale. A plausible reading: squeezing per-unit rates forces
the computation to spread across *more* units, nudging dimensionality up. So in
this simplified 16-direction task the penalty is not what controls dimensionality;
the task is simple enough (2-D velocity, 16 clean conditions) that even the
unregularized solution is low-dimensional, and the Sussillo 2015 effect needs a
harder task (EMG/muscle output, richer condition set) to bite. We **keep** the
penalty (README-mandated, harmless at λ=1e-3, and it does bound rate magnitude),
and flag the rationale mismatch under "Proposed deviations" D2.

### Rejected alternatives
- **Discrete vanilla RNN (as in M1) for reaching:** rejected — the README fixes the
  reaching RNN as continuous-time, and continuous dynamics make the M4 flow field /
  Jacobian classification cleaner.

## M4 · Reaching RNN fixed points + flow field (Stage 2 part 2)
_(q(x) objective; IC sampling rule; tolerance from q-distribution; clustering;
Jacobian stability classes; adversarial check: FP stability to IC re-sampling +
hand-vs-reference Hungarian alignment; numbers.)_

## M5 · Unified viewer + deploy
_(space asymmetry handled honestly; matched visual scale only; deploy notes.)_

---

## Differential-test log

| # | Test | Input (identical to both) | Metric / tolerance | Result |
|---|------|---------------------------|--------------------|--------|
| M0-A | hand closed-form `M` vs Antin CG `skew_sym_regress` | same reduced `(X, Ẋ)` | `‖ΔM‖/‖M‖` small; plane angle < 5° | 3.8e-5; **0.0008°** ✅ |
| M0-A′ | vectorized-LS vs Sylvester (two hand closed forms) | same `(X, Ẋ)` | agree < 1e-8 | 9e-16 ✅ |
| M0-B | full hand `jpca()` vs `JPCA.fit()` | same raw `datas` | plane angle < 5°; freq < 5% | **0.0008°**; **0.000%** ✅ |
| M0-truth | clean single-plane rotation | pure `R(ω)` orbits | recovered ω = sin(ω) < 0.5% | 0.0000% ✅ |
| M1 | our finder vs `pytorch-fixed-point-analysis` | same trained flip-flop RNN | ref points → our corners < 0.15 | precision **5e-4**, coverage 6/8 ✅ |
| M2 | hand `jpca()` vs `JPCA.fit()` on REAL data | same Churchland 2012 rates | plane angle < 5°; freq < 5% | **0.0001°**; **0.000%** ✅ |

Run: `python tests/test_m0_jpca.py`; `REF_FPA_DIR=… python tests/test_m1_flipflop.py`.
(M2 re-runs M0-B on MC_Maze.)

## Proposed deviations (I propose, the user decides)
_(Any change to the FIXED commitments — jPCA steps, fixed-point objective + IC
rule, differential-test tolerances, flip-flop 8-corner gate, train-on-task-not-
spikes, scope boundary — goes here with rationale. Do NOT silently change these.)_

**D1 — Stage-1 dataset: MC_Maze → Churchland 2012 (FLAGGED, needs your call).**
MC_Maze (README's named primary) is unreachable in this environment: the DANDI
API returns 403 (egress policy), and HuggingFace/Zenodo are blocked too. The
README's own fallback rule says to use an alternative *public* reaching dataset
rather than synthetic. I chose **Churchland et al. 2012** (`exampleData.mat`, via
`raw.githubusercontent.com`), because it is (a) the exact Stage-1 reproduction
target the README names, (b) 108 reach conditions × 218 neurons — plenty for
jPCA, and (c) the data Antin's reference loader reads, so the differential test
runs on identical real input. This is arguably *more* faithful to "reproduce
Churchland 2012" than MC_Maze, and it is reversible (swap `data.load_mc_maze`
back in if the policy changes). **If you specifically need MC_Maze** (e.g. you can
grant DANDI egress or provide a local copy), say so and I'll switch — the M0/M2
pipeline is dataset-agnostic. Proceeding on Churchland 2012 for now.

**D2 — metabolic-reg rationale does not reproduce here (FLAGGED, not a change).**
I am **keeping** the metabolic penalty (README-mandatory). But its stated
justification — "without it the network finds a non-biological high-dimensional
solution" — does **not** hold in my simplified 16-direction reach task: the penalty
compresses rate magnitude (as expected) but effective dimensionality does not drop
— PR stays ~2.5 and if anything nudges up as λ grows (see M3 sweep table). The
penalty bounds rates but is not load-bearing for dimensionality here. Two honest options if you
want the penalty to *matter* (either is scope-expanding, so I did not take it):
(a) harden the task toward Sussillo 2015 (EMG/muscle-like output, more conditions,
longer horizon) so the unregularized solution actually inflates in dimension; or
(b) accept it as a faithful-but-inert term and move on. I recommend (b) for this
iteration. Your call.

## Honesty citations (in scope to mention, out of scope to run)
- Lebedev et al. (2019) — critique of the cross-condition-mean subtraction step.
- Elsayed & Cunningham (2017) — low-D/rotation may be a byproduct.
- Maheswaranathan et al. (2019) — geometric match is cheap (universality).

## Next iteration (noted, deliberately NOT built now)
_(shuffle/TME null models, DSA (Ostrow et al. 2023), cross-day/cross-task
alignment, LFADS/GPFA, disease data.)_

## References
See README "Key references" for exact citations.
