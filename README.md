# neural-dynamics

> 🇰🇷 한국어 번역본: [README.ko.md](README.ko.md) (구현 현황·다음 할 일 부록 포함)

# Project: Reproducing Motor Cortex Dynamics + Interactive Web Visualization

## Purpose
Reproduce two classic results in motor-cortex population dynamics and build an
interactive, shareable web visualization that lets you SEE the dynamics.
This is for implementation + learning. For every non-trivial design choice,
explain in NOTES.md and code comments WHY this choice and what you rejected.

- Stage 1 (descriptive): PCA + jPCA on real M1 data → rotational dynamics
  (Churchland et al. 2012, Nature 487:51-56)
- Stage 2 (mechanistic): fixed-point analysis of a task-trained RNN → the local
  structure that GENERATES the rotation
  (Sussillo & Barak 2013, Neural Computation 25(3):626-649;
   Sussillo et al. 2015, Nat Neurosci 18(7):1025-1033)

Conceptual relationship to respect throughout:
- jPCA is DESCRIPTIVE. It fits one global linear system to trajectory geometry.
  Requires no equations. Applies to BOTH brain data and RNN hidden states.
- Fixed-point analysis is MECHANISTIC. It needs the system's equations, so it
  applies to the RNN ONLY (we do not have the brain's equations).
- Fixed points GENERATE the rotation that jPCA DESCRIBES. One explains the other;
  they are not two things "compared against each other."
- Therefore: brain gets trajectories + jPCA plane only (no fixed points).
  RNN gets trajectories + jPCA + fixed points + flow field.
  The viewer must make this asymmetry visible.

## Development environment: Google Colab (primary) + static web app (deploy)
Split the system into two layers connected by JSON. Do not make them compete.

- COLAB NOTEBOOK = the kitchen. Runs the full pipeline, teaches the math, gives
  immediate inline feedback via plotly 3D (orbit built in) + ipywidgets sliders.
  Exports results to /data/*.json. Structure every section as:
  (1) markdown cell: the math and the "why", 
  (2) code cell: implementation, 
  (3) inline plotly cell: see the result immediately, 
  (4) markdown cell: interpretation + failure modes observed.
  First cell: mount Google Drive and cache MC_Maze + trained weights there, so
  sessions do not re-download/re-train (Colab filesystem is ephemeral).

- STATIC WEB APP = the plate. React + Vite + react-three-fiber. Loads the exported
  JSON and renders with Three.js. No live compute, no server, no browser storage.
  Deploy to Hugging Face Spaces (Static SDK) for a shareable URL with ML-community
  discoverability. (GitHub Pages is an acceptable fallback.)

Rationale: Colab interactive outputs die with the session and are not a shareable
link. plotly gives the "see it now" loop during development; Three.js gives the
deployable artifact. Build the plotly version first; upgrade to Three.js if time
remains.

## Repository structure
/notebook
  motor_dynamics.ipynb   # the Colab notebook (kitchen)
/python                  # importable modules the notebook calls (keep logic here, not in cells)
  data.py                # MC_Maze loading (nlb_tools), 20ms bins, Gaussian smooth (σ=40ms)
  pca_jpca.py            # PCA + jPCA (hand implementation, see spec below)
  rnn.py                 # 3-bit flip-flop RNN + reaching RNN
  fixedpoints.py         # PyTorch-autograd fixed/slow point finder + Jacobian + stability
  flowfield.py           # vector field dx/dt on a grid in PCA plane
  export.py              # serialize results to web JSON
/web                     # Vite + react-three-fiber app (plate)
/data                    # exported JSON
NOTES.md                 # math + design rationale + rejected alternatives (learning artifact)

## Differential-test policy (IMPORTANT — this is how we make the hand code falsifiable)
For the two hard algorithms, implement BY HAND as the primary path, then verify
against a maintained reference on IDENTICAL input. The reference is the oracle,
not a fallback. If hand and reference disagree beyond tolerance, one is wrong —
debug before proceeding. Keep both in the code; do not comment the reference out.

- jPCA reference: Benjamin Antin's `jPCA` Python package (pip). 
  Agreement test: principal angle between hand-derived and reference jPC planes
  < 5 degrees; recovered rotation frequencies match within 5%.
- Fixed-point reference: a lightweight PyTorch implementation of Sussillo & Barak
  (e.g. the `tripdancer0916/pytorch-fixed-point-analysis` repo). Do NOT pull in
  the TensorFlow FixedPointFinder — it fights the PyTorch stack. 
  Agreement test: on the flip-flop, both find 8 stable fixed points near cube
  corners; on the reaching RNN, fixed-point sets match after Hungarian alignment
  within tolerance.

## Data
- Primary: MC_Maze (Neural Latents Benchmark '21, Pei et al. 2021), via nlb_tools.
  Align on movement onset. Use prep + movement epochs. 20 ms bins, Gaussian smooth
  σ=40 ms, soft-normalize firing rates (Churchland-style: rate/(range+5)).
- If MC_Maze access fails, report it explicitly and propose an alternative PUBLIC
  reaching dataset. Do NOT substitute synthetic data — reproduction becomes meaningless.

## Pipeline (implementation order = milestones, each with a falsifiable pass/fail)

### M0. jPCA hand implementation, spec at sentence granularity
1. Take condition-averaged firing rates X_c(t) per condition c.
2. Soft-normalize each neuron; subtract the cross-condition mean at each timepoint
   (this is the step Lebedev et al. 2019 criticized — note it in NOTES, keep it for
   faithful reproduction, and save a before/after comparison).
3. PCA on the stacked, preprocessed data; keep top k=6 PCs. Project to get X (k-dim).
4. Compute state derivative Ẋ by finite difference.
5. Fit Ẋ = M X constrained to M skew-symmetric (M = -Mᵀ): vectorize, solve the
   constrained least squares in closed form (this is the crux — get the constraint
   right, cite Churchland 2012 supplementary methods).
6. Eigendecompose M; eigenvalues are purely imaginary (±iω pairs). The pair with
   largest |ω| defines the top jPC plane.
7. Project trajectories onto the jPC plane; they should sweep out rotations.
PASS/FAIL: fit R² of Ẋ=MX reported; top rotation-plane variance fraction reported;
differential test vs Antin `jPCA` passes.

### M1. Instrument calibration: flip-flop (known answer) — DO THIS FIRST
- Train a small vanilla tanh RNN on the 3-bit flip-flop task (Sussillo & Barak 2013).
- Run fixedpoints.py.
- FALSIFICATION TEST: exactly 8 stable fixed points, located near the corners of a
  cube in state space. If not, STOP and debug the finder before touching reaching data.
- Render these 8 fixed points + trajectories in the minimal viewer (this also
  validates the visualization pipeline on a known-answer case).

### M2. Brain data: PCA + jPCA (Stage 1)
- Apply M0 pipeline to MC_Maze.
- Brain viewer: per-condition trajectories + highlighted jPCA rotation plane.
  No fixed points (no equations).

### M3. Reaching RNN + jPCA (Stage 2, part 1)
- KEY CONCEPT (make explicit in NOTES): the RNN is trained on the BEHAVIORAL TASK,
  NOT on neural data. Inputs = target/condition cue + go signal. Output = hand
  velocity (or EMG). We then ask whether the EMERGENT hidden dynamics resemble M1.
  (This is the Sussillo et al. 2015 method; conflating it with "fit the RNN to
  spikes" is the classic fatal misunderstanding.)
- RNN spec: 256 units, tanh, continuous-time dx/dt = -x + W_rec·φ(x) + W_in·u + b,
  z = W_out·x. Metabolic regularization (L2 on firing rates) is MANDATORY — without
  it the network finds a non-biological high-dimensional solution.
- PASS/FAIL: report task performance (velocity R²) and require it above threshold
  BEFORE any dynamics analysis (analyzing a network that failed the task is meaningless).
- Apply the same PCA/jPCA to hidden states → does the RNN rotate too?

### M4. Reaching RNN fixed points + flow field (Stage 2, part 2)
- fixedpoints.py: minimize q(x) = ½‖dx/dt‖². Optimizer: Adam then optional L-BFGS
  polish. Sample initial conditions ONLY from hidden states visited on real trials
  (+ small noise) — not uniformly, which yields spurious slow points.
- Allow slow points; set the tolerance from the DISTRIBUTION of q values, do not
  hard-code an absolute threshold (network-dependent). De-duplicate nearby minima by
  clustering.
- Classify each point by Jacobian J = ∂(dx/dt)/∂x eigenvalues:
  stable / unstable / saddle / rotational (complex pair with near-zero real part).
- flowfield.py: dx/dt on a grid in the top PCA plane.
- RNN viewer: trajectories + flow field + fixed points colored by stability class.

### M5. Unified viewer + deploy
- Toggle: Brain | RNN | side-by-side. Note: the two live in DIFFERENT spaces; render
  each in its own jPCA space and match only the visual scale — do not force them into
  one coordinate system. The viewer should surface this limitation honestly.
- Deploy static app to Hugging Face Spaces (Static SDK).

## Interactive visualization spec (first-class deliverable — invest here)
Required interactions:
- OrbitControls (rotate/zoom/pan).
- Time scrubber + play/pause: trajectories draw over time.
- Condition selector: show/hide individual reach conditions.
- View toggle: Brain(M1) | RNN | side-by-side.
- RNN-only: toggle flow field; toggle fixed points; on fixed-point hover, tooltip
  shows Jacobian eigenvalues + stability class.
- Toggle jPCA rotation-plane highlight.
Representation:
- Trajectories as smooth TubeGeometry, conditions distinguished by hue.
- Fixed points colored by stability class.
Aesthetics (Apple-grade, restrained): dark background, subtle grid, high contrast,
clean typographic axis labels/legend. No gratuitous effects. Render brain and RNN at
matched visual scale so divergences are visible to the eye.

JSON schema (example):
{ "trajectories": {cond_id: [[x,y,z],...]},
  "fixed_points": [{"pos":[x,y,z], "eigs":[[re,im],...], "stability":"saddle"}],
  "flow_field": [{"pos":[x,y,z], "vec":[dx,dy,dz]}],
  "jpca_plane": [[...],[...]], "meta": {...} }

## Learning requirements (mandatory — this is half the point)
- NOTES.md: for each stage, the math and "why this choice + what I rejected".
- Comments explain WHY, not WHAT.
- Treat jPCA preprocessing, fixed-point vs slow-point, and Jacobian stability
  classification in particular depth.

## Scope boundary
This iteration ends at Stage 2 (reproduction + viewer). Shuffle/TME null models, DSA,
cross-day/cross-task alignment, LFADS/GPFA, and disease data are the NEXT iteration —
do not add them now, as they explode scope. If you find yourself wanting to add them,
note the idea in NOTES.md under "Next iteration" and move on.

## Key references (exact citations)
Reproduction targets:
- Churchland et al. (2012), Nature 487(7405):51-56 — jPCA / rotational dynamics (Stage 1)
- Sussillo & Barak (2013), Neural Computation 25(3):626-649 — fixed-point method (Stage 2)
- Sussillo, Churchland, Kaufman, Shenoy (2015), Nat Neurosci 18(7):1025-1033 —
  task-trained RNN reproduces M1 rotations (direct template)
Methods / tools:
- Golub & Sussillo (2018), JOSS 3(31):1003 — FixedPointFinder (method reference)
- Mante, Sussillo, Shenoy, Newsome (2013), Nature 503(7474):78-84 — fixed points dissect computation
- Pei et al. (2021), NeurIPS Datasets & Benchmarks — MC_Maze / NLB
Conceptual framing (NOTES.md):
- Shenoy, Sahani, Churchland (2013), Annu Rev Neurosci 36:337-359 — dynamical systems view
- Vyas, Golub, Sussillo, Shenoy (2020), Annu Rev Neurosci 43:249-275 — computation through dynamics
Honesty citations (mention in NOTES even though out of scope):
- Elsayed & Cunningham (2017), Nat Neurosci 20:1310-1318 — low-D/rotation may be a byproduct
- Maheswaranathan et al. (2019), NeurIPS 2019:15603-15615 — geometric match is cheap (universality)
- Ostrow, Eisen, Kozachkov, Fiete (2023), NeurIPS 2023, arXiv:2306.10168 — DSA, the right
  brain↔RNN comparison tool (next iteration)
