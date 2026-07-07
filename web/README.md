# Motor Cortex Dynamics — web viewer

A self-contained, dependency-free 3D viewer for the exported dynamics. Custom
Canvas renderer (no Three.js / no external libraries), so `index.html` runs
identically as a local file, on a static host, or as a Claude Artifact.

## What it shows
- **Brain (M1)** — Churchland 2012 reach trajectories + the jPCA rotation plane.
  No fixed points, no flow field: we have no equations for the brain.
- **RNN** — task-trained reaching RNN: trajectories + jPCA plane + the fixed
  point(s) + the flow field `dx/dt`. Hover a fixed point for its Jacobian
  eigenvalues and stability class.
- **Side by side** — both at matched visual scale. They live in *different* state
  spaces (each in its own jPCA coordinates); the viewer says so and never forces
  them into one frame.

Interactions: orbit (drag) · zoom (wheel) · time scrubber + play · condition
count · layer toggles (plane / flow / fixed points).

## Build
`index.html` is generated — data is inlined so the page is fully standalone:

```bash
python scripts/build_rnn_export.py   # writes data/rnn.json (needs cache/reaching_metab.pt)
python scripts/build_viewer.py       # inlines data/*.json into web/index.html
```

`viewer_template.html` is the source (with `__BRAIN_JSON__` / `__RNN_JSON__`
placeholders); `index.html` and `artifact_body.html` are build outputs.

## Deploy — Hugging Face Spaces (Static SDK)
1. Create a Space → SDK **Static**.
2. Add `index.html` at the repo root of the Space (it is fully self-contained —
   no other files needed).
   ```bash
   git clone https://huggingface.co/spaces/<user>/motor-cortex-dynamics
   cp web/index.html motor-cortex-dynamics/index.html
   cd motor-cortex-dynamics && git add index.html && git commit -m "viewer" && git push
   ```
3. The Space serves it at `https://<user>-motor-cortex-dynamics.static.hf.space`.

GitHub Pages is an equivalent fallback: drop `index.html` on a `gh-pages` branch.
