"""flowfield.py — vector field dx/dt sampled on a grid (RNN only).

Evaluates the RNN's velocity field dx/dt on a 2D grid spanning the top PCA plane
(back-projected to full state space), so the viewer can render the flow that the
fixed points organize. Like fixedpoints.py, this needs the closed-form F(x) and
therefore applies to the RNN ONLY.

Planned public surface (M4):
    flow_on_grid(rnn, plane_basis, center, extent, n)
        -> grid positions (in plane coords) + dx/dt vectors (projected to plane)
"""

# TODO(M4): sample dx/dt on a grid in the top PCA plane. See README M4.
