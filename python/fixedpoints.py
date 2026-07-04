"""fixedpoints.py — fixed/slow point finder + Jacobian (the MECHANISTIC layer).

Applies to the RNN ONLY: it operates on the vector field F(x) = dx/dt, which we
have in closed form because we built the network. It (a) minimizes the speed
q(x) = 1/2 ||dx/dt||^2 to locate fixed points (q ~ 0) and slow points, and
(b) linearizes there via the Jacobian J = d(dx/dt)/dx to classify stability.
The brain has no evaluable/differentiable F(x), so this layer never touches
brain data — that asymmetry is the whole point (README lines 18-27).

FIXED commitments (README M1, M4 — do not silently change):
  - ICs sampled ONLY from hidden states visited on real trials (+ small noise),
    never uniformly (uniform yields spurious slow points).
  - Tolerance set from the DISTRIBUTION of q values, not a hard-coded absolute.
  - De-duplicate nearby minima by clustering.
  - Flip-flop gate: exactly 8 stable fixed points near cube corners, else STOP.

Reference oracle: a PyTorch Sussillo & Barak implementation (e.g.
tripdancer0916/pytorch-fixed-point-analysis). NOT the TF FixedPointFinder.

Planned public surface (M1, M4):
    find_fixed_points(rnn, initial_states, ...)  -> candidate minima + q values
    jacobian(rnn, x)                             -> J at x (autograd)
    classify(J)                                  -> stable/unstable/saddle/rotational
    dedup_cluster(points, ...)                   -> unique fixed points
"""

# TODO(M1): finder + Jacobian + stability + 8-corner gate.  TODO(M4): reaching RNN.
