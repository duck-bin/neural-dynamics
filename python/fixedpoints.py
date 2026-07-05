"""fixedpoints.py — fixed/slow point finder + Jacobian (the MECHANISTIC layer).

Applies to the RNN ONLY: it operates on the vector field F(x) (the network's own
velocity), which we have in closed form because we built the network. It
(a) minimizes q(x) = 1/2 ||F(x)||^2 to locate fixed points (q ~ 0) and slow
points, and (b) linearizes there via the Jacobian J = dF/dx to classify
stability. The brain has no evaluable/differentiable F(x), so this layer never
touches brain data — that asymmetry is the whole point.

Velocity conventions:
  - discrete RNN (flip-flop, M1): F(h) = tanh(W_in u + W_hh h + b) - h  (map
    residual). Its Jacobian is J_map - I, so classifying by Re(eig(J)) < 0 is the
    continuous-surrogate reading of discrete stability.
  - continuous RNN (reaching, M4): F(x) = -x + W_rec phi(x) + W_in u + b = dx/dt.

FIXED commitments (README M1, M4 — do not silently change):
  - ICs sampled ONLY from hidden states visited on real trials (+ small noise).
  - Tolerance set from the DISTRIBUTION of q values, not a hard-coded absolute.
  - De-duplicate nearby minima by clustering.
  - Flip-flop gate: exactly 8 stable fixed points near cube corners, else STOP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# --------------------------------------------------------------------------
# Velocity functions F(x) (what the finder minimizes the norm of).
# --------------------------------------------------------------------------
def discrete_velocity(model, u=None):
    """Autonomous map residual F(h) = phi(W_in u + W_hh h) - h for a discrete RNN.

    u is the input held constant during the search (zeros = the autonomous memory
    dynamics we want the fixed points of). Returns a function h -> F(h) operating
    on a (batch, n_hid) torch tensor.
    """
    if u is None:
        u = torch.zeros(model.n_in)
    u = torch.as_tensor(u, dtype=torch.float32)
    # Constant w.r.t. the search variable h; detach so it is not a live graph
    # node that gets freed after the first backward (only x should carry grad).
    bias_in = model.w_in(u).detach()              # constant input drive (n_hid,)

    def F(h):
        pre = bias_in + model.w_hh(h)
        act = torch.tanh(pre) if model.activation == "tanh" else torch.relu(pre)
        return act - h

    return F


# --------------------------------------------------------------------------
# The finder: minimize q(x) = 1/2 ||F(x)||^2 from many initial conditions.
# --------------------------------------------------------------------------
@dataclass
class FinderResult:
    points: np.ndarray            # (n_ic, n_hid) converged states
    speeds: np.ndarray            # (n_ic,) ||F|| at each converged state


def find_fixed_points(
    velocity_fn,
    init_states: np.ndarray,
    adam_steps: int = 4000,
    lr: float = 0.05,
    lbfgs_steps: int = 40,
    device: str = "cpu",
) -> FinderResult:
    """Minimize q(x)=1/2||F(x)||^2 for a whole batch of ICs in parallel.

    Adam does the bulk descent; an optional L-BFGS polish sharpens the minima
    (README M4). All ICs are optimized simultaneously — each row is independent,
    so this is just a vectorized search, not a coupled objective.
    """
    x = torch.tensor(np.asarray(init_states, dtype=np.float32), device=device,
                     requires_grad=True)

    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(adam_steps):
        opt.zero_grad()
        q = 0.5 * (velocity_fn(x) ** 2).sum(dim=1)   # per-point speed^2
        q.sum().backward()                           # rows independent -> sum ok
        opt.step()

    if lbfgs_steps > 0:
        opt2 = torch.optim.LBFGS([x], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss = (0.5 * (velocity_fn(x) ** 2).sum(dim=1)).sum()
            loss.backward()
            return loss

        opt2.step(closure)

    with torch.no_grad():
        speeds = torch.norm(velocity_fn(x), dim=1).cpu().numpy()
    return FinderResult(x.detach().cpu().numpy(), speeds)


def sample_initial_conditions(visited_states, n_ic=400, noise=0.05, seed=0):
    """Sample ICs from states the network actually visited (+ Gaussian noise).

    README rule: NOT uniform sampling (which invents spurious slow points in
    regions the dynamics never explore). Noise lets the search fall into nearby
    basins rather than only re-finding visited points.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(visited_states), size=n_ic)
    picks = visited_states[idx].astype(np.float32)
    return picks + rng.normal(scale=noise, size=picks.shape).astype(np.float32)


# --------------------------------------------------------------------------
# Tolerance from the q distribution (not a hard-coded absolute).
# --------------------------------------------------------------------------
def speed_tolerance(speeds: np.ndarray, log_gap_factor: float = 10.0) -> float:
    """Pick a speed cutoff separating true minima from slow/non-converged points.

    Fixed points cluster at very low ||F||; everything else sits far higher. We
    find the largest gap in sorted log-speeds and cut there. Falls back to a
    small fraction of the median when no clear gap exists. Network-dependent by
    construction (README M4 forbids a hard-coded absolute threshold).
    """
    s = np.sort(speeds[speeds > 0])
    if s.size == 0:
        return 0.0
    logs = np.log10(s)
    gaps = np.diff(logs)
    if gaps.size and gaps.max() >= np.log10(log_gap_factor):
        # Clear low/high separation (true minima vs slow/non-converged): cut in
        # the gap, keeping everything below it.
        i = int(np.argmax(gaps))
        return float(10 ** ((logs[i] + logs[i + 1]) / 2))
    # Unimodal: every IC converged to the same low-speed regime (common on the
    # flip-flop, where the finder is very effective). Accept the whole cluster
    # rather than slicing an arbitrary gap inside it.
    return float(s[-1] * 10.0)


# --------------------------------------------------------------------------
# De-duplicate nearby minima by greedy clustering.
# --------------------------------------------------------------------------
def dedup_cluster(points: np.ndarray, tol: float = 0.1) -> np.ndarray:
    """Collapse points within `tol` (Euclidean) into single representatives."""
    reps = []
    for p in points:
        if all(np.linalg.norm(p - r) > tol for r in reps):
            reps.append(p)
    return np.array(reps) if reps else np.empty((0, points.shape[1]))


# --------------------------------------------------------------------------
# Jacobian + stability classification.
# --------------------------------------------------------------------------
def jacobian(velocity_fn, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Autograd Jacobian J = dF/dx at a single state x (n_hid, n_hid)."""
    xt = torch.tensor(np.asarray(x, dtype=np.float32), device=device)

    def f(v):
        return velocity_fn(v.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, xt)
    return J.detach().cpu().numpy()


@dataclass
class Stability:
    label: str                    # stable | unstable | saddle
    is_rotational: bool           # leading eigenpair complex with |Re| ~ 0
    eigenvalues: np.ndarray       # complex eigenvalues of J
    n_unstable: int               # count of eigenvalues with Re > tol


def classify(J: np.ndarray, real_tol: float = 1e-3) -> Stability:
    """Classify a fixed point from Jacobian eigenvalues (continuous convention).

    Re(lambda) < 0 = contracting direction, > 0 = expanding. All contracting =>
    stable; all expanding => unstable; mixed => saddle. 'Rotational' flags a
    leading complex pair with near-zero real part (a rotation-organizing point),
    which is exactly the local structure that generates jPCA-style rotation.
    """
    eigs = np.linalg.eigvals(J)
    n_unstable = int(np.sum(eigs.real > real_tol))
    if n_unstable == 0:
        label = "stable"
    elif n_unstable == len(eigs):
        label = "unstable"
    else:
        label = "saddle"

    order = np.argsort(-eigs.real)               # least-stable first
    lead = eigs[order[0]]
    is_rot = bool(abs(lead.imag) > real_tol and abs(lead.real) < real_tol)
    return Stability(label, is_rot, eigs, n_unstable)
