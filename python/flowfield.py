"""flowfield.py — the RNN velocity field dx/dt sampled on a 2D grid (RNN only).

Evaluates the network's own velocity F(x) = dx/dt on a grid spanning a 2D plane
(the top jPCA plane) embedded in full hidden space, and projects the resulting
velocity back onto that plane. The viewer draws these as arrows, so the rotation
that the fixed point organizes becomes literally visible as swirling flow. Like
fixedpoints.py this needs the closed-form F(x), so it applies to the RNN ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FlowField:
    coords: np.ndarray            # (n*n, 2) grid positions in plane coordinates
    vectors: np.ndarray           # (n*n, 2) dx/dt projected onto the plane
    extent: float                 # half-width of the grid in plane units


def flow_on_grid(velocity_fn, origin, plane, extent, n=21, device="cpu") -> FlowField:
    """Sample dx/dt on an n x n grid in the plane through `origin` spanned by `plane`.

    velocity_fn: x (batch, N) torch tensor -> dx/dt (batch, N).
    origin: (N,) center of the plane (e.g. the mean hidden state).
    plane:  (N, 2) orthonormal basis of the plane in hidden space.
    A grid point at plane coords (a, b) maps to x = origin + a*e1 + b*e2; we
    evaluate F(x) there and project it back: (F·e1, F·e2).
    """
    origin = np.asarray(origin, dtype=np.float32)
    plane = np.asarray(plane, dtype=np.float32)               # (N, 2)
    gs = np.linspace(-extent, extent, n)
    A, B = np.meshgrid(gs, gs)
    ab = np.stack([A.ravel(), B.ravel()], axis=1)             # (n*n, 2)

    X = (origin[None, :] + ab @ plane.T).astype(np.float32)   # (n*n, N) embed to hidden space
    with torch.no_grad():
        V = velocity_fn(torch.tensor(X, device=device)).cpu().numpy()
    Vp = V @ plane                                            # (n*n, 2) project onto plane
    return FlowField(ab, Vp, float(extent))


def jpca_plane_in_hidden(jpca_result) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, plane) in hidden space from a JPCAResult.

    origin = the PCA mean (center of the reduced space in hidden coordinates);
    plane  = the two jPC axes lifted from PCA space back to hidden space and
    re-orthonormalized. Convenience for wiring the flow field to M3's jPCA.
    """
    pca = jpca_result.pre.pca
    plane_hidden = pca.components_.T @ jpca_result.plane       # (N, k)(k, 2) -> (N, 2)
    q, _ = np.linalg.qr(plane_hidden)
    return pca.mean_.astype(np.float32), q[:, :2].astype(np.float32)
