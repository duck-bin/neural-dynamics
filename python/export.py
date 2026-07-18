"""export.py — serialize pipeline results to web JSON (Colab -> static app).

The notebook (kitchen) and the web app (plate) are connected ONLY by JSON. Brain
exports carry trajectories + jpca_plane and NO fixed_points/flow_field; RNN
exports carry all four — the file contents encode the brain/RNN asymmetry the
viewer must show (the brain has no equations, so no mechanistic layer).

3D embedding: jPCA gives a 2D rotation plane living in the k-dim PCA space. For a
3D viewer we build an orthonormal basis [jPC1, jPC2, u3] where u3 is the highest-
variance PCA direction orthogonal to the plane, and project trajectories onto it.
The jPCA plane is then simply the z = 0 plane, trivial for the viewer to
highlight.

JSON schema (README):
    { "trajectories": {cond_id: [[x,y,z], ...]},
      "fixed_points": [{"pos":[x,y,z], "eigs":[[re,im],...], "stability":"saddle"}],
      "flow_field":   [{"pos":[x,y,z], "vec":[dx,dy,dz]}],
      "jpca_plane":   [[...],[...]],
      "meta": {...} }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _embed_basis(reduced: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Orthonormal (k x 3) basis: the two jPC plane axes + top orthogonal PC."""
    C, T, k = reduced.shape
    Xr = reduced.reshape(C * T, k)
    # Variance in each PCA dim after removing the plane component.
    resid = Xr - (Xr @ plane) @ plane.T
    # Highest-variance orthogonal direction via SVD of the residual.
    _, _, vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False)
    u3 = vt[0]
    u3 = u3 - plane @ (plane.T @ u3)
    u3 = u3 / (np.linalg.norm(u3) + 1e-12)
    return np.column_stack([plane[:, 0], plane[:, 1], u3])   # (k, 3)


def export_brain(path, jpca_result, conditions, meta=None) -> dict:
    """Write brain JSON: trajectories + jpca_plane, NO fixed points/flow field."""
    res = jpca_result
    basis = _embed_basis(res.pre.reduced, res.plane)         # (k, 3)
    C, T, k = res.pre.reduced.shape
    coords = res.pre.reduced.reshape(C * T, k) @ basis        # (C*T, 3)
    coords = coords.reshape(C, T, 3)

    # Key by condition INDEX (unique); labels may be empty/duplicated in the data.
    trajectories = {str(c): np.round(coords[c], 5).tolist() for c in range(C)}
    labels = [str(conditions[c]) for c in range(C)] if conditions else None
    payload = {
        "trajectories": trajectories,
        "labels": labels,
        "fixed_points": [],           # brain has no equations -> no mechanism
        "flow_field": [],
        "jpca_plane": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],   # the z=0 plane in this basis
        "meta": {
            "kind": "brain",
            "has_mechanism": False,
            "n_conditions": C,
            "fit_R2": round(float(res.fit_R2), 4),
            "plane_var_frac": round(float(res.plane_var_frac), 4),
            **(meta or {}),
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))
    return payload


def export_rnn(path, jpca_result, conditions, fixed_points=None, flow_field=None,
               meta=None) -> dict:
    """Write RNN JSON: trajectories + jpca_plane + fixed_points + flow_field.

    fixed_points: list of dicts {pos(k-dim), eigs, stability}; positions are
    projected into the same 3D basis. flow_field similarly. (Wired at M4.)
    """
    res = jpca_result
    basis = _embed_basis(res.pre.reduced, res.plane)
    C, T, k = res.pre.reduced.shape
    coords = (res.pre.reduced.reshape(C * T, k) @ basis).reshape(C, T, 3)
    trajectories = {str(c): np.round(coords[c], 5).tolist() for c in range(C)}
    labels = [str(conditions[c]) for c in range(C)] if conditions else None

    fps = []
    for fp in (fixed_points or []):
        pos3 = (np.asarray(fp["pos"]) @ basis).tolist()
        fps.append({"pos": np.round(pos3, 5).tolist(),
                    "eigs": fp.get("eigs", []), "stability": fp.get("stability", "")})
    flows = []
    for fv in (flow_field or []):
        p3 = (np.asarray(fv["pos"]) @ basis).tolist()
        v3 = (np.asarray(fv["vec"]) @ basis).tolist()
        flows.append({"pos": np.round(p3, 5).tolist(), "vec": np.round(v3, 5).tolist()})

    payload = {
        "trajectories": trajectories,
        "labels": labels,
        "fixed_points": fps,
        "flow_field": flows,
        "jpca_plane": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "meta": {"kind": "rnn", "has_mechanism": True, "n_conditions": C,
                 "fit_R2": round(float(res.fit_R2), 4),
                 "plane_var_frac": round(float(res.plane_var_frac), 4),
                 **(meta or {})},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))
    return payload


def _hidden_basis(traj_hidden: np.ndarray, plane: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Orthonormal (N x 3) basis in HIDDEN space: jPC1, jPC2, + top orthogonal dir.

    Used for the RNN export, where trajectories, fixed points, and the flow field
    all live in raw hidden space and must share ONE linear projection so they
    overlay correctly (the k-space `_embed_basis` cannot place raw-hidden fixed
    points without re-applying jPCA preprocessing, which is not a clean map).
    """
    C, T, N = traj_hidden.shape
    Xc = traj_hidden.reshape(C * T, N) - origin
    resid = Xc - (Xc @ plane) @ plane.T
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    u3 = vt[0] - plane @ (plane.T @ vt[0])
    u3 = u3 / (np.linalg.norm(u3) + 1e-12)
    return np.column_stack([plane[:, 0], plane[:, 1], u3])   # (N, 3)


def export_rnn_hidden(path, traj_hidden, plane, origin, fixed_points, flow_field,
                      meta=None) -> dict:
    """RNN JSON from HIDDEN-space inputs, all projected through ONE shared basis.

    traj_hidden : (C, T, N) raw condition-averaged hidden states.
    plane       : (N, 2) orthonormal jPC plane in hidden space.
    origin      : (N,) center of the projection (e.g. hidden mean).
    fixed_points: list of {'x': (N,), 'eigs': [[re,im],...], 'stability': str,
                           'is_spiral': bool}.
    flow_field  : {'pos': (M, N), 'vec': (M, N)} sampled dx/dt (or None).
    """
    traj_hidden = np.asarray(traj_hidden, float)
    plane = np.asarray(plane, float)
    origin = np.asarray(origin, float)
    C, T, N = traj_hidden.shape
    basis = _hidden_basis(traj_hidden, plane, origin)         # (N, 3)

    coords = ((traj_hidden.reshape(C * T, N) - origin) @ basis).reshape(C, T, 3)
    trajectories = {str(c): np.round(coords[c], 5).tolist() for c in range(C)}

    fps = []
    for fpt in (fixed_points or []):
        pos3 = ((np.asarray(fpt["x"], float) - origin) @ basis)
        fps.append({"pos": np.round(pos3, 5).tolist(),
                    "eigs": fpt.get("eigs", []),
                    "stability": fpt.get("stability", ""),
                    "is_spiral": bool(fpt.get("is_spiral", False))})

    flows = []
    if flow_field is not None:
        P = (np.asarray(flow_field["pos"], float) - origin) @ basis
        V = np.asarray(flow_field["vec"], float) @ basis
        for p3, v3 in zip(P, V):
            flows.append({"pos": np.round(p3, 5).tolist(), "vec": np.round(v3, 5).tolist()})

    payload = {
        "trajectories": trajectories,
        "labels": None,
        "fixed_points": fps,
        "flow_field": flows,
        "jpca_plane": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],     # jPC plane = z=0 in this basis
        "meta": {"kind": "rnn", "has_mechanism": True, "n_conditions": C, **(meta or {})},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))
    return payload
