"""pca_jpca.py — PCA + hand-implemented jPCA (the DESCRIPTIVE layer).

jPCA fits ONE global linear system to trajectory geometry, constrained to be
skew-symmetric so its dynamics are pure rotation: Xdot = X M^T with M = -M^T
(equivalently dx/dt = M x). It needs no equations of the underlying system,
only observed trajectories + their finite-difference derivatives, which is why
it applies to BOTH brain data and RNN states.

Hand implementation is the PRIMARY path; Benjamin Antin's `jPCA` package is the
differential-test ORACLE (see tests/test_m0_jpca.py). Both stay in the code.

Convention (matches Antin, so M is directly comparable):
    state rows in X (T x k), derivative rows in Xdot (T x k),
    model  Xdot = X @ M.T   <=>   dx/dt = M x,   M skew-symmetric.

The crux is step 5 (README M0): fit the skew-symmetric M by CONSTRAINED least
squares, solved in CLOSED FORM. We vectorize over a basis of skew-symmetric
matrices and solve one linear least-squares system. Antin instead runs an
iterative CG solver on the same objective; two independent solvers agreeing is
a strong correctness check. (Cite: Churchland et al. 2012, Nature, supp. methods.)

Algorithm steps are a FIXED commitment (README M0, steps 1-7). Do not silently
alter the ordering, the cross-condition-mean subtraction, or the skew-symmetric
constraint; propose changes in NOTES "Proposed deviations".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------
# Step 2 preprocessing: soft-normalize, then subtract the cross-condition mean.
# --------------------------------------------------------------------------
def soft_normalize(datas: np.ndarray, const: float = 5.0) -> np.ndarray:
    """Churchland-style soft normalization of firing rates.

    Divide each neuron by its full-range (max - min across ALL conditions and
    times) plus a constant. WHY the constant: pure range normalization would
    blow up low-firing / low-modulation units (dividing by a tiny range makes
    noise look like signal); the +const softens that so quiet neurons stay
    quiet instead of dominating. const<=0 skips the step.

    datas: (C, T, N) condition x time x neuron.
    """
    if const <= 0:
        return datas.astype(float, copy=True)
    fr_range = datas.max(axis=(0, 1)) - datas.min(axis=(0, 1))  # (N,)
    return datas / (fr_range + const)


def subtract_cc_mean(datas: np.ndarray) -> np.ndarray:
    """Subtract the cross-condition mean at each timepoint (README step 2).

    cc_mean[t, n] = mean over conditions of datas[:, t, n]. This removes the
    condition-independent component (the part every reach shares) so jPCA sees
    only condition-DEPENDENT structure.

    NOTE (Lebedev et al. 2019): this step is contested — it can manufacture
    apparent rotation. We KEEP it for faithful reproduction of Churchland 2012
    and test its effect adversarially at M2 (rotation with vs without it).
    """
    return datas - datas.mean(axis=0, keepdims=True)


@dataclass
class Preprocessed:
    """Result of preprocessing: reduced data + the PCA needed to back-project."""
    reduced: np.ndarray          # (C, T, k) data in top-k PCA space
    pca: PCA                     # fitted sklearn PCA (components_ is k x N)
    full_data_var: float         # total variance of preprocessed (pre-PCA) data
    pca_var: np.ndarray          # variance captured by each of the k PCs


def preprocess(
    datas: np.ndarray,
    num_pcs: int = 6,
    soft_norm_const: float = 5.0,
    subtract_ccm: bool = True,
) -> Preprocessed:
    """README steps 1-3: soft-normalize -> subtract CC mean -> PCA to k dims.

    datas: (C, T, N). Returns reduced (C, T, k) plus the fitted PCA.
    subtract_ccm is exposed so M2's adversarial check can toggle it.
    """
    datas = np.asarray(datas, dtype=float)
    C, T, N = datas.shape

    datas = soft_normalize(datas, soft_norm_const)
    if subtract_ccm:
        datas = subtract_cc_mean(datas)

    X_full = datas.reshape(C * T, N)
    full_data_var = float(np.sum(np.var(X_full, axis=0)))

    pca = PCA(n_components=num_pcs)
    reduced = pca.fit_transform(X_full).reshape(C, T, num_pcs)
    return Preprocessed(reduced, pca, full_data_var, pca.explained_variance_)


# --------------------------------------------------------------------------
# Step 4: state derivative by first difference, WITHIN each condition.
# --------------------------------------------------------------------------
def finite_diff(reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-condition (X, Xdot). Xdot is a first difference (dt = 1 bin).

    Differences are taken WITHIN a condition only — never across the seam
    between two conditions — so we drop each condition's last sample from X.
    reduced: (C, T, k) -> X (C*(T-1), k), Xdot (C*(T-1), k).
    """
    X = np.concatenate([c[:-1] for c in reduced], axis=0)
    Xdot = np.concatenate([np.diff(c, axis=0) for c in reduced], axis=0)
    return X, Xdot


# --------------------------------------------------------------------------
# Step 5 (the crux): fit Xdot = X M^T with M skew-symmetric, in closed form.
# --------------------------------------------------------------------------
def _skew_basis(k: int) -> np.ndarray:
    """Orthogonal basis of k x k skew-symmetric matrices: m = k(k-1)/2 of them.

    Basis element for pair (a<b) has +1 at (a,b) and -1 at (b,a). Every skew
    matrix is a unique linear combination of these, so fitting M reduces to
    fitting m free coefficients.
    """
    idx = np.triu_indices(k, k=1)
    m = len(idx[0])
    E = np.zeros((m, k, k))
    for p, (a, b) in enumerate(zip(*idx)):
        E[p, a, b] = 1.0
        E[p, b, a] = -1.0
    return E


def fit_skew_symmetric_M(X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    """Closed-form constrained least squares for skew-symmetric M.

    Minimize ||X M^T - Xdot||_F^2 over M = -M^T. Writing M = sum_p theta_p E_p
    for the skew basis {E_p}, the model is LINEAR in theta:
        X M^T = sum_p theta_p (X E_p^T) = -sum_p theta_p (X E_p)   (E_p^T = -E_p)
    so we stack columns g_p = vec(-X E_p) into G and solve the ordinary least
    squares G theta = vec(Xdot). This is the "vectorize + closed-form" solve of
    README step 5 — one lstsq, no iteration. (Antin solves the same objective by
    iterative CG; tests/test_m0_jpca.py checks the two agree.)
    """
    T, k = X.shape
    E = _skew_basis(k)                      # (m, k, k)
    # g_p = vec(-X E_p); columns of G. -X E_p has shape (T, k) -> flatten.
    G = np.stack([(-(X @ E[p])).ravel() for p in range(E.shape[0])], axis=1)
    theta, *_ = np.linalg.lstsq(G, Xdot.ravel(), rcond=None)
    M = np.einsum("p,pij->ij", theta, E)
    # Clean tiny numerical asymmetry; the parametrization guarantees skew.
    return 0.5 * (M - M.T)


def _fit_skew_symmetric_M_sylvester(X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    """Alternative closed form via the normal equations (internal cross-check).

    Optimality of ||X M^T - Xdot||^2 over skew M requires skew(grad) = 0, which
    reduces to the Sylvester equation  C M + M C = D^T - D  with C = X^T X and
    D = X^T Xdot. Solved by scipy; the solution is automatically skew when C is
    positive definite. Kept only so the test can confirm both closed forms match.
    """
    from scipy.linalg import solve_sylvester

    C = X.T @ X
    D = X.T @ Xdot
    M = solve_sylvester(C, C, D.T - D)
    return 0.5 * (M - M.T)


# --------------------------------------------------------------------------
# Step 6: eigendecompose M -> the top rotation (jPC) plane and its frequency.
# --------------------------------------------------------------------------
def jpc_plane_from_M(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Top jPC plane basis (k x 2, orthonormal) and its angular freq |omega|.

    A real skew-symmetric M has purely imaginary eigenvalues in conjugate pairs
    (+/- i*omega). The pair with the largest |omega| spans the fastest-rotating
    plane; its eigenvector v = u + i w gives real spanning vectors u, w. We
    orthonormalize [u, w] (QR) so the returned basis is clean for projection and
    for principal-angle comparison (span is what matters).
    """
    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(np.abs(eigvals.imag))[::-1]
    top = order[0]
    omega = float(abs(eigvals[top].imag))            # rad per bin (dt = 1)

    v = eigvecs[:, top]
    plane = np.column_stack([v.real, v.imag])
    Q, _ = np.linalg.qr(plane)                        # k x 2 orthonormal
    return Q[:, :2], omega


# --------------------------------------------------------------------------
# Fit quality + geometry metrics.
# --------------------------------------------------------------------------
def fit_R2(X: np.ndarray, Xdot: np.ndarray, M: np.ndarray) -> float:
    """Fraction of derivative 'energy' explained by the skew-symmetric fit.

    R2 = 1 - ||Xdot - X M^T||^2 / ||Xdot||^2 (through the origin: the linear
    dynamical model has no intercept). This is the headline M0 number.
    """
    resid = Xdot - X @ M.T
    return 1.0 - float(np.sum(resid ** 2) / np.sum(Xdot ** 2))


def plane_variance_fraction(reduced: np.ndarray, plane: np.ndarray) -> float:
    """Fraction of the reduced data's variance captured by a 2D plane."""
    C, T, k = reduced.shape
    Xr = reduced.reshape(C * T, k)
    proj = Xr @ plane                                 # (C*T, 2)
    return float(np.sum(np.var(proj, axis=0)) / np.sum(np.var(Xr, axis=0)))


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B.

    Orthonormalize each, then the singular values of Qa^T Qb are the cosines of
    the principal angles. Used as the differential-test metric vs Antin: the max
    angle must be < 5 degrees. Basis/sign/rotation within each plane is ignored.
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


# --------------------------------------------------------------------------
# Full pipeline (steps 1-7).
# --------------------------------------------------------------------------
@dataclass
class JPCAResult:
    M: np.ndarray                 # skew-symmetric dynamics matrix (k x k)
    plane: np.ndarray             # top jPC plane basis in PCA space (k x 2)
    plane_neuron: np.ndarray      # same plane back-projected to neuron space (N x 2)
    omega: float                  # angular frequency of top plane (rad / bin)
    fit_R2: float                 # R2 of Xdot = X M^T
    plane_var_frac: float         # variance fraction captured by the plane (PCA space)
    projected: np.ndarray         # trajectories on the plane (C, T, 2)
    pre: Preprocessed


def jpca(
    datas: np.ndarray,
    num_pcs: int = 6,
    soft_norm_const: float = 5.0,
    subtract_ccm: bool = True,
) -> JPCAResult:
    """End-to-end hand jPCA (README M0 steps 1-7). datas: (C, T, N)."""
    pre = preprocess(datas, num_pcs, soft_norm_const, subtract_ccm)
    X, Xdot = finite_diff(pre.reduced)
    M = fit_skew_symmetric_M(X, Xdot)
    plane, omega = jpc_plane_from_M(M)
    r2 = fit_R2(X, Xdot, M)
    var_frac = plane_variance_fraction(pre.reduced, plane)

    C, T, k = pre.reduced.shape
    projected = (pre.reduced.reshape(C * T, k) @ plane).reshape(C, T, 2)
    plane_neuron = pre.pca.components_.T @ plane      # (N x k)(k x 2) = N x 2

    return JPCAResult(M, plane, plane_neuron, omega, r2, var_frac, projected, pre)
