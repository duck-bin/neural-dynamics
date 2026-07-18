"""M0 differential test: hand jPCA vs Benjamin Antin's `jPCA` (the oracle).

We validate the hand implementation three ways on synthetic data whose dominant
rotation frequency and plane are KNOWN:

  Layer A (the crux, isolated): feed IDENTICAL (X, Xdot) to the hand closed-form
      skew-symmetric solver and to Antin's iterative CG solver. Two independent
      solvers of the same constrained problem must land on the same M.
  Layer B (full pipeline): feed IDENTICAL raw datas to hand `jpca()` and to
      `JPCA.fit()`. Top jPC planes must agree (principal angle < 5 deg) and the
      rotation frequencies must match within 5%.
  Ground truth: for a CLEAN single-plane rotation, the recovered frequency must
      equal sin(omega_true) (see note below), confirming absolute correctness.

Note on the frequency bias: with a FORWARD first difference (what Churchland and
Antin use), a rotation of omega rad/bin gives states X(t+1) = R(omega) X(t), so
Xdot = (R(omega) - I) X. The skew-symmetric part of R(omega) - I is
[[0,-sin w],[sin w,0]], whose eigenvalues are +/- i*sin(omega). Thus the fit
recovers sin(omega), not omega -- a KNOWN, ~omega^3/6 bias (1.5% at 0.3 rad/bin),
shared exactly by the reference oracle. We keep the forward difference for
faithful reproduction and record the bias in NOTES; a central difference would
reduce it (a documented, not-yet-taken option).

Reported numbers: fit R^2 of Xdot=X M^T, top rotation-plane variance fraction,
principal angles, and frequency matches.

Synthetic data is used ONLY to unit-test the algorithm against the oracle with a
known answer. It is NOT a stand-in for brain data — the real reproduction runs
this same hand-vs-Antin test on MC_Maze at M2 (see README "Data").
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python import pca_jpca as J  # noqa: E402

from jPCA import JPCA  # noqa: E402
from jPCA.regression import skew_sym_regress  # noqa: E402


# --------------------------------------------------------------------------
def make_synthetic(seed=0):
    """Condition-averaged, firing-rate-like data with a KNOWN dominant rotation.

    Latents: a dominant 2D rotation at omega_true, a slower weaker 2D rotation,
    and a non-rotational decaying mode. Conditions differ by initial phase. A
    large condition-INDEPENDENT ramp is added (removed later by CC-mean
    subtraction). Everything is embedded into N neurons via random loadings with
    a positive baseline and mild noise.
    """
    rng = np.random.default_rng(seed)
    C, T, N = 8, 40, 30
    t = np.arange(T)

    omega_true = 0.30   # rad/bin — the dominant rotation we must recover
    omega2 = 0.10       # slower, weaker rotation (a distractor plane)

    phases1 = np.linspace(0, 2 * np.pi, C, endpoint=False)
    phases2 = rng.uniform(0, 2 * np.pi, C)
    decay_sign = rng.uniform(-1, 1, C)

    L = 5
    latents = np.zeros((C, T, L))
    for c in range(C):
        latents[c, :, 0] = 1.00 * np.cos(omega_true * t + phases1[c])
        latents[c, :, 1] = 1.00 * np.sin(omega_true * t + phases1[c])
        latents[c, :, 2] = 0.35 * np.cos(omega2 * t + phases2[c])
        latents[c, :, 3] = 0.35 * np.sin(omega2 * t + phases2[c])
        latents[c, :, 4] = 0.30 * decay_sign[c] * np.exp(-t / 12.0)

    W = rng.normal(size=(L, N))                      # latent -> neuron loadings
    baseline = rng.uniform(5, 25, size=N)            # rate-like positive offset
    shared_ramp = np.outer(0.8 * t / T, rng.normal(size=N))  # cond-independent

    datas = np.zeros((C, T, N))
    for c in range(C):
        datas[c] = baseline + latents[c] @ W + shared_ramp
        datas[c] += rng.normal(scale=0.05, size=(T, N))
    return datas, omega_true


def principal_angle_max(A, B):
    return float(J.principal_angles_deg(A, B).max())


def ground_truth_frequency():
    """Clean single-plane rotation: recovered omega must equal sin(omega_true).

    No distractor plane, no decay, no soft-norm -- so the ONLY departure from the
    injected omega is the forward-difference bias, whose exact prediction is
    sin(omega). Passing this to ~0 error is the absolute-correctness proof for the
    skew fit + eigendecomposition. Returns (max err vs sin, worst raw bias vs true).
    """
    err_vs_sin, bias_vs_true = [], []
    for w in (0.05, 0.10, 0.20, 0.30):
        C, T = 6, 60
        t = np.arange(T)
        datas = np.zeros((C, T, 2))
        for c in range(C):
            ph = 2 * np.pi * c / C
            datas[c, :, 0] = np.cos(w * t + ph)
            datas[c, :, 1] = np.sin(w * t + ph)
        X, Xdot = J.finite_diff(datas)
        _, w_rec = J.jpc_plane_from_M(J.fit_skew_symmetric_M(X, Xdot))
        err_vs_sin.append(abs(w_rec - np.sin(w)) / np.sin(w))
        bias_vs_true.append(abs(w_rec - w) / w)
    return max(err_vs_sin), max(bias_vs_true)


def main():
    datas, omega_true = make_synthetic()
    C, T, N = datas.shape
    print(f"synthetic data: C={C} conditions, T={T} bins, N={N} neurons")
    print(f"injected dominant omega = {omega_true:.4f} rad/bin\n")

    # ----- hand full pipeline -----
    res = J.jpca(datas, num_pcs=6, soft_norm_const=5.0, subtract_ccm=True)

    # ----- Antin full pipeline on IDENTICAL raw datas -----
    datas_list = [datas[c] for c in range(C)]
    times = list(range(T))
    ref = JPCA(num_jpcs=2)
    ref_proj, ref_full_var, ref_pca_var, ref_jpca_var = ref.fit(
        datas_list, pca=True, num_pcs=6, subtract_cc_mean=True,
        tstart=0, tend=T - 1, times=times, align_axes_to_data=True,
        soft_normalize=5,
    )
    _, omega_ref = J.jpc_plane_from_M(ref.M_skew)

    # ----- Layer A: identical (X, Xdot) -> hand closed form vs Antin CG -----
    X, Xdot = J.finite_diff(res.pre.reduced)
    M_hand = J.fit_skew_symmetric_M(X, Xdot)
    M_syl = J._fit_skew_symmetric_M_sylvester(X, Xdot)
    M_ref = skew_sym_regress(X, Xdot)
    a_hand_ref = np.linalg.norm(M_hand - M_ref) / np.linalg.norm(M_ref)
    a_hand_syl = np.linalg.norm(M_hand - M_syl) / np.linalg.norm(M_hand)
    plane_hand_A, _ = J.jpc_plane_from_M(M_hand)
    plane_ref_A, _ = J.jpc_plane_from_M(M_ref)
    ang_A = principal_angle_max(plane_hand_A, plane_ref_A)

    # ----- Layer B: full-pipeline plane agreement (same reduced space) -----
    ang_B = principal_angle_max(res.plane, ref.jpcs)
    freq_err_ref = abs(res.omega - omega_ref) / omega_ref

    # ----- ground truth on a clean single-plane rotation (sin-law) -----
    gt_err_vs_sin, gt_bias_vs_true = ground_truth_frequency()

    # ----- unconstrained ceiling for context (how much the skew constraint costs)
    M_unc = np.linalg.lstsq(X, Xdot, rcond=None)[0].T   # Xdot = X M^T
    r2_unc = 1.0 - np.sum((Xdot - X @ M_unc.T) ** 2) / np.sum(Xdot ** 2)

    print("=== Reported numbers (M0 pass/fail) ===")
    print(f"fit R^2 (Xdot = X M^T, skew)        : {res.fit_R2:.4f}")
    print(f"  unconstrained ceiling R^2          : {r2_unc:.4f}")
    print(f"top rotation-plane variance fraction: {res.plane_var_frac:.4f}")
    print(f"recovered omega (hand)  [rad/bin]   : {res.omega:.4f}")
    print(f"recovered omega (Antin) [rad/bin]   : {omega_ref:.4f}")
    print()
    print("--- Layer A: identical (X,Xdot), hand closed-form vs Antin CG ---")
    print(f"||M_hand - M_antin|| / ||M_antin||  : {a_hand_ref:.2e}")
    print(f"||M_hand - M_sylvester|| (2 closed) : {a_hand_syl:.2e}")
    print(f"principal angle(plane_hand,plane_ref): {ang_A:.4f} deg")
    print()
    print("--- Layer B: full pipeline vs Antin (< 5 deg, < 5%) ---")
    print(f"principal angle(plane) hand vs Antin : {ang_B:.4f} deg")
    print(f"freq match hand vs Antin             : {100*freq_err_ref:.3f} %")
    print()
    print("--- Ground truth: clean single-plane rotation (sin-law) ---")
    print(f"max |omega_rec - sin(omega)| / sin   : {100*gt_err_vs_sin:.4f} %  (target ~0)")
    print(f"raw forward-diff bias vs true omega  : {100*gt_bias_vs_true:.3f} %  (expected ~omega^2/6)")
    print()

    checks = {
        "LayerA: hand ~ Antin M (<1e-3)": a_hand_ref < 1e-3,
        "LayerA: two closed forms agree (<1e-8)": a_hand_syl < 1e-8,
        "LayerA: plane angle < 5 deg": ang_A < 5.0,
        "LayerB: plane angle < 5 deg": ang_B < 5.0,
        "LayerB: freq within 5%": freq_err_ref < 0.05,
        "Truth: omega_rec == sin(omega) (<0.5%)": gt_err_vs_sin < 0.005,
    }
    print("=== PASS/FAIL ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nM0 differential test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
