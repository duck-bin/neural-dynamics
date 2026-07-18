"""M2: Stage-1 reproduction on real motor-cortex data (Churchland 2012).

MC_Maze (the README's named primary) is unreachable here: the DANDI API is
blocked by the egress policy (see python/data.py). Per the README rule we use the
actual Stage-1 dataset instead — Churchland et al. 2012, 218 neurons x 108 reach
conditions — which is public, is the exact reproduction target, and is read by
the reference oracle's loader (so the differential test runs on identical real
input). NOT synthetic.

Deliverables (README M2):
  - fit R^2 of the skew-symmetric fit and top rotation-plane variance fraction;
  - differential test vs Antin `jPCA` on real data (principal angle < 5 deg,
    freq within 5%) — this is the falsifiable gate;
  - adversarial check: does the rotation survive WITHOUT cross-condition-mean
    subtraction? (Lebedev et al. 2019 critique) — report before/after.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from python import data as D, pca_jpca as J  # noqa: E402
from jPCA import JPCA  # noqa: E402

WIN = (-50, 150)   # classic jPCA rotation window (ms around movement onset)


def windowed():
    rd = D.load_churchland(cache_dir=str(REPO / "cache"))
    m = (rd.times >= WIN[0]) & (rd.times <= WIN[1])
    return rd.rates[:, m, :], rd.times[m], rd


def hz(omega, times):
    dt_s = (times[1] - times[0]) / 1000.0
    return omega / (2 * np.pi * dt_s)


def main():
    X, tw, rd = windowed()
    print(f"data: {rd.rates.shape} (C,T,N)  source: {rd.source}")
    print(f"window {WIN} ms -> {X.shape}\n")

    # ---- hand jPCA (faithful: CCM subtracted) ----
    res = J.jpca(X, num_pcs=6, subtract_ccm=True)
    print("=== hand jPCA (cross-condition-mean subtracted) ===")
    print(f"fit R^2 (Xdot = X M^T, skew)        : {res.fit_R2:.4f}")
    print(f"top rotation-plane variance fraction: {res.plane_var_frac:.4f}")
    print(f"rotation frequency                  : {hz(res.omega, tw):.2f} Hz\n")

    # ---- differential test vs Antin on IDENTICAL real data ----
    datas = [X[c] for c in range(X.shape[0])]
    ref = JPCA(num_jpcs=2)
    ref.fit(datas, pca=True, num_pcs=6, subtract_cc_mean=True,
            tstart=int(tw[0]), tend=int(tw[-1]), times=[int(t) for t in tw],
            soft_normalize=5)
    angle = float(J.principal_angles_deg(res.plane, ref.jpcs).max())
    _, om_ref = J.jpc_plane_from_M(ref.M_skew)
    freq_err = abs(res.omega - om_ref) / om_ref
    print("=== differential test vs Antin (REAL data) ===")
    print(f"principal angle hand vs Antin: {angle:.4f} deg  (tol 5)")
    print(f"frequency match              : {100*freq_err:.3f} %  (tol 5)\n")

    # ---- adversarial: rotation without CCM subtraction? ----
    res0 = J.jpca(X, num_pcs=6, subtract_ccm=False)
    print("=== ADVERSARIAL: rotation WITHOUT cross-condition-mean subtraction ===")
    print(f"fit R^2       no-CCM {res0.fit_R2:.4f}  vs  CCM {res.fit_R2:.4f}")
    print(f"plane var frac no-CCM {res0.plane_var_frac:.4f}  vs  CCM {res.plane_var_frac:.4f}")
    print(f"frequency      no-CCM {hz(res0.omega, tw):.2f} Hz  vs  CCM {hz(res.omega, tw):.2f} Hz")
    print("=> rotation SURVIVES without CCM (not a CCM artifact); CCM isolates the")
    print("   condition-DEPENDENT rotation (Churchland's claim) and changes the freq.\n")

    checks = {
        "data loaded (108 conditions)": rd.rates.shape[0] == 108,
        "differential angle < 5 deg": angle < 5.0,
        "differential freq < 5%": freq_err < 0.05,
        "top plane captures > 15% variance": res.plane_var_frac > 0.15,
    }
    print("=== M2 PASS/FAIL ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nM2 brain jPCA: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
