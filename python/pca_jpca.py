"""pca_jpca.py — PCA + hand-implemented jPCA (the DESCRIPTIVE layer).

jPCA fits ONE global linear system to trajectory geometry, constrained to be
skew-symmetric so its dynamics are pure rotation: Xdot = M X, M = -M^T. It needs
no equations of the underlying system, only observed trajectories + their finite-
difference derivatives, which is why it applies to BOTH brain data and RNN states.

Hand implementation is the PRIMARY path; Benjamin Antin's `jPCA` package is the
differential-test ORACLE (not a fallback). Both stay in the code.

Planned public surface (implemented in M0):
    soft_normalize(...)                  -> per-neuron soft-normalized rates
    preprocess(X_c, subtract_ccm=True)   -> stacked, mean-subtracted matrix
    pca(X, k=6)                          -> (scores, components, var_explained)
    fit_skew_symmetric_M(X, Xdot)        -> M via constrained least squares
                                            (closed form; Churchland 2012 supp.)
    jpca(X_c, k=6)                        -> planes, freqs, projections, fit R^2
    principal_angle(plane_a, plane_b)    -> differential-test metric (degrees)

Algorithm steps are a FIXED commitment (README M0, steps 1-7). Do not silently
alter the ordering, the cross-condition-mean subtraction, or the skew-symmetric
constraint; propose changes in NOTES "Proposed deviations".
"""

# TODO(M0): implement jPCA by hand + Antin differential test. See README M0.
