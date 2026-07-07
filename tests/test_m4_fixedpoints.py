"""M4: reaching RNN fixed points + flow field (the mechanism behind the rotation).

We find fixed/slow points of the RNN's AUTONOMOUS velocity field
    F(x) = -x + W_rec tanh(x)        (input held at 0)
minimizing q(x) = 1/2||F||^2 from ICs sampled on real movement-epoch states,
classify each by its Jacobian, and sample the flow field in the jPCA plane.

Scientific payoff: the reaching RNN is organized by a SINGLE dominant fixed point
— an unstable spiral / rotational saddle whose leading eigenpair is complex
(e.g. +0.5 +/- 1.3i). That complex pair IS the local rotation that jPCA
describes at M2/M3. Contrast M1, whose flip-flop had 8 point attractors: different
computation, different fixed-point topology.

Differential test note: the M1 reference oracle (pytorch-fixed-point-analysis) is a
DISCRETE-map finder and cannot analyze this continuous-time RNN. So M4's "does the
port agree?" is answered by an INDEPENDENT method on the same field — SciPy's
Newton/hybrid root finder (scipy.optimize.root) — plus IC-resampling stability.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from python import rnn, fixedpoints as fp, flowfield as ff, pca_jpca as J  # noqa: E402

CKPT = REPO / "cache" / "reaching_metab.pt"


def load_model():
    model = rnn.ReachingRNN(n_hid=256)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def autonomous_velocity(model):
    uc = torch.zeros(model.n_in)
    return lambda x: model.velocity(x, uc)


def find_fps(model, F, seed=1, n_ic=800):
    with torch.no_grad():
        u_np, _, _ = rnn.generate_reaching(16, reps=6, T=60, go=20, seed=3)
        X, _ = model(torch.from_numpy(u_np))
    visited = X[:, 22:45, :].reshape(-1, model.n_hid).numpy()   # movement epoch
    ic = fp.sample_initial_conditions(visited, n_ic=n_ic, noise=0.3, seed=seed)
    res = fp.find_fixed_points(F, ic, adam_steps=4000, lr=0.02, lbfgs_steps=40)
    tol = fp.speed_tolerance(res.speeds)
    uniq = fp.dedup_cluster(res.points[res.speeds < tol], tol=1.0)
    return uniq, res, visited


def newton_confirm(model, F, x0):
    """Independent oracle: SciPy Newton/hybrid root of F(x)=0 from a perturbed start."""
    from scipy.optimize import root

    def fun(xnp):
        with torch.no_grad():
            return F(torch.tensor(xnp[None, :], dtype=torch.float32)).numpy()[0]

    sol = root(fun, x0, method="hybr", tol=1e-10)
    return sol.x, float(np.linalg.norm(fun(sol.x)))


def main():
    model = load_model()
    F = autonomous_velocity(model)
    uniq, res, visited = find_fps(model, F, seed=1)

    # field scale for context: how slow are the minima vs a typical state?
    with torch.no_grad():
        raw = torch.norm(F(torch.tensor(visited[:500], dtype=torch.float32)), dim=1).numpy()
    print(f"raw field ||F|| median over visited states: {np.median(raw):.2f}")
    print(f"unique fixed/slow points: {len(uniq)}\n")

    rows = []
    for pt in uniq:
        st = fp.classify(fp.jacobian(F, pt), real_tol=1e-2)
        speed = float(np.linalg.norm(F(torch.tensor(pt[None], dtype=torch.float32)).detach().numpy()))
        rows.append((pt, st, speed))
        print(f"|x|={np.linalg.norm(pt):5.2f}  {st.label:10s} "
              f"spiral={st.is_spiral}  n_unstable={st.n_unstable}  "
              f"lead={st.lead_eig.real:+.2f}{st.lead_eig.imag:+.2f}j  q_speed={speed:.1e}")
    print()

    # ---- Adversarial 1: IC re-sampling stability ----
    uniq2, _, _ = find_fps(model, F, seed=777)
    if len(uniq) and len(uniq2):
        D = np.linalg.norm(uniq[:, None, :] - uniq2[None, :, :], axis=2)
        resample_maxmatch = float(D.min(axis=1).max())
    else:
        resample_maxmatch = np.inf
    print(f"[adversarial] IC re-sampling: each seed-1 point re-found within "
          f"{resample_maxmatch:.3f} by seed-777\n")

    # ---- Adversarial 2: independent Newton confirmation (the 'port') ----
    pt0 = uniq[0]
    xnew, resid = newton_confirm(model, F, pt0 + np.random.default_rng(0).normal(scale=0.5, size=pt0.shape))
    newton_dist = float(np.linalg.norm(xnew - pt0))
    print(f"[adversarial] SciPy Newton from a perturbed start: residual ||F||={resid:.1e}, "
          f"distance to our point {newton_dist:.3f}\n")

    # ---- Flow field in the RNN jPCA plane ----
    with torch.no_grad():
        u_np, _, _ = rnn.generate_reaching(16, reps=1, T=60, go=20, noise=0.0, seed=0)
        Xc, _ = model(torch.from_numpy(u_np))
    jr = J.jpca(Xc.numpy()[:, 15:45, :], num_pcs=6, subtract_ccm=True)
    origin, plane = ff.jpca_plane_in_hidden(jr)
    field = ff.flow_on_grid(F, origin, plane, extent=3.0, n=15)
    speeds = np.linalg.norm(field.vectors, axis=1)
    # rotational signature: mean tangential (curl-like) component about the center
    r = field.coords
    tang = (r[:, 0] * field.vectors[:, 1] - r[:, 1] * field.vectors[:, 0])
    rot_frac = float(np.mean(tang) / (np.mean(np.linalg.norm(r, axis=1) * speeds) + 1e-9))
    print(f"flow field: {len(field.coords)} grid points, mean |dx/dt|_plane={speeds.mean():.2f}, "
          f"net rotational component (curl sign) = {rot_frac:+.2f}\n")

    checks = {
        "found >= 1 fixed point": len(uniq) >= 1,
        "leading eigenpair complex (rotation generator)": any(s.is_spiral for _, s, _ in rows),
        "IC re-sampling stable (< 0.5)": resample_maxmatch < 0.5,
        "Newton confirms our point (< 0.2)": newton_dist < 0.2 and resid < 1e-6,
        "flow field is rotational (|curl| > 0.1)": abs(rot_frac) > 0.1,
    }
    print("=== M4 PASS/FAIL ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nM4 reaching RNN fixed points: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
