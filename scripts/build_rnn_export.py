"""Build data/rnn.json for the web viewer: trajectories + fixed point + flow field.

Everything is projected through ONE shared hidden-space basis (jPC1, jPC2, top
orthogonal dir), so the flow field, the fixed point, and the trajectories overlay
correctly. The rotation plane is computed on RAW hidden states (no soft-norm/CCM)
so it stays a clean linear map on the same space the RNN velocity lives in.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from python import rnn, fixedpoints as fp, pca_jpca as J, export as E  # noqa: E402

WIN = slice(15, 45)


def main():
    model = rnn.ReachingRNN(n_hid=256)
    model.load_state_dict(torch.load(REPO / "cache" / "reaching_metab.pt", map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    uc = torch.zeros(model.n_in)
    F = lambda x: model.velocity(x, uc)  # noqa: E731

    # condition-averaged raw hidden states, movement window
    with torch.no_grad():
        u_np, _, _ = rnn.generate_reaching(16, reps=1, T=60, go=20, noise=0.0, seed=0)
        Xc = model(torch.from_numpy(u_np))[0].numpy()[:, WIN, :]     # (16, T, 256)

    # rotation plane on RAW hidden states (linear map -> consistent projection)
    jr = J.jpca(Xc, num_pcs=6, soft_norm_const=0.0, subtract_ccm=False)
    origin = jr.pre.pca.mean_.astype(float)
    plane = jr.pre.pca.components_.T @ jr.plane                       # (256, 2)
    plane, _ = np.linalg.qr(plane)
    plane = plane[:, :2]

    # standard-preprocessing jPCA (for the reported rotation numbers)
    jr_std = J.jpca(Xc, num_pcs=6, subtract_ccm=True)

    # fixed point(s)
    vis = Xc.reshape(-1, 256)
    ic = fp.sample_initial_conditions(vis, n_ic=800, noise=0.3, seed=1)
    r = fp.find_fixed_points(F, ic, adam_steps=4000, lr=0.02, lbfgs_steps=40)
    uniq = fp.dedup_cluster(r.points[r.speeds < fp.speed_tolerance(r.speeds)], tol=1.0)
    fixed_points = []
    lead = None
    for pt in uniq:
        st = fp.classify(fp.jacobian(F, pt), real_tol=1e-2)
        order = np.argsort(-st.eigenvalues.real)[:8]                  # top 8 eigenvalues
        eigs = [[round(float(st.eigenvalues[i].real), 3),
                 round(float(st.eigenvalues[i].imag), 3)] for i in order]
        fixed_points.append({"x": pt, "eigs": eigs, "stability": st.label,
                             "is_spiral": st.is_spiral})
        lead = st.lead_eig

    # flow field: grid in the plane, sized to the trajectory spread so it covers
    # the rotation region -> embed to hidden -> raw dx/dt
    proj = (Xc.reshape(-1, 256) - origin) @ plane
    ext = float(1.15 * np.abs(proj).max())
    n = 17
    gs = np.linspace(-ext, ext, n)
    A, B = np.meshgrid(gs, gs)
    ab = np.stack([A.ravel(), B.ravel()], 1)
    pos_hidden = origin[None, :] + ab @ plane.T
    with torch.no_grad():
        vec_hidden = F(torch.tensor(pos_hidden, dtype=torch.float32)).numpy()
    flow = {"pos": pos_hidden, "vec": vec_hidden}

    meta = {
        "fit_R2": round(float(jr_std.fit_R2), 4),
        "plane_var_frac": round(float(jr_std.plane_var_frac), 4),
        "n_fixed_points": len(fixed_points),
        "lead_eig": [round(lead.real, 3), round(lead.imag, 3)] if lead else None,
        "source": "task-trained reaching RNN (256-unit continuous-time tanh)",
        "note": "RNN lives in its OWN jPCA space; matched to brain by visual scale only.",
    }
    payload = E.export_rnn_hidden(str(REPO / "data" / "rnn.json"),
                                  Xc, plane, origin, fixed_points, flow, meta)
    print(f"wrote data/rnn.json: {len(payload['trajectories'])} trajectories, "
          f"{len(payload['fixed_points'])} fixed points, {len(payload['flow_field'])} flow arrows")
    print(f"meta: {payload['meta']}")


if __name__ == "__main__":
    main()
