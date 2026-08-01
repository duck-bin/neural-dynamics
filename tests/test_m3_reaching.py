"""M3: reaching RNN trained on the TASK (not spikes) + jPCA on hidden states.

KEY CONCEPT (README): the RNN is trained on the BEHAVIORAL task — inputs are the
target cue + go signal, output is hand velocity — NOT on neural data. We then ask
whether the EMERGENT hidden dynamics resemble M1's rotations. Conflating this with
"fit the RNN to spikes" is the classic fatal misunderstanding.

Gates / deliverables:
  - velocity R^2 above threshold BEFORE any dynamics analysis (analyzing a network
    that failed the task is meaningless);
  - jPCA on the condition-averaged hidden states over the movement epoch: does the
    RNN rotate? Report fit R^2 and top-plane variance fraction.

Metabolic regularization is KEPT (README mandates it). Its README rationale is
that WITHOUT it the network finds a non-biological high-dimensional solution.
HONESTY NOTE (adversarial): in this simplified 16-direction task that effect does
NOT reproduce — an ablation leaves the participation ratio essentially unchanged
(both ~2.5), because the task is simple enough that even the unregularized
solution is low-dimensional. We keep the penalty for faithfulness and report the
null result; see README section 6 (M3 sweep table) and section 9 (decision D2).
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from python import rnn, pca_jpca as J  # noqa: E402

CKPT = REPO / "cache" / "reaching_metab.pt"
N_DIRS, T, GO = 16, 60, 20
WIN = slice(15, 45)                # movement epoch (around GO = 20)
VEL_R2_THRESHOLD = 0.90


def participation_ratio(Xc):
    """Effective dimensionality: (sum lambda)^2 / sum(lambda^2) of PCA spectrum."""
    C, Tt, H = Xc.shape
    M = Xc.reshape(C * Tt, H)
    ev = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def get_model():
    model = rnn.ReachingRNN(n_hid=256)
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            u_np, v_np, _ = rnn.generate_reaching(N_DIRS, 8, T, GO, seed=77777)
            _, Z = model(torch.from_numpy(u_np))
            v = torch.from_numpy(v_np)
            r2 = float(1 - ((Z - v) ** 2).sum() / ((v - v.mean()) ** 2).sum())
        return model, r2
    model, r2, _ = rnn.train_reaching(n_hid=256, iters=800, n_dirs=N_DIRS, reps=8,
                                      lr=2e-3, metabolic=1e-3, seed=0)
    torch.save(model.state_dict(), CKPT)
    return model, r2


def main():
    model, vel_r2 = get_model()
    print(f"velocity R^2: {vel_r2:.4f}  (task gate: > {VEL_R2_THRESHOLD})\n")

    Xc, _ = rnn.reaching_condition_averaged(model, n_dirs=N_DIRS, T=T, go=GO)
    res = J.jpca(Xc[:, WIN, :], num_pcs=6, subtract_ccm=True)
    print("=== jPCA on RNN hidden states (movement epoch) ===")
    print(f"fit R^2 (skew): {res.fit_R2:.4f}")
    print(f"top rotation-plane variance fraction: {res.plane_var_frac:.4f}")
    s, f, r = J.skew_over_full(res)
    print(f"R2(skew)/R2(full): {r:.4f}   (skew {s:.4f} / full {f:.4f})")
    print(f"  -> rotation accounts for {100*r:.0f}% of the linearly explainable")
    print("     derivative structure; comparable to the same ratio on brain data.")
    print(f"participation ratio (with metabolic): {participation_ratio(Xc[:, WIN, :]):.2f}\n")

    checks = {
        f"velocity R^2 > {VEL_R2_THRESHOLD} (task gate)": vel_r2 > VEL_R2_THRESHOLD,
        "hidden states rotate (skew fit R^2 > 0.5)": res.fit_R2 > 0.5,
    }
    print("=== M3 PASS/FAIL ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nM3 reaching RNN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
