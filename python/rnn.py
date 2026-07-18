"""rnn.py — task-trained RNNs (the systems whose MECHANISM we can inspect).

M1 (this file, first half): a small vanilla tanh RNN trained on the 3-bit
flip-flop memory task (Sussillo & Barak 2013). This is the KNOWN-ANSWER
instrument that calibrates the fixed-point finder: a correct network must hold
2^3 = 8 memory states, which appear as 8 stable fixed points at the corners of a
cube in readout space.

Discrete vanilla update (matches Sussillo & Barak and the reference oracle
tripdancer0916/pytorch-fixed-point-analysis):
    h_{t+1} = tanh(W_in u_t + W_hh h_t + b),   z_t = W_out h_t
The autonomous velocity used for fixed-point analysis is the map residual
    F(h) = tanh(W_in u + W_hh h + b) - h            (see fixedpoints.py)

`FlipFlopRNN` deliberately exposes `.w_in/.w_hh/.w_out/.activation/.n_hid` with
the SAME names as the reference model, so the reference `FixedPoint` finder can
run on our trained network unchanged — that is what makes the M1 differential
test a true apples-to-apples comparison.

M3 (added later): a 256-unit continuous-time reaching RNN, trained on the TASK
(target + go -> hand velocity), NOT on spikes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 3-bit flip-flop task
# --------------------------------------------------------------------------
def generate_flipflop(
    batch: int,
    length: int = 300,
    n_bits: int = 3,
    p_pulse: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 3-bit flip-flop trials.

    Each of `n_bits` channels independently receives sparse +/-1 pulses (prob
    `p_pulse` per step). The target for a channel is the sign of the MOST RECENT
    pulse on that channel, held until the next pulse — i.e. a 1-bit memory. With
    3 bits the network must maintain one of 2^3 = 8 states.

    Memory starts at 0 and the target is 0 until the first pulse on a channel:
    before any input the bit is genuinely UNKNOWABLE, so asking the network to
    guess it just injects irreducible error (this inflated MSE badly until fixed).

    Returns inputs (batch, length, n_bits) and targets (batch, length, n_bits).
    """
    rng = np.random.default_rng(seed)
    inputs = np.zeros((batch, length, n_bits), dtype=np.float32)
    targets = np.zeros((batch, length, n_bits), dtype=np.float32)

    for b in range(batch):
        for bit in range(n_bits):
            state = 0.0                              # no memory until first pulse
            for t in range(length):
                if rng.random() < p_pulse:
                    state = float(rng.choice([-1.0, 1.0]))  # a pulse sets the bit
                    inputs[b, t, bit] = state
                targets[b, t, bit] = state
    return inputs, targets


# --------------------------------------------------------------------------
# Vanilla tanh RNN (API-compatible with the reference oracle model)
# --------------------------------------------------------------------------
class FlipFlopRNN(nn.Module):
    def __init__(self, n_in=3, n_hid=100, n_out=3, activation="tanh"):
        super().__init__()
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.activation = activation
        # Bias on both linears mirrors the reference model exactly.
        self.w_in = nn.Linear(n_in, n_hid, bias=True)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=True)
        self.w_out = nn.Linear(n_hid, n_out, bias=True)
        self._init_recurrent()

    def _init_recurrent(self):
        # Orthogonal recurrent init with gain slightly ABOVE 1: a memory task
        # needs the recurrent map to PRESERVE state across long gaps between
        # pulses. Gain < 1 contracts and forgets (observed: MSE stuck ~0.18);
        # gain ~1.3 retains information while staying trainable.
        nn.init.orthogonal_(self.w_hh.weight, gain=1.3)
        nn.init.zeros_(self.w_hh.bias)

    def _phi(self, x):
        return torch.tanh(x) if self.activation == "tanh" else torch.relu(x)

    def step(self, u_t, h):
        """One recurrent step: h_{t+1} = phi(W_in u + W_hh h + b)."""
        return self._phi(self.w_in(u_t) + self.w_hh(h))

    def forward(self, inputs, hidden=None):
        """inputs: (batch, length, n_in). Returns (hidden_list, output_list, h).

        Signature/returns match the reference model so downstream code (and the
        reference fixed-point finder's expectations) line up.
        """
        B, L, _ = inputs.shape
        if hidden is None:
            hidden = inputs.new_zeros(B, self.n_hid)
        hidden_list = inputs.new_zeros(L, B, self.n_hid)
        output_list = inputs.new_zeros(L, B, self.n_out)
        u = inputs.permute(1, 0, 2)                  # (L, B, n_in)
        for t in range(L):
            hidden = self.step(u[t], hidden)
            hidden_list[t] = hidden
            output_list[t] = self.w_out(hidden)
        return hidden_list.permute(1, 0, 2), output_list.permute(1, 0, 2), hidden


def train_flipflop(
    n_hid=100,
    iters=2000,
    batch=64,
    length=300,
    lr=1e-3,
    seed=0,
    device="cpu",
    log_every=200,
):
    """Train FlipFlopRNN on the 3-bit flip-flop. Returns (model, final_mse, log).

    A modest metabolic penalty (L2 on firing rates) is included even here: it
    keeps hidden magnitudes bounded so the 8 memory states sit at clean, well-
    separated corners rather than drifting out along saturating tanh tails.
    """
    torch.manual_seed(seed)
    model = FlipFlopRNN(n_hid=n_hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    mse = nn.MSELoss()
    log = []
    for it in range(iters):
        x_np, y_np = generate_flipflop(batch, length, seed=seed + it + 1)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        hidden_list, out, _ = model(x)
        # Tiny metabolic term only (keeps states bounded). The MANDATORY metabolic
        # regularization is an M3/reaching-RNN requirement; a large penalty here
        # would smear the memory attractors we need for the 8-corner gate.
        loss = mse(out, y) + 1e-5 * hidden_list.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            log.append((it, float(loss.detach())))
    # Report task MSE (without the metabolic term) on a fresh batch.
    with torch.no_grad():
        x_np, y_np = generate_flipflop(batch, length, seed=99999)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, out, _ = model(x)
        final_mse = float(mse(out, y))
    return model, final_mse, log


def collect_hidden_states(model, n_trials=32, length=300, seed=123, device="cpu"):
    """Run trials and return visited hidden states, shape (n_trials*length, n_hid).

    These seed the fixed-point search (README M4 rule: sample ICs from states the
    network actually visits, not uniformly).
    """
    with torch.no_grad():
        x_np, _ = generate_flipflop(n_trials, length, seed=seed)
        x = torch.from_numpy(x_np).to(device)
        hidden_list, _, _ = model(x)
    return hidden_list.reshape(-1, model.n_hid).cpu().numpy()


# ==========================================================================
# M3/M4: center-out reaching task + 256-unit continuous-time RNN
# ==========================================================================
def generate_reaching(n_dirs=8, reps=8, T=60, go=20, dt=1.0, tau=10.0,
                      peak=14, width=6.0, noise=0.02, seed=None):
    """Delayed center-out reach: target cue held throughout, GO gates movement.

    Inputs u (B, T, 3) = [cos(theta), sin(theta), go(t)]; go steps 0->1 at t=go.
    Target output vel (B, T, 2) = a bell-shaped speed bump in direction theta,
    ZERO before GO (the delay/prep period). The prep->movement structure is what
    makes the emergent dynamics rotational (Churchland/Sussillo).

    Returns u, vel, dir_idx (condition = reach direction).
    """
    rng = np.random.default_rng(seed)
    dirs = np.arange(n_dirs)
    thetas = 2 * np.pi * dirs / n_dirs
    B = n_dirs * reps
    t = np.arange(T)
    bump = np.where(t >= go, np.exp(-((t - go - peak) ** 2) / (2 * width ** 2)), 0.0)

    u = np.zeros((B, T, 3), dtype=np.float32)
    vel = np.zeros((B, T, 2), dtype=np.float32)
    dir_idx = np.zeros(B, dtype=int)
    b = 0
    for _ in range(reps):
        for k in range(n_dirs):
            th = thetas[k]
            u[b, :, 0] = np.cos(th)
            u[b, :, 1] = np.sin(th)
            u[b, t >= go, 2] = 1.0
            vel[b, :, 0] = bump * np.cos(th)
            vel[b, :, 1] = bump * np.sin(th)
            dir_idx[b] = k
            b += 1
    u += rng.normal(scale=noise, size=u.shape).astype(np.float32)
    return u, vel, dir_idx


class ReachingRNN(nn.Module):
    """256-unit continuous-time tanh RNN (Euler-integrated).

        dx/dt = -x + W_rec phi(x) + W_in u + b ,   z = W_out x ,   phi = tanh
        x_{t+1} = x_t + (dt/tau) dx/dt

    We expose `velocity(x, u)` = dx/dt for the M4 fixed-point finder, and rates()
    = phi(x) for the metabolic penalty.
    """

    def __init__(self, n_in=3, n_hid=256, n_out=2, dt=1.0, tau=10.0):
        super().__init__()
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.alpha = dt / tau
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_rec = nn.Linear(n_hid, n_hid, bias=True)   # bias = b
        self.w_out = nn.Linear(n_hid, n_out, bias=False)
        nn.init.normal_(self.w_rec.weight, std=1.0 / np.sqrt(n_hid))  # ~unit spectral radius
        nn.init.zeros_(self.w_rec.bias)

    def velocity(self, x, u_t):
        """dx/dt = -x + W_rec phi(x) + W_in u + b (continuous field; used at M4)."""
        return -x + self.w_rec(torch.tanh(x)) + self.w_in(u_t)

    def forward(self, u):
        """u: (B, T, n_in) -> hidden X (B, T, n_hid), output z (B, T, n_out)."""
        B, T, _ = u.shape
        x = u.new_zeros(B, self.n_hid)
        X = u.new_zeros(B, T, self.n_hid)
        Z = u.new_zeros(B, T, self.n_out)
        up = u.permute(1, 0, 2)
        for t in range(T):
            x = x + self.alpha * self.velocity(x, up[t])
            X[:, t, :] = x
            Z[:, t, :] = self.w_out(x)
        return X, Z


def train_reaching(n_hid=256, iters=1500, n_dirs=8, reps=8, T=60, go=20,
                   lr=2e-3, metabolic=1e-3, seed=0, device="cpu", log_every=250):
    """Train ReachingRNN on the reach task. Returns (model, vel_R2, log).

    Loss = MSE(velocity) + metabolic * mean(rate^2). The metabolic term is
    MANDATORY: without it the network finds a non-biological high-dimensional
    solution (README M3); with it, rates stay bounded and the dynamics are
    low-dimensional and rotational.
    """
    torch.manual_seed(seed)
    model = ReachingRNN(n_hid=n_hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    log = []
    for it in range(iters):
        u_np, v_np, _ = generate_reaching(n_dirs, reps, T, go, seed=seed + it + 1)
        u = torch.from_numpy(u_np).to(device)
        v = torch.from_numpy(v_np).to(device)
        X, Z = model(u)
        task = ((Z - v) ** 2).mean()
        metab = metabolic * torch.tanh(X).pow(2).mean()
        loss = task + metab
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            log.append((it, float(task.detach()), float(metab.detach())))

    with torch.no_grad():
        u_np, v_np, _ = generate_reaching(n_dirs, reps, T, go, seed=77777)
        u = torch.from_numpy(u_np).to(device); v = torch.from_numpy(v_np).to(device)
        _, Z = model(u)
        ss_res = ((Z - v) ** 2).sum()
        ss_tot = ((v - v.mean()) ** 2).sum()
        vel_r2 = float(1 - ss_res / ss_tot)
    return model, vel_r2, log


def reaching_condition_averaged(model, n_dirs=8, T=60, go=20, device="cpu"):
    """Condition-averaged hidden states (n_dirs, T, n_hid) for jPCA on the RNN."""
    with torch.no_grad():
        u_np, _, didx = generate_reaching(n_dirs, reps=1, T=T, go=go, noise=0.0, seed=0)
        X, _ = model(torch.from_numpy(u_np).to(device))
    return X.cpu().numpy(), u_np       # already one trial per direction (reps=1)
