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
