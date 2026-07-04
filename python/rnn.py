"""rnn.py — task-trained RNNs (the systems whose MECHANISM we can inspect).

Two networks, both trained on a TASK (never on neural spikes):
  - Flip-flop RNN (M1): small vanilla tanh RNN, 3-bit flip-flop memory task
    (Sussillo & Barak 2013). Known-answer instrument calibration for the
    fixed-point finder and the viewer.
  - Reaching RNN (M3): 256-unit continuous-time tanh RNN,
        dx/dt = -x + W_rec phi(x) + W_in u + b,   z = W_out x
    inputs = target/condition cue + go signal, output = hand velocity (or EMG).
    Metabolic regularization (L2 on firing rates) is MANDATORY — without it the
    network finds a non-biological high-dimensional solution (README M3).

KEY CONCEPT (see NOTES M3): we train on the BEHAVIORAL TASK and then ask whether
the EMERGENT hidden dynamics resemble M1. Conflating this with "fit the RNN to
spikes" is the classic fatal misunderstanding.

Planned public surface (M1, M3):
    FlipFlopRNN, train_flipflop(...)
    ReachingRNN,  train_reaching(...)   # returns model + velocity R^2 (task gate)
"""

# TODO(M1): flip-flop RNN + trainer.  TODO(M3): reaching RNN + trainer.
