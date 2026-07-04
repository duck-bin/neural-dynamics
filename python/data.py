"""data.py — MC_Maze loading and preprocessing (Stage 1 data path).

Loads MC_Maze (Neural Latents Benchmark '21, Pei et al. 2021) via nlb_tools,
aligns on movement onset (prep + movement epochs), bins spikes at 20 ms,
Gaussian-smooths (sigma = 40 ms), and soft-normalizes firing rates
Churchland-style: rate / (range + 5).

WHY a separate module: keep data I/O + preprocessing out of notebook cells so
the pipeline is importable and the preprocessing is auditable in one place
(the cross-condition-mean step is contested — see NOTES "M2 / adversarial").

Planned public surface (implemented in M2):
    load_mc_maze(cache_dir)              -> raw bundle (cached on Drive)
    condition_average(bundle, ...)       -> X_c(t): (conditions, time, neurons)
    soft_normalize(rates, floor=5.0)     -> normalized rates
"""

# TODO(M2): implement MC_Maze loading + preprocessing. See README "Data".
