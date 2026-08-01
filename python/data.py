"""data.py — real motor-cortex reaching data for Stage 1 (the DESCRIPTIVE layer).

Primary intent per the README was MC_Maze (NLB '21) via nlb_tools/DANDI. In this
execution environment the DANDI API (api.dandiarchive.org) is blocked by the
egress policy (403 on CONNECT), so nlb_tools cannot fetch MC_Maze. Per the README
rule ("if MC_Maze access fails, report it and use an alternative PUBLIC reaching
dataset; do NOT substitute synthetic data") we fall back to the **Churchland et
al. 2012** dataset itself — the exact Stage-1 reproduction target (Nature 487:51),
218 neurons x 108 maze/reach conditions of condition-averaged, smoothed firing
rates. It is arguably MORE faithful for "reproduce Churchland 2012" than MC_Maze,
and it is the data Benjamin Antin's reference loader consumes, so the M0
differential test runs on identical real input. See README section 9 (limitations
and open decisions, D1).

The rates are already trial-averaged and smoothed (as published), aligned around
movement onset (times -50..550 ms, 10 ms bins spanning prep + movement). The
jPCA-specific preprocessing (soft-normalize, cross-condition-mean subtraction,
PCA) is applied downstream in pca_jpca.py, exactly as for any jPCA input.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

# raw.githubusercontent.com is reachable through the egress proxy (DANDI/S3 and
# HuggingFace are not). This mirrors Churchland's jPCA `exampleData.mat`.
CHURCHLAND_URL = (
    "https://raw.githubusercontent.com/nwb4edu/nwb4edu.github.io/"
    "master/Lesson_5/exampleData.mat"
)


@dataclass
class ReachData:
    rates: np.ndarray            # (C, T, N) condition-averaged firing rates
    times: np.ndarray            # (T,) ms relative to movement onset
    conditions: list[str]        # length-C condition labels
    source: str                  # provenance string


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_churchland(cache_dir="cache", filename="churchland_exampleData.mat") -> ReachData:
    """Load the Churchland 2012 reaching data as (C, T, N) firing rates.

    Downloads to cache on first use (Colab: point cache_dir at Drive). Parses the
    published `Data` struct: each entry is one condition with an (T x N) rate
    matrix `A` and a per-bin `times` vector.
    """
    path = Path(cache_dir) / filename
    if not path.exists():
        _download(CHURCHLAND_URL, path)

    mat = sio.loadmat(str(path))
    conditions = mat["Data"][0]                       # struct array, one per condition
    times = np.array([float(t[0]) for t in conditions[0][1]])

    rates = np.stack([np.asarray(cond[0], dtype=float) for cond in conditions], axis=0)
    labels = []
    for cond in conditions:
        # Condition label lives in a later struct field when present; fall back to index.
        lab = ""
        for fld in cond:
            if isinstance(fld, np.ndarray) and fld.dtype.kind in "US" and fld.size:
                lab = str(fld.ravel()[0])
                break
        labels.append(lab)

    return ReachData(rates=rates, times=times,
                     conditions=labels or [str(i) for i in range(len(conditions))],
                     source="Churchland et al. 2012 (Nature 487:51) exampleData.mat")


def load_mc_maze(cache_dir="cache"):
    """MC_Maze via nlb_tools/DANDI — BLOCKED in this environment.

    Kept so the intended primary path is documented and trivially re-enabled if
    the egress policy changes. api.dandiarchive.org returns 403 (policy denial),
    so we raise with a clear pointer to the Churchland fallback rather than fail
    opaquely inside nlb_tools.
    """
    raise RuntimeError(
        "MC_Maze requires the DANDI API (api.dandiarchive.org), which is blocked "
        "by the egress policy in this environment (403 on CONNECT). Use "
        "load_churchland() — the actual Stage-1 (Churchland 2012) dataset. See "
        "README section 9 (limitations / open decision D1)."
    )
