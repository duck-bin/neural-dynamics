"""neural-dynamics: importable pipeline modules (the 'kitchen' logic).

Keep algorithm logic here, not in notebook cells, so the pipeline stays
importable, testable, and reproducible across Colab sessions.

Modules:
    data.py        MC_Maze loading + preprocessing            (M2)
    pca_jpca.py    PCA + hand-implemented jPCA (+ Antin ref)  (M0, M2, M3)
    rnn.py         flip-flop RNN + reaching RNN               (M1, M3)
    fixedpoints.py fixed/slow point finder + Jacobian         (M1, M4)
    flowfield.py   vector field dx/dt on a grid               (M4)
    export.py      serialize results to web JSON              (M2, M4, M5)
"""
