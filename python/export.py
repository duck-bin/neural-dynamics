"""export.py — serialize pipeline results to web JSON (Colab -> static app).

The notebook (kitchen) and the web app (plate) are connected ONLY by JSON. This
module writes the schema the viewer consumes. Brain exports carry trajectories +
jpca_plane and NO fixed_points/flow_field; RNN exports carry all four — the file
contents encode the brain/RNN asymmetry the viewer must show.

JSON schema (README lines 167-171):
    { "trajectories": {cond_id: [[x,y,z], ...]},
      "fixed_points": [{"pos":[x,y,z], "eigs":[[re,im],...], "stability":"saddle"}],
      "flow_field":   [{"pos":[x,y,z], "vec":[dx,dy,dz]}],
      "jpca_plane":   [[...],[...]],
      "meta": {...} }

Planned public surface (M2, M4, M5):
    export_brain(path, ...)   # trajectories + jpca_plane + meta (no mechanism)
    export_rnn(path, ...)     # + fixed_points + flow_field
"""

# TODO(M2/M4/M5): serialize results to the web JSON schema above.
