# README figures

Images referenced by the top-level README. Keep them small (a few hundred KB each) and regenerate
them from a scored run rather than hand-picking a good view.

Three slots are filled by `scripts/render_figure.py`, which draws every panel from the dataset's own
camera and writes `figures.json` beside them with the commit, the views and the camera-agreement
check:

- `scan65_render_vs_gt.png`  - photograph, splat render, and the shaded mesh, one camera per row
- `scan65_mesh.png`          - the mesh itself, shaded; the point is that nothing is extracted
- `scan65_x0_vs_deformed.png`- the initial mesh against the deformed one

The checked-in panels use the historical `outputs/deform_scan65` run. Its checkpoint predates
saved frame references, so replay is approximate; `figures.json` records source and rendering
provenance. Replace these panels only after the corrected full run is scored.

Optional illustrations, not completion requirements:

- `scan65_chamfer.png`       - DTU error coloured on the surface
- `edit_propagation.png`     - the DEC brush against a Euclidean brush of matched radius
                               (an archived image exists in `outputs/edit_demo/`; regeneration
                               also needs the original appearance checkpoint's initial mesh)
