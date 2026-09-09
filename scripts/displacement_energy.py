"""Roughness of the deformation ``d = x - x0``, measured on the initial mesh's operators.

The Sobolev metric's claim is that it filters the *roughness of the displacement field*, not that
it lands somewhere better. That claim lives here rather than in the Chamfer number: the metric arm
and the penalty arm can score the same distance while travelling along fields of quite different
frequency content, and it is the frequency content the transfer function ``1/(1+t lambda)`` acts on.

Reports, per deformed mesh, the mean displacement, the Dirichlet quotient ``tr(d^T L0 d)/tr(d^T M d)``
(the mean squared frequency in ``d``), ``|L0 d|/|d|``, and the tangential fraction. Every mesh must
share ``x0``'s vertex count and connectivity - a fixed-connectivity run, which is the model.

Run: ``.venv/bin/python scripts/displacement_energy.py --x0 <ply> <deformed.ply> [...]``
"""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
from omegaconf import OmegaConf

from dec3dgs.mesh import Mesh
from dec3dgs.metric import displacement_energy

REPO = Path(__file__).resolve().parent.parent


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument(
        "--x0", required=True, help="the initial mesh the run started from"
    )
    cli.add_argument(
        "meshes", nargs="+", help="deformed meshes, same connectivity as x0"
    )
    cli.add_argument("--out", default=None, help="write the table as JSON here")
    args = cli.parse_args()
    cfg = OmegaConf.load(REPO / args.config)

    V0, F0 = igl.read_triangle_mesh(args.x0)
    mesh0 = Mesh(
        V0,
        F0,
        laplacian=cfg.dec.laplacian,
        dual_type=cfg.dec.dual.type,
        dual_clamp=cfg.dec.dual.clamp,
    )
    print(f"x0: {len(V0)} verts, {len(F0)} faces  ({args.x0})")
    print(
        f"\n{'mesh':28s} {'mean |d|':>10s} {'dirichlet':>11s} {'|L0d|/|d|':>10s} "
        f"{'tangential':>11s}"
    )
    rows = {}
    for p in args.meshes:
        V, F = igl.read_triangle_mesh(p)
        if V.shape != V0.shape or not np.array_equal(F, F0):
            # the quotient is only meaningful in x0's basis, so a reindexed mesh is not comparable
            print(f"{Path(p).name:28s}  SKIPPED: connectivity differs from x0")
            continue
        e = displacement_energy(mesh0, V - V0)
        rows[p] = e
        print(
            f"{Path(p).name:28s} {e['mean_disp']:10.3e} {e['dirichlet']:11.3e} "
            f"{e['curvature_ratio']:10.3f} {e['tangential']:11.3f}"
        )
    if len(rows) > 1:
        ref = next(iter(rows))
        print(f"\nDirichlet relative to {Path(ref).name}:")
        for p, e in rows.items():
            print(
                f"  {Path(p).name:28s} {rows[ref]['dirichlet'] / e['dirichlet']:6.2f}x smoother"
            )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
