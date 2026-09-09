"""Curvature tests.

Sphere convergence (H, K -> 1 with monotone error decay), the H-sign convention,
flat-disk exactness, and Gauss-Bonnet at machine precision (combinatorial identity,
not an approximation)."""

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import icosphere_arrays
from dec3dgs.mesh import (
    Mesh,
    angle_defect,
    gaussian_curvature,
    mean_curvature,
    vertex_normals,
)


def test_sphere_curvature_convergence():
    # Sphere curvature error should decrease with refinement and finish below 1%.
    rows = []
    for sub in range(2, 6):
        m = Mesh(*icosphere_arrays(sub))
        rows.append(
            (
                sub,
                len(m.verts),
                np.abs(mean_curvature(m) - 1.0).mean(),
                np.abs(gaussian_curvature(m) - 1.0).mean(),
            )
        )
    eH, eK = [r[2] for r in rows], [r[3] for r in rows]
    assert all(a > b for a, b in zip(eH, eH[1:]))
    assert all(a > b for a, b in zip(eK, eK[1:]))
    assert eH[-1] < 0.01 and eK[-1] < 0.01

    lines = ["subdivision,verts,H_mean_rel_err,K_mean_rel_err"]
    lines += [f"{s},{v},{h:.6e},{k:.6e}" for s, v, h, k in rows]
    out = Path("outputs/curvature")
    out.mkdir(parents=True, exist_ok=True)
    (out / "curvature_convergence.csv").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def test_sphere_H_sign(icosphere):
    # The sign convention, asserted once: convex with outward normals is positive.
    m = Mesh(*icosphere)
    n = vertex_normals(m)
    assert (np.einsum("ij,ij->i", n, m.verts) > 0.9).all()  # outward on the unit sphere
    assert (mean_curvature(m) > 0).all()


def test_flat_disk_curvature_vanishes(flat_disk):
    # On a flat interior, angle defect and the normal area gradient are both zero.
    m = Mesh(*flat_disk)
    interior = np.setdiff1d(np.arange(len(m.verts)), m.boundary_vertices)
    assert np.abs(mean_curvature(m)[interior]).max() < 1e-8
    assert np.abs(gaussian_curvature(m)[interior]).max() < 1e-8


def test_vertex_normal_cancellation_is_finite_zero():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    faces = np.array(
        [[0, 1, 2], [0, 3, 4]], dtype=np.int32
    )  # +z and -z cancel at vertex 0
    n = vertex_normals(Mesh(verts, faces))
    assert np.isfinite(n).all()
    assert np.array_equal(n[0], np.zeros(3))


@pytest.mark.parametrize("name, chi", [("icosphere", 2), ("torus", 0)])
def test_gauss_bonnet_closed(name, chi, request):
    m = Mesh(*request.getfixturevalue(name))
    assert abs(angle_defect(m).sum() - 2 * np.pi * chi) < 1e-9


def test_gauss_bonnet_disk_with_boundary(flat_disk):
    # Gauss-Bonnet: sum_int(2pi - sum th) + sum_bnd(pi - sum th) = 2pi chi.
    m = Mesh(*flat_disk)
    defect = angle_defect(m)
    bnd = m.boundary_vertices
    interior = np.setdiff1d(np.arange(len(m.verts)), bnd)
    total = defect[interior].sum() + (defect[bnd] - np.pi).sum()
    assert abs(total - 2 * np.pi) < 1e-9
