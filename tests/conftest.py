"""Shared test meshes: unit icosphere (chi = 2), torus (chi = 0), flat unit disk (chi = 1).

Fixtures return raw ``(verts, faces)`` arrays - constructing ``Mesh`` and asserting its
invariants is the tests' job.  Session-scoped, so each mesh is built once per run.
"""

import numpy as np
import pytest
import trimesh
from scipy.spatial import Delaunay

ICOSPHERE_SUBDIVISIONS = 3  # default resolution; curvature sweeps parameterize 2..5
DISK_RINGS = 6
DISK_PTS_PER_RING = 8  # ring i carries 8*i points; the rim has DISK_RINGS * 8


def icosphere_arrays(subdivisions=ICOSPHERE_SUBDIVISIONS):
    """Near-equilateral triangulation of the unit sphere (also our Delaunay reference)."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


@pytest.fixture(scope="session")
def icosphere():
    return icosphere_arrays()


@pytest.fixture(scope="session")
def torus():
    m = trimesh.creation.torus(major_radius=1.0, minor_radius=0.4)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


@pytest.fixture(scope="session")
def flat_disk():
    """Unit disk in the z = 0 plane: center plus concentric regular rings of points,
    Delaunay-triangulated on the 2D coordinates.  Planar Delaunay by construction (all
    cotangent weights non-negative), deterministic, and the only fixture with boundary.
    """
    pts = [np.zeros((1, 2))]
    for i in range(1, DISK_RINGS + 1):
        k = DISK_PTS_PER_RING * i
        th = 2 * np.pi * np.arange(k) / k
        pts.append(i / DISK_RINGS * np.c_[np.cos(th), np.sin(th)])
    xy = np.vstack(pts)
    faces = Delaunay(xy).simplices
    # Flip clockwise qhull triangles so the disk normals point along +z.
    u, v = xy[faces[:, 1]] - xy[faces[:, 0]], xy[faces[:, 2]] - xy[faces[:, 0]]
    cw = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] < 0
    faces[cw] = faces[cw][:, ::-1]
    verts = np.c_[xy, np.zeros(len(xy))]
    return verts, faces.astype(np.int32)
