"""Vector-heat parallel transport tests: trivial holonomy on a flat region, and holonomy
around a spherical octant loop = its spherical excess (from the snapped vertices)."""

import numpy as np

from tests.conftest import icosphere_arrays
from dec3dgs.mesh import Mesh, VectorTransport


def signed_angle(a, b):
    """Signed rotation (radians) taking 2-vector ``a`` to ``b``."""
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def test_flat_disk_transport_is_parallel(flat_disk):
    # Transport from the disk centre should give one parallel field on the interior.
    m = Mesh(*flat_disk)
    vt = VectorTransport(m)
    field3 = vt.to_3d(vt.transport(0, (1.0, 0.0)))
    field3 /= np.linalg.norm(field3, axis=1, keepdims=True)
    interior = np.setdiff1d(np.arange(len(m.verts)), m.boundary_vertices)
    cos = field3[interior] @ field3[0]
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 1.0


def test_sphere_octant_holonomy_matches_excess():
    # Transport around the spherical octant should rotate by its spherical excess.
    m = Mesh(*icosphere_arrays(4))
    vt = VectorTransport(m)
    V = m.verts
    p1, p2, p3 = (
        int(np.argmin(np.linalg.norm(V - t, axis=1)))
        for t in ([0, 0, 1], [1, 0, 0], [0, 1, 0])
    )

    v1 = np.array([1.0, 0.0])
    v2 = vt.transport(p1, v1)[p2]
    v3 = vt.transport(p2, v2)[p3]
    v_back = vt.transport(p3, v3)[p1]
    holonomy = signed_angle(v1, v_back)

    # Solid angle: tan(E/2) = |a.(b x c)| / (1 + a.b + b.c + c.a).
    a, b, c = V[p1], V[p2], V[p3]
    excess = 2 * np.arctan2(abs(a @ np.cross(b, c)), 1 + a @ b + b @ c + c @ a)

    # Transport preserves magnitude, and the rotation matches the excess within 5%.
    assert np.isclose(np.linalg.norm(v_back), 1.0, atol=1e-6)
    assert abs(abs(holonomy) - excess) / excess < 0.05
