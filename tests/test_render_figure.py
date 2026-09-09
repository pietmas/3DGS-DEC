"""The figure's camera is the dataset's camera: the forward projection, pinned by hand.

``project_points`` is what the figure uses to check that an independently rasterised mesh sits
where our convention says it does. If the projection is itself wrong the check passes on a wrong
picture, so it is pinned here on cameras whose answer can be written down: a canonical pinhole, a
rotated and translated one, and a round trip against ``unproject_median_depth``, which is the map
the surfel rasterizer's own depth buffer is read with. CPU, no data, no CUDA.
"""

import numpy as np
import torch

from dec3dgs.render import project_points, to_raster_camera, unproject_median_depth

F, CX, CY = 100.0, 320.0, 240.0
K = np.array([[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]])


def test_canonical_pinhole_and_the_y_down_convention():
    """Camera at the origin looking down +z: the pinhole is (f x/z + c_x, f y/z + c_y)."""
    X = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 5.0]])
    uv, z = project_points(K, np.eye(4), X)
    assert np.allclose(z, 5.0)
    assert np.allclose(uv[0], [CX, CY])  # the optical axis hits the principal point
    assert np.allclose(uv[1], [CX + F * 1 / 5, CY + F * 2 / 5])
    # +y is DOWN: a point on the +y side of the axis lands below the principal point
    assert uv[1, 1] > CY


def test_skew_is_carried():
    """DTU's K comes out of an RQ decomposition with a small skew; K is used verbatim."""
    Ks = K.copy()
    Ks[0, 1] = 0.5
    X = np.array([[1.0, 2.0, 5.0]])
    uv, _ = project_points(Ks, np.eye(4), X)
    assert np.allclose(uv[0], [CX + F * 1 / 5 + 0.5 * 2 / 5, CY + F * 2 / 5])


def test_c2w_columns_are_the_camera_axes():
    """A point down the camera's own +z axis lands on the principal point at its own depth,
    and a step along the camera's +x moves the pixel by ``f a / d`` - which is what fixes the
    convention as ``x_cam = R^T (x - t)`` rather than its transpose."""
    R = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )  # 90 deg about +y
    assert np.isclose(np.linalg.det(R), 1.0)
    t = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, t

    d, a = 7.0, 0.75
    X = np.stack([t + d * R[:, 2], t + d * R[:, 2] + a * R[:, 0]])
    uv, z = project_points(K, c2w, X)
    assert np.allclose(z, d)
    assert np.allclose(uv[0], [CX, CY])
    assert np.allclose(uv[1], [CX + F * a / d, CY])


def test_a_point_behind_the_camera_has_negative_depth():
    """Signed depth, so the caller can drop what is behind the camera instead of trusting a
    pixel the projection would happily produce for it."""
    _, z = project_points(K, np.eye(4), np.array([[0.0, 0.0, -5.0]]))
    assert z[0] < 0


def test_round_trip_against_the_rasterizer_depth_map():
    """``unproject_median_depth`` is how the rasterizer's depth buffer becomes world points; it
    must be this projection's inverse, or the splat panel and the mesh panel are two cameras.
    A pixel at a chosen depth is unprojected and must project back onto itself."""
    W, H = 64, 48
    f = 50.0
    Kc = np.array(
        [[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]]
    )  # centred: the FoV model
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, np.array([0.3, -0.2, 0.5])
    cam = to_raster_camera(Kc, c2w, W, H, 0.01, 100.0, "cpu")

    depth = torch.full((1, H, W), 2.5)
    pts = unproject_median_depth(depth, cam).numpy().reshape(-1, 3)
    uv, z = project_points(Kc, c2w, pts)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    assert np.allclose(z, 2.5, atol=1e-4)  # a z-depth map, not a range map
    assert np.abs(uv[:, 0] - gx.ravel()).max() < 1e-3
    assert np.abs(uv[:, 1] - gy.ravel()).max() < 1e-3
