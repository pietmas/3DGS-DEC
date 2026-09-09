"""Loader tests. Skip cleanly on a dataless checkout."""

from pathlib import Path

import numpy as np
import pytest

from dec3dgs.datasets import (
    dtu_renormalise,
    load_dtu,
    load_dtu_gt_points,
    load_dtu_scale_mat,
    load_nerf_synthetic,
)

REPO = Path(__file__).resolve().parent.parent
NERF_ROOT = REPO / "data" / "nerf_synthetic"
DTU_ROOT = REPO / "data" / "dtu"
DTU_OFFICIAL = DTU_ROOT / "Official"
DTU_SCAN, DTU_SCAN_ID = "scan65", 65

needs_lego = pytest.mark.skipif(
    not (NERF_ROOT / "lego" / "transforms_train.json").is_file(),
    reason="nerf_synthetic/lego not downloaded",
)
needs_dtu = pytest.mark.skipif(
    not (DTU_ROOT / DTU_SCAN / "cameras.npz").is_file(),
    reason=f"dtu/{DTU_SCAN} not downloaded",
)
needs_dtu_gt = pytest.mark.skipif(
    not (DTU_OFFICIAL / "Points" / "stl" / f"stl{DTU_SCAN_ID:03d}_total.ply").is_file(),
    reason="official DTU GT points not downloaded",
)


@pytest.fixture(scope="module")
def lego():
    return load_nerf_synthetic(NERF_ROOT, "lego", white_background=True)


@pytest.fixture(scope="module")
def dtu():
    return load_dtu(DTU_ROOT, DTU_SCAN)


@needs_lego
def test_lego_splits(lego):
    assert len(lego.split["train"]) == 100
    assert len(lego.split["val"]) == 100
    assert len(lego.split["test"]) == 200


@needs_lego
def test_lego_images(lego):
    n = lego.images.shape[0]
    assert lego.images.shape == (n, 800, 800, 3)
    assert lego.images.dtype == np.float32
    assert lego.images.min() >= 0.0 and lego.images.max() <= 1.0
    assert lego.masks.shape == (n, 800, 800) and lego.masks.dtype == bool


@needs_lego
def test_lego_cameras(lego):
    K, c2w = lego.cameras.K, lego.cameras.c2w
    assert K[0, 0, 0] > 0
    R = c2w[:, :3, :3]
    eye = np.einsum("nij,nkj->nik", R, R)
    # Camera JSON has about seven significant digits, so allow its rounding error.
    assert np.allclose(eye, np.eye(3), atol=1e-5), "rotations not orthonormal"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)
    # The Blender cameras sit about 4.03 units from the origin.
    radii = np.linalg.norm(c2w[:, :3, 3], axis=1)
    assert np.allclose(radii, 4.03, atol=0.05), f"radii {radii.min()}..{radii.max()}"


@needs_dtu
def test_dtu_counts_and_masks(dtu):
    n = dtu.images.shape[0]
    assert dtu.masks.shape[0] == n
    assert dtu.masks.dtype == bool
    assert all(m.any() for m in dtu.masks), "empty mask"


@needs_dtu
@needs_dtu_gt
def test_dtu_reprojection():
    dtu = load_dtu(DTU_ROOT, DTU_SCAN)
    pts = load_dtu_gt_points(DTU_OFFICIAL, DTU_SCAN_ID)
    scale = load_dtu_scale_mat(DTU_ROOT, DTU_SCAN)
    center_raw = (pts.min(0) + pts.max(0)) / 2
    # raw DTU world -> IDR-normalized frame the cameras live in
    center = np.linalg.solve(scale, np.append(center_raw, 1.0))[:3]
    for i in range(3):
        w2c = np.linalg.inv(dtu.cameras.c2w[i])
        cam = w2c[:3, :3] @ center + w2c[:3, 3]
        assert cam[2] > 0, f"GT center behind camera {i}"
        uv = dtu.cameras.K[i] @ cam
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        assert 0 <= u < dtu.cameras.width and 0 <= v < dtu.cameras.height, (
            f"GT center projects outside image {i}: ({u:.1f}, {v:.1f})"
        )


@needs_dtu
@needs_dtu_gt
def test_dtu_gt_roundtrip():
    """Log GT point count + bounding box."""
    dtu = load_dtu(DTU_ROOT, DTU_SCAN)
    pts = load_dtu_gt_points(DTU_OFFICIAL, DTU_SCAN_ID)
    assert pts.shape[0] > 1_000_000, "DTU STL references are dense (millions of points)"
    bbox_min, bbox_max = pts.min(0), pts.max(0)
    assert np.all(np.isfinite(bbox_min)) and np.all(bbox_max > bbox_min)
    print(
        f"\nDTU {DTU_SCAN}: {dtu.images.shape[0]} images "
        f"{dtu.cameras.width}x{dtu.cameras.height}, "
        f"GT points {pts.shape[0]:,}, bbox min {bbox_min}, max {bbox_max}"
    )


def _scale_mat(s, t):
    m = np.eye(4)
    m[:3, :3] *= s
    m[:3, 3] = t
    return m


def test_dtu_renormalise_is_exact_and_world_preserving():
    """The defining property: both normalisations name the same DTU world point.

    A scale_mat is normalised -> world, so re-normalising must leave the world image fixed.
    That identity is what makes a mesh trained under one release scoreable under the other.
    """
    rng = np.random.default_rng(0)
    V = rng.normal(size=(64, 3))
    src, dst = (
        _scale_mat(246.40544, [-37.542286, -42.644344, 653.20886]),
        _scale_mat(223.3332, [43.391823, -18.587917, 613.8096]),
    )  # the two scan65 releases
    out = dtu_renormalise(V, src, dst)
    world_src = V * src[0, 0] + src[:3, 3]
    world_dst = out * dst[0, 0] + dst[:3, 3]
    assert np.allclose(world_src, world_dst, atol=1e-9), (
        "renormalisation moved the world point"
    )


def test_dtu_renormalise_identity_and_roundtrip():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(32, 3))
    a, b = (
        _scale_mat(246.40544, [-37.5, -42.6, 653.2]),
        _scale_mat(223.3332, [43.4, -18.6, 613.8]),
    )
    assert np.allclose(dtu_renormalise(V, a, a), V, atol=1e-12)  # same frame: a no-op
    assert np.allclose(dtu_renormalise(dtu_renormalise(V, a, b), b, a), V, atol=1e-9)


@needs_dtu
def test_dtu_principal_point_is_centred(dtu):
    """The FoV camera model puts the principal point at the image centre by construction, so the
    loader must hand it one that is actually there - within the half-pixel the crop can achieve."""
    K = dtu.cameras.K
    assert abs(K[:, 0, 2].mean() - dtu.cameras.width / 2) < 1.0
    assert abs(K[:, 1, 2].mean() - dtu.cameras.height / 2) < 1.0
    assert dtu.images.shape[1:3] == (dtu.cameras.height, dtu.cameras.width)
    assert dtu.masks.shape[1:3] == (dtu.cameras.height, dtu.cameras.width)


@needs_dtu
def test_dtu_crop_reproduces_the_2dgs_release(dtu):
    """The crop is not a choice of ours: it is what the 2DGS DTU release already did. Their
    scan65 images are ours cropped 46 left / 38 top, so agreeing with them bit-for-bit is the
    strongest available check that the window and the shifted K describe the same camera."""
    release = REPO / "data" / "dtu_colmap" / DTU_SCAN / "images" / "0000.png"
    if not release.is_file():
        pytest.skip("2DGS DTU release not on disk")
    from PIL import Image

    ref = np.asarray(Image.open(release).convert("RGB"), dtype=np.float32) / 255.0
    assert (dtu.cameras.height, dtu.cameras.width) == ref.shape[:2] == (1162, 1554)
    assert np.abs(dtu.images[0] - ref).max() < 1e-6
