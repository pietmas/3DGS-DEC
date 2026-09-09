"""Splat derivation tests: scales follow the frame (not M's eigenvectors), sphere
isotropy + radial frames, cylinder anisotropy, the umbilic Gram-Schmidt fallback, and
obtuse/sliver robustness (finite, sigma > 0, mu pulled inside the face)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dec3dgs.mesh import Mesh
from dec3dgs.splats import derive_splats, face_shape_operators, tangential_scales

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "default.yaml").splats


def test_torch_circumcenter_falls_back_before_float32_division_nan():
    from dec3dgs.splats import _circumcenters_torch

    # Positive-area triangle whose Heron-style barycentric denominator cancels in float32.
    p = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0e-12, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    cc = _circumcenters_torch(p)
    assert torch.isfinite(cc).all()
    cc.sum().backward()
    assert torch.isfinite(p.grad).all()


def cylinder_arrays(n_theta=24, n_rows=8, radius=1.0):
    """Open tube, CCW seen from outside, axial spacing = 2x the circumferential arc."""
    dz = 2.0 * (2 * np.pi * radius / n_theta)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    ring = np.c_[radius * np.cos(th), radius * np.sin(th), np.zeros(n_theta)]
    verts = np.vstack([ring + [0, 0, i * dz] for i in range(n_rows)])
    v = lambda i, j: i * n_theta + (j % n_theta)  # noqa: E731
    faces = []
    for i in range(n_rows - 1):
        for j in range(n_theta):
            v00, v01 = v(i, j), v(i, j + 1)
            v10, v11 = v(i + 1, j), v(i + 1, j + 1)
            faces += [(v00, v01, v11), (v00, v11, v10)]
    return verts, np.asarray(faces, dtype=np.int32)


def test_config_carries_the_splat_keys():
    assert CFG.gamma == pytest.approx(0.5)
    assert CFG.eigengap_min == pytest.approx(1e-4)
    assert CFG.sigma_max_rel == pytest.approx(3.0)


def test_scales_follow_frame_not_dual_eigenvectors():
    # At 45 degrees to diag(4, 1), both projected extents are 2.5, unlike the eigenvalues.
    M = np.diag([4.0, 1.0, 0.0])[None]
    c = 1.0 / np.sqrt(2.0)
    e1, e2, n = [c, c, 0.0], [-c, c, 0.0], [0.0, 0.0, 1.0]
    frame = np.stack([e1, e2, n], axis=1)[None]
    sigma = tangential_scales(M, frame, gamma=1.0)
    assert sigma == pytest.approx(np.full((1, 2), np.sqrt(2.5)))
    assert np.abs(sigma - 2.0).min() > 0.4 and np.abs(sigma - 1.0).min() > 0.4


def test_sphere_shape_operator_is_identity(icosphere):
    # On a sphere S=+I; check median convergence since irregular vertices retain bias.
    S3 = face_shape_operators(Mesh(*icosphere))
    assert np.trace(S3, axis1=1, axis2=2).mean() / 2 == pytest.approx(1.0, abs=0.05)
    err3 = np.median(np.abs(S3 - np.eye(2)).max(axis=(1, 2)))
    from tests.conftest import icosphere_arrays

    S4 = face_shape_operators(Mesh(*icosphere_arrays(4)))
    err4 = np.median(np.abs(S4 - np.eye(2)).max(axis=(1, 2)))
    assert err4 < 0.65 * err3 < 0.05


def test_sphere_isotropy_and_radial_frames(icosphere):
    sp = derive_splats(Mesh(*icosphere), CFG)
    assert np.isfinite(sp.sigma).all() and (sp.sigma > 0).all()
    ratio = sp.sigma.max(axis=1) / sp.sigma.min(axis=1)
    assert np.median(ratio) < 1.1
    # frame normal vs outward radial direction at the splat center
    radial = sp.mu / np.linalg.norm(sp.mu, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", sp.frame[:, :, 2], radial)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 5.0
    # frames are honest rotations (the rasterizer quaternion depends on it)
    gram = np.einsum("fij,fik->fjk", sp.frame, sp.frame)
    assert np.abs(gram - np.eye(3)).max() < 1e-10
    assert np.linalg.det(sp.frame) == pytest.approx(np.ones(len(sp.frame)))


def test_cylinder_frames_and_anisotropy():
    n_theta, n_rows = 24, 8
    mesh = Mesh(*cylinder_arrays(n_theta, n_rows))
    sp = derive_splats(mesh, CFG)
    # interior = no vertex on the two boundary rings (full dual cells, clean normals)
    row = mesh.faces // n_theta
    interior = (row.min(axis=1) >= 1) & (row.max(axis=1) <= n_rows - 2)
    assert interior.sum() > 100
    # On a cylinder e1 is circumferential and e2 is axial.
    z = np.array([0.0, 0.0, 1.0])
    assert np.abs(sp.frame[interior, :, 0] @ z).max() < np.sin(np.radians(10))
    assert np.abs(sp.frame[interior, :, 1] @ z).min() > np.cos(np.radians(10))
    # the long scale axis is e2 - the flat (axial) direction, from the stretched cells
    assert (sp.sigma[interior, 1] > sp.sigma[interior, 0]).all()


def test_umbilic_fallback_is_gram_schmidt_exactly(flat_disk):
    mesh = Mesh(*flat_disk)
    sp = derive_splats(mesh, CFG)
    for arr in (sp.mu, sp.frame, sp.sigma, sp.quat_wxyz):
        assert np.isfinite(arr).all()
    # A flat face should recover the Gram-Schmidt frame exactly.
    p = mesh.verts[mesh.faces]
    t1 = p[:, 1] - p[:, 0]
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    assert np.array_equal(sp.frame[:, :, 0], t1)
    assert np.array_equal(
        sp.frame[:, :, 2], np.tile([0.0, 0.0, 1.0], (len(mesh.faces), 1))
    )


def test_sigma_reference_is_global_or_local_and_only_local_survives_a_bimodal_mesh():
    """The clamp's reference decides whether one region's resolution can set another's coverage.

    Specimen: two scales in one array - a coarse population and a fine one, i.e. a mesh with a
    genuine size gradient. Under
    ``median`` the ceiling is dragged down by the fine majority and cuts every coarse face; under
    ``local`` each face is judged against its own ``h_f`` and keeps its coverage."""
    from dec3dgs.splats import clamp_sigma

    h_f = np.concatenate([np.full(20, 1.0), np.full(80, 0.1)])  # 20 coarse, 80 fine
    sigma = 0.5 * np.column_stack([h_f, h_f])  # each face wants sigma ~ h_f / 2
    cfg = OmegaConf.create({"sigma_max_rel": 3.0, "sigma_ref": "median"})

    med = clamp_sigma(sigma.copy(), h_f, cfg)
    assert np.allclose(med[20:], sigma[20:])  # the fine majority is untouched
    assert (med[:20] < sigma[:20]).all() and np.allclose(
        med[:20], 0.3
    )  # ...the coarse tail is cut

    cfg.sigma_ref = "local"
    loc = clamp_sigma(sigma.copy(), h_f, cfg)
    assert np.allclose(loc, sigma)  # h_f/2 < 3 h_f on every face
    # and the guard still bites where it must: a dual cell that dwarfs the face's own area
    assert clamp_sigma(np.array([[10.0, 10.0]]), np.array([0.1]), cfg)[
        0, 0
    ] == pytest.approx(0.3)

    cfg.sigma_ref = "bogus"
    with pytest.raises(ValueError):
        clamp_sigma(sigma.copy(), h_f, cfg)


def test_obtuse_and_sliver_robustness(icosphere):
    verts, faces = icosphere
    rng = np.random.default_rng(0)
    verts = verts + 0.02 * rng.normal(size=verts.shape)
    # Move the apex near the opposite edge while keeping the area positive.
    i, j, k = faces[0]
    n = np.cross(verts[j] - verts[i], verts[k] - verts[i])
    verts[i] = 0.5 * (verts[j] + verts[k]) + 1e-3 * n / np.linalg.norm(n)

    mesh = Mesh(verts, faces)
    p = verts[faces]
    u, w = p[:, [1, 2, 0]] - p, p[:, [2, 0, 1]] - p
    cos = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    assert (np.arccos(np.clip(cos, -1, 1)).max(axis=1) > np.pi / 2).mean() > 0
    sp = derive_splats(mesh, CFG)
    assert np.isfinite(sp.sigma).all() and (sp.sigma > 0).all()
    assert np.isfinite(sp.mu).all() and np.isfinite(sp.quat_wxyz).all()
    # Check nonnegative barycentric coordinates after pulling the circumcentre into its face.
    from dec3dgs.mesh import circumcenters

    assert np.linalg.norm(circumcenters(mesh)[0] - p[0].mean(axis=0)) > 1.0
    n_f = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    bary = np.stack(
        [
            np.einsum(
                "ij,ij->i",
                np.cross(p[:, (k + 1) % 3] - sp.mu, p[:, (k + 2) % 3] - sp.mu),
                n_f,
            )
            for k in range(3)
        ],
        axis=1,
    ) / (n_f * n_f).sum(axis=1, keepdims=True)
    assert bary.min() > -1e-9
    # The cap uses mesh-scale area, so a sliver's own area does not collapse its splat.
    h_eq = np.sqrt(
        4.0
        * (0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1))
        / np.sqrt(3.0)
    )
    assert (sp.sigma <= CFG.sigma_max_rel * np.median(h_eq) + 1e-12).all()
    assert sp.sigma[0].max() > CFG.sigma_max_rel * h_eq[0]
