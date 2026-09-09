"""Zero extents and orthogonal references must not destroy a valid tangent frame."""

import numpy as np
import torch
from omegaconf import OmegaConf

from dec3dgs.mesh import Mesh, vertex_normals
from dec3dgs.splats import (
    _shape_operator_frame,
    _tangential_scales_torch,
    _vertex_normals_torch,
    clamp_sigma,
)


def test_orthogonal_reference_keeps_principal_direction():
    S = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    t1, t2, n = torch.eye(3).unbind()
    e = _shape_operator_frame(S, t1[None], t2[None], n[None], t2[None], 1e-4, 1e-3)
    assert torch.allclose(e, t1[None])
    e.sum().backward()
    assert torch.isfinite(S.grad).all()


def test_reference_normal_to_face_falls_back_to_tangent():
    S = torch.eye(2)[None].requires_grad_()
    t1, t2, n = torch.eye(3).unbind()
    e = _shape_operator_frame(S, t1[None], t2[None], n[None], n[None], 1e-4, 1e-3)
    assert torch.allclose(e, t1[None])
    e.sum().backward()
    assert torch.isfinite(S.grad).all()


def test_zero_dual_extent_has_finite_backward():
    M = torch.zeros(1, 3, 3, requires_grad=True)
    sigma = _tangential_scales_torch(M, torch.eye(3)[None], 0.5)
    sigma.sum().backward()
    assert torch.equal(sigma, torch.zeros_like(sigma))
    assert torch.isfinite(M.grad).all()


def test_even_face_count_uses_same_median_in_both_paths():
    cfg = OmegaConf.create({"sigma_ref": "median", "sigma_max_rel": 1.0})
    h = np.array([1.0, 3.0, 5.0, 7.0])
    sigma = np.full((4, 2), 10.0)
    assert np.array_equal(
        clamp_sigma(sigma, h, cfg),
        clamp_sigma(torch.tensor(sigma), torch.tensor(h), cfg).numpy(),
    )


def test_cancelling_vertex_normals_match_reference():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2], [1, 0, 3]])
    verts = torch.tensor(V, requires_grad=True)
    normal = _vertex_normals_torch(verts, torch.tensor(F))
    assert np.allclose(normal.detach().numpy(), vertex_normals(Mesh(V, F)))
    normal.sum().backward()
    assert torch.isfinite(verts.grad).all()
