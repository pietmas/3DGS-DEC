"""A checkpoint preserves the renderer state, including its detached frame reference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dec3dgs.checkpoint import load_state, save_state


def model():
    return SimpleNamespace(
        faces_t=torch.tensor([[0, 1, 2]]),
        x0=torch.eye(3),
        verts=torch.eye(3).requires_grad_(),
        sh_dc=torch.zeros(1, 1, 3),
        sh_rest=torch.zeros(1, 15, 3),
        opacity_logit=torch.zeros(1, 1),
        frame_mode="gram_schmidt",
        aniso=0.25,
        ref_e1=torch.tensor([[0.0, 1.0, 0.0]]),
        mesh=SimpleNamespace(faces=np.array([[0, 1, 2]])),
    )


def test_renderer_state_round_trip(tmp_path):
    source = model()
    source.verts.data.add_(0.1)
    save_state(source, tmp_path, 7)
    target = model()
    target.aniso = 1.0
    target.ref_e1 = None
    assert load_state(target, tmp_path) == 7
    assert torch.equal(target.verts, source.verts)
    assert torch.equal(target.ref_e1, source.ref_e1)
    assert target.frame_mode == "gram_schmidt"
    assert target.aniso == 0.25


def test_reindexed_mesh_is_rejected_before_parameters_change(tmp_path):
    source = model()
    source.faces_t = source.faces_t.flip(1)
    save_state(source, tmp_path, 7)
    target = model()
    before = target.verts.detach().clone()
    with pytest.raises(ValueError, match="connectivity"):
        load_state(target, tmp_path)
    assert torch.equal(target.verts, before)


def test_malformed_cochain_is_rejected_before_parameters_change(tmp_path):
    source = model()
    source.verts.data.add_(1)
    source.sh_rest = torch.zeros(2, 15, 3)
    save_state(source, tmp_path, 7)
    target = model()
    with pytest.raises(ValueError, match="sh_rest"):
        load_state(target, tmp_path)
    assert torch.equal(target.verts, target.x0)
