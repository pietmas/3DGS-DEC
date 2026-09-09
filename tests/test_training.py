"""Exercise training bookkeeping with real Adam updates and a small CPU image surrogate."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dec3dgs import deform


def test_mesh_export_preserves_checkpoint_precision(tmp_path):
    import igl

    verts = np.array(
        [[0.123456789, 0, 0], [0, 1.23456789, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]])
    path = tmp_path / "mesh.ply"
    assert deform.write_mesh(verts, faces, path)
    read, triangles = igl.read_triangle_mesh(str(path))
    assert np.array_equal(read, verts)
    assert np.array_equal(triangles, faces)


@pytest.fixture
def cpu_loop(monkeypatch):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    cfg.training.max_steps = 3
    cfg.training.log_every = 1
    cfg.training.eval_every = 10
    cfg.deform.metric.enabled = False
    cfg.deform.checkpoint_every = 0
    for name in ("lambda_lap", "lambda_normal", "lambda_curv", "lambda_distortion"):
        cfg.deform[name] = 0
    cfg.field.lambda4 = 0
    verts = torch.eye(3).requires_grad_()
    model = SimpleNamespace(
        verts=verts,
        x0=verts.detach().clone(),
        faces_t=torch.tensor([[0, 1, 2]]),
        pairs_t=torch.empty(0, 2),
        mesh=SimpleNamespace(faces=np.array([[0, 1, 2]]), verts=np.eye(3)),
        sh_dc=torch.zeros(1, 1, 3, requires_grad=True),
        sh_rest=torch.zeros(1, 15, 3, requires_grad=True),
        opacity_logit=torch.zeros(1, 1, requires_grad=True),
        data=SimpleNamespace(split={"train": [0]}),
        silhouette_iou=lambda: [1.0],
        render_view=lambda *a, **kw: {"render": verts.sum()},
        gt=lambda i: torch.tensor(0.0),
        eval_psnr=lambda split: 0.0,
        frame_mode="gram_schmidt",
        ref_e1=None,
        aniso=1.0,
    )
    monkeypatch.setattr(deform, "Model", lambda *a, **kw: model)
    monkeypatch.setattr(
        deform, "photometric", lambda p, g, weight: ((p - g) ** 2, (p - g) ** 2, p * 0)
    )
    monkeypatch.setattr(
        deform,
        "SummaryWriter",
        lambda *a: SimpleNamespace(add_scalar=lambda *a: None, close=lambda: None),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    return cfg, model


def test_early_failure_saves_actual_completed_step(cpu_loop, monkeypatch, tmp_path):
    cfg, model = cpu_loop

    def fail_second(step, **terms):
        if step == 2:
            raise RuntimeError("injected non-finite loss")

    monkeypatch.setattr(deform, "assert_finite", fail_second)
    with pytest.raises(RuntimeError, match="diverged at step 2"):
        deform.deform_main(cfg, tmp_path, 0)
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert state["step"] == metrics["completed_steps"] == 1
    assert metrics["requested_steps"] == 3
    assert metrics["diverged"] == 2
    assert not torch.equal(state["verts"], model.x0)


def test_warmup_checkpoint_records_round_splats(cpu_loop, tmp_path):
    cfg, _ = cpu_loop
    deform.deform_main(cfg, tmp_path, 0)
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    assert state["step"] == 3
    assert state["frame_mode"] == "gram_schmidt"
    assert state["aniso"] == 0.0
