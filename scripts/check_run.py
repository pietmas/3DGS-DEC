"""Check that a completed deformation run is finite and its saved artifacts agree."""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
import torch
from omegaconf import OmegaConf

from dec3dgs.mesh import Mesh

REPO = Path(__file__).resolve().parent.parent


def check_run(run):
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/default.yaml"),
        OmegaConf.load(run / "config.yaml"),
    )
    metrics = json.loads((run / "metrics.json").read_text())
    state = torch.load(run / "state.pt", map_location="cpu", weights_only=True)
    expected = int(cfg.training.max_steps)
    if metrics.get("diverged") is not None:
        raise ValueError("run diverged")
    if state["step"] != expected or metrics.get("completed_steps") != expected:
        raise ValueError("run did not complete its configured budget")
    if metrics["seed"] != cfg.seed or not np.isfinite(metrics["train_psnr"]):
        raise ValueError("seed disagrees with config or train PSNR is non-finite")
    for name in ("verts", "x0", "sh_dc", "sh_rest", "opacity_logit"):
        if not torch.isfinite(state[name]).all():
            raise ValueError(f"non-finite checkpoint tensor: {name}")
    V, F = igl.read_triangle_mesh(str(run / "mesh.ply"))
    V0, F0 = igl.read_triangle_mesh(str(REPO / cfg.training.mesh))
    if not np.array_equal(F, F0) or not np.array_equal(F, state["faces"].numpy()):
        raise ValueError("mesh connectivity differs between artifacts")
    if not np.allclose(
        V, state["verts"].numpy(), rtol=0, atol=cfg.validation.mesh_atol
    ):
        raise ValueError("mesh vertices differ from the checkpoint")
    if not np.array_equal(V0.astype(np.float32), state["x0"].numpy()):
        raise ValueError("initial mesh differs from the checkpoint")
    mesh = Mesh(V, F)
    if mesh.boundary_edges.size:
        raise ValueError("reconstruction has boundary edges")
    with np.load(run / "history.npz") as history:
        if int(history["diverged"]) != 0:
            raise ValueError("history records divergence")
        for name in history.files:
            if not np.isfinite(history[name]).all():
                raise ValueError(f"non-finite history: {name}")
        steps = history["step"]
        if np.any(np.diff(steps) <= 0) or np.any(steps > expected):
            raise ValueError("history has invalid step ordering")
    return {
        "steps": expected,
        "vertices": len(V),
        "faces": len(F),
        "train_psnr": metrics["train_psnr"],
        "gradient_repairs": metrics["n_skipped"],
        "mesh_serialization_error": float(np.abs(V - state["verts"].numpy()).max()),
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    print(json.dumps(check_run(Path(args.run)), indent=2))


if __name__ == "__main__":
    main()
