"""Smoke render: vanilla 2DGS on a NeRF-synthetic scene, GTX-1050 budget.

Drives the pinned official 2DGS trainer (baselines/2d-gaussian-splatting) with the
budget knobs from configs/default.yaml. gsplat cannot compile for Pascal (sm_61),
so the official repo is the vehicle.

A monkeypatched hard cap on densification (smoke_render.cap_max_splats) keeps the
model inside the 1050's 3 GiB. Peak VRAM and final splat count are printed and
appended to <model_path>/smoke_summary.txt.

Usage:
    python scripts/smoke_render.py --config configs/default.yaml --seed 0
"""

import argparse
import random
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
GS2D = REPO / "baselines" / "2d-gaussian-splatting"


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument("--seed", type=int, default=None)
    cli.add_argument(
        "--steps", type=int, default=None, help="override smoke_render.max_steps"
    )
    cli.add_argument(
        "--out", default="smoke", help="output directory name under outputs/"
    )
    cli_args = cli.parse_args()

    cfg = OmegaConf.load(REPO / cli_args.config)
    seed = cfg.seed if cli_args.seed is None else cli_args.seed
    cfg.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cli_args.steps is not None:
        cfg.smoke_render.max_steps = cli_args.steps
    scene = cfg.smoke_render.scene
    source = REPO / cfg.data.nerf_synthetic_root / scene
    out = REPO / "outputs" / cli_args.out
    out.mkdir(parents=True, exist_ok=True)

    # provenance: config copy + git commit + seed
    OmegaConf.save(cfg, out / "config.yaml")
    commit = (
        subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "no-git"
    )
    (out / "provenance.txt").write_text(f"commit: {commit}\nseed: {seed}\n")

    sys.path.insert(0, str(GS2D))
    import train as gs_train  # noqa: E402  (the 2DGS repo's train.py)
    from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
    from scene.gaussian_model import GaussianModel  # noqa: E402

    # no GUI server: neutralize the viewer hooks
    gs_train.network_gui.conn = None
    gs_train.network_gui.try_connect = lambda *a, **k: None

    # hard cap on splat growth to stay inside the 1050's VRAM
    cap = int(cfg.smoke_render.cap_max_splats)
    orig_densify = GaussianModel.densify_and_prune

    def capped_densify(self, *a, **k):
        if self.get_xyz.shape[0] >= cap:
            return None  # stop growing; smoke run does not need further pruning
        return orig_densify(self, *a, **k)

    GaussianModel.densify_and_prune = capped_densify

    max_steps = int(cfg.smoke_render.max_steps)
    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args(
        [
            "--source_path",
            str(source),
            "--model_path",
            str(out),
            "--sh_degree",
            str(cfg.smoke_render.sh_degree),
            "--iterations",
            str(max_steps),
            "--eval",
            "--data_device",
            "cpu",  # keep the image stack out of the 3 GiB VRAM
        ]
    )
    if cfg.data.white_background:
        args.white_background = True

    test_iters = list(range(500, max_steps + 1, 500))
    print(
        f"smoke render: scene={scene} steps={max_steps} cap={cap} "
        f"sh_degree={cfg.smoke_render.sh_degree} seed={seed}"
    )
    gs_train.training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        test_iters,
        [max_steps],
        [],
        None,
    )

    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    summary = (
        f"peak VRAM allocated: {peak_mib:.0f} MiB (card: 3072 MiB)\n"
        f"cap_max_splats: {cap}\nseed: {seed}\nsteps: {max_steps}\n"
    )
    print(summary)
    (out / "smoke_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
