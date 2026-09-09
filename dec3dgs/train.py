"""The training loop, in two modes.

*Fixed mesh* (default): geometry is constant and the only parameters are the dual 0-cochains,
SH ``(F,16,3)`` and opacity logit ``(F,1)``. *Deforming mesh* (``--deform``, in :mod:`.deform`):
the vertices join them as the geometric parameter, the splats are re-derived from the live mesh
each render, and the geometric losses fair the result. Loss is ``(1-lambda_dssim) L1 + lambda_dssim
(1-SSIM)`` plus whichever regularisers are switched on. The silhouette gate runs first in both
modes and fails fast if the derived splats miss the masks.

Run: ``python -m dec3dgs.train --config configs/default.yaml [--seed S] [--out DIR] [--deform]``
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .deform import deform_main
from .losses import l_field, photometric, tv_dc
from .model import REPO, Model, assert_finite, git_commit


def _appearance_main(cfg, out: Path, seed: int):
    """The fixed-mesh loop: the geometry is a constant and only the cochains move."""
    sa = cfg.training
    model = Model(cfg)
    ious = model.silhouette_iou()
    print("silhouette IoU:", " ".join(f"{v:.3f}" for v in ious))
    assert min(ious) >= sa.silhouette_iou_min, (
        f"silhouette IoU {min(ious):.3f} < {sa.silhouette_iou_min} - "
        "splat placement is wrong, look at the derivation, not the renderer"
    )

    opt = torch.optim.Adam(
        [
            {"params": [model.sh_dc], "lr": sa.lr.sh_dc, "name": "sh_dc"},
            {"params": [model.sh_rest], "lr": sa.lr.sh_rest, "name": "sh_rest"},
            {
                "params": [model.opacity_logit],
                "lr": sa.lr.opacity_logit,
                "name": "opacity_logit",
            },
        ],
        eps=1e-15,
    )

    writer = SummaryWriter(str(out))
    train_idx = [int(v) for v in model.data.split["train"]]
    lam4, mu_field = float(cfg.field.lambda4), float(cfg.field.mu)
    nfac = model.mu.shape[
        0
    ]  # per-face normalization calibrates lam4; it does not ensure resolution independence
    ema = None
    for step in range(1, int(sa.max_steps) + 1):
        i = random.choice(train_idx)
        loss, l1, ssim = photometric(
            model.render_view(i)["render"], model.gt(i), sa.lambda_dssim
        )
        lf = None
        if lam4 > 0:
            lf = (
                l_field(
                    torch.cat([model.sh_dc, model.sh_rest], dim=1),
                    model.opacity_logit,
                    model.L2t,
                    mu_field,
                )
                / nfac
            )
            assert_finite(step, l1=l1, ssim=ssim, l_field=lf)
            loss = loss + lam4 * lf
        else:
            assert_finite(step, l1=l1, ssim=ssim)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        ema = float(loss) if ema is None else 0.99 * ema + 0.01 * float(loss)
        if step % int(sa.log_every) == 0:
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/loss_ema", ema, step)
            writer.add_scalar("train/l1", float(l1), step)
            writer.add_scalar("train/ssim", float(ssim), step)
            if lf is not None:
                writer.add_scalar("train/l_field", float(lf), step)
            with torch.no_grad():
                writer.add_scalar(
                    "train/tv_dc", float(tv_dc(model.sh_dc, model.pairs)), step
                )
            print(f"step {step:6d}  loss {float(loss):.4f}  ema {ema:.4f}")
        if step % int(sa.eval_every) == 0:
            val = model.eval_psnr("val")
            writer.add_scalar("val/psnr", val, step)
            print(f"step {step:6d}  val PSNR {val:.2f} dB")

    with torch.no_grad():
        final_tv = float(tv_dc(model.sh_dc, model.pairs))
    metrics = {
        "seed": seed,
        "commit": git_commit(),
        "lambda4": lam4,
        "silhouette_iou": ious,
        "train_psnr": model.eval_psnr(
            "train"
        ),  # the only photometric readout on DTU, which holds nothing out
        "val_psnr": model.eval_psnr("val"),
        "test_psnr": model.eval_psnr("test"),
        "tv_dc": final_tv,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    torch.save(
        {
            "step": int(sa.max_steps),
            "sh_dc": model.sh_dc.detach().cpu(),
            "sh_rest": model.sh_rest.detach().cpu(),
            "opacity_logit": model.opacity_logit.detach().cpu(),
        },
        out / "state.pt",
    )
    OmegaConf.save(cfg, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    writer.close()
    print(
        f"done: test PSNR {metrics['test_psnr']:.2f} dB, "
        f"peak VRAM {metrics['peak_vram_mib']} MiB -> {out}"
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument("--seed", type=int, default=None)
    cli.add_argument("--out", default=None)
    cli.add_argument(
        "--deform",
        action="store_true",
        help="unfreeze the vertices: optimise the mesh, not just its appearance",
    )
    cli.add_argument(
        "--no-warmup",
        action="store_true",
        help="ablation: everything on from step 0 (no cheaper fixed-frame warmup)",
    )
    args = cli.parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO / "configs/default.yaml"), OmegaConf.load(REPO / args.config)
    )
    sa = cfg.training
    seed = cfg.seed if args.seed is None else args.seed
    cfg.seed = seed
    if min(int(sa.max_steps), int(sa.log_every), int(sa.eval_every)) <= 0:
        raise ValueError("training step budget, logging and evaluation intervals must be positive")
    if args.deform and min(
        int(cfg.deform.reassemble_every), int(cfg.deform.metric.refactor_every)
    ) <= 0:
        raise ValueError("operator rebuild intervals must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(args.out) if args.out else REPO / sa.out
    if any((out / name).exists() for name in ("state.pt", "metrics.json", "mesh.ply")):
        raise FileExistsError(f"run artifacts already exist in {out}; choose a new --out")
    cfg.training.out = str(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    if args.deform:
        deform_main(cfg, out, seed, no_warmup=args.no_warmup)
    else:
        _appearance_main(cfg, out, seed)


if __name__ == "__main__":
    main()
