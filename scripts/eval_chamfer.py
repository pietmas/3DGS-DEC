"""Chamfer distance to the DTU ground truth, under the official protocol.

Drives the reference port shipped with the pinned 2DGS checkout rather than reimplementing it -
a reimplemented metric can be silently wrong against the one number that validates the harness.
The port culls the mesh by the object masks, maps to the DTU frame (mm), and reports accuracy
(reconstruction -> GT), completeness (GT -> reconstruction) and their mean, dropping pairs beyond
``max_dist_mm``. Note ``pcu.chamfer_distance`` is 2x this (a sum, not a mean); see
``tests/test_eval_chamfer.py``.

Run: ``.venv/bin/python scripts/eval_chamfer.py --mesh <ply> --out <dir>``
"""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import igl
import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
EVAL_DTU = REPO / "baselines/2d-gaussian-splatting/scripts/eval_dtu"


def seeded_reference():
    """Run the pinned scorer with a seeded shuffle; legacy np.random.seed does not seed default_rng.

    Kept in a child process because the reference scorer starts a multiprocessing pool.
    Its geometry protocol is unchanged; only its entropy-seeded sampling order is fixed.
    """
    seed, script, *args = sys.argv[1:]
    factory = np.random.default_rng
    sys.argv = [script, *args]
    with patch(
        "numpy.random.default_rng",
        lambda value=None: factory(int(seed) if value is None else value),
    ):
        runpy.run_path(script, run_name="__main__")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "?"


def _prepare_scan(scan: str, data_root: Path, workspace: Path) -> Path:
    """The reference culler globs ``<dir>/scan<id>/images``; our layout calls it ``image``.

    Symlinks rather than copies - the scan is 119 MiB and read-only here."""
    d = workspace / scan
    d.mkdir(parents=True, exist_ok=True)
    src = data_root / scan
    for link, target in (
        ("images", src / "image"),
        ("mask", src / "mask"),
        ("cameras.npz", src / "cameras.npz"),
    ):
        p = d / link
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(target.resolve())
    return d


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument(
        "--mesh", required=True, help="mesh to score, in the normalised (IDR) frame"
    )
    cli.add_argument(
        "--out", required=True, help="artifact dir: results.json + sidecar"
    )
    cli.add_argument("--scan", default=None, help="overrides eval.dtu_scan")
    cli.add_argument("--seed", type=int, default=None)
    args = cli.parse_args()

    cfg = OmegaConf.load(REPO / args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    cfg.seed = seed
    ev = cfg.eval
    scan = args.scan or ev.dtu_scan
    scan_id = int("".join(c for c in scan if c.isdigit()))
    mesh, out = Path(args.mesh).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists() or (out / "chamfer.json").exists():
        raise FileExistsError(
            f"evaluation artifacts already exist in {out}; choose a new --out"
        )
    if not mesh.is_file():
        raise FileNotFoundError(f"no mesh at {mesh}")

    data_root = REPO / ev.dtu_root
    official = REPO / ev.dtu_official
    for p in (
        official / "ObsMask" / f"ObsMask{scan_id}_10.mat",
        official / "ObsMask" / f"Plane{scan_id}.mat",
        official / "Points" / "stl" / f"stl{scan_id:03d}_total.ply",
    ):
        if not p.is_file():
            raise FileNotFoundError(f"missing official DTU asset: {p}")

    # Call the culler directly; its CLI shells out to an unchecked bare python command.
    sys.path.insert(0, str(EVAL_DTU))
    from evaluate_single_scene import cull_scan

    prepared = _prepare_scan(scan, data_root, out / "_prepared")
    culled = out / "culled_mesh.ply"
    print(f"culling {mesh.name} against {scan} masks (49 views) ...")
    cull_scan(str(scan_id), str(mesh), str(culled), instance_dir=str(prepared))

    cmd = [
        sys.executable,
        "-c",
        "from scripts.eval_chamfer import seeded_reference; seeded_reference()",
        str(seed),
        str(EVAL_DTU / "eval.py"),
        "--data",
        str(culled),
        "--scan",
        str(scan_id),
        "--mode",
        "mesh",
        "--dataset_dir",
        str(official),
        "--vis_out_dir",
        str(out),
        "--downsample_density",
        str(ev.downsample_density),
        "--max_dist",
        str(ev.max_dist_mm),
    ]
    print("running the reference metric: " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=REPO)

    r = json.loads((out / "results.json").read_text())
    V, F = igl.read_triangle_mesh(str(mesh))
    metrics = {
        "scan": scan,
        "seed": seed,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "accuracy_mm": r["mean_d2s"],  # reconstruction -> GT
        "completeness_mm": r["mean_s2d"],  # GT -> reconstruction
        "chamfer_mm": r["overall"],  # the published convention: their mean
        "mesh": str(mesh.relative_to(REPO) if mesh.is_relative_to(REPO) else mesh),
        "faces": int(F.shape[0]),
        "vertices": int(V.shape[0]),
        "downsample_density": float(ev.downsample_density),
        "max_dist_mm": float(ev.max_dist_mm),
        "commit": _git_commit(),
    }
    (out / "chamfer.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"\naccuracy {metrics['accuracy_mm']:.4f} mm | completeness "
        f"{metrics['completeness_mm']:.4f} mm | chamfer {metrics['chamfer_mm']:.4f} mm "
        f"({metrics['faces']} faces) -> {out}"
    )


if __name__ == "__main__":
    main()
