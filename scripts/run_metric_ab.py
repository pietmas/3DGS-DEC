"""The matched comparison behind the project's one surviving positive claim.

``--arms metric_vs_penalty`` - does the metric beat the displacement-Laplacian penalty?

  ``penalty``       lambda_lap = 100, no metric      - the incumbent
  ``metric_cotan``  (M + t L0)^-1, mesh's own L0     - the proposal
  ``metric_comb``   (M + t L0)^-1, graph Laplacian   - the combinatorial null, re-tested here
  ``control``       neither                          - what the fairing was buying at all

Arms are matched by **displacement**, not by ``lr``: each is probed once, then its rate is scaled
so every arm travels as far as ``penalty`` does. Comparing a metric to a penalty at equal ``lr``
compares two different distances, which is not the question.

**Read the Chamfer output against the noise floor.** The three-seed control spread on scan65 is
0.032 mm, and the metric-vs-penalty Chamfer difference is 1.5x that at n=1 - **not reportable**.
The claim that survives is the displacement smoothness: run ``scripts/displacement_energy.py`` on
the arms' final meshes, where the same effect clears its own noise by a factor of ~74. That is the
whole lesson - measure the mechanism, not its consequence.

The curvature-aware ``*1^kappa`` arms were removed. That question is settled: the star is inert on
the surface and adverse on the triangulation, so ``lambda_normal = 0.05`` stays.

Run: ``python -m scripts.run_metric_ab --run`` then ``--report``.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "metric_ab"

# Override only the settings that define each arm.
ARM_SETS = {
    # is the metric better than the penalty? Four arms, lr calibrated (see below).
    "metric_vs_penalty": {
        "penalty": {"deform.lambda_lap": 100.0, "deform.metric.enabled": False},
        "metric_cotan": {
            "deform.lambda_lap": 0.0,
            "deform.metric.enabled": True,
            "deform.metric.laplacian": "cotan",
            "deform.metric.anisotropy": "none",
        },
        "metric_comb": {
            "deform.lambda_lap": 0.0,
            "deform.metric.enabled": True,
            "deform.metric.laplacian": "combinatorial",
        },
        "control": {"deform.lambda_lap": 0.0, "deform.metric.enabled": False},
    },
}
BASELINE = "penalty"  # the arm whose mean_disp the others are matched to

# a compressed schedule so the probe's steps land past the warmup, in the regime that matters
PROBE = {
    "training.max_steps": 250,
    "deform.warmup.warmup_steps": 50,
    "deform.warmup.n_ramp": 50,
    "training.log_every": 50,
    "training.eval_every": 10**9,
    "deform.checkpoint_every": 0,
}


def write_config(out, name, overrides, extra=None):
    cfg = OmegaConf.load(REPO / "configs" / "default.yaml")
    for k, v in {**overrides, **(extra or {})}.items():
        OmegaConf.update(cfg, k, v)
    path = out / name / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    return path


def train(out, name, config, seed):
    """Run one arm; return its metrics, or ``None`` if it produced none. Not ``check=True`` - a
    diverged arm is a result, not a reason to lose the arms after it."""
    out = out / name
    # -u: python block-buffers stdout when redirected, which once hid a whole run's progress
    r = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "dec3dgs.train",
            "--deform",
            "--config",
            str(config),
            "--out",
            str(out),
            "--seed",
            str(seed),
        ],
        cwd=REPO,
    )
    mp = out / "metrics.json"
    if r.returncode != 0:
        print(
            f"!! arm {name!r} exited {r.returncode}"
            f"{' (metrics.json written anyway)' if mp.exists() else ' with no metrics.json'}"
        )
    return json.loads(mp.read_text()) if mp.exists() else None


def calibrate(arms, out, seed, lr0):
    """One probe per arm at the shared ``lr0``, then ``lr = lr0 * disp_baseline / disp_arm``, then
    one verification probe. Writes ``lr.json``, which ``--run`` consumes and does not recompute."""
    disp = {}
    for name, ov in arms.items():
        cfg = write_config(out, f"probe_{name}", ov, {**PROBE, "deform.lr_verts": lr0})
        disp[name] = train(out, f"probe_{name}", cfg, seed)["mean_disp"]
        print(f"probe {name:13s} lr {lr0:.2e} -> mean_disp {disp[name]:.4e}")

    lr = {n: lr0 * disp[BASELINE] / d for n, d in disp.items()}
    for name in arms:
        if name == BASELINE:
            continue
        cfg = write_config(
            out, f"verify_{name}", arms[name], {**PROBE, "deform.lr_verts": lr[name]}
        )
        d = train(out, f"verify_{name}", cfg, seed)["mean_disp"]
        print(
            f"verify {name:13s} lr {lr[name]:.2e} -> mean_disp {d:.4e} "
            f"(target {disp[BASELINE]:.4e}, off by {100 * (d / disp[BASELINE] - 1):+.1f} %)"
        )
    # Keep the calibrated rate; a residual mismatch measures Adam's nonlinearity.
    (out / "lr.json").write_text(
        json.dumps({"lr0": lr0, "probe_mean_disp": disp, "lr": lr}, indent=2) + "\n"
    )
    print(f"-> {out / 'lr.json'}")


def run(arms, out, seed, lr0):
    lrs = (
        json.loads((out / "lr.json").read_text())["lr"]
        if (out / "lr.json").exists()
        else {}
    )
    for name, ov in arms.items():
        # an arm may pin its own lr; only fall back to the calibrated one when it does not
        extra = (
            {} if "deform.lr_verts" in ov else {"deform.lr_verts": lrs.get(name, lr0)}
        )
        cfg = write_config(out, name, ov, extra)
        m = train(out, name, cfg, seed)
        if m is None:
            print(f"{name:13s} NO METRICS - see the arm's log")
            continue
        print(
            f"{name:13s} train {m['train_psnr']:.2f} dB  disp {m['mean_disp']:.3e}  "
            f"skipped {m['n_skipped']}  diverged {m['diverged']}"
        )


def report(arms, out):
    """Displacement roughness per arm, then Chamfer. In that order deliberately: the roughness is
    the mechanism the metric acts on and it resolves at ~5 % noise, while Chamfer is a downstream
    scalar whose metric-vs-penalty difference sits at 1.5x the seed spread and cannot carry a
    claim. Both are recorded, neither is gated - no arm here is tuned against either."""
    meshes = [
        str(out / n / "mesh.ply") for n in arms if (out / n / "mesh.ply").exists()
    ]
    if not meshes:
        raise SystemExit("no arm meshes on disk - run --run first")
    x0 = REPO / OmegaConf.load(REPO / "configs" / "default.yaml").training.mesh
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.displacement_energy",
            "--x0",
            str(x0),
            *meshes,
            "--out",
            str(out / "displacement_energy.json"),
        ],
        cwd=REPO,
        check=True,
    )
    for name in arms:
        mesh = out / name / "mesh.ply"
        if mesh.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.eval_chamfer",
                    "--mesh",
                    str(mesh),
                    "--out",
                    str(out / name / "chamfer"),
                ],
                cwd=REPO,
                check=True,
            )
    print(
        f"\n{'arm':14s} {'train PSNR':>11s} {'mean_disp':>11s} {'skipped':>8s} "
        f"{'chamfer':>9s} {'accuracy':>9s} {'complete':>9s}"
    )
    for name in arms:
        mp, cp = out / name / "metrics.json", out / name / "chamfer" / "chamfer.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        c = json.loads(cp.read_text()) if cp.exists() else {}
        nan = float("nan")
        print(
            f"{name:14s} {m['train_psnr']:11.2f} {m['mean_disp']:11.3e} "
            f"{m['n_skipped']:8d} {c.get('chamfer_mm', nan):9.4f} "
            f"{c.get('accuracy_mm', nan):9.4f} {c.get('completeness_mm', nan):9.4f}"
        )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--arms", choices=sorted(ARM_SETS), default="metric_vs_penalty")
    cli.add_argument(
        "--calibrate",
        action="store_true",
        help="probe runs -> lr.json: match every arm's displacement to the baseline's",
    )
    cli.add_argument("--run", action="store_true", help="the scored arms")
    cli.add_argument(
        "--report", action="store_true", help="displacement roughness + chamfer + table"
    )
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument(
        "--lr0", type=float, default=None, help="shared lr (default: deform.lr_verts)"
    )
    args = cli.parse_args()

    arms, out = ARM_SETS[args.arms], OUT / args.arms
    lr0 = args.lr0 or float(
        OmegaConf.load(REPO / "configs" / "default.yaml").deform.lr_verts
    )
    if args.calibrate:
        calibrate(arms, out, args.seed, lr0)
    if args.run:
        run(arms, out, args.seed, lr0)
    if args.report:
        report(arms, out)
    if not (args.calibrate or args.run or args.report):
        cli.error("pick at least one of --calibrate / --run / --report")


if __name__ == "__main__":
    main()
