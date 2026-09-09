"""The deforming-mesh loop: connectivity fixed, vertices free.

The vertices are the geometry, the splats are re-derived from them at every render, and
``L_photo + lambda1 L_lap + lambda2 L_normal + lambda_dist L_distortion`` fair the result. A
two-stage :class:`WarmupSchedule` holds a fixed Gram-Schmidt frame with isotropic scales while the
mesh settles, then ramps on the shape-operator orientation, the anisotropy, ``L_curv`` and
``L_field``.

Descent may be taken in the Sobolev metric rather than the Euclidean one: ``x <- x - eta (M + t
L_D)^-1 g``. That is a change of *metric*, not an added penalty, and it replaces ``lambda1``.
"""

import json
import random
from pathlib import Path

import igl
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import save_state
from .losses import l_curv, l_distortion, l_field, l_lap, l_normal, photometric
from .mesh import Mesh, cotan_laplacian_torch
from .model import (
    FG_ALPHA,
    Model,
    assert_finite,
    build_metric,
    git_commit,
    reassemble,
)
from .render import face_id_from_depth
from .splats import face_adjacency_weights, face_normals_torch


def repair_nonfinite_grads(**params) -> list[tuple[str, np.ndarray]]:
    """Zero the non-finite **rows** of every parameter's gradient, in place, and name what was hit.
    Run **before** ``opt.step()`` - a bad gradient must never be written into the mesh - and before
    the metric solve, so a NaN is caught as itself instead of being smeared over the 1-ring.

    Zeroing only the offending rows, rather than dropping the whole update, is what keeps a *local*
    defect local: the handful of vertices carrying a bad gradient stay put for one step while the
    rest of the mesh moves, and the fairing around them pulls the pocket out of its degeneracy.
    Dropping the update freezes all of them instead, so the state that produced the NaN is still
    there at the next step - a deadlock, measured twice on scan65: an identical burst on 5 of 76522
    rows (0.007 %) repeated for 21 consecutive steps and ended the run, once in warmup and once in
    the ramp.

    The **indices**, not just their count, because a count is not an identity: a burst of 21 skips
    each reporting "12 rows" was read as *the same twelve vertices every step*, and the mechanism
    was inferred from that rather than measured."""
    hit = []
    for name, p in params.items():
        if p.grad is None:
            continue
        rows = ~torch.isfinite(p.grad.reshape(p.grad.shape[0], -1)).all(1)
        if rows.any():
            p.grad[rows] = 0.0
            hit.append((name, torch.nonzero(rows).flatten().cpu().numpy()))
    return hit


def write_mesh(V: np.ndarray, faces: np.ndarray, path: Path) -> bool:
    """Write ``V, faces`` to ``path``, refusing a mesh with any non-finite vertex.

    ``igl.write_triangle_mesh`` emits a *structurally corrupt* ply from non-finite verts - every
    face collapses to ``[0,0,0]``, so it reads back as plausible vertices with zero-area faces, a
    booby trap that once cost a wrong diagnosis. A diverged mesh is meaningless anyway; skip it."""
    if np.isfinite(V).all():
        igl.write_triangle_mesh(str(path), V, faces, igl.FileEncoding.Binary)
        return True
    print(
        f"WARNING: {int((~np.isfinite(V).all(1)).sum())} non-finite vertices - {path.name} "
        f"not written"
    )
    return False


class WarmupSchedule:
    """The two-stage deform schedule, a pure function of ``step``. A geometric **warmup** of
    ``warmup_steps`` runs with a fixed Gram-Schmidt frame and isotropic in-plane scales, faring the
    mesh under ``L_photo + L_lap + L_normal`` alone; then a linear **ramp** over ``n_ramp`` steps
    brings the shape-operator orientation, the anisotropy, ``L_curv`` and ``L_field`` up from zero,
    so nothing switches discontinuously. ``disabled=True`` is the ablation - everything on from
    step 0.

    The warmup is a **compute optimisation**, not a stability mechanism: the fixed frame costs ~5x
    less per step than the shape operator, and its isotropy (anisotropy 0 at ramp start) is what
    lets the orientation switch on continuously. Its length is a **fixed budget** by design. An
    earlier version ended it adaptively when a step-displacement EMA fell below ``eps_warmup``; that
    was withdrawn because the displacement has no settling knee (it drifts ~20 % with no clear
    event), so a threshold near it fired essentially at random across objects - early on scan65,
    never on lego. A fixed budget is reproducible."""

    def __init__(self, wcfg, disabled: bool = False):
        self.n_ramp = max(1, int(wcfg.n_ramp))
        self.disabled = disabled
        self.ramp_start = (
            0 if disabled else int(wcfg.warmup_steps)
        )  # last warmup step; ramping begins after

    def alpha(self, step: int) -> float:
        """Ramp fraction in ``[0, 1]``: 0 during warmup, linear over ``n_ramp`` after it ends,
        and 1 from step 0 when disabled (everything on at once)."""
        if self.disabled:
            return 1.0
        return min(1.0, max(0, step - self.ramp_start) / self.n_ramp)

    def warming_up(self, step: int) -> bool:
        return not self.disabled and step <= self.ramp_start

    def frame_mode(self, step: int) -> str:
        """Gram-Schmidt during warmup, shape-operator once ramping - a hard flip, safe because
        the anisotropy is 0 at ramp start so the frame orientation does not yet reach the render."""
        return "gram_schmidt" if self.warming_up(step) else "shape_operator"


def _resolve_lambdas(cfg, metric, lam_lap, lam_normal):
    """The metric *replaces* penalties rather than joining them: running ``(M + tL)^-1`` and
    ``lambda1 L_lap`` together states the same smoothness requirement twice, once as a metric and
    once as a force. Reports what it switched off."""
    dv = cfg.deform
    theta_c = None if dv.metric.theta_c is None else float(dv.metric.theta_c)
    pct = (
        None
        if dv.metric.theta_c_percentile is None
        else float(dv.metric.theta_c_percentile)
    )
    # no knee (theta_c None and no percentile) is g == 1, i.e. isotropic - not an aniso arm
    aniso = dv.metric.anisotropy == "curvature" and (
        theta_c is not None or pct is not None
    )
    if lam_lap > 0:
        print(
            f"metric on ({dv.metric.laplacian}, t={float(dv.metric.t):.2e}): "
            f"lambda_lap {lam_lap} -> 0"
        )
        lam_lap = 0.0
    if aniso and lam_normal > 0:
        knee = (
            f"p{pct:g}->{metric.theta_c:.2f}" if pct is not None else f"{theta_c:.2f}"
        )
        if bool(dv.metric.replace_lambda_normal):
            print(
                f"anisotropy on ({dv.metric.modulator}, theta_c={knee}): "
                f"lambda_normal {lam_normal} -> 0"
            )
            lam_normal = 0.0
        else:
            print(
                f"anisotropy on ({dv.metric.modulator}, theta_c={knee}) WITH "
                f"lambda_normal={lam_normal} (replace_lambda_normal off): running both"
            )
    return lam_lap, lam_normal


def deform_main(cfg, out: Path, seed: int, no_warmup: bool = False):
    """The deforming-mesh loop. The in-graph ``L0_x`` is rebuilt every step; the detached robust
    ``L2`` coupling, the curvature weight ``kappa`` and the circumcenter ``cKDTree`` (``L_curv``'s
    face-ID lookup) are reassembled every ``reassemble_every`` ramping steps. ``no_warmup`` is the
    ablation."""
    sa, dv = cfg.training, cfg.deform
    model = Model(cfg, deform=True)
    faces_np = model.mesh.faces
    # Count normalization calibrates the weights; it does not ensure resolution independence.
    nfac0, nvert0, nedge0 = len(faces_np), len(model.mesh.verts), len(model.pairs_t)
    device = model.faces_t.device
    kappa_t, face_tree = (
        None,
        None,
    )  # built at ramp start (L_curv/L_field off in warmup)

    ious = model.silhouette_iou()
    print("silhouette IoU:", " ".join(f"{v:.3f}" for v in ious))
    # Check placement before learned opacity changes the thresholded silhouette.
    assert min(ious) >= sa.silhouette_iou_min, (
        f"silhouette IoU {min(ious):.3f} < {sa.silhouette_iou_min} - placement is wrong"
    )

    opt = torch.optim.Adam(
        [
            {"params": [model.verts], "lr": float(dv.lr_verts), "name": "verts"},
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
    # plain ints: numpy int64 lands in metrics.json (via a skip record) and is not JSON-serialisable
    train_idx = [int(v) for v in model.data.split["train"]]
    lam_lap = float(dv.lambda_lap)
    lam_normal, delta_huber = float(dv.lambda_normal), float(dv.delta_huber)
    lam_curv = float(dv.lambda_curv)
    lam_dist = float(dv.lambda_distortion)
    lam4, mu_field = float(cfg.field.lambda4), float(cfg.field.mu)
    reassemble_every = int(dv.reassemble_every)
    checkpoint_every = int(dv.checkpoint_every)
    sched = WarmupSchedule(dv.warmup, disabled=no_warmup)
    metric, refactor_every = None, int(dv.metric.refactor_every)
    if bool(dv.metric.enabled):
        metric = build_metric(model.mesh, dv)
        lam_lap, lam_normal = _resolve_lambdas(cfg, metric, lam_lap, lam_normal)

    ramp_steps, transition_logged, diverged = 0, no_warmup, None
    skipped = []  # steps whose gradient needed repair (rows zeroed)
    n_consecutive, last_repaired = (
        0,
        -2,
    )  # consecutive repairs: the stuck-state detector
    force_reassemble = False
    hist = {
        k: []
        for k in (
            "step",
            "loss",
            "loss_ema",
            "mean_disp",
            "step_disp",
            "alpha",
            "l_lap",
            "l_normal",
            "l_curv",
            "l_field",
            "l_distortion",
        )
    }
    ema = None
    completed_steps = 0
    for step in range(1, int(sa.max_steps) + 1):
        alpha = sched.alpha(step)
        model.frame_mode = sched.frame_mode(step)
        if not sched.warming_up(step):
            if ramp_steps % reassemble_every == 0 or force_reassemble:
                model.reassemble_reference()  # transported frame for the umbilic blend
                model.L2t, kappa_t, face_tree = reassemble(
                    model.verts, faces_np, cfg, device
                )
                if force_reassemble:
                    print(f"step {step:6d}  forced reassembly after a skip")
                force_reassemble = False  # detached rebuild, frozen within the interval
            ramp_steps += 1
            if not transition_logged:
                print(
                    f"step {step:6d}  warmup -> ramp (fixed budget {sched.ramp_start})"
                )
                transition_logged = True

        i = random.choice(train_idx)
        gt = model.gt(i)
        rendered = model.render_view(i, aniso=alpha)
        loss, l1, ssim = photometric(rendered["render"], gt, sa.lambda_dssim)
        terms = {"l1": l1, "ssim": ssim}
        llap = None
        if lam_lap > 0:
            L0_x, M = cotan_laplacian_torch(model.verts, model.faces_t)
            llap = l_lap(model.verts, model.x0, L0_x, M) / nvert0
            terms["l_lap"] = llap
            loss = loss + lam_lap * llap
        lnorm = None
        if lam_normal > 0:
            fn = face_normals_torch(model.verts, model.faces_t)
            w = face_adjacency_weights(
                model.verts.detach(), model.faces_t, model.pairs_t
            )
            lnorm = l_normal(fn, model.pairs_t, w, delta_huber) / nedge0
            terms["l_normal"] = lnorm
            loss = loss + lam_normal * lnorm
        ldist = None
        if lam_dist > 0:
            ldist = l_distortion(
                model.verts, model.faces_t, float(dv.distortion_eps_rel)
            )
            terms["l_distortion"] = ldist
            loss = loss + lam_dist * ldist
        lcurv = None
        if alpha * lam_curv > 0:
            face_id = face_id_from_depth(
                rendered["depth"], rendered["alpha"], model.cams[i], face_tree, FG_ALPHA
            )
            lcurv = l_curv(rendered["render"], gt, face_id, kappa_t)
            terms["l_curv"] = lcurv
            loss = loss + alpha * lam_curv * lcurv
        lf = None
        if alpha * lam4 > 0:
            lf = (
                l_field(
                    torch.cat([model.sh_dc, model.sh_rest], dim=1),
                    model.opacity_logit,
                    model.L2t,
                    mu_field,
                )
                / nfac0
            )
            terms["l_field"] = lf
            loss = loss + alpha * lam4 * lf
        prev = model.verts.detach().clone()
        try:
            assert_finite(step, **terms)
            loss.backward()
        except RuntimeError as e:  # params already poisoned: the repair below failed
            diverged = step
            print(f"step {step:6d}  DIVERGED: {e}")
            break
        repaired = repair_nonfinite_grads(
            verts=model.verts,
            sh_dc=model.sh_dc,
            sh_rest=model.sh_rest,
            opacity_logit=model.opacity_logit,
        )
        if repaired:
            # Zero only bad rows; consecutive repairs detect a genuinely stuck state.
            n_consecutive = n_consecutive + 1 if last_repaired == step - 1 else 1
            first_of_burst, last_repaired = n_consecutive == 1, step
            skipped.append(
                {
                    "step": step,
                    "view": i,
                    "repairs": [
                        {"param": name, "n": len(rows), "rows": rows.tolist()}
                        for name, rows in repaired
                    ],
                }
            )  # identities, not just the count
            for name, rows in repaired:
                print(
                    f"step {step:6d}  repaired: zeroed non-finite gradient rows on '{name}' "
                    f"({len(rows)}/{getattr(model, name).shape[0]}) {rows[:8].tolist()}"
                )
            if (
                first_of_burst
                and bool(dv.reassemble_on_skip)
                and not sched.warming_up(step)
            ):
                force_reassemble = (
                    True  # refresh the frozen operators around the bad pocket
                )
            if n_consecutive > int(dv.max_skipped_steps):
                diverged = step
                print(
                    f"step {step:6d}  DIVERGED: {n_consecutive} consecutive steps needed "
                    f"gradient repair - zeroing the rows is not clearing it, so the state is "
                    f"genuinely stuck, not transient"
                )
                break
        else:
            n_consecutive = 0
        if metric is not None:
            # Repair NaNs before the solve can spread them; Adam takes the filtered gradient.
            if metric.geometry_dependent and step % refactor_every == 0:
                metric = build_metric(
                    Mesh(
                        model.verts.detach().cpu().numpy(),
                        faces_np,
                        laplacian=cfg.dec.laplacian,
                        dual_type=cfg.dec.dual.type,
                        dual_clamp=cfg.dec.dual.clamp,
                    ),
                    dv,
                )
            metric.apply_(model.verts.grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
        completed_steps = step
        model.aniso = alpha
        step_disp = float((model.verts.detach() - prev).norm(dim=1).mean())

        ema = float(loss) if ema is None else 0.99 * ema + 0.01 * float(loss)
        if step % int(sa.log_every) == 0:
            disp = float((model.verts - model.x0).norm(dim=1).mean())
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/loss_ema", ema, step)
            writer.add_scalar("train/l1", float(l1), step)
            writer.add_scalar("train/ssim", float(ssim), step)
            writer.add_scalar("train/mean_disp", disp, step)
            writer.add_scalar("train/step_disp", step_disp, step)
            writer.add_scalar("sched/alpha", alpha, step)
            for name, t in (
                ("l_lap", llap),
                ("l_normal", lnorm),
                ("l_distortion", ldist),
                ("l_curv", lcurv),
                ("l_field", lf),
            ):
                if t is not None:
                    writer.add_scalar(f"train/{name}", float(t), step)

            def f0(t):
                return float(t) if t is not None else 0.0

            for k, v in (
                ("step", step),
                ("loss", float(loss)),
                ("loss_ema", ema),
                ("mean_disp", disp),
                ("step_disp", step_disp),
                ("alpha", alpha),
                ("l_lap", f0(llap)),
                ("l_normal", f0(lnorm)),
                ("l_curv", f0(lcurv)),
                ("l_field", f0(lf)),
                ("l_distortion", f0(ldist)),
            ):
                hist[k].append(v)
            mode = "warm " if sched.warming_up(step) else "ramp "
            print(
                f"step {step:6d}  {mode} loss {float(loss):.4f}  ema {ema:.4f}  "
                f"disp {disp:.2e}  sdisp {step_disp:.2e}  alpha {alpha:.2f}"
            )
        if step % int(sa.eval_every) == 0:
            val = model.eval_psnr("val")
            writer.add_scalar("val/psnr", val, step)
            print(f"step {step:6d}  val PSNR {val:.2f} dB")
        if checkpoint_every and step % checkpoint_every == 0:
            # Save intermediate meshes so interrupted runs still have inspectable geometry.
            if write_mesh(
                model.verts.detach().cpu().numpy(),
                faces_np,
                out / f"mesh_{step:06d}.ply",
            ):
                print(f"step {step:6d}  checkpoint -> mesh_{step:06d}.ply")

    V = model.verts.detach().cpu().numpy()
    write_mesh(V, faces_np, out / "mesh.ply")
    np.savez(
        out / "history.npz",
        ramp_start=(sched.ramp_start or 0),
        diverged=(diverged or 0),
        **{k: np.asarray(v) for k, v in hist.items()},
    )
    metrics = {
        "completed_steps": completed_steps,
        "requested_steps": int(sa.max_steps),
        "seed": seed,
        "commit": git_commit(),
        "warmup_disabled": no_warmup,
        "ramp_start": sched.ramp_start,
        "diverged": diverged,
        "n_skipped": len(skipped),
        "skipped": skipped,  # step, param, count, view and row IDs; expect ~0-1 per 1e4
        "metric": (
            None
            if metric is None
            else {
                "laplacian": metric.laplacian,
                "t": metric.t,
                "anisotropy": metric.anisotropy,
                "theta_c": metric.theta_c,
                "theta_c_percentile": metric.theta_c_percentile,
                "modulator": metric.modulator,
            }
        ),
        "lambda_lap": lam_lap,
        "lambda_normal": lam_normal,
        "lambda_curv": lam_curv,
        "lambda4": lam4,
        "lambda_distortion": lam_dist,
        "silhouette_iou": ious,
        "mean_disp": float(np.linalg.norm(V - model.x0.cpu().numpy(), axis=1).mean()),
        "train_psnr": model.eval_psnr("train") if diverged is None else float("nan"),
        "val_psnr": model.eval_psnr("val") if diverged is None else float("nan"),
        "test_psnr": model.eval_psnr("test") if diverged is None else float("nan"),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    save_state(model, out, completed_steps)
    OmegaConf.save(cfg, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    writer.close()
    if diverged is not None:
        raise RuntimeError(f"training diverged at step {diverged}; diagnostics saved to {out}")
    print(
        f"done: test PSNR {metrics['test_psnr']:.2f} dB, mean disp "
        f"{metrics['mean_disp']:.2e}, peak VRAM {metrics['peak_vram_mib']} MiB -> {out}"
    )
