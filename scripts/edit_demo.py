"""Edit-time geodesic diffusion - the colour-editing demonstration.

Editing through the reconstruction coupling propagates Euclidean, at every coupling strength: a
penalty is a preference the photometric term can outvote, so nothing forbids a colour field that
jumps a thin gap. Editing instead by a direct dual heat step ``(I + t M2^-1 L2)^-1``
(``dec3dgs.mesh.dual_heat_diffuse``) propagates geodesically, ``sqrt(t) = r`` the brush
radius. We paint a splat near a thin gap and compare the DEC brush to a Euclidean brush of
matched radius: leakage = brush mass reaching the across-gap set (Euclidean-near yet
geodesic-far), fixed on geometry before either brush is applied.

Run: ``python scripts/edit_demo.py --config configs/default.yaml [--seed S] [--out DIR]``
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import potpourri3d as pp3d
import torch
from omegaconf import OmegaConf
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dec3dgs.mesh import face_areas  # noqa: E402
from dec3dgs.render import render  # noqa: E402
from dec3dgs.model import Model  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C0 = 0.28209479177387814  # SH band-0 constant: rendered rgb = sh_dc * C0 + 0.5


def checkpoint_config(cfg):
    """Appearance must be rendered on the mesh and cameras that trained it."""
    checkpoint = REPO / cfg.edit.checkpoint
    stored = OmegaConf.load(checkpoint / "config.yaml")
    if "training" not in stored and "stage_a" in stored:
        stored.training = stored.stage_a
        stored.data.kind = "nerf_synthetic"  # the legacy fixed-mesh runs were Blender scenes
    if "training" not in stored:
        raise ValueError("appearance checkpoint has no training configuration")
    resolved = OmegaConf.merge(cfg, stored, {"edit": cfg.edit})
    mesh = REPO / resolved.training.mesh
    if not mesh.is_file():
        raise FileNotFoundError(f"checkpoint's initial mesh is missing: {mesh}")
    return resolved


def _git_commit() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return r.stdout.strip() or "no-git"


def _best_camera(cams, mu_s, normal_s):
    """The front-facing camera projecting the source nearest the image centre.
    Occlusion is left to the eye; a surface stud is almost always visible."""
    best, best_score = None, -np.inf
    for i in range(len(cams.c2w)):
        R, t = cams.c2w[i][:3, :3], cams.c2w[i][:3, 3]
        xc = R.T @ (mu_s - t)  # world -> OpenCV camera frame
        if xc[2] <= 0:
            continue
        facing = float(normal_s @ (t - mu_s)) / np.linalg.norm(t - mu_s)
        if facing <= 0:
            continue
        u = cams.K[i][0, 0] * xc[0] / xc[2] + cams.K[i][0, 2]
        v = cams.K[i][1, 1] * xc[1] / xc[2] + cams.K[i][1, 2]
        if not (0 <= u < cams.width and 0 <= v < cams.height):
            continue
        centred = 1.0 - np.hypot(u - cams.width / 2, v - cams.height / 2) / cams.width
        score = facing + centred
        if score > best_score:
            best, best_score = i, score
    return best


def _painted_dc(model, weight, paint_dc):
    """Alpha-blend DC toward ``paint_dc``, alpha = brush weight peak-normalised to 1."""
    a = torch.tensor(
        weight / weight.max(), dtype=torch.float32, device=model.mu.device
    )[:, None, None]
    paint = torch.tensor(paint_dc, dtype=torch.float32, device=model.mu.device)
    return (1.0 - a) * model.sh_dc.detach() + a * paint


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument("--seed", type=int, default=None)
    cli.add_argument("--out", default=None)
    args = cli.parse_args()
    cfg = OmegaConf.load(REPO / args.config)
    cfg = checkpoint_config(cfg)
    ed = cfg.edit
    seed = cfg.seed if args.seed is None else args.seed
    cfg.seed = seed
    rng = np.random.default_rng(seed)
    out = Path(args.out) if args.out else REPO / ed.out
    out.mkdir(parents=True, exist_ok=True)

    model = Model(cfg)
    st = torch.load(
        REPO / ed.checkpoint / "state.pt",
        map_location=model.mu.device,
        weights_only=True,
    )
    if "verts" in st:
        raise ValueError("edit_demo requires an appearance-only checkpoint on its original mesh")
    with torch.no_grad():
        model.sh_dc.copy_(st["sh_dc"].to(model.mu.device))
        model.sh_rest.copy_(st["sh_rest"].to(model.mu.device))
        model.opacity_logit.copy_(st["opacity_logit"].to(model.mu.device))

    mesh = model.mesh
    faces, verts = mesh.faces, mesh.verts
    mu = model.mu.cpu().numpy()  # splat positions
    fn = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]
    )
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    tree = cKDTree(mu)
    geo = pp3d.MeshHeatMethodDistanceSolver(verts, faces)

    def geo_from_face(face):
        distances = geo.compute_distance_multisource(faces[face].tolist())
        return distances[faces].mean(1)

    r = float(ed.radius)
    t = r * r
    c_geo = float(ed.across_geo_factor)
    paint_dc = (np.asarray(ed.paint_rgb, dtype=np.float64) - 0.5) / C0

    # t is fixed, so factorise (M2 + t L2) once and reuse across sources.
    area = face_areas(mesh)
    solve = factorized((mesh.M2 + t * mesh.L2).tocsc())
    n_nb = int(ed.n_neighbors)

    def brushes(s):
        """DEC and Euclidean brush weights at source ``s`` over its neighbourhood, plus
        the across-gap set (Euclidean-near, geodesic-far)."""
        nb = tree.query(mu[s], k=n_nb + 1)[1]
        d_geo_nb = geo_from_face(s)[nb]
        d_euc_nb = np.linalg.norm(mu[nb] - mu[s], axis=1)
        delta = np.zeros(len(faces))
        delta[s] = 1.0
        w_dec = solve(area * delta)[nb]
        w_euc = np.exp(-0.5 * (d_euc_nb / r) ** 2)
        across = (d_euc_nb < r) & (d_geo_nb > c_geo * r)
        return nb, d_geo_nb, d_euc_nb, w_dec, w_euc, across

    def leakage(w, across):
        return float(w[across].sum() / w.sum()) if w.sum() > 0 else 0.0

    # source screen (geometry only, seeded) + leakage table
    rows = []
    tried = 0
    while len(rows) < int(ed.n_sources) and tried < 3000:
        s = int(rng.integers(len(faces)))
        tried += 1
        nb, d_geo_nb, d_euc_nb, w_dec, w_euc, across = brushes(s)
        rho = spearmanr(d_geo_nb, d_euc_nb).statistic
        if abs(rho) >= float(ed.decouple_max) or across.sum() == 0:
            continue
        rows.append(
            {
                "source": s,
                "screen_rho": round(float(rho), 4),
                "across_faces": int(across.sum()),
                "leak_dec": round(leakage(w_dec, across), 4),
                "leak_euc": round(leakage(w_euc, across), 4),
            }
        )
    if not rows:
        raise RuntimeError("no faces passed the registered decoupling screen; no demonstration produced")
    # figure = the framed source with the strongest gap; the number is the median.
    framed = [
        (row["leak_euc"], row["source"])
        for row in rows
        if _best_camera(model.data.cameras, mu[row["source"]], fn[row["source"]])
        is not None
    ]
    if not framed:
        raise RuntimeError("no screened source is visible in a dataset camera")
    figure_src = max(framed)[1]

    # the money shot: original vs DEC-brushed vs Euclidean-brushed
    s = figure_src
    cam = _best_camera(model.data.cameras, mu[s], fn[s])
    nb, d_geo_nb, d_euc_nb, w_dec_nb, w_euc_nb, across = brushes(s)
    # full-mesh weights for rendering (the neighbourhood was only the leakage window)
    delta = np.zeros(len(faces))
    delta[s] = 1.0
    w_dec = solve(area * delta)
    d_euc_all = np.linalg.norm(mu - mu[s], axis=1)
    w_euc = np.exp(-0.5 * (d_euc_all / r) ** 2)

    def shot(weight):
        img = render(
            model.mu,
            model.quat,
            model.sigma,
            torch.cat([_painted_dc(model, weight, paint_dc), model.sh_rest], dim=1),
            model.opacity_logit,
            model.cams[cam],
            model.bg,
            cfg.render.sh_degree,
        )["render"]
        return img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    orig = (
        model.render_view(cam)["render"]
        .detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    dec_img, euc_img = shot(w_dec), shot(w_euc)

    _write_outputs(
        out,
        cfg,
        seed,
        rows,
        figure_src,
        orig,
        dec_img,
        euc_img,
        leakage(w_dec_nb, across),
        leakage(w_euc_nb, across),
    )


def _write_outputs(
    out, cfg, seed, rows, figure_src, orig, dec_img, euc_img, leak_dec, leak_euc
):
    with open(out / "leakage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    med_dec = float(np.median([row["leak_dec"] for row in rows]))
    med_euc = float(np.median([row["leak_euc"] for row in rows]))
    metrics = {
        "seed": seed,
        "commit": _git_commit(),
        "figure_source": int(figure_src),
        "n_sources": len(rows),
        "radius": float(cfg.edit.radius),
        "median_leak_dec": round(med_dec, 4),
        "median_leak_euc": round(med_euc, 4),
        "figure_leak_dec": round(leak_dec, 4),
        "figure_leak_euc": round(leak_euc, 4),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    OmegaConf.save(cfg, out / "config.yaml")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, title in zip(
        ax,
        (orig, dec_img, euc_img),
        (
            "original",
            f"DEC brush (geodesic)\nleakage {leak_dec:.1%}",
            f"Euclidean brush\nleakage {leak_euc:.1%}",
        ),
    ):
        a.imshow(img)
        a.set_title(title)
        a.axis("off")
    fig.suptitle(
        f"edit-time diffusion near a gap  (median over {len(rows)} sources: "
        f"DEC {med_dec:.1%} vs Euclidean {med_euc:.1%})"
    )
    fig.tight_layout()
    fig.savefig(out / "edit_propagation.png", dpi=130)

    print(
        f"figure source {figure_src}: leak_dec {leak_dec:.3f}  leak_euc {leak_euc:.3f}"
    )
    print(
        f"median over {len(rows)} sources: leak_dec {med_dec:.3f}  leak_euc {med_euc:.3f}"
    )
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
