"""The README panels: the photograph, the splat render, and the mesh itself, from one camera.

Every panel is drawn from the dataset's own ``(K, c2w)`` for the chosen view - OpenCV convention,
DTU in the IDR-normalised frame. The splat panel goes through the surfel rasterizer exactly as
training did; the mesh panel is an offscreen rasterisation of the **same vertices** the splats are
derived from, handed the same intrinsics and the same world-to-camera matrix. A second camera path
invented for the figure would make the comparison meaningless, so the agreement is measured rather
than asserted: ``render.project_points`` puts the mesh's vertices in pixel coordinates with our own
convention, and the independently rasterised silhouette has to sit on them.

The mesh panel exists to make one point: nothing is extracted afterwards. The surface *is* the
model. The photograph is shown as the loss saw it, background masked to white, since a mesh-bound
splat set has nothing to explain a table with.

Run: ``.venv/bin/python scripts/render_figure.py --run outputs/deform_scan65 --out docs/images``
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dec3dgs.model import REPO, Model, git_commit  # noqa: E402
from dec3dgs.checkpoint import load_state  # noqa: E402
from dec3dgs.render import project_points  # noqa: E402


class ShadedMesh:
    """Offscreen diffuse rasterisation of a triangle mesh in the dataset's camera.

    Open3D's Filament backend renders headless over EGL on this box; writing a z-buffer of our own
    would only re-derive what an existing dependency already does. ``setup_camera`` takes the
    intrinsics and the world-to-camera matrix in the OpenCV convention, which is what
    ``CameraSet`` stores, so no convention is translated on the way in. The sun is placed on the
    camera axis, so shading reads as relief rather than as a lighting choice."""

    def __init__(self, width, height, fg):
        self.width, self.height = int(width), int(height)
        self.r = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat.base_color = (*[float(c) for c in fg.mesh_rgb], 1.0)
        self.r.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.intensity = float(fg.light_intensity)

    def __call__(self, verts, faces, K, c2w):
        """-> (H,W,3) float image in [0,1] and its (H,W) bool silhouette."""
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        m.compute_vertex_normals()
        self.r.scene.clear_geometry()
        self.r.scene.add_geometry("mesh", m, self.mat)
        # the camera looks down +z, so the light travels with it
        self.r.scene.scene.set_sun_light(
            c2w[:3, 2].tolist(), [1.0, 1.0, 1.0], self.intensity
        )
        self.r.setup_camera(K, np.linalg.inv(c2w), self.width, self.height)
        rgb = np.asarray(self.r.render_to_image(), dtype=np.float32) / 255.0
        # normalised device depth: 1.0 is the far plane, i.e. nothing was drawn there
        depth = np.asarray(self.r.render_to_depth_image(), dtype=np.float32)
        return rgb, depth < 1.0


def alignment(silhouette, mask, K, c2w, verts):
    """Does the rasterised mesh sit where the camera says it should?

    Two independent readings. The IoU of the silhouette against the dataset mask says the geometry
    is in the photograph's place; the bounding-box drift compares that silhouette with our own
    forward projection of the vertices, so a convention error in either path shows up as pixels
    rather than as a picture that merely looks plausible. A closed surface projects to a set whose
    extent is its silhouette's, back-facing vertices included, so the two boxes must agree - once
    the projected box is cut to the image rectangle, because on this object the skull leaves the
    frame in most views and only the rasterizer knows to stop at the border."""
    h, w = silhouette.shape
    uv, z = project_points(K, c2w, verts)
    uv = uv[z > 0]
    ys, xs = np.nonzero(silhouette)
    if len(xs) == 0 or len(uv) == 0:
        return 0.0, float("inf")
    drawn = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    projected = np.clip(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        [0.0, 0.0, 0.0, 0.0],
        [w - 1.0, h - 1.0, w - 1.0, h - 1.0],
    )
    iou = float((silhouette & mask).sum() / (silhouette | mask).sum())
    return iou, float(np.abs(drawn - projected).max())


def panel_grid(rows, col_titles, suptitle, path, fg):
    """One figure: a row per view, a column per panel, no axes and no interpolation games."""
    w, h = (float(v) for v in fg.panel_size)
    fig, ax = plt.subplots(
        len(rows),
        len(col_titles),
        figsize=(w * len(col_titles), h * len(rows)),
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax[r][c].imshow(np.clip(img, 0.0, 1.0))
            ax[r][c].axis("off")
            if r == 0:
                ax[r][c].set_title(col_titles[c])
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=int(fg.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument("--run", default=None, help="run directory holding state.pt")
    cli.add_argument("--out", default=None, help="where the panels land")
    args = cli.parse_args()

    cfg = OmegaConf.load(REPO / args.config)
    fg = cfg.figures
    run = Path(args.run) if args.run else REPO / fg.run
    out = Path(args.out) if args.out else REPO / fg.out
    out.mkdir(parents=True, exist_ok=True)
    # Use the run's geometry settings and override only the figure options.
    cfg = OmegaConf.merge(OmegaConf.load(run / "config.yaml"), {"figures": fg})
    run_metrics = json.loads((run / "metrics.json").read_text())
    if run_metrics.get("diverged") is not None:
        raise ValueError("cannot present a diverged run as a completed reconstruction")

    model = Model(cfg, deform=True)
    step = load_state(model, run)
    cams = model.data.cameras
    faces = model.mesh.faces
    x0 = model.mesh.verts
    verts = model.verts.detach().cpu().numpy().astype(np.float64)

    train = model.data.split["train"]
    n = int(fg.n_views)
    if n <= 0 or len(train) == 0:
        raise ValueError("figures.n_views and the training view count must be positive")
    views = [int(i) for i in train[:: max(1, len(train) // n)][:n]]
    print(f"{model.data.name}: step {step}, views {views}")

    # Release the mesh renderer before splat rendering to keep GPU memory down.
    shade = ShadedMesh(cams.width, cams.height, fg)
    checks, surface, initial = [], [], []
    for i in views:
        rgb, sil = shade(verts, faces, cams.K[i], cams.c2w[i])
        surface.append(rgb)
        initial.append(shade(x0, faces, cams.K[i], cams.c2w[i])[0])
        iou, drift = alignment(sil, model.data.masks[i], cams.K[i], cams.c2w[i], verts)
        checks.append(
            {"view": i, "silhouette_iou": round(iou, 4), "drift_px": round(drift, 2)}
        )
        print(f"  view {i:3d}  silhouette IoU {iou:.3f}  bbox drift {drift:5.2f} px")
    del shade

    photo = [model.data.images[i] for i in views]
    with torch.no_grad():
        splat = [
            model.render_view(i)["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            for i in views
        ]

    worst = max(c["drift_px"] for c in checks)
    aligned = worst <= float(fg.align_tol_px)
    print(
        f"camera agreement: worst drift {worst:.2f} px "
        f"({'PASS' if aligned else 'FAIL'} at {float(fg.align_tol_px):g} px)"
    )

    scene = model.data.name
    for name, rows, titles, caption in (
        (
            "render_vs_gt",
            list(zip(photo, splat, surface)),
            ("photograph", "splats", "the mesh itself"),
            "one camera, three panels - nothing between the mesh and the render",
        ),
        (
            "mesh",
            [surface],
            [f"view {i}" for i in views],
            f"the reconstruction is the mesh, shaded - {len(faces)} faces, nothing extracted",
        ),
        (
            "x0_vs_deformed",
            list(zip(initial, surface)),
            ("x0", "deformed"),
            "the mesh the run started from, and where the photometric gradient moved it",
        ),
    ):
        panel_grid(rows, titles, f"{scene}: {caption}", out / f"{scene}_{name}.png", fg)

    (out / "figures.json").write_text(
        json.dumps(
            {
                "commit": git_commit(),
                "source_commit": run_metrics.get("commit"),
                "seed": run_metrics.get("seed"),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "run": str(run),
                "step": step,
                "views": views,
                "aligned": aligned,
                "align_tol_px": float(fg.align_tol_px),
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    if not aligned:
        raise RuntimeError(f"camera alignment failed: {worst:.2f} px exceeds {fg.align_tol_px}")


if __name__ == "__main__":
    main()
