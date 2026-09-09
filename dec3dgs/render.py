"""Rendering interface wrapping the 2DGS surfel rasterization.

The torch boundary starts here (fp32). Cameras are stored OpenCV ``c2w`` and converted to
the Inria rasterizer's transposed matrices in ``to_raster_camera`` alone (validated by the
``__main__`` parity check). A splat is a flat surfel (``scales`` tangential, wxyz quaternion
of ``[e1 e2 n]``); opacity crosses as its logit, sigmoid applied at the rasterizer call.
``diff_surfel_rasterization`` is imported lazily so the CPU pytest suite never touches CUDA.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RasterCamera:
    """One view, in the rasterizer's own vocabulary (all tensors fp32, on device)."""

    width: int
    height: int
    tanfovx: float
    tanfovy: float
    world_view_transform: torch.Tensor  # (4,4) w2c, transposed
    full_proj_transform: torch.Tensor  # (4,4) (P @ w2c), transposed
    campos: torch.Tensor  # (3,) camera center, world


def _projection(
    znear: float, zfar: float, tanfovx: float, tanfovy: float
) -> np.ndarray:
    """The repo's ``getProjectionMatrix``: perspective, z in [0,1], +z forward."""
    P = np.zeros((4, 4))
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def to_raster_camera(
    K: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    znear: float,
    zfar: float,
    device,
) -> RasterCamera:
    """OpenCV ``(K, c2w)`` -> rasterizer camera. Assumes a centered principal point
    (true for Blender; the FoV-only rasterizer interface cannot express otherwise)."""
    assert abs(K[0, 2] - width / 2) < 1.0 and abs(K[1, 2] - height / 2) < 1.0, (
        "principal point not centered: the FoV camera model cannot represent it"
    )
    tanfovx = width / (2.0 * K[0, 0])
    tanfovy = height / (2.0 * K[1, 1])
    w2c = np.linalg.inv(c2w)
    full = _projection(znear, zfar, tanfovx, tanfovy) @ w2c

    def f32(a):
        return torch.tensor(a.T, dtype=torch.float32, device=device)

    return RasterCamera(
        width=int(width),
        height=int(height),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        world_view_transform=f32(w2c),
        full_proj_transform=f32(full),
        campos=torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device),
    )


def render(
    mu: torch.Tensor,
    quat_wxyz: torch.Tensor,
    sigma: torch.Tensor,
    sh: torch.Tensor,
    opacity_logit: torch.Tensor,
    cam: RasterCamera,
    bg_color: torch.Tensor,
    sh_degree: int,
) -> dict:
    """One forward pass: (N,3) centers, (N,4) wxyz quats, (N,2) scales, (N,16,3) SH,
    (N,1) opacity logits -> ``render`` (3,H,W) and ``alpha`` (1,H,W)."""
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    settings = GaussianRasterizationSettings(
        image_height=cam.height,
        image_width=cam.width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=sh_degree,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    image, radii, allmap = rasterizer(
        means3D=mu,
        means2D=torch.zeros_like(mu),
        shs=sh,
        colors_precomp=None,
        opacities=torch.sigmoid(opacity_logit),
        scales=sigma,
        rotations=quat_wxyz,
        cov3D_precomp=None,
    )
    return {"render": image, "alpha": allmap[1:2], "depth": allmap[5:6], "radii": radii}


def project_points(
    K: np.ndarray, c2w: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points ``(N,3)`` -> pixel coordinates ``(N,2)`` and camera depth ``(N,)``.

    The forward map of the stored convention, and the inverse of
    :func:`unproject_median_depth`. OpenCV ``c2w`` carries the camera axes as its columns, so
    ``x_cam = R^T (x - t)``, and the pinhole reads ``(u, v)^T = K_2 (x, y)^T / z + (c_x, c_y)^T``
    with +y **down**: a point on the camera's +y side lands below the principal point. ``K`` is used
    verbatim, skew and off-centre principal point included - unlike the rasterizer's FoV camera,
    which cannot express either. Depth is signed, so a point behind the camera returns ``z < 0``
    rather than a plausible pixel."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    xc = (np.asarray(X, dtype=np.float64) - t) @ R
    z = xc[:, 2]
    uv = (xc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z


def unproject_median_depth(depth: torch.Tensor, cam: RasterCamera) -> torch.Tensor:
    """Median-depth map (1, H, W) -> world points (H, W, 3). Reuses the 2DGS ray recipe
    (``utils/point_utils.depths_to_points``): recover the intrinsics from the projection, turn each
    pixel into a camera ray, scale it by the per-pixel depth, map to world by ``c2w``. Our
    ``RasterCamera`` stores the same transposed ``w2c``/full-proj matrices the 2DGS camera does, so
    the recipe carries over verbatim (only the hard-coded ``cuda`` device is replaced by the map's)."""
    device = depth.device
    W, H = cam.width, cam.height
    c2w = torch.linalg.inv(cam.world_view_transform.T)
    ndc2pix = torch.tensor(
        [[W / 2, 0, 0, W / 2], [0, H / 2, 0, H / 2], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).T
    intrins = ((c2w.T @ cam.full_proj_transform) @ ndc2pix)[:3, :3].T
    gx, gy = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    pix = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    rays_d = pix @ torch.linalg.inv(intrins).T @ c2w[:3, :3].T
    pts = depth.reshape(-1, 1) * rays_d + c2w[:3, 3]
    return pts.reshape(H, W, 3)


def face_id_from_depth(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    cam: RasterCamera,
    face_tree,
    alpha_thresh: float,
) -> torch.Tensor:
    """Per-pixel face index (H, W) long, ``-1`` at background - the fallback for the primitive-index
    buffer the pinned ``diff-surfel-rasterization`` does not expose, and which we will not rebuild
    the extension to add. Unproject the median depth and label each foreground pixel (``alpha > alpha_thresh``)
    by its nearest splat centre, since ``mu_k`` *is* the circumcenter of face ``k``. ``face_tree`` is
    a ``scipy.spatial.cKDTree`` over ``mu`` (detached, rebuilt per reassembly interval by the caller,
    like the other robust quantities), so this call is a per-view query, not a per-view rebuild."""
    pts = unproject_median_depth(depth, cam)
    fg = alpha[0] > alpha_thresh
    face_id = pts.new_full(fg.shape, -1, dtype=torch.long)
    if fg.any():
        _, idx = face_tree.query(pts[fg].detach().cpu().numpy())
        face_id[fg] = torch.as_tensor(idx, dtype=torch.long, device=pts.device)
    return face_id


def load_2dgs_ply(path) -> dict:
    """A 2DGS checkpoint ``.ply`` -> numpy fp32 dict. Undoes storage quirks: scales are
    logs (-> exp), ``f_rest`` is channel-major ``(N,3,15)`` (-> ``(N,15,3)``), quaternions
    unnormalised wxyz."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    n = v.count

    def grab(prefix):
        names = sorted(
            (prop.name for prop in v.properties if prop.name.startswith(prefix)),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        return np.stack([v[name] for name in names], axis=1)

    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)[:, None, :]
    rest = grab("f_rest_").reshape(n, 3, -1).transpose(0, 2, 1)
    quat = grab("rot_")
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {
        "mu": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32),
        "sh": np.concatenate([dc, rest], axis=1).astype(np.float32),
        "opacity_logit": np.asarray(v["opacity"], dtype=np.float32)[:, None],
        "sigma": np.exp(grab("scale_")).astype(np.float32),
        "quat_wxyz": quat.astype(np.float32),
    }


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(((a - b) ** 2).mean()))


def _parity_main():
    """Parity check: our wrapper vs the repo pipeline on 3 test views, PSNR between them
    above ``render.parity_min_psnr`` on every view."""
    import argparse
    import sys
    from argparse import ArgumentParser, Namespace

    from omegaconf import OmegaConf

    from .datasets import load_nerf_synthetic

    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    args = cli.parse_args()
    repo = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(repo / args.config)

    sys.path.insert(0, str(repo / "baselines" / "2d-gaussian-splatting"))
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel, render as repo_render
    from scene import Scene

    ckpt = repo / cfg.mesh_extract.checkpoint
    parser = ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    ns = parser.parse_args([])
    stored = eval((ckpt / "cfg_args").read_text(), {"Namespace": Namespace})
    for k, val in vars(stored).items():
        setattr(ns, k, val)
    dataset, pipe = lp.extract(ns), pp.extract(ns)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    test_cams = scene.getTestCameras()

    data = load_nerf_synthetic(
        repo / cfg.data.nerf_synthetic_root,
        Path(dataset.source_path).name,
        cfg.data.white_background,
    )
    ply = (
        max(
            (ckpt / "point_cloud").glob("iteration_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[1]),
        )
        / "point_cloud.ply"
    )
    sp = {k: torch.tensor(a, device="cuda") for k, a in load_2dgs_ply(ply).items()}
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0] * 3,
        dtype=torch.float32,
        device="cuda",
    )

    views = [0, len(test_cams) // 2, len(test_cams) - 1]
    ok = True
    with torch.no_grad():
        for i in views:
            j = data.split["test"][i]
            cam = to_raster_camera(
                data.cameras.K[j],
                data.cameras.c2w[j],
                data.cameras.width,
                data.cameras.height,
                cfg.render.znear,
                cfg.render.zfar,
                "cuda",
            )
            ours = render(
                sp["mu"],
                sp["quat_wxyz"],
                sp["sigma"],
                sp["sh"],
                sp["opacity_logit"],
                cam,
                bg,
                cfg.render.sh_degree,
            )["render"]
            theirs = repo_render(test_cams[i], gaussians, pipe, bg)["render"]
            gt = torch.from_numpy(data.images[j]).permute(2, 0, 1).cuda()
            p = _psnr(ours, theirs)
            ok &= p >= cfg.render.parity_min_psnr
            print(
                f"view {test_cams[i].image_name}: ours-vs-repo {p:.2f} dB "
                f"(gt alignment {_psnr(gt, test_cams[i].original_image.cuda()):.2f} dB)"
            )
    print("parity PASS" if ok else "parity FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _parity_main()
