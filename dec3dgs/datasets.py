"""Data loaders for NeRF-synthetic (Blender) and DTU (IDR-preprocessed) objects.

Cameras are stored as OpenCV/COLMAP ``c2w`` (+x right, +y down, +z forward): Blender's
OpenGL matrices are flipped with ``c2w[:, 1:3] *= -1``, DTU cameras returned in the IDR
normalized frame (object inside the unit sphere). Blender RGBA is composited onto a white
background when ``white_background`` (matching 2DGS/3DGS eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_BLENDER_LAYOUT = """\
expected NeRF-synthetic layout under {root}:
  {scene}/
    transforms_train.json  transforms_val.json  transforms_test.json
    train/r_*.png  val/r_*.png  test/r_*.png   (800x800 RGBA)"""

_DTU_LAYOUT = """\
expected DTU (IDR-preprocessed) layout under {root}:
  {scan}/
    image/*.png   (1600x1200)
    mask/*.png
    cameras.npz   (world_mat_<i>, scale_mat_<i>)"""

_DTU_GT_LAYOUT = """\
expected official DTU layout under {root}:
  Points/stl/stl{scan_id:03d}_total.ply"""


@dataclass
class CameraSet:
    K: np.ndarray  # (N,3,3) float64 intrinsics
    c2w: np.ndarray  # (N,4,4) float64 camera-to-world, OpenCV convention
    width: int
    height: int


@dataclass
class ObjectData:
    name: str
    images: np.ndarray  # (N,H,W,3) float32 in [0,1], background composited
    masks: np.ndarray  # (N,H,W) bool
    cameras: CameraSet
    split: dict  # {"train": idx_array, "val": ..., "test": ...}


def _read_image(path: Path) -> np.ndarray:
    """PNG -> float32 array in [0,1], shape (H,W,C)."""
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def load_nerf_synthetic(root: Path, scene: str, white_background: bool) -> ObjectData:
    root = Path(root)
    scene_dir = root / scene
    splits = ("train", "val", "test")
    for s in splits:
        if not (scene_dir / f"transforms_{s}.json").is_file():
            raise FileNotFoundError(
                f"missing {scene_dir / f'transforms_{s}.json'}\n"
                + _BLENDER_LAYOUT.format(root=root, scene=scene)
            )

    images, masks, c2ws = [], [], []
    split_idx: dict[str, np.ndarray] = {}
    camera_angle_x = None
    for s in splits:
        meta = json.loads((scene_dir / f"transforms_{s}.json").read_text())
        camera_angle_x = float(meta["camera_angle_x"])
        start = len(images)
        for frame in meta["frames"]:
            img_path = scene_dir / (frame["file_path"].lstrip("./") + ".png")
            if not img_path.is_file():
                raise FileNotFoundError(
                    f"missing {img_path}\n"
                    + _BLENDER_LAYOUT.format(root=root, scene=scene)
                )
            rgba = _read_image(img_path)
            assert rgba.ndim == 3 and rgba.shape[2] == 4, (
                f"{img_path}: expected RGBA, got shape {rgba.shape}"
            )
            alpha = rgba[..., 3:4]
            bg = 1.0 if white_background else 0.0
            images.append(rgba[..., :3] * alpha + bg * (1.0 - alpha))
            masks.append(alpha[..., 0] > 0)

            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            c2w[:3, 1:3] *= -1.0  # OpenGL -> OpenCV
            c2ws.append(c2w)
        split_idx[s] = np.arange(start, len(images))

    images = np.stack(images).astype(np.float32)
    masks = np.stack(masks)
    c2w = np.stack(c2ws)
    n, h, w = images.shape[:3]

    fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:] = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])

    return ObjectData(
        name=scene,
        images=images,
        masks=masks,
        cameras=CameraSet(K=K, c2w=c2w, width=w, height=h),
        split=split_idx,
    )


def _decompose_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P (3,4) = K [R|t] -> (K (3,3), c2w (4,4)), K with positive diagonal."""
    import scipy.linalg

    K, R = scipy.linalg.rq(P[:3, :3])
    # fix RQ sign ambiguity so diag(K) > 0 and det(R) = +1
    signs = np.sign(np.diag(K))
    K = K * signs[None, :]
    R = R * signs[:, None]
    if np.linalg.det(R) < 0:
        K, R = -K, -R
    t = np.linalg.solve(K, P[:3, 3])
    K = K / K[2, 2]

    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return K, c2w


def load_dtu_scale_mat(root: Path, scan: str) -> np.ndarray:
    """The (4,4) scale_mat mapping IDR-normalized (unit sphere) coords to raw DTU world."""
    cam_path = Path(root) / scan / "cameras.npz"
    if not cam_path.is_file():
        raise FileNotFoundError(
            f"missing {cam_path}\n" + _DTU_LAYOUT.format(root=root, scan=scan)
        )
    return np.load(cam_path)["scale_mat_0"].astype(np.float64)


def dtu_renormalise(V: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Re-express points from one DTU normalisation in another, given the two scale_mats.

    A scale_mat is the similarity carrying normalised coords to raw DTU world (mm), so two
    releases calibrated alike but normalised to different spheres differ by ``dst^-1 . src``
    on their normalised frames. The 2DGS COLMAP release and the IDR release are exactly that
    pair: byte-identical ``world_mat``, different ``scale_mat``. Both maps are uniform scale
    plus translation, so the composition is exact:

        x_dst = (s_src . x_src + t_src - t_dst) / s_dst
    """
    s_src, t_src = float(src[0, 0]), src[:3, 3]
    s_dst, t_dst = float(dst[0, 0]), dst[:3, 3]
    return (np.asarray(V, dtype=np.float64) * s_src + t_src - t_dst) / s_dst


def _centre_principal_point(images, masks, K, h, w):
    """Crop the largest window centred on the principal point, and shift ``K`` to match.

    DTU is calibrated with an off-centre principal point (scan65: (823.2, 619.1) in 1600x1200).
    The pinhole model of every splat rasterizer is an FoV pair, which puts the principal point at
    the image centre *by construction* and cannot represent the offset - ``to_raster_camera``
    asserts on it rather than render a silently shifted image. Cropping symmetrically about the
    principal point is the standard reconciliation, and it is not our invention: it reproduces the
    2DGS DTU release **bit-exactly** (scan65 -> 1554x1162, cropped 46 left / 38 top; verified
    against their images at mean |diff| = 0).

    The camera is unchanged - a crop only renames pixel coordinates, so the rays are the same and
    the frame is untouched. Only the image window shrinks, by 3 % here.
    """
    cx, cy = float(K[:, 0, 2].mean()), float(K[:, 1, 2].mean())
    w2, h2 = int(round(2 * min(cx, w - cx))), int(round(2 * min(cy, h - cy)))
    x0, y0 = int(round(cx - w2 / 2)), int(round(cy - h2 / 2))
    images = images[:, y0 : y0 + h2, x0 : x0 + w2]
    masks = masks[:, y0 : y0 + h2, x0 : x0 + w2]
    K = K.copy()
    K[:, 0, 2] -= x0
    K[:, 1, 2] -= y0
    return images, masks, K, (h2, w2)


def load_dtu(root: Path, scan: str) -> ObjectData:
    root = Path(root)
    scan_dir = root / scan
    cam_path = scan_dir / "cameras.npz"
    img_dir, mask_dir = scan_dir / "image", scan_dir / "mask"
    if not (cam_path.is_file() and img_dir.is_dir() and mask_dir.is_dir()):
        raise FileNotFoundError(
            f"missing pieces under {scan_dir}\n"
            + _DTU_LAYOUT.format(root=root, scan=scan)
        )

    cams = np.load(cam_path)
    n = sum(1 for k in cams.files if k.startswith("world_mat_") and "inv" not in k)
    img_paths = sorted(img_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    assert len(img_paths) == len(mask_paths) == n, (
        f"{scan_dir}: {len(img_paths)} images, {len(mask_paths)} masks, "
        f"{n} world_mat_* entries - must all match"
    )

    images, masks = [], []
    for ip, mp in zip(img_paths, mask_paths):
        images.append(_read_image(ip)[..., :3])
        m = _read_image(mp)
        masks.append((m if m.ndim == 2 else m[..., 0]) > 0.5)
    images = np.stack(images).astype(np.float32)
    masks = np.stack(masks)
    h, w = images.shape[1:3]

    K = np.zeros((n, 3, 3), dtype=np.float64)
    c2w = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        P = (cams[f"world_mat_{i}"] @ cams[f"scale_mat_{i}"])[:3, :4]
        K[i], c2w[i] = _decompose_P(P)

    images, masks, K, (h, w) = _centre_principal_point(images, masks, K, h, w)

    idx = np.arange(n)
    return ObjectData(
        name=scan,
        images=images,
        masks=masks,
        cameras=CameraSet(K=K, c2w=c2w, width=w, height=h),
        split={"train": idx, "val": idx[:0], "test": idx[:0]},
    )


def load_dtu_gt_points(official_root: Path, scan_id: int) -> np.ndarray:
    """Official GT reference scan as (M,3) float64, RAW DTU world coordinates (mm)."""
    ply_path = Path(official_root) / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(
            f"missing {ply_path}\n"
            + _DTU_GT_LAYOUT.format(root=official_root, scan_id=scan_id)
        )
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    return np.asarray(pcd.points, dtype=np.float64)
