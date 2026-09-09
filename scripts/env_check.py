"""Environment verification: imports, versions, GPU identity, CUDA + rasterizer smoke ops.

Exit 0 only if every dependency imports, the GTX 1050 is visible with capability
(6,1), and both a CUDA matmul and a diff-surfel-rasterization render actually run.

Note: gsplat is NOT in this stack - its backward kernels require compute >= 7.0
(Volta) and cannot compile for the Pascal GTX 1050. The renderer is the official
2DGS repo's diff-surfel-rasterization.
"""

import importlib
import importlib.metadata
import math
import sys

# import name -> distribution name (for version lookup)
DEPS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diff_surfel_rasterization": "diff_surfel_rasterization",
    "simple_knn": "simple-knn",
    "robust_laplacian": "robust_laplacian",
    "potpourri3d": "potpourri3d",
    "igl": "libigl",
    "open3d": "open3d",
    "trimesh": "trimesh",
    "point_cloud_utils": "point_cloud_utils",
    "lpips": "lpips",
    "torchmetrics": "torchmetrics",
    "skimage": "scikit-image",
    "omegaconf": "omegaconf",
    "tensorboard": "tensorboard",
    "polyscope": "polyscope",
    "scipy": "scipy",
    "numpy": "numpy",
    "pytest": "pytest",
    "plyfile": "plyfile",
    "cv2": "opencv-python",
}

failures = []


def check(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - collect everything, report at the end
        failures.append(f"{name}: {type(e).__name__}: {e}")


def surfel_render(torch):
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    N, H, W = 1000, 400, 400
    fov = math.radians(60.0)
    tanfov = math.tan(fov / 2)
    znear, zfar = 0.01, 100.0
    # perspective projection as in 2DGS graphics_utils.getProjectionMatrix
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = 1 / tanfov
    proj[1, 1] = 1 / tanfov
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    view = torch.eye(4, device="cuda")
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfov,
        tanfovy=tanfov,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view.T,
        projmatrix=(view.T @ proj.T),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    means3D = torch.randn(N, 3, device="cuda") * 0.3
    means3D[:, 2] += 3.0  # in front of the camera
    out = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D, requires_grad=True),
        shs=None,
        colors_precomp=torch.rand(N, 3, device="cuda"),
        opacities=torch.rand(N, 1, device="cuda"),
        scales=torch.rand(N, 2, device="cuda") * 0.05,  # 2D scales: surfels
        rotations=torch.nn.functional.normalize(
            torch.randn(N, 4, device="cuda"), dim=-1
        ),
        cov3D_precomp=None,
    )
    color = out[0]
    torch.cuda.synchronize()
    assert color.shape == (3, H, W), f"unexpected render shape {tuple(color.shape)}"
    print(f"diff-surfel render ok: {tuple(color.shape)} mean={float(color.mean()):.4f}")


def main():
    modules = {}
    for import_name in DEPS:
        check(
            f"import {import_name}",
            lambda n=import_name: modules.__setitem__(n, importlib.import_module(n)),
        )

    print(f"{'package':<28} version")
    print("-" * 40)
    for import_name, dist_name in DEPS.items():
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(modules.get(import_name), "__version__", "?")
        status = "" if import_name in modules else "  IMPORT FAILED"
        print(f"{dist_name:<28} {version}{status}")

    if "torch" in modules:
        torch = modules["torch"]

        def gpu_checks():
            assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"\nGPU: {name}  capability: {cap}")
            assert "1050" in name, f"expected the GTX 1050, got {name!r}"
            assert cap == (6, 1), f"expected capability (6,1), got {cap}"
            free, total = torch.cuda.mem_get_info()
            print(f"VRAM: {free / 2**20:.0f} MiB free / {total / 2**20:.0f} MiB total")

        check("GPU identity", gpu_checks)

        def cuda_matmul():
            x = torch.randn(1024, 1024, device="cuda") @ torch.randn(
                1024, 1024, device="cuda"
            )
            torch.cuda.synchronize()
            print(f"CUDA matmul ok: {float(x.sum()):.3f}")

        check("CUDA matmul", cuda_matmul)

        if "diff_surfel_rasterization" in modules:
            check("diff-surfel render", lambda: surfel_render(torch))

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
