"""Frozen mesh: 2DGS checkpoint -> closed oriented 2-manifold.

Pipeline on `lego`: TSDF-fuse the checkpoint's depth renders, crop to the splat-center
bbox (kills depth-dust spikes), close the unobserved underside with screened Poisson,
crop the Poisson balloon, cap the cut loops with centroid fans, drop-and-cap residual
non-manifold/misoriented edges, then decimate to budget with ``igl.qslim``. The result is
gated through ``dec3dgs.mesh.Mesh`` (manifoldness + orientation), which asserts zero
boundary edges, positive areas, and signed volume > 0; genus is logged, not asserted.

Usage:
    python scripts/extract_mesh.py --config configs/default.yaml --seed 0 [--view]
"""

import argparse
import json
import random
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import igl
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
GS2D = REPO / "baselines" / "2d-gaussian-splatting"
sys.path.insert(0, str(REPO))

from dec3dgs.datasets import dtu_renormalise, load_dtu_scale_mat  # noqa: E402
from dec3dgs.mesh import Mesh, face_areas  # noqa: E402


# fusion
def fuse_tsdf(cfg):
    """TSDF-fuse the checkpoint's depth renders -> raw (V, F), scene name, crop box,
    resolved TSDF knobs."""
    sys.path.insert(0, str(GS2D))
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel, render
    from scene import Scene
    from utils.mesh_utils import GaussianExtractor

    ckpt = REPO / cfg.mesh_extract.checkpoint
    parser = ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    # cfg_args is the repo's own record of the training invocation; replay it verbatim.
    stored = eval((ckpt / "cfg_args").read_text(), {"Namespace": Namespace})
    for k, v in vars(stored).items():
        setattr(args, k, v)
    dataset, pipe = lp.extract(args), pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    extractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg)
    gaussians.active_sh_degree = 0  # geometry only; SH bands beyond DC are irrelevant
    extractor.reconstruction(scene.getTrainCameras())

    me = cfg.mesh_extract
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    c, h = (xyz.min(0) + xyz.max(0)) / 2, (xyz.max(0) - xyz.min(0)) / 2
    box = (c - me.crop_expand * h, c + me.crop_expand * h)
    opac = gaussians.get_opacity.detach().cpu().numpy()[:, 0]
    support = xyz[opac >= me.support_opacity_min]
    depth_trunc = 2.0 * extractor.radius if me.depth_trunc < 0 else me.depth_trunc
    voxel = depth_trunc / me.mesh_res if me.voxel_size < 0 else me.voxel_size
    sdf_trunc = 5.0 * voxel if me.sdf_trunc < 0 else me.sdf_trunc
    fused = extractor.extract_mesh_bounded(
        voxel_size=voxel, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
    )
    knobs = {
        "voxel_size": voxel,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "crop_box": np.array(box).tolist(),
        "n_support": len(support),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    return fused, Path(dataset.source_path), box, support, knobs


# combinatorial helpers
def edge_stats(F):
    """Undirected edges with their face count and orientation signsum."""
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    sign = np.where(he[:, 0] < he[:, 1], 1, -1)
    edges, inv = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(edges))
    ss = np.bincount(inv, weights=sign, minlength=len(edges))
    return edges, inv, cnt, ss


def n_defects(F):
    _, _, cnt, ss = edge_stats(F)
    return int((cnt == 1).sum() + (cnt > 2).sum() + ((cnt == 2) & (ss != 0)).sum())


def compact(V, F):
    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3).astype(np.int32)


def largest_component(V, F):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)
    )
    labels, counts, _ = m.cluster_connected_triangles()
    return compact(V, F[np.asarray(labels) == int(np.argmax(counts))])


def crop(V, F, box):
    lo, hi = box
    outside = ((V < lo) | (V > hi)).any(axis=1)
    return compact(V, F[~outside[F].any(axis=1)])


def trim_to_support(V, F, support, dist):
    """Drop faces farther than ``dist`` from every confident splat center - the pointwise
    support criterion, which removes the Poisson balloon between object and crop planes."""
    from scipy.spatial import cKDTree

    far = cKDTree(support).query(V, workers=-1)[0] > dist
    return compact(V, F[~far[F].any(axis=1)])


# repair
def drop_bad_faces(F, widen):
    """Remove faces touching non-manifold or misoriented edges; ``widen`` cuts the whole
    vertex 1-ring, to break configurations a fan cap would otherwise recreate."""
    edges, inv, cnt, ss = edge_stats(F)
    bad_edge = (cnt > 2) | ((cnt == 2) & (ss != 0))
    if widen:
        bad_vert = np.zeros(F.max() + 1, dtype=bool)
        bad_vert[edges[bad_edge]] = True
        return F[~bad_vert[F].any(axis=1)]
    return F[~bad_edge[inv].reshape(-1, 3).any(axis=1)]


def fan_fill(V, F):
    """Cap every boundary loop with a centroid fan: for boundary halfedge a->b the cap
    triangle (c, b, a) opposes the face's a->b, so the capped edge becomes interior."""
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    heset = set(map(tuple, he))
    bnd = [(a, b) for (a, b) in heset if (b, a) not in heset]
    if not bnd:
        return V, F, 0
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(a, []).append(b)
    unused = set(bnd)
    Vs, newF, n_caps = [V], [], 0
    for e0 in list(bnd):
        if e0 not in unused:
            continue
        loop, (a, b) = [], e0
        while (a, b) in unused:
            unused.discard((a, b))
            loop.append(a)
            a, b = b, next((h for h in nxt[b] if (b, h) in unused), nxt[b][0])
        n_caps += 1
        c = len(V) + len(Vs) - 1
        Vs.append(V[loop].mean(axis=0, keepdims=True))
        for i, a in enumerate(loop):
            newF.append((c, loop[(i + 1) % len(loop)], a))
    return np.vstack(Vs), np.vstack([F, np.array(newF, dtype=F.dtype)]), n_caps


def make_closed(V, F, max_rounds):
    """Drop-and-cap until no boundary / non-manifold / misoriented edges remain."""
    prev = None
    for it in range(max_rounds):
        F = drop_bad_faces(F, widen=(prev is not None and it > 1))
        V, F = largest_component(V, F)
        V, F, n = fan_fill(V, F)
        bad = n_defects(F)
        print(f"  repair round {it}: +{n} caps, {bad} defective edges left")
        if bad == 0:
            return V, F
        prev = bad
    raise RuntimeError(f"repair did not converge in {max_rounds} rounds")


# validation
def validate(V, F, cfg):
    """The extraction gate: Mesh constructor + closedness/orientation asserts + stats."""
    me = cfg.mesh_extract
    mesh = Mesh(
        V,
        F,
        laplacian=cfg.dec.laplacian,
        dual_type=cfg.dec.dual.type,
        dual_clamp=cfg.dec.dual.clamp,
    )
    assert mesh.boundary_edges.size == 0, "boundary edges survived repair"
    assert (face_areas(mesh) > 0).all(), "zero-area face survived repair"
    vol = (
        float(np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]]))) / 6.0
    )
    assert vol > 0, "signed volume <= 0: normals point inward"
    rel = abs(len(F) - me.target_faces) / me.target_faces
    assert rel <= me.target_faces_tol, (
        f"face count {len(F)} misses target {me.target_faces} by {rel:.1%}"
    )

    # Corner angles: at corner c, the two edges leaving it span the angle.
    p = V[F]
    u, w = p[:, [1, 2, 0]] - p, p[:, [2, 0, 1]] - p
    cosang = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    hist, bin_edges = np.histogram(ang.min(axis=1), bins=np.arange(0.0, 65.0, 5.0))
    chi = mesh.euler_characteristic
    nm_verts = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)
    ).get_non_manifold_vertices()
    return {
        "n_verts": len(V),
        "n_faces": len(F),
        "euler_characteristic": chi,
        "genus": (2 - chi) // 2 if len(nm_verts) == 0 else None,
        "signed_volume": vol,
        "obtuse_fraction": float((ang.max(axis=1) > 90.0).mean()),
        "min_angle_deg": float(ang.min()),
        "min_angle_hist": {
            "bin_edges_deg": bin_edges.tolist(),
            "counts": hist.tolist(),
        },
        "nonmanifold_vertices": len(nm_verts),  # pinch points: legal edges, log loudly
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="configs/default.yaml")
    cli.add_argument("--seed", type=int, default=None)
    cli.add_argument("--out", default=None, help="output mesh path; defaults to mesh_extract.out")
    cli.add_argument(
        "--view",
        action="store_true",
        help="polyscope eyeball pass on the validated mesh",
    )
    cli_args = cli.parse_args()

    cfg = OmegaConf.load(REPO / cli_args.config)
    seed = cfg.seed if cli_args.seed is None else cli_args.seed
    cfg.seed = seed
    path = Path(cli_args.out) if cli_args.out else REPO / cfg.mesh_extract.out
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"mesh artifacts already exist at {path}; choose a new --out")
    cfg.mesh_extract.out = str(path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    me = cfg.mesh_extract
    if me.method not in ("tsdf", "poisson"):
        raise ValueError(f"mesh_extract.method must be tsdf|poisson, got {me.method!r}")

    fused, source, box, support, knobs = fuse_tsdf(cfg)
    scene = source.name
    fused.remove_duplicated_vertices()
    fused.remove_duplicated_triangles()
    fused.remove_degenerate_triangles()
    fused.remove_unreferenced_vertices()

    V = np.asarray(fused.vertices, dtype=np.float64)
    F = np.asarray(fused.triangles, dtype=np.int32)
    if me.method == "poisson":
        # Poisson consumes the cropped surface samples, not the mesh
        fused.compute_vertex_normals()
        inside = ((V >= box[0]) & (V <= box[1])).all(axis=1)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[inside]))
        pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(fused.vertex_normals)[inside]
        )
        poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(me.poisson_depth)
        )
        V = np.asarray(poisson.vertices, dtype=np.float64)
        F = np.asarray(poisson.triangles, dtype=np.int32)
    V, F = crop(V, F, box)  # Poisson balloon cut here too
    V, F = trim_to_support(V, F, support, me.support_dist)
    V, F = largest_component(V, F)
    print(f"{me.method}+cropped+trimmed: V={len(V)} F={len(F)}")

    V, F = make_closed(V, F, int(me.repair_max_rounds))
    U, G, _, _ = igl.qslim(V, F, int(me.target_faces))
    V, F = compact(U, G.astype(np.int32))
    V, F = make_closed(V, F, int(me.repair_max_rounds))  # qslim can chip an edge
    if np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) < 0:
        F = np.ascontiguousarray(F[:, ::-1])  # consistent but inward: flip globally

    if me.dtu_renormalise:
        # Convert the checkpoint frame to the camera and Chamfer frame once.
        V = dtu_renormalise(
            V,
            load_dtu_scale_mat(source.parent, scene),
            load_dtu_scale_mat(REPO / cfg.data.dtu_root, scene),
        )

    stats = validate(V, F, cfg)
    print(
        f"validated: F={stats['n_faces']} chi={stats['euler_characteristic']} "
        f"genus={stats['genus']} vol={stats['signed_volume']:.3f} "
        f"obtuse={stats['obtuse_fraction']:.3f} "
        f"min_angle={stats['min_angle_deg']:.2f}deg "
        f"nm_verts={stats['nonmanifold_vertices']}"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    written = o3d.io.write_triangle_mesh(
        str(path),
        o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)
        ),
    )
    if not written:
        raise OSError(f"could not write mesh to {path}")
    commit = (
        subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "no-git"
    )
    sidecar = {
        "commit": commit,
        "seed": seed,
        "scene": scene,
        "method": str(me.method),
        "tsdf": knobs,
        "stats": stats,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {path} (+ sidecar)")

    if cli_args.view:
        import polyscope as ps

        ps.init()
        ps.register_surface_mesh(f"{scene}_mesh", V, F)
        ps.show()


if __name__ == "__main__":
    main()
