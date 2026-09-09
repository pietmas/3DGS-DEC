"""The trained state: geometry, the cochains defined over it, and the cameras that see it.

One splat per face throughout. With ``deform=False`` the mesh is frozen and ``(mu, quat, sigma)``
are constants derived once; with ``deform=True`` the vertices are the trainable geometry and the
splats are re-derived from them at every render, so the photometric gradient reaches the surface.

Also holds the per-interval reassembly: the operators that are *detached* by design (the robust
``L2`` coupling, the curvature weight, the circumcenter lookup) and are rebuilt on a cadence rather
than differentiated through.
"""

import math
import subprocess
from pathlib import Path

import igl
import numpy as np
import torch
from scipy.spatial import cKDTree

from .datasets import load_dtu, load_nerf_synthetic
from .losses import curv_weight
from .mesh import Mesh, circumcenters
from .metric import SobolevMetric
from .render import render, to_raster_camera
from .splats import (
    derive_splats,
    derive_splats_torch,
    dual_edges,
    interior_face_pairs,
    pull_into_face,
    transported_reference_frame,
)

REPO = Path(__file__).resolve().parent.parent

FG_ALPHA = 0.5  # Foreground cutoff for face lookup, matching the silhouette check.


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(((a - b) ** 2).mean()))


def assert_finite(step: int, **terms):
    """The per-step NaN hook: name the offending term, don't just crash later."""
    for name, t in terms.items():
        if not torch.isfinite(t).all():
            raise RuntimeError(f"non-finite loss term '{name}' at step {step}")


def git_commit() -> str:
    """The commit a run was executed at, recorded beside its metrics."""
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return r.stdout.strip() or "no-git"


class Model:
    """Geometry + trainable cochains + cameras, wired once. With ``deform=True`` the vertices
    become the trainable geometry and the splats are re-derived from them each render; with
    ``deform=False`` the mesh is frozen and the splats are constants."""

    def __init__(self, cfg, device="cuda", deform=False):
        self.cfg = cfg
        self.deform = deform
        self.aniso = 1.0
        sa = cfg.training
        white = bool(cfg.data.white_background)
        if cfg.data.kind == "nerf_synthetic":
            self.data = load_nerf_synthetic(
                REPO / cfg.data.nerf_synthetic_root, sa.scene, white
            )
        elif cfg.data.kind == "dtu":
            self.data = load_dtu(REPO / cfg.data.dtu_root, sa.scene)
            # Mask the background; mesh-bound splats can only explain the object.
            self.data.images = np.where(
                self.data.masks[..., None], self.data.images, 1.0 if white else 0.0
            ).astype(np.float32)
        else:
            raise ValueError(
                f"data.kind must be nerf_synthetic|dtu, got {cfg.data.kind!r}"
            )
        V, F = igl.read_triangle_mesh(str(REPO / sa.mesh))
        mesh = Mesh(
            V,
            F,
            laplacian=cfg.dec.laplacian,
            dual_type=cfg.dec.dual.type,
            dual_clamp=cfg.dec.dual.clamp,
        )
        self.mesh = mesh
        self.faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=device)

        def const(a):
            return torch.tensor(a, dtype=torch.float32, device=device)

        if deform:
            # Keep x0 fixed and cache the connectivity shared by every update.
            self.x0 = torch.tensor(mesh.verts, dtype=torch.float32, device=device)
            self.verts = self.x0.clone().requires_grad_(True)
            self.pairs_t = interior_face_pairs(self.faces_t)
            self.frame_mode = "gram_schmidt"  # warmup default; the schedule flips it
            self.ref_e1 = None  # transported blend reference, built on first use
        else:
            sp = derive_splats(mesh, cfg.splats)
            self.mu, self.quat, self.sigma = (
                const(sp.mu),
                const(sp.quat_wxyz),
                const(sp.sigma),
            )
        # L2 crosses the torch boundary exactly here, once
        L2 = mesh.L2.tocsr()
        self.L2t = torch.sparse_csr_tensor(
            L2.indptr,
            L2.indices,
            L2.data.astype(np.float32),
            size=L2.shape,
            device=device,
        )
        self.pairs = torch.tensor(dual_edges(mesh), dtype=torch.long, device=device)

        n = len(mesh.faces)  # one splat per face, both modes
        self.sh_dc = torch.zeros((n, 1, 3), device=device, requires_grad=True)
        self.sh_rest = torch.zeros((n, 15, 3), device=device, requires_grad=True)
        logit = math.log(sa.opacity_init / (1.0 - sa.opacity_init))
        self.opacity_logit = torch.full(
            (n, 1), logit, device=device, requires_grad=True
        )

        cams = self.data.cameras
        self.cams = [
            to_raster_camera(
                cams.K[i],
                cams.c2w[i],
                cams.width,
                cams.height,
                cfg.render.znear,
                cfg.render.zfar,
                device,
            )
            for i in range(len(cams.c2w))
        ]
        self.bg = const([1.0] * 3 if cfg.data.white_background else [0.0] * 3)

    def reassemble_reference(self):
        """Rebuild the transported reference frame for the umbilic blend on the detached current
        mesh - the same per-interval, frozen-within-an-interval cadence as the robust ``L2``
        coupling."""
        ref = transported_reference_frame(
            self.verts.detach().cpu().numpy(), self.mesh.faces
        )
        self.ref_e1 = torch.tensor(ref, dtype=torch.float32, device=self.faces_t.device)

    def splats(self, aniso: float = 1.0):
        """The (mu, quat, sigma) fed to the rasterizer: re-derived from the live vertices
        in deform mode (grad flows to the mesh), frozen constants otherwise. ``aniso`` in
        ``[0, 1]`` is the anisotropy ramp: at ``1`` the derived tangential scales are
        used verbatim, at ``0`` the splat is round (both scales = the per-face mean), so during
        the warmup the render is independent of the (then arbitrary) in-plane frame orientation
        and the orientation can be switched on continuously as ``aniso`` rises from 0."""
        if self.deform:
            if self.frame_mode == "shape_operator" and self.ref_e1 is None:
                self.reassemble_reference()  # first-use build; the loop reassembles per interval
            st = derive_splats_torch(
                self.verts,
                self.faces_t,
                self.cfg.splats,
                self.frame_mode,
                self.pairs_t,
                self.ref_e1,
                float(self.cfg.deform.eigengap_clip),
            )
            sigma = st.sigma
            if aniso < 1.0:
                iso = sigma.mean(
                    dim=1, keepdim=True
                )  # round splat of the same mean extent
                sigma = aniso * sigma + (1.0 - aniso) * iso
            return st.mu, st.quat_wxyz, sigma
        return self.mu, self.quat, self.sigma

    def render_view(self, i: int, opacity_logit=None, aniso: float | None = None) -> dict:
        aniso = self.aniso if aniso is None else aniso
        mu, quat, sigma = self.splats(aniso)
        return render(
            mu,
            quat,
            sigma,
            torch.cat([self.sh_dc, self.sh_rest], dim=1),
            self.opacity_logit if opacity_logit is None else opacity_logit,
            self.cams[i],
            self.bg,
            self.cfg.render.sh_degree,
        )

    def gt(self, i: int) -> torch.Tensor:
        return (
            torch.from_numpy(self.data.images[i])
            .permute(2, 0, 1)
            .to(self.faces_t.device)
        )

    def silhouette_iou(self, n_views: int = 4) -> list[float]:
        """Rendered alpha (the model as initialised) vs dataset mask, evenly spaced
        train views - the placement gate that runs before training."""
        train = self.data.split["train"]
        ious = []
        with torch.no_grad():
            for i in train[:: max(1, len(train) // n_views)][:n_views]:
                pred = self.render_view(int(i))["alpha"][0] > 0.5
                mask = torch.from_numpy(self.data.masks[i]).to(pred.device)
                ious.append(float((pred & mask).sum() / (pred | mask).sum()))
        return ious

    def eval_psnr(self, split: str) -> float:
        idx = self.data.split[split]
        if len(idx) == 0:
            return float(
                "nan"
            )  # DTU holds nothing out; its geometry protocol trains on all 49
        with torch.no_grad():
            return float(
                np.mean(
                    [psnr(self.render_view(int(i))["render"], self.gt(i)) for i in idx]
                )
            )


def reassemble_from_mesh(mesh: Mesh, cfg, device):
    """The detached, per-interval quantities read off a **given** mesh (robust /
    intrinsic-Delaunay, no gradient, frozen within a reassembly interval). Returns the robust ``L2``
    coupling (``L_field``), the per-face curvature weight ``kappa`` and a ``cKDTree`` over the
    circumcenters (``L_curv``'s face-ID lookup - ``mu_k`` *is* the circumcenter of face ``k``)."""
    L2 = mesh.L2.tocsr()
    L2t = torch.sparse_csr_tensor(
        L2.indptr, L2.indices, L2.data.astype(np.float32), size=L2.shape, device=device
    )
    kappa = torch.tensor(
        curv_weight(mesh, cfg.deform), dtype=torch.float32, device=device
    )
    # Use face centroids for non-finite dual points before building the lookup tree.
    mu = pull_into_face(mesh, circumcenters(mesh, on_degenerate="centroid"))
    finite = np.isfinite(mu).all(axis=1)
    if not finite.all():
        mu[~finite] = mesh.verts[mesh.faces[~finite]].mean(axis=1)
    tree = cKDTree(mu)
    return L2t, kappa, tree


def reassemble(verts: torch.Tensor, faces_np, cfg, device):
    """Reassemble on the *current* deforming vertices (fixed topology). Builds the ``Mesh`` from the
    live vertices and the frozen ``faces_np``, then defers to :func:`reassemble_from_mesh`."""
    mesh = Mesh(
        verts.detach().cpu().numpy(),
        faces_np,
        laplacian=cfg.dec.laplacian,
        dual_type=cfg.dec.dual.type,
        dual_clamp=cfg.dec.dual.clamp,
    )
    return reassemble_from_mesh(mesh, cfg, device)


def build_metric(mesh: Mesh, dv) -> SobolevMetric:
    """The Sobolev metric factored on ``mesh``. At module scope so the deform loop and any
    edit-time rebuild share one construction: a rebuild refactorises ``(M + t L_D)`` on the
    **current** mesh, never by reusing a frozen earlier factor."""
    theta_c = None if dv.metric.theta_c is None else float(dv.metric.theta_c)
    pct = (
        None
        if dv.metric.theta_c_percentile is None
        else float(dv.metric.theta_c_percentile)
    )
    return SobolevMetric(
        mesh,
        float(dv.metric.t),
        dv.metric.laplacian,
        dv.metric.anisotropy,
        theta_c,
        dv.metric.modulator,
        pct,
    )
