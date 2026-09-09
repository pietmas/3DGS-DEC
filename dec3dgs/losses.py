"""Photometric loss and the ``L_field`` coupling (Dirichlet energy on the dual complex).

``L_field = sum_l c^(l)T L2 c^(l) + mu * oT L2 o`` over the 48 SH channels and the
opacity logit. Its Euclidean gradient is ``2 L2 c``; surface heat flow additionally applies
``M2^-1``, and Adam is not an Euler heat step. ``L2`` is a cached fp32 sparse copy.
"""

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure

from .mesh import gaussian_curvature, mean_curvature


def photometric(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float):
    """(3,H,W) pair in [0,1] -> (loss, l1, ssim); loss = (1-lambda)*L1 + lambda*(1-SSIM)."""
    l1 = (pred - gt).abs().mean()
    ssim = structural_similarity_index_measure(pred[None], gt[None], data_range=1.0)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim), l1, ssim


def l_field(
    sh: torch.Tensor, opacity_logit: torch.Tensor, L2t: torch.Tensor, mu: float
) -> torch.Tensor:
    """``sum_l c^(l)T L2 c^(l) + mu oT L2 o`` for SH ``(F,16,3)`` and logits ``(F,1)``."""
    X = sh.reshape(sh.shape[0], -1)
    return (X * (L2t @ X)).sum() + mu * (opacity_logit * (L2t @ opacity_logit)).sum()


def l_lap(
    verts: torch.Tensor, x0: torch.Tensor, L0_x: torch.Tensor, M: torch.Tensor
) -> torch.Tensor:
    """``||L0_x (x - x0)||^2_M`` - the M-norm of the Laplacian of the *displacement*
    ``x - x0``, a translation-invariant (**not** rotation-invariant) displacement fairing,
    **not** thin-plate. It charges roughness of how far each vertex has moved from the frozen
    init, so global translation (``delta = const`` is in ``ker L0``) is free while wrinkles
    cost; unlike ``||L0 x||^2`` it does not deflate the mesh toward its centroid. ``L0_x``,
    ``M`` are the in-graph plain-cotangent pair (``cotan_laplacian_torch``), so the fairing
    direction is recomputed from the live geometry each step.

    The matvec ``L0_x delta`` is a scatter over ``L0_x``'s COO triplets, **not**
    ``torch.sparse.mm``: the latter's backward materialises a dense ``(V, V)`` gradient of the
    sparse operand (~V^2, OOM at 1e5 faces), while ``index_add`` on the values keeps both
    passes O(nnz)."""
    delta = verts - x0
    idx, w = L0_x.indices(), L0_x.values()
    Ld = torch.zeros_like(delta).index_add(0, idx[0], w[:, None] * delta[idx[1]])
    return (M * (Ld**2).sum(-1)).sum()


def l_normal(
    face_normals: torch.Tensor,
    adjacency: torch.Tensor,
    w: torch.Tensor,
    delta_huber: float,
) -> torch.Tensor:
    """``sum_{i~j} w_ij psi(angle(n_i, n_j))`` - a **robust** Dirichlet energy of the face-normal
    field, the discrete counterpart of the total curvature ``int |grad n|^2``. ``psi`` is the
    Huber function: quadratic below the knee ``delta_huber`` (small disagreements = noise, faired
    strongly), linear above it (a large disagreement = a designed crease, penalised only to first
    order so its dihedral survives). The plain Dirichlet ``||n_i - n_j||^2`` (``psi = (.)^2``) is
    the ``delta_huber -> inf`` limit and erases creases; the knee is where "sharp" is defined.

    ``adjacency`` (E_int, 2) are face pairs sharing an edge. This is the extrinsic dihedral angle
    in ambient space, not Levi-Civita transport of tangent vectors. ``w`` (E_int,) are detached
    weights rebuilt each training step; only ``face_normals`` carries gradient to the vertices.

    The angle is ``atan2(||n_i x n_j||, n_i . n_j)`` rather than ``acos(n_i . n_j)``, whose
    derivative blows up at ``+-1`` (parallel/antiparallel normals). The cross-product norm is
    floored before the square root so a perfectly flat mesh (``n_i x n_j = 0``) gives a finite
    (zero) gradient instead of ``0/0``: the clamp saturates and kills the offending term, which is
    correct since ``psi'(0) = 0``."""
    ni, nj = face_normals[adjacency[:, 0]], face_normals[adjacency[:, 1]]
    tiny = torch.finfo(face_normals.dtype).tiny
    sin = torch.linalg.cross(ni, nj).pow(2).sum(-1).clamp_min(tiny).sqrt()
    theta = torch.atan2(sin, (ni * nj).sum(-1))
    psi = torch.where(
        theta <= delta_huber, 0.5 * theta**2, delta_huber * (theta - 0.5 * delta_huber)
    )
    return (w * psi).sum()


def l_distortion(
    verts: torch.Tensor, faces: torch.Tensor, eps_rel: float = 1.0e-6
) -> torch.Tensor:
    """``mean_f (E_conf(f) - 1)`` with ``E_conf = (a^2 + b^2 + c^2) / (4 sqrt3 A)`` - the conformal
    MIPS distortion of each triangle, and the objective's only term that charges *element shape*.

    ``E_conf >= 1`` with equality iff the triangle is equilateral (AM-GM on the three edge lengths
    against Heron), it is scale-invariant, and it **diverges as ``A -> 0``**: a face cannot be
    flattened to nothing without paying unboundedly for it. That is the point. Every other geometric
    term in this objective measures the *surface* - ``L_normal`` the normal field, the Sobolev metric
    the smoothness of the displacement - and a displacement field can be perfectly smooth and still
    shear triangles, because triangle shape is set by the field's **gradient**, not the field.
    Measured: smoothing the displacement takes the badly-conditioned face count from 967 to 1640. So nothing in the objective opposed degeneracy and the descent
    manufactured it continuously, until a face with no normal ended the run. With this term the
    sliver population becomes an equilibrium set by ``lambda_distortion`` against the photometric
    force - a trade one can tune - rather than a quantity that only ever grows.

    Normalised by the face **count**, deliberately, and this is the one place the count is right:
    ``E_conf`` is a dimensionless per-element shape statistic, not a density being integrated, so
    the honest summary is "the average element", each element counting once. Area-weighting would
    be exactly wrong - it would let the smallest slivers, the ones that matter, weigh nothing.

    ``eps_rel`` regularises the area as ``A_eps = sqrt(A^2 + (eps_rel * mean(a^2+b^2+c^2))^2)``
    rather than clamping it. Clamping fixes the forward and leaves the backward carrying
    ``1/A_clamped``, the same trap ``face_normals_torch`` was caught by; the regularised area has
    derivative ``A / A_eps <= 1`` everywhere, so the barrier stays finite in fp32 at exactly zero
    area while remaining ``A`` to nine digits on every healthy face. The reference scale is the mean
    **squared edge length** (an area, dimensionally) and not the mean area, because the mean area is
    exactly what vanishes in the degenerate limit this term exists to survive - a regulariser that
    can itself go to zero regularises nothing.

    It is therefore a **soft** barrier: a large enough photometric gradient can still push a
    triangle through, and the hard guarantee - a step limiter on the largest ``alpha`` keeping every
    face's area above ``beta A_f`` - is the named fallback, not this."""
    p = verts[faces]
    e = p[:, [1, 2, 0]] - p  # (F, 3, 3) the three edges
    ell2 = (e**2).sum(-1).sum(-1)  # a^2 + b^2 + c^2
    area = 0.5 * torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).norm(dim=1)
    eps = (eps_rel * ell2.mean()).detach()
    area_eps = torch.sqrt(area**2 + eps**2)
    return (ell2 / (4.0 * np.sqrt(3.0) * area_eps) - 1.0).mean()


def curv_weight(mesh, cfg) -> np.ndarray:
    """Per-face curvature density ``kappa = |H| h_bar + |K| h_bar^2``, (F,) - the dimensionless
    weight ``l_curv`` folds into the photometric residual so appearance capacity flows to where the
    surface bends. Dimensional analysis is the design: ``[H] = 1/length`` (mean curvature),
    ``[K] = 1/length^2`` (Gaussian), and the local edge length ``[h_bar] = length``, so both
    products are pure numbers and ``kappa`` is **invariant under a global rescale** ``x -> s x`` (a
    2x-printed copy gets the same map) - the reason ``h_bar`` sits *inside* the formula rather than
    as a global constant. ``H, K`` are the vertex curvature fields averaged onto each face (the value
    read at the centroid), ``h_bar`` the face's mean edge length; clamped above by ``cfg.curv_clip``
    so a near-degenerate sliver cannot dominate. Numpy on the detached current mesh, per-interval: a
    *weight* need not be differentiable in ``x`` - it modulates a residual whose gradient to the
    vertices already flows through the derivation."""
    H = np.abs(mean_curvature(mesh)[mesh.faces].mean(axis=1))
    K = np.abs(gaussian_curvature(mesh)[mesh.faces].mean(axis=1))
    v = mesh.verts[mesh.faces]  # (F, 3, 3)
    h_bar = np.linalg.norm(v[:, [1, 2, 0]] - v, axis=2).mean(
        axis=1
    )  # mean of the 3 edge lengths
    return np.minimum(H * h_bar + K * h_bar**2, cfg.curv_clip)


def l_curv(
    pred: torch.Tensor, gt: torch.Tensor, face_id: torch.Tensor, kappa: torch.Tensor
) -> torch.Tensor:
    """Curvature-weighted photometric residual: a ``kappa``-weighted mean of the per-pixel L1 error,
    so error at a fillet costs more than the same error on a flat panel. ``pred``, ``gt`` are
    (3, H, W) in [0, 1]; ``face_id`` (H, W) long is the face each pixel hits (``< 0`` at background,
    which drops out - no surface, no weight); ``kappa`` (F,) is the detached per-face density from
    ``curv_weight``. Normalised by the total weight, so a *constant* ``kappa`` reduces to the plain
    masked mean L1 - the reweighting redistributes capacity, it does not rescale the loss. Only
    ``pred`` carries gradient; ``kappa`` is a weight and ``face_id`` an index."""
    resid = (pred - gt).abs().mean(0)  # (H, W) per-pixel L1
    valid = face_id >= 0
    w = torch.where(valid, kappa[face_id.clamp_min(0)], torch.zeros_like(resid))
    return (w * resid).sum() / w.sum().clamp_min(torch.finfo(resid.dtype).tiny)


def tv_dc(sh_dc: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Total variation of the DC band over the dual edges, ``sum ||c_i - c_j||1`` - a
    smoothness diagnostic, reported never optimised."""
    d = sh_dc[pairs[:, 0], 0] - sh_dc[pairs[:, 1], 0]
    return d.abs().sum()
