"""Splat parameters derived from the mesh: position, frame, tangential scale.

A splat is a decorated dual vertex, everything read off the mesh, nothing optimised.
The numpy path (``derive_splats``) is the frozen-mesh reference; the torch mirror
(``derive_splats_torch``) is the same geometry re-implemented so that photometric gradients
reach the vertex positions when the mesh itself is being optimised.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .mesh import Mesh, VectorTransport, circumcenters, face_areas, vertex_normals


@dataclass(frozen=True)
class SplatParams:
    """Everything derived, frozen, float64."""

    mu: np.ndarray  # (F, 3) circumcenters
    frame: np.ndarray  # (F, 3, 3) columns [e1 e2 n]
    sigma: np.ndarray  # (F, 2) tangential scales, clamped
    quat_wxyz: np.ndarray  # (F, 4) frame as rasterizer quaternion, wxyz order


def _face_bases(mesh: Mesh):
    """Gram-Schmidt basis per face: t1 = first edge, n = normal, t2 = n x t1."""
    p = mesh.verts[mesh.faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t1 = p[:, 1] - p[:, 0]
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n, t1), n


def pull_into_face(mesh: Mesh, mu: np.ndarray) -> np.ndarray:
    """Pull an escaped (obtuse) circumcenter along the segment to the barycenter until
    it re-enters the face; inside faces are untouched, bitwise."""
    p = mesh.verts[mesh.faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # 2A * unit normal
    b = np.stack(
        [
            np.einsum(
                "ij,ij->i", np.cross(p[:, (k + 1) % 3] - mu, p[:, (k + 2) % 3] - mu), n
            )
            for k in range(3)
        ],
        axis=1,
    ) / (n * n).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):  # the discarded b >= 0 branch may hit 1/0
        t = np.where(b < 0, 1.0 / (1.0 - 3.0 * b), 1.0).min(axis=1)
    g = p.mean(axis=1)
    return np.where(t[:, None] == 1.0, mu, g + t[:, None] * (mu - g))


def face_shape_operators(mesh: Mesh) -> np.ndarray:
    """Per-face symmetric shape operator (F, 2, 2) in the Gram-Schmidt basis: solve
    ``dn = S de`` in least squares over the three edges (sign makes convex positive)."""
    t1, t2, _ = _face_bases(mesh)
    nv = vertex_normals(mesh)
    p = mesh.verts[mesh.faces]
    e = p[:, [1, 2, 0]] - p  # (F, 3, 3) edges 0->1, 1->2, 2->0
    dn = nv[mesh.faces][:, [1, 2, 0]] - nv[mesh.faces]
    u = np.stack(
        [np.einsum("fej,fj->fe", e, t1), np.einsum("fej,fj->fe", e, t2)], axis=-1
    )  # (F, 3, 2)
    d = np.stack(
        [np.einsum("fej,fj->fe", dn, t1), np.einsum("fej,fj->fe", dn, t2)], axis=-1
    )

    F = len(mesh.faces)
    A = np.zeros((F, 6, 3))
    A[:, 0::2, 0] = u[..., 0]  # row 2m:   u_x a + u_y b       = d_x
    A[:, 0::2, 1] = u[..., 1]
    A[:, 1::2, 1] = u[..., 0]  # row 2m+1:       u_x b + u_y c = d_y
    A[:, 1::2, 2] = u[..., 1]
    abc = (np.linalg.pinv(A) @ d.reshape(F, 6, 1))[..., 0]
    S = np.empty((F, 2, 2))
    S[:, 0, 0], S[:, 0, 1] = abc[:, 0], abc[:, 1]
    S[:, 1, 0], S[:, 1, 1] = abc[:, 1], abc[:, 2]
    return S


def dual_edges(mesh: Mesh) -> np.ndarray:
    """Interior primal edges as face pairs (E_int, 2) - the dual 1-skeleton graph."""
    coo = mesh.d1.tocoo()  # rows = faces, cols = edges
    order = np.argsort(coo.col, kind="stable")
    fs, es = coo.row[order], coo.col[order]
    interior = (np.bincount(es, minlength=mesh.d1.shape[1]) == 2)[es]
    return fs[interior].reshape(-1, 2)


def dual_cell_metric(mesh: Mesh, mu: np.ndarray) -> np.ndarray:
    """``M_k = sum ell ell^T`` over the dual edges leaving ``mu_k``, (F, 3, 3)
    (rank-deficient next to a boundary, where scales are not claimed)."""
    pairs = dual_edges(mesh)
    ell = mu[pairs[:, 1]] - mu[pairs[:, 0]]
    outer = ell[:, :, None] * ell[:, None, :]
    M = np.zeros((len(mesh.faces), 3, 3))
    np.add.at(M, pairs[:, 0], outer)
    np.add.at(M, pairs[:, 1], outer)
    return M


def tangential_scales(M: np.ndarray, frame: np.ndarray, gamma: float) -> np.ndarray:
    """``sigma_i = gamma * sqrt(e_i^T M e_i)`` along the two tangential frame columns -
    the along-frame extent of the dual cell, not an eigenvalue of M."""
    e = frame[..., :2]  # (F, 3, 2)
    q = np.einsum("fia,fij,fja->fa", e, M, e)
    return gamma * np.sqrt(np.maximum(q, 0.0))


def clamp_sigma(sigma, h_f, cfg):
    """Cap the tangential scales at ``sigma_max_rel`` times a reference length - the sliver/fan-cap
    guard, so a face whose dual cell is degenerate does not get a splat sized by that degeneracy.
    Works on numpy arrays and on torch tensors alike (only ``median``/``minimum`` are used, and both
    libraries spell them the same), which is what keeps the two derivation paths one rule.

    ``cfg.sigma_ref`` picks the reference. ``median`` - the mesh-wide median ``h``, the default and
    the right rule while element size is roughly uniform, as it is on a mesh whose connectivity never
    moves. ``local`` - the face's own equilateral-equivalent edge ``h_f = sqrt(4A/sqrt3)``, which
    makes the ceiling scale-local so one region's resolution cannot set another's coverage; it is
    *stricter* exactly where the guard is meant to bite (a sliver's ``h_f`` has collapsed with its
    area) and looser on a legitimately coarse face. Kept for a mesh with a genuine size gradient."""
    ref = str(cfg.sigma_ref)
    if ref == "median":
        # torch.median chooses the lower middle element; numpy averages the middle pair.
        h = h_f.quantile(0.5) if torch.is_tensor(h_f) else np.median(h_f)
    elif ref == "local":
        h = h_f[:, None]
    else:
        raise ValueError(f"splats.sigma_ref must be median|local, got {ref!r}")
    return (
        sigma.minimum(cfg.sigma_max_rel * h)
        if hasattr(sigma, "minimum")
        else np.minimum(sigma, cfg.sigma_max_rel * h)
    )


def derive_splats(mesh: Mesh, cfg) -> SplatParams:
    """Forward derivation mesh -> splats; ``cfg`` is the ``splats:`` config node."""
    mu = pull_into_face(mesh, circumcenters(mesh))
    t1, t2, n = _face_bases(mesh)

    S = face_shape_operators(mesh)
    w, vec = np.linalg.eigh(S)  # ascending eigenvalues
    order = np.argsort(-np.abs(w), axis=1)  # e1 = larger |kappa|
    w = np.take_along_axis(w, order, axis=1)
    vec = np.take_along_axis(vec, order[:, None, :], axis=2)
    e1 = vec[:, 0, 0, None] * t1 + vec[:, 1, 0, None] * t2
    e1 *= np.where(np.einsum("ij,ij->i", e1, t1) >= 0, 1.0, -1.0)[:, None]
    umbilic = np.abs(w[:, 0] - w[:, 1]) < cfg.eigengap_min
    e1[umbilic] = t1[umbilic]  # deterministic umbilic fallback
    frame = np.stack([e1, np.cross(n, e1), n], axis=2)

    sigma = tangential_scales(dual_cell_metric(mesh, mu), frame, cfg.gamma)
    sigma = clamp_sigma(sigma, np.sqrt(4.0 * face_areas(mesh) / np.sqrt(3.0)), cfg)

    quat = Rotation.from_matrix(frame).as_quat()  # scipy returns xyzw
    return SplatParams(
        mu=mu,
        frame=frame,
        sigma=sigma,
        quat_wxyz=np.ascontiguousarray(quat[:, [3, 0, 1, 2]]),
    )


# Torch carries gradients to vertices; NumPy is the numerical refrence.


@dataclass(frozen=True)
class SplatTensors:
    """The derivation's torch output, all grad-tracking in the live vertices."""

    mu: torch.Tensor  # (F, 3) means3D (circumcenters, pulled into the face)
    frame: torch.Tensor  # (F, 3, 3) columns [e1 e2 n]
    sigma: torch.Tensor  # (F, 2) tangential scales, clamped
    quat_wxyz: torch.Tensor  # (F, 4) frame as rasterizer quaternion, wxyz order


def interior_face_pairs(faces) -> torch.Tensor:
    """Interior edges as face pairs ``(E_int, 2)`` - the dual 1-skeleton, the torch
    counterpart of ``dual_edges``. Pure topology (constant in training), so it is computed
    once in numpy and returned as a long tensor; pair order is irrelevant downstream (the
    dual-cell metric sums a sign-independent outer product into both faces)."""
    fn = faces.detach().cpu().numpy().astype(np.int64)
    F = len(fn)
    he = fn[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inv = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    fs, es = np.repeat(np.arange(F), 3)[order], inv[order]
    interior = (np.bincount(es, minlength=inv.max() + 1) == 2)[es]
    return torch.as_tensor(fs[interior].reshape(-1, 2), device=faces.device)


def face_cross_torch(verts, faces):
    """The unnormalised area vector ``(v1-v0) x (v2-v0)`` per face (F, 3), norm ``2A_f``.

    Split out of :func:`face_normals_torch` so a degeneracy check reads the *same* expression the
    loss divides by: "this face has no normal" is exactly ``|cross| == 0`` in the parameter's own
    dtype, and the face area comes off the same tensor for free."""
    p = verts[faces]
    return torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def face_normals_torch(verts, faces):
    """Unit face normals (F, 3), differentiable in the vertices - the same normal that is the
    third frame column, computed on its own so ``l_normal`` need not re-run the derivation.

    A degenerate face has no normal, and dividing by its zero norm gives ``0/0``. The honest answer
    is the **zero vector** - there is no direction - which ``l_normal`` then reads at
    ``atan2(tiny, 0) = pi/2`` and penalises, rather than propagating a NaN through the whole sum.
    An unguarded division here previously ended two long runs at step ~2800.

    The denominator is **replaced**, not clamped, and the difference is the whole point. Clamping to
    ``tiny`` fixes only the forward: the backward carries ``d(fn/n)/dfn = 1/n``, so a clamped
    denominator still hands back ``1/tiny ~ 8e37`` and overflows to ``inf`` on the way down. Putting
    a plain ``1`` there for a face with no normal makes the gradient of that face's (identically
    zero) normal finite and its contribution vanish, which is what a face with no direction should
    contribute."""
    fn = face_cross_torch(verts, faces)
    norm = fn.norm(dim=1, keepdim=True)
    return fn / torch.where(norm > 0.0, norm, torch.ones_like(norm))


def face_adjacency_weights(verts, faces, pairs):
    """Dirichlet weight per face-adjacency edge (E_int,): ``|e| / |c_f - c_g|`` with ``c`` the
    face centroids and ``e`` the shared primal edge - the barycentric dual conductance, positive
    by construction. These are the ``l_normal`` edge weights; detached, per-interval (the mesh
    moves slowly), like the robust coupling operator - only the face normals carry gradient."""
    centroids = verts[faces].mean(dim=1)
    dual_len = (centroids[pairs[:, 0]] - centroids[pairs[:, 1]]).norm(dim=1)
    ff, gg = faces[pairs[:, 0]], faces[pairs[:, 1]]
    shared = ff[(ff[:, :, None] == gg[:, None, :]).any(dim=2)].reshape(
        -1, 2
    )  # 2 common verts
    edge_len = (verts[shared[:, 0]] - verts[shared[:, 1]]).norm(dim=1)
    return edge_len / dual_len


def _vertex_normals_torch(verts, faces):
    """Area-weighted unit vertex normals (V, 3) - torch mirror of ``vertex_normals``."""
    p = verts[faces]
    fn = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # area-weighted
    n = torch.zeros_like(verts).index_add(
        0, faces.reshape(-1), fn.repeat_interleave(3, dim=0)
    )
    norm = n.norm(dim=1, keepdim=True)
    return n / torch.where(norm > 0, norm, torch.ones_like(norm))


def face_frame_torch(verts, faces):
    """Per-face ``(fn, n, t1, t2)``: the area-weighted normal (its half-norm is the area), the
    unit normal, the first-edge direction, and their cross - the Gram-Schmidt basis every
    downstream derivation starts from."""
    p = verts[faces]
    fn = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n = fn / fn.norm(dim=1, keepdim=True)
    t1 = p[:, 1] - p[:, 0]
    t1 = t1 / t1.norm(dim=1, keepdim=True)
    return fn, n, t1, torch.linalg.cross(n, t1)


def shape_operator_system(verts, faces, t1, t2):
    """The per-face least-squares system ``A (a,b,c)^T = d`` behind the shape operator: ``A``
    (F, 6, 3) from the three in-plane edge vectors, ``d`` (F, 6, 1) from the matching normal
    differences, two rows per edge. Split out so ``A``'s conditioning can be censused on its own:
    the SVD backward behind ``pinv`` carries ``1/(s_i^2 - s_j^2)``, so a sliver hands back an
    enormous gradient while its forward value stays finite."""
    nv = _vertex_normals_torch(verts, faces)
    p = verts[faces]
    e = p[:, [1, 2, 0]] - p  # (F, 3, 3) edges
    dn = nv[faces][:, [1, 2, 0]] - nv[faces]
    u = torch.stack([(e * t1[:, None]).sum(-1), (e * t2[:, None]).sum(-1)], dim=-1)
    d = torch.stack([(dn * t1[:, None]).sum(-1), (dn * t2[:, None]).sum(-1)], dim=-1)

    F = faces.shape[0]
    A = torch.zeros((F, 6, 3), dtype=verts.dtype, device=verts.device)
    A[:, 0::2, 0] = u[..., 0]  # row 2m:   u_x a + u_y b       = d_x
    A[:, 0::2, 1] = u[..., 1]
    A[:, 1::2, 1] = u[..., 0]  # row 2m+1:       u_x b + u_y c = d_y
    A[:, 1::2, 2] = u[..., 1]
    return A, d.reshape(F, 6, 1)


def _shape_operators_torch(verts, faces, t1, t2):
    """Per-face symmetric shape operator (F, 2, 2) in the (t1, t2) basis - torch mirror of
    ``face_shape_operators``, solving ``dn = S de`` over the three edges via ``pinv`` (the
    same min-norm least squares as the numpy path, differentiable through the SVD)."""
    A, d = shape_operator_system(verts, faces, t1, t2)
    abc = (torch.linalg.pinv(A) @ d)[..., 0]
    return torch.stack(
        [
            torch.stack([abc[:, 0], abc[:, 1]], -1),
            torch.stack([abc[:, 1], abc[:, 2]], -1),
        ],
        dim=1,
    )


def _circumcenters_torch(p):
    """Face circumcenters (F, 3) from the corner triple ``p`` (F, 3, 3) - torch mirror of
    ``circumcenters`` (barycentric form, obtuse centres allowed, pulled in afterwards)."""
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    a2 = ((b - c) ** 2).sum(-1)
    b2 = ((c - a) ** 2).sum(-1)
    c2 = ((a - b) ** 2).sum(-1)
    bary = torch.stack(
        [a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)], dim=1
    )
    denom = bary.sum(dim=1, keepdim=True)
    # Guard the denominator before division; masking NaNs afterwards breaks autograd.
    good = torch.isfinite(denom) & (denom.abs() > torch.finfo(p.dtype).tiny)
    safe = torch.where(good, denom, torch.ones_like(denom))
    cc = ((bary / safe)[:, :, None] * p).sum(dim=1)
    return torch.where(good, cc, p.mean(dim=1))


def _pull_into_face_torch(p, mu):
    """Pull an obtuse circumcenter along the segment to the barycentre until it re-enters
    the face - torch mirror of ``pull_into_face``. Inside faces give ``t = 1`` identically
    (all barycentric coords > 0), so the result is ``mu`` with the correct derivative; the
    ``t < 1`` denominator ``1 - 3b`` is guarded so the masked branch carries no NaN grad."""
    n = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # 2A * unit normal
    b = torch.stack(
        [
            (
                torch.linalg.cross(p[:, (k + 1) % 3] - mu, p[:, (k + 2) % 3] - mu) * n
            ).sum(-1)
            for k in range(3)
        ],
        dim=1,
    ) / (n * n).sum(dim=1, keepdim=True)
    neg = b < 0
    denom = torch.where(neg, 1.0 - 3.0 * b, torch.ones_like(b))  # >= 1 where used
    t = torch.where(neg, 1.0 / denom, torch.ones_like(b)).min(dim=1).values
    g = p.mean(dim=1)
    return g + t[:, None] * (mu - g)


def _tangential_scales_torch(M, frame, gamma):
    """``sigma_a = gamma sqrt(e_a^T M e_a)`` along the two tangential frame columns - torch
    mirror of ``tangential_scales``."""
    e = frame[..., :2]  # (F, 3, 2)
    q = torch.einsum("fia,fij,fja->fa", e, M, e)
    return gamma * _sqrt_positive_part(q)


def _sqrt_positive_part(x):
    """``sqrt(x)`` where ``x > 0`` and 0 elsewhere, but with a *zero* backward on the
    non-positive part - a plain ``sqrt(clamp(x, 0))`` has an infinite gradient at 0, which
    the unselected quaternion branches below turn into ``0 * inf = NaN``."""
    out = torch.zeros_like(x)
    pos = x > 0
    out[pos] = torch.sqrt(x[pos])
    return out


def _matrix_to_quat_wxyz(R):
    """Rotation matrices (F, 3, 3) -> unit quaternions (F, 4) in wxyz order,
    differentiably. The largest-diagonal branch is selected to keep the division
    well-conditioned (the standard trick; every branch is smooth in its own region)."""
    m = R.reshape(-1, 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    cand = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=1,
    ) / (2.0 * q_abs[:, :, None].clamp_min(0.1))
    pick = q_abs.argmax(dim=1)
    return cand[torch.arange(cand.shape[0], device=R.device), pick]


def _shape_operator_frame(S, t1, t2, n, ref_e1, eigengap_min, eigengap_clip):
    """In-plane frame direction ``e1`` (F, 3) from the symmetric 2x2 shape operator, made
    backward-safe on umbilic faces. The eigenvector Jacobian of ``S`` carries a
    ``1/(kappa1 - kappa2)`` that blows up as the surface goes umbilic (``kappa1 -> kappa2``:
    planes, spheres, any smooth patch), where the eigenframe is also geometrically
    undetermined - the spinning-frame / exploding-gradient of principal-direction fields.
    Two coupled remedies:

    - **analytic eigenvector angle** ``theta = 1/2 atan2(2b, a - c)`` for ``[[a, b], [b, c]]``,
      with its *gradient* scaled by ``min(1, gap/eigengap_clip)`` (``gap = |kappa1 - kappa2|``)
      so ``|d theta| ~ 1/gap`` is clipped from below - the closed form is exactly what lets
      the clip land on the gap rather than inside an opaque ``eigh`` backward;
    - **smooth blend** toward the reference direction ``ref_e1`` with weight ``beta(gap)``, a
      smoothstep reaching pure-reference at full umbilicity (``gap -> 0``) and pure-eigenframe
      at ``gap >= eigengap_min``, so the undetermined eigenframe is replaced by the coherent,
      non-spinning reference field there (a hard switch, as the numpy path uses, has an
      ill-defined derivative at the threshold; the blend is smooth).

    ``e1`` is ordered to the larger-``|kappa|`` direction (``+pi/2`` when the trace is
    negative) and sign-aligned to ``ref_e1`` - matching the numpy path up to the per-column
    sign the tests compare. The ``atan2`` input is guarded so a *fully* isotropic ``S``
    (``gap = 0`` exactly, e.g. a plane) carries a finite - here zero - gradient rather than
    the ``0/0`` an unguarded ``atan2`` would hand back."""
    a, b, c = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    x, y = a - c, 2.0 * b
    g2 = x * x + y * y  # gap^2; smooth at 0 (no sqrt on the graph)
    gap = torch.sqrt(g2)  # only used detached, so sqrt'(0) never bites
    # Rotate by pi/2 for negative trace to select the larger-|kappa| direction.
    quarter = torch.where(
        a + c >= 0, torch.zeros_like(a), torch.full_like(a, 0.5 * torch.pi)
    )
    deg = g2 < (eigengap_min * eigengap_min) * 1e-8  # only an *exactly* isotropic S
    x_in = torch.where(deg, torch.ones_like(x), x)  # keep atan2 off (0, 0)
    y_in = torch.where(deg, torch.zeros_like(y), y)
    theta = 0.5 * torch.atan2(y_in, x_in) + quarter
    factor = (
        (gap / eigengap_clip).clamp(max=1.0).detach()
    )  # in [0, 1], = 1 above the clip
    theta = (
        factor * theta + (1.0 - factor) * theta.detach()
    )  # scale d theta, keep the value
    e1 = torch.cos(theta)[:, None] * t1 + torch.sin(theta)[:, None] * t2

    ref = (
        ref_e1 - (ref_e1 * n).sum(-1, keepdim=True) * n
    )  # project ref into the face plane
    norm = ref.norm(dim=-1, keepdim=True)
    ref = torch.where(norm > 0, ref / torch.where(norm > 0, norm, torch.ones_like(norm)), t1)
    # An orthogonal reference leaves the sign free; it must never zero a unit eigenvector.
    e1 = torch.where((e1 * ref).sum(-1, keepdim=True) >= 0, e1, -e1)
    s = (g2 / eigengap_min**2).clamp(max=1.0)
    beta = 1.0 - s * s * (
        3.0 - 2.0 * s
    )  # smoothstep: 1 at gap 0, 0 at gap>=eigengap_min
    e1 = beta[:, None] * ref + (1.0 - beta)[:, None] * e1
    e1 = (
        e1 - (e1 * n).sum(-1, keepdim=True) * n
    )  # re-project and renormalise -> unit, _|_ n
    return e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _project_to_faces(mesh, ref: np.ndarray) -> np.ndarray:
    """Project a per-face direction ``(F, 3)`` into each face plane and unit-normalise, with
    ``t1`` filling the field's unavoidable zeros (a hairy-ball zero, or a fixed axis parallel
    to the normal)."""
    p = mesh.verts[mesh.faces]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    ref = ref - (ref * fn).sum(axis=1, keepdims=True) * fn
    norm = np.linalg.norm(ref, axis=1, keepdims=True)
    return np.where(norm > 1e-9, ref / norm, _face_bases(mesh)[0])


def transported_reference_frame(verts, faces) -> np.ndarray:
    """A globally coherent per-face tangent direction ``(F, 3)`` for the umbilic blend:
    seed one tangent vector and carry it over the surface by vector-heat
    parallel transport, so umbilic regions - where the eigenframe spins - blend toward a
    smooth, non-spinning field instead of an incoherent per-face default. Detached and
    per-interval (potpourri3d, CPU, non-differentiable), like the robust coupling operator;
    only faces below the blend threshold actually use it.

    The vector-heat solver (geometry-central) needs a **vertex-manifold** mesh; the Poisson-
    extracted reconstruction meshes carry a handful of bowtie vertices that ``robust_laplacian``
    tolerates but it does not. When the solver cannot be built, fall back to projecting a fixed
    world axis (the one least aligned with the mean normal) into each face plane - still coherent
    and non-spinning wherever the normal varies smoothly, and manifold-agnostic."""
    mesh = Mesh(verts, faces)
    try:
        vt = VectorTransport(mesh)
        field = vt.to_3d(vt.transport(0, (1.0, 0.0)))  # one seed carried everywhere
        return _project_to_faces(mesh, field[mesh.faces].mean(axis=1))
    except RuntimeError:  # non-manifold: geometry-central refuses
        p = mesh.verts[mesh.faces]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        axis = np.eye(3)[
            np.abs(fn.mean(axis=0)).argmin()
        ]  # world axis least aligned with the surface
        return _project_to_faces(mesh, np.broadcast_to(axis, fn.shape).copy())


def derive_splats_torch(
    verts,
    faces,
    cfg,
    frame_mode="gram_schmidt",
    pairs=None,
    ref_e1=None,
    eigengap_clip=None,
) -> SplatTensors:
    """Differentiable derivation ``x -> (mu, frame, sigma)`` on live vertices; ``cfg`` is
    the ``splats:`` config node. ``frame_mode``:

    - ``"gram_schmidt"`` - the in-plane frame is the Gram-Schmidt basis (e1 = first edge);
      the warmup mode, free of the shape-operator feedback loop.
    - ``"shape_operator"`` - e1 is the larger-``|kappa|`` eigenvector of the shape operator,
      computed by the analytic 2x2 eigendecomposition with the eigengap clip and the smooth
      blend toward ``ref_e1`` on umbilic faces (``_shape_operator_frame``) - the
      trained mode, backward-safe where the surface is umbilic.

    ``pairs`` (interior face pairs, from ``interior_face_pairs``) is precomputed once in
    training and passed in; if ``None`` it is derived from ``faces``. ``ref_e1`` (F, 3) is the
    detached, per-interval transported reference for the umbilic blend (``None`` -> the
    first-edge direction ``t1``, the warmup/test default); ``eigengap_clip`` is the lower gap
    clip for the frame Jacobian (``None`` -> ``cfg.eigengap_min``, an inert clip).
    """
    p = verts[faces]  # (F, 3, 3)
    fn, n, t1, t2 = face_frame_torch(verts, faces)

    mu = _pull_into_face_torch(p, _circumcenters_torch(p))

    if frame_mode == "gram_schmidt":
        e1 = t1
    elif frame_mode == "shape_operator":
        S = _shape_operators_torch(verts, faces, t1, t2)
        ref = t1 if ref_e1 is None else ref_e1
        clip = cfg.eigengap_min if eigengap_clip is None else eigengap_clip
        e1 = _shape_operator_frame(S, t1, t2, n, ref, cfg.eigengap_min, clip)
    else:
        raise ValueError(
            f"frame_mode must be 'gram_schmidt' or 'shape_operator', got {frame_mode!r}"
        )
    frame = torch.stack([e1, torch.linalg.cross(n, e1), n], dim=2)

    if pairs is None:
        pairs = interior_face_pairs(faces)
    ell = mu[pairs[:, 1]] - mu[pairs[:, 0]]
    outer = ell[:, :, None] * ell[:, None, :]
    Mk = torch.zeros((faces.shape[0], 3, 3), dtype=verts.dtype, device=verts.device)
    Mk = Mk.index_add(0, pairs[:, 0], outer).index_add(0, pairs[:, 1], outer)
    sigma = _tangential_scales_torch(Mk, frame, cfg.gamma)
    area = 0.5 * fn.norm(dim=1)
    sigma = clamp_sigma(sigma, torch.sqrt(4.0 * area / np.sqrt(3.0)), cfg)

    return SplatTensors(
        mu=mu, frame=frame, sigma=sigma, quat_wxyz=_matrix_to_quat_wxyz(frame)
    )
