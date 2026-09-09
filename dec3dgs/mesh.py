"""Triangle mesh and its DEC operators (incidence d0/d1, cotangent Laplacian, dual L2).

Conventions: L0 is PSD with ``L0 @ 1 = 0`` (``igl.cotmatrix`` returns the negative -
always wrap); faces CCW / outward normals, edges sorted ``i < j``; the unit sphere has
``H = +1``; float64 on CPU.
"""

from dataclasses import dataclass, field
from functools import cached_property

import igl
import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import torch
from scipy import sparse
from scipy.sparse.linalg import factorized


@dataclass
class Mesh:
    """Triangle mesh with its incidence operators, built eagerly on construction.

    ``d0`` (E, V) and ``d1`` (F, E) are the signed incidence matrices; ``edges`` are
    sorted ``i < j``. The constructor checks manifoldness and orientation consistency.
    """

    verts: np.ndarray
    faces: np.ndarray
    laplacian: str = "robust"  # L0/M assembly, from config key dec.laplacian
    dual_type: str = "circumcentric"  # L2 dual mesh, from config key dec.dual.type
    dual_clamp: float = (
        0.5  # lower clamp on the edge star, from config key dec.dual.clamp
    )
    edges: np.ndarray = field(init=False)
    d0: sparse.csr_matrix = field(init=False)
    d1: sparse.csr_matrix = field(init=False)

    def __post_init__(self):
        self.verts = np.ascontiguousarray(self.verts, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if self.verts.ndim != 2 or self.verts.shape[1] != 3:
            raise ValueError(f"verts must be (V, 3), got {self.verts.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {self.faces.shape}")
        if (self.faces == np.roll(self.faces, -1, axis=1)).any():
            raise ValueError("degenerate face: a vertex appears twice")
        if self.laplacian not in ("robust", "igl"):
            raise ValueError(
                f"laplacian must be 'robust' or 'igl', got {self.laplacian!r}"
            )
        if self.dual_type not in ("circumcentric", "barycentric"):
            raise ValueError(
                "dual_type must be 'circumcentric' or 'barycentric', "
                f"got {self.dual_type!r}"
            )
        self._build_incidence()

    def _build_incidence(self):
        V, F = len(self.verts), len(self.faces)
        # The three directed edges of every face, in CCW order: 0->1, 1->2, 2->0.
        halfedges = self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)  # (3F, 2)
        # +1 where the traversal already runs i -> j with i < j (the stored orientation).
        sign = np.where(halfedges[:, 0] < halfedges[:, 1], 1, -1).astype(np.int8)
        self.edges, edge_of = np.unique(
            np.sort(halfedges, axis=1), axis=0, return_inverse=True
        )
        E = len(self.edges)

        # Each edge has at most two faces, with opposite traversal on interior edges.
        count = np.bincount(edge_of, minlength=E)
        signsum = np.bincount(edge_of, weights=sign, minlength=E).astype(np.int64)
        bad = np.flatnonzero(count > 2)
        if bad.size:
            i, j = self.edges[bad[0]]
            raise ValueError(
                f"non-manifold edge ({i}, {j}): {count[bad[0]]} incident faces"
            )
        bad = np.flatnonzero((count == 2) & (signsum != 0))
        if bad.size:
            i, j = self.edges[bad[0]]
            raise ValueError(f"inconsistent face orientation across edge ({i}, {j})")

        rows = np.repeat(np.arange(E), 2)
        vals = np.tile(np.array([-1, 1], dtype=np.int8), E)
        self.d0 = sparse.csr_matrix((vals, (rows, self.edges.ravel())), shape=(E, V))
        self.d1 = sparse.csr_matrix(
            (sign, (np.repeat(np.arange(F), 3), edge_of)), shape=(F, E)
        )

    @property
    def euler_characteristic(self) -> int:
        return len(self.verts) - len(self.edges) + len(self.faces)

    @property
    def boundary_edges(self) -> np.ndarray:
        """Indices of edges claimed by one face only (one nonzero in their d1 column)."""
        return np.flatnonzero(self.d1.getnnz(axis=0) == 1)

    @property
    def boundary_vertices(self) -> np.ndarray:
        return np.unique(self.edges[self.boundary_edges])

    # Cache stiffness and mass; this mesh instance is immutable.

    @cached_property
    def _robust_pair(self):
        L, M = robust_laplacian.mesh_laplacian(self.verts, self.faces)
        return L.tocsr(), M.tocsr()

    @property
    def L0(self) -> sparse.csr_matrix:
        """Cotangent Laplacian on vertex functions, (V, V), PSD with ``L0 @ 1 = 0``.
        Robust intrinsic-Delaunay assembly by default (weights >= 0); ``M^-1 L0`` is the
        strong-form Laplace-Beltrami."""
        return self._robust_pair[0] if self.laplacian == "robust" else self.L0_igl

    @property
    def M(self) -> sparse.csr_matrix:
        """Lumped mass, diagonal of vertex areas - the Hodge star on 0-forms, pairing
        with ``L0`` so ``M^-1 L0`` is the strong-form Laplacian."""
        return self._robust_pair[1] if self.laplacian == "robust" else self.M_igl

    @cached_property
    def L0_igl(self) -> sparse.csr_matrix:
        """Plain cotangent Laplacian via igl, flipped to the PSD sign - cross-check and
        ablation only; its weights go negative on non-Delaunay edges."""
        return (-igl.cotmatrix(self.verts, self.faces)).tocsr()

    @cached_property
    def M_igl(self) -> sparse.csr_matrix:
        """Voronoi-lumped mass via igl; cross-check only (same total area as ``M``,
        lumped differently)."""
        return igl.massmatrix(
            self.verts, self.faces, igl.MASSMATRIX_TYPE_VORONOI
        ).tocsr()

    # Faces are dual vertices; interior edges connect them in L2.

    def _dual_weights(self) -> np.ndarray:
        """Per-edge dual weight ``w*`` (E,), zeroed on boundary edges (so ``L2 @ 1 = 0``).

        Circumcentric: reciprocal cotangent star ``w* = 1 / max((cot a + cot b)/2, clamp)``.
        The clamp is a conductance *ceiling* ``w* <= 1/clamp`` - it makes degenerate
        (non-Delaunay) edges ordinary, not ``1/eps`` superconductors that would sink fp32.
        Barycentric: ``w* = |e| / |c_f - c_g|`` from centroids - positive, less accurate.
        """
        i, j = self.edges[:, 0], self.edges[:, 1]
        if self.dual_type == "circumcentric":
            star = -np.asarray(self.L0_igl[i, j]).ravel()  # (cot a + cot b)/2 per edge
            w = 1.0 / np.maximum(star, self.dual_clamp)
        else:
            centroids = self.verts[self.faces].mean(axis=1)
            dual_len = np.linalg.norm(self.d1.T @ centroids, axis=1)  # |c_f - c_g|
            edge_len = np.linalg.norm(self.verts[j] - self.verts[i], axis=1)
            w = edge_len / dual_len
        w[self.boundary_edges] = 0.0
        return w

    @cached_property
    def L2(self) -> sparse.csr_matrix:
        """Face (dual 0-form) Laplacian ``d1 diag(w*) d1^T``, (F, F), symmetric PSD with
        ``L2 @ 1 = 0`` - the dual-graph Laplacian on the face-adjacency graph."""
        return (self.d1 @ sparse.diags(self._dual_weights()) @ self.d1.T).tocsr()

    @cached_property
    def M2(self) -> sparse.csr_matrix:
        """Face mass, diagonal of face areas; ``M2^-1 L2`` is the strong-form face
        Laplacian."""
        return sparse.diags(face_areas(self)).tocsr()


# Curvature and dual geometry - derived quantities, read off the operators above.


def face_areas(mesh: Mesh) -> np.ndarray:
    """Triangle areas, (F,) - half the cross-product magnitude of two edges."""
    v0, v1, v2 = (mesh.verts[mesh.faces[:, k]] for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def circumcenters(mesh: Mesh, on_degenerate: str = "raise") -> np.ndarray:
    """Face circumcenters, (F, 3) - the dual vertices and the splat positions. Obtuse faces put the
    centre outside the triangle (allowed); a **degenerate** face has no circumcentre.

    ``on_degenerate``: ``"raise"`` (default) refuses a degenerate face - the strict DEC contract, so
    the pure-DEC and census paths catch degeneracy rather than propagate a NaN. ``"centroid"``
    returns that face's centroid instead - a finite dual point inside the face - for the deform
    reassembly, whose face-ID tree must survive a moving mesh that occasionally drives a sliver to
    zero area. Degeneracy is detected by the circumcentre coming out **non-finite**: the barycentric
    denominator is ``16*Area^2``, which cancels to ``0`` on a collinear face and even on an extreme
    sliver whose cross-product area is still positive - a case the old ``face_areas <= 0`` guard let
    slip through into a NaN."""
    a, b, c = (mesh.verts[mesh.faces[:, k]] for k in range(3))
    a2 = np.sum((b - c) ** 2, axis=1)
    b2 = np.sum((c - a) ** 2, axis=1)
    c2 = np.sum((a - b) ** 2, axis=1)
    bary = np.stack(
        [a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        bary = bary / bary.sum(axis=1, keepdims=True)
        cc = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    bad = ~np.isfinite(cc).all(axis=1)
    if bad.any():
        if on_degenerate == "raise":
            raise ValueError("degenerate (zero-area) face: circumcenter undefined")
        if on_degenerate != "centroid":
            raise ValueError(
                f"on_degenerate must be 'raise' or 'centroid', got {on_degenerate!r}"
            )
        cc[bad] = ((a + b + c) / 3.0)[bad]
    return cc


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted unit vertex normals, (V, 3), outward (raw cross products accumulate
    the area weight for free)."""
    v0, v1, v2 = (mesh.verts[mesh.faces[:, k]] for k in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(mesh.verts)
    np.add.at(n, mesh.faces.ravel(), np.repeat(fn, 3, axis=0))
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    # Use a zero normal when incident face normals cancel exactly.
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0.0)


# The curvature-modulated Hodge star on 1-forms, and the anisotropic Laplacian it builds.


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle between rows of ``a`` and ``b``, in ``[0, pi)``."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))


def dihedral_angles(mesh: Mesh) -> np.ndarray:
    """Unsigned angle between the two incident face normals per edge, (E,), in ``[0, pi)``;
    zero on boundary edges."""
    v0, v1, v2 = (mesh.verts[mesh.faces[:, k]] for k in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)

    inc = mesh.d1.T.tocsr()  # (E, F), one row per edge
    interior = np.flatnonzero(np.diff(inc.indptr) == 2)
    f = inc.indices[inc.indptr[interior]]
    g = inc.indices[inc.indptr[interior] + 1]

    theta = np.zeros(len(mesh.edges))
    theta[interior] = _angle_between(fn[f], fn[g])
    return theta


def gauss_map_angles(mesh: Mesh) -> np.ndarray:
    """Angle between the two *vertex* normals per edge, (E,), in ``[0, pi)`` - how far the
    Gauss map turns along the edge. The primal-edge counterpart of ``dihedral_angles``."""
    n = vertex_normals(mesh)
    return _angle_between(n[mesh.edges[:, 0]], n[mesh.edges[:, 1]])


def star1_kappa(
    mesh: Mesh, theta_c: float | None = None, modulator: str = "gauss"
) -> tuple[np.ndarray, ...]:
    """The robust ``L0`` weights modulated by a Perona-Malik stop, as ``(rows, cols, weights)``.

    ``*1^kappa_ij = g(phi_ij) * w_ij`` with ``g(phi) = 1 / (1 + (phi/theta_c)^2)``: flat edges
    conduct (``g ~ 1``), creases insulate (``g ~ 0``), so the crease survives. ``theta_c = None``
    is ``g = 1`` and recovers ``mesh.L0`` exactly. ``modulator`` picks the bending measure;
    ``gauss`` is on the primal edge where ``*1`` lives, ``dihedral`` on the dual (see
    ``bending_angles``). The star comes from the robust ``L0`` - its weights are already
    nonnegative, so no clamp is needed to stay PSD.
    """
    r, c, w, angle = bending_angles(mesh, modulator)
    if theta_c is None:
        return r, c, w
    if theta_c <= 0:
        raise ValueError(f"theta_c must be positive, got {theta_c}")
    return r, c, w / (1.0 + (angle / theta_c) ** 2)


def bending_angles(mesh: Mesh, modulator: str = "gauss") -> tuple[np.ndarray, ...]:
    """``(rows, cols, weights, angle)`` on ``L0``'s off-diagonal pattern - the star and the
    bending measure that modulates it, on the same index pairs."""
    if modulator not in ("gauss", "dihedral"):
        raise ValueError(f"modulator must be 'gauss' or 'dihedral', got {modulator!r}")
    L = mesh.L0.tocoo()
    off = L.row != L.col
    r, c, w = L.row[off], L.col[off], -L.data[off]  # weight = -offdiagonal
    if modulator == "gauss":
        n = vertex_normals(mesh)
        angle = _angle_between(n[r], n[c])
    else:
        # dihedral is defined on mesh edges only; pairs L0 couples but the mesh does not join read 0
        i, j = mesh.edges[:, 0], mesh.edges[:, 1]
        T = sparse.coo_matrix(
            (dihedral_angles(mesh), (i, j)), shape=mesh.L0.shape
        ).tocsr()
        angle = np.asarray((T + T.T)[r, c]).ravel()
    return r, c, w, angle


def theta_c_from_percentile(
    mesh: Mesh, percentile: float, modulator: str = "gauss"
) -> float:
    """The knee at the given percentile of ``phi`` over ``L0``'s coupled pairs.

    Reads the knee off the mesh's own bending distribution, so it insulates a fixed *fraction*
    of edges (the bendiest ``100 - percentile`` %) whatever the mesh looks like. Recompute it as
    the mesh moves - a literal knee throttles ever harder as ``phi`` grows.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    return float(np.percentile(bending_angles(mesh, modulator)[3], percentile))


def anisotropic_laplacian(
    mesh: Mesh, theta_c: float | None = None, modulator: str = "gauss"
) -> sparse.csr_matrix:
    """``L_D = diag(rowsum W) - W`` with ``W = *1^kappa``, (V, V), symmetric PSD, ``L_D @ 1 = 0``.

    Graph-Laplacian assembly rather than ``d0^T (.) d0``, since the star lives on ``L0``'s own
    (intrinsic) pattern where ``d0`` does not apply. PSD for any nonnegative ``W``, so
    ``M + t L_D`` stays SPD however hard ``g`` insulates.
    """
    r, c, w = star1_kappa(mesh, theta_c, modulator)
    W = sparse.coo_matrix((w, (r, c)), shape=mesh.L0.shape).tocsr()
    return (sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()


def mean_curvature(mesh: Mesh, return_vector: bool = False) -> np.ndarray:
    """Signed mean curvature per vertex (unit sphere gives ``H = +1``): the normal
    component of ``Hn = (M^-1 L0 verts) / 2``. Boundary rows are unreliable; callers mask
    them. ``return_vector=True`` also returns ``Hn``, (V, 3)."""
    Hn = (mesh.L0 @ mesh.verts) / (2.0 * mesh.M.diagonal()[:, None])
    H = np.einsum("ij,ij->i", Hn, vertex_normals(mesh))
    return (H, Hn) if return_vector else H


def angle_defect(mesh: Mesh) -> np.ndarray:
    """Integrated Gaussian curvature per vertex: ``2 pi - sum of incident angles``
    (interior formula on boundary vertices too)."""
    return igl.gaussian_curvature(mesh.verts, mesh.faces)


def gaussian_curvature(mesh: Mesh) -> np.ndarray:
    """Pointwise Gaussian curvature: angle defect over vertex area."""
    return angle_defect(mesh) / mesh.M.diagonal()


# Dual heat flow - the edit-time geodesic brush.


def dual_heat_diffuse(
    mesh: Mesh, c: np.ndarray, t: float, mode: str = "implicit"
) -> np.ndarray:
    """Diffuse a dual 0-cochain ``c`` - (F,) or (F, k) - by heat time ``t``: the
    edit-time geodesic brush, ``sqrt(t)`` the brush radius.

    ``implicit`` (default): backward-Euler step, solved as ``(M2 + t L2) c_new = M2 c``
    (SPD M-matrix, so nonnegative and geodesic). ``explicit``: forward step
    ``c - t M2^-1 L2 c``. Both conserve mass (``1^T L2 = 0``).
    """
    c = np.ascontiguousarray(c, dtype=np.float64)
    area = face_areas(mesh)
    if mode == "explicit":
        Lc = mesh.L2 @ c
        return c - t * (Lc / area[:, None] if c.ndim == 2 else Lc / area)
    if mode != "implicit":
        raise ValueError(f"mode must be 'implicit' or 'explicit', got {mode!r}")
    solve = factorized((mesh.M2 + t * mesh.L2).tocsc())
    if c.ndim == 1:
        return solve(area * c)
    return np.column_stack([solve(area * c[:, k]) for k in range(c.shape[1])])


# Parallel transport - the connection, not just the metric.


class VectorTransport:
    """Parallel transport of tangent vectors by the vector heat method (Sharp, Soliman
    & Crane), wrapping ``potpourri3d``.

    A 2-vector is a pair of coefficients in the per-vertex frame ``(basisX, basisY)``
    the solver returns (``basisN`` is the outward normal). The frames are shared, so a
    vector read off at ``q`` feeds straight back as a second leg's source.
    """

    def __init__(self, mesh: Mesh):
        self._solver = pp3d.MeshVectorHeatSolver(mesh.verts, mesh.faces)
        self.basisX, self.basisY, self.basisN = self._solver.get_tangent_frames()

    def transport(self, src_vertex: int, vec2) -> np.ndarray:
        """Transport the frame 2-vector ``vec2``, sitting at ``src_vertex``, to all
        vertices; returns ``(V, 2)`` coefficients, each in its own vertex's frame."""
        a, b = vec2
        return np.asarray(
            self._solver.transport_tangent_vector(int(src_vertex), [float(a), float(b)])
        )

    def to_3d(self, field2: np.ndarray) -> np.ndarray:
        """Embed a per-vertex frame field ``(V, 2)`` back into ``R^3``, ``(V, 3)``."""
        field2 = np.asarray(field2)
        return field2[:, 0, None] * self.basisX + field2[:, 1, None] * self.basisY


# Differentiable cotangent assembly, checked against the NumPy operators above.


def cotan_laplacian_torch(verts, faces):
    """Plain cotangent Laplacian ``L0_x(x)`` and lumped mass ``M(x)``, assembled inside
    the torch graph so both carry gradient to the vertex positions - ``verts`` (V, 3) float and ``faces`` (F, 3) long
    are torch tensors.

    Returns ``(L0_x, M)``:

    - ``L0_x`` (V, V) sparse COO, PSD with ``L0_x @ 1 = 0``; edge weight
      ``w_ij = 1/2 (cot a + cot b)`` from the *plain* cotangent, so weights may go negative
      on non-Delaunay edges. That is allowed here: ``L0_x`` is only ever the gradient source
      inside squared-norm losses, never a coupling - the robust ``Mesh.L0`` stays the PSD
      coupling operator.
    - ``M`` (V,) lumped barycentric vertex mass (one third of each incident face area) -
      the diagonal Hodge star paired with ``L0_x`` in the ``M``-norm.

    No hand-written backward: ``cot t = (a . b) / ||a x b||`` is a smooth rational function
    of the three face vertices, scattered into the fixed ``(V, V)`` sparsity pattern.
    """
    V = verts.shape[0]
    p = verts[faces]  # (F, 3, 3)
    tiny = torch.finfo(verts.dtype).tiny

    def cot(a, b):  # cot of the angle between a and b: (a.b) / ||a x b|| = (a.b) / 2A
        return (a * b).sum(-1) / torch.linalg.cross(a, b).norm(dim=-1).clamp_min(tiny)

    # Half of each corner cotangent weights the opposite edge.
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    c0 = cot(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    c1 = cot(p[:, 2] - p[:, 1], p[:, 0] - p[:, 1])
    c2 = cot(p[:, 0] - p[:, 2], p[:, 1] - p[:, 2])
    w = 0.5 * torch.cat([c0, c1, c2])  # (3F,)
    i = torch.cat([f1, f2, f0])  # opposite-edge endpoints
    j = torch.cat([f2, f0, f1])

    rows = torch.cat([i, j, i, j])
    cols = torch.cat([j, i, i, j])
    vals = torch.cat([-w, -w, w, w])  # off-diagonal -w, diagonal +w
    L0_x = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (V, V)).coalesce()

    area = 0.5 * torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).norm(dim=-1)
    M = torch.zeros(V, dtype=verts.dtype, device=verts.device).index_add(
        0, faces.reshape(-1), (area / 3.0).repeat_interleave(3)
    )
    return L0_x, M
