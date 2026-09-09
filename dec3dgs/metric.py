"""The Sobolev (H^1) metric on vertex positions - the Laplacian as a *metric*, not a penalty.

Gradient descent is only defined relative to an inner product: ``x <- x - eta g`` silently means
``g`` is the gradient with respect to the **Euclidean** product on coordinates, in which two
adjacent vertices are as independent as two on opposite sides of the object. That is the wrong
geometry for a surface, and its symptom is roughness: nothing in the update says a vertex should
move with its neighbours.

Give the vertex space the H^1 product instead,

    <u, v>_{H^1} = u^T (M + t L0) v ,

with ``M`` the L^2 (mass) product on 0-forms and ``L0`` the Dirichlet form. The Riemannian
gradient is the ordinary one pulled back through the metric, so the update becomes

    x  <-  x  -  eta (M + t L0)^{-1} g .

``t`` carries units of **area**, so ``sqrt(t)`` is a *length* - the smoothing radius on the
surface. Two consequences worth stating, because both are exact and both are tested:

- On an eigenpair ``L0 phi = lambda M phi`` the operator acts as ``(M + tL0)^{-1} M phi =
  phi / (1 + t lambda)``: a low-pass whose transfer function is written down, not tuned. Rough
  components are attenuated by their own frequency, smooth ones pass.
- ``L0 @ 1 = 0``, so ``(M + tL0)^-1 M 1 = 1``. The mass-weighted filter preserves
  constant displacement fields; applying the inverse to an unweighted constant need not.

This is the same linear system as one implicit heat step - ``dual_heat_diffuse`` solves
``(M2 + tL2) c = M2 c0`` on the dual - with a different right-hand side. It replaces ``L_lap``:
a penalty shapes *where* you land, a metric shapes *how you travel*, and the roughness we are
fighting is a property of the path.

``anisotropy = "curvature"`` modulates the star to insulate creases (``mesh.star1_kappa``), which
was meant to replace ``L_normal`` too. On scan65 it does not, so it defaults off and ``L_normal``
stays; the construction is kept as an ablation.

The preconditioner is **not** differentiated through. It is applied to ``.grad`` after
``backward()`` and before ``opt.step()``, so it needs no autograd and ``L0`` is assembled
detached, from the numpy path that the DEC tests already pin.
Training uses Adam on this preconditioned gradient, so its actual update is not the plain
gradient-descent formula above and does not have that formula's exact transfer function.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .mesh import Mesh, anisotropic_laplacian, theta_c_from_percentile, vertex_normals


def combinatorial_laplacian(faces: np.ndarray, n_verts: int) -> sparse.csr_matrix:
    """Graph Laplacian ``D - A`` of the edge graph, (V, V), PSD with ``L @ 1 = 0``.

    The ablation stiffness: no geometry, so it is constant under fixed topology."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.ones(len(e))
    A = sparse.coo_matrix(
        (
            np.concatenate([w, w]),
            (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])),
        ),
        shape=(n_verts, n_verts),
    ).tocsr()
    return (sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsr()


def displacement_energy(mesh: Mesh, d: np.ndarray) -> dict:
    """Roughness of the displacement field ``d = x - x0``, on ``x0``'s operators.

    Read on the **initial** mesh, so the eigenbasis is a fixed frequency reference rather than one
    that travels with ``x``.

    - ``dirichlet``: the Rayleigh quotient ``tr(d^T L0 d) / tr(d^T M d)``, units ``1/length^2``.
      On an eigenpair ``L0 phi = lambda M phi`` it returns ``lambda``, so it is the mean squared
      frequency in ``d`` - exactly what ``1/(1+t lambda)`` attenuates.
    - ``curvature_ratio``: ``|L0 d| / |d|``, the same without the mass weighting.
    - ``tangential``: fraction of ``|d|^2`` in the tangent plane, against the *discrete* vertex
      normal. Tangential motion leaves the surface alone and shears the triangulation, so this
      separates a surface that moved from one that slid.
    """
    d = np.ascontiguousarray(d, dtype=np.float64)
    L0, M = mesh.L0, mesh.M
    quad = float(np.einsum("ij,ij->", d, L0 @ d))
    mass = float(np.einsum("ij,ij->", d, M @ d))
    norm2 = float((d**2).sum())
    n = vertex_normals(mesh)
    normal2 = float((np.einsum("ij,ij->i", d, n) ** 2).sum())
    return {
        "mean_disp": float(np.linalg.norm(d, axis=1).mean()),
        "dirichlet": quad / mass if mass > 0 else float("nan"),
        "curvature_ratio": (
            float(np.linalg.norm(L0 @ d) / np.sqrt(norm2))
            if norm2 > 0
            else float("nan")
        ),
        "tangential": 1.0 - normal2 / norm2 if norm2 > 0 else float("nan"),
    }


class SobolevMetric:
    """Prefactored ``(M + t L0)``, applied to a ``(V, 3)`` gradient.

    ``laplacian``: ``"cotan"`` takes the mesh's own ``L0`` (robust intrinsic-Delaunay by
    default, so the weights stay nonnegative and the matrix stays an M-matrix); ``"combinatorial"``
    takes the topology-only graph Laplacian. Both are PSD and both annihilate constants, so
    ``M + tL0`` is symmetric **positive definite** for any ``t >= 0`` and positive vertex masses.
    This ensures a unique gradient solve, not an injective or non-collapsing mesh update.

    ``anisotropy = "curvature"`` swaps ``L0`` for the crease-insulating ``L_D`` (cotan only).
    ``theta_c = None`` recovers ``mesh.L0`` exactly, so it and ``anisotropy = "none"`` are the
    same operator - the knee is then the only variable between the two. ``theta_c_percentile``
    overrides ``theta_c`` with the knee read off this mesh at each refactorisation; prefer it,
    since a literal knee only suits the mesh it was picked on.
    """

    def __init__(
        self,
        mesh: Mesh,
        t: float,
        laplacian: str = "cotan",
        anisotropy: str = "none",
        theta_c: float | None = 0.35,
        modulator: str = "gauss",
        theta_c_percentile: float | None = None,
    ):
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"heat time must be finite and nonnegative, got {t}")
        if theta_c_percentile is not None:
            theta_c = theta_c_from_percentile(mesh, theta_c_percentile, modulator)
        if anisotropy not in ("none", "curvature"):
            raise ValueError(
                f"anisotropy must be 'none' or 'curvature', got {anisotropy!r}"
            )
        if anisotropy == "curvature" and laplacian != "cotan":
            raise ValueError(
                "anisotropy='curvature' modulates the cotangent star, so it needs "
                f"laplacian='cotan', got {laplacian!r}"
            )
        if laplacian == "cotan":
            L = (
                mesh.L0
                if anisotropy == "none"
                else anisotropic_laplacian(mesh, theta_c, modulator)
            )
        elif laplacian == "combinatorial":
            L = combinatorial_laplacian(mesh.faces, len(mesh.verts))
        else:
            raise ValueError(
                f"laplacian must be 'cotan' or 'combinatorial', got {laplacian!r}"
            )
        self.laplacian = laplacian
        self.anisotropy = anisotropy
        self.theta_c = None if theta_c is None else float(theta_c)
        self.theta_c_percentile = theta_c_percentile
        self.modulator = modulator
        self.t = float(t)
        self._solve = factorized((mesh.M + self.t * L).tocsc())

    @property
    def geometry_dependent(self) -> bool:
        """Whether a moved mesh invalidates the factorisation. The combinatorial arm depends on
        connectivity alone only for its stiffness; this ablation deliberately also freezes its
        initial mass matrix. The cotangent arm updates both mass and stiffness."""
        return self.laplacian == "cotan"

    def apply(self, g: np.ndarray) -> np.ndarray:
        """``(M + tL0)^{-1} g``, column by column - the three coordinates are independent, the
        metric couples *vertices*, not components."""
        g = np.ascontiguousarray(g, dtype=np.float64)
        return np.column_stack([self._solve(g[:, k]) for k in range(g.shape[1])])

    def apply_(self, grad) -> None:
        """In-place on a torch ``(V, 3)`` gradient: down to CPU float64 for the solve, back to
        the tensor's own device and dtype."""
        out = self.apply(grad.detach().cpu().numpy())
        grad.copy_(grad.new_tensor(out))
