"""DEC operator tests: incidence and d1@d0=0, L0/mass, the intrinsic-Delaunay safeguard,
the dual L2, and the edit-time brush dual_heat_diffuse."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import sparse
from scipy.sparse.linalg import spsolve

import potpourri3d as pp3d

import dec3dgs
from dec3dgs.mesh import Mesh, circumcenters, dual_heat_diffuse, face_areas

MESHES = ["icosphere", "torus", "flat_disk"]
EULER = {"icosphere": 2, "torus": 0, "flat_disk": 1}


def get_mesh(request, name):
    return Mesh(*request.getfixturevalue(name))


def test_package_imports():
    assert dec3dgs.__version__


@pytest.mark.parametrize("name", MESHES)
def test_fixture_sanity(name, request):
    verts, faces = request.getfixturevalue(name)
    assert len(verts) > 0 and len(faces) > 0
    assert faces.max() < len(verts)


@pytest.mark.parametrize("name", MESHES)
def test_dd_is_exactly_zero(name, request):
    # Incidence cancels exactly around each face; no tolerance needed.
    m = get_mesh(request, name)
    dd = m.d1 @ m.d0
    dd.eliminate_zeros()
    assert dd.nnz == 0


@pytest.mark.parametrize("name", MESHES)
def test_euler_characteristic(name, request):
    assert get_mesh(request, name).euler_characteristic == EULER[name]


@pytest.mark.parametrize("name", ["icosphere", "torus"])
def test_closed_meshes_have_no_boundary(name, request):
    assert len(get_mesh(request, name).boundary_edges) == 0


def test_disk_boundary_is_the_outer_ring(flat_disk):
    # The unit disk's boundary is exactly the vertices at radius 1.
    m = Mesh(*flat_disk)
    rim = np.flatnonzero(np.linalg.norm(m.verts[:, :2], axis=1) > 1 - 1e-9)
    assert rim.size > 0
    assert np.array_equal(m.boundary_vertices, rim)


def test_flipped_face_is_rejected(icosphere):
    verts, faces = icosphere
    flipped = faces.copy()
    flipped[0] = flipped[0, ::-1]
    with pytest.raises(ValueError, match="orientation"):
        Mesh(verts, flipped)


# Cotangent Laplacian L0 and mass matrix M.


@pytest.mark.parametrize("name", MESHES)
def test_L0_kills_constants(name, request):
    # Rows sum to zero: constants are in the kernel of the Dirichlet form.
    m = get_mesh(request, name)
    assert np.abs(m.L0 @ np.ones(len(m.verts))).max() < 1e-10


@pytest.mark.parametrize("name", MESHES)
def test_L0_symmetric(name, request):
    m = get_mesh(request, name)
    assert abs(m.L0 - m.L0.T).max() < 1e-12


def test_L0_psd_spectrum(icosphere):
    # Use dense eigvalsh to check PSD and the constant null mode.
    m = Mesh(*icosphere)
    eigs = np.linalg.eigvalsh(m.L0.toarray())
    assert eigs[0] >= -1e-9


def test_L0_energy_nonnegative(icosphere):
    m = Mesh(*icosphere)
    rng = np.random.default_rng(0)
    for x in rng.standard_normal((10, len(m.verts))):
        assert x @ (m.L0 @ x) >= 0


def test_L0_cross_check_robust_vs_igl(icosphere):
    # Check Delaunay weights first, so intrinsic flips do not change the comparison.
    m = Mesh(*icosphere)
    off_diagonal = m.L0_igl - sparse.diags(m.L0_igl.diagonal())
    assert off_diagonal.max() <= 1e-12
    assert abs(m.L0 - m.L0_igl).max() / abs(m.L0).max() < 1e-6
    # Lumpings differ pointwise but share the total (= surface area); log, don't gate.
    a, b = m.M.diagonal(), m.M_igl.diagonal()
    assert np.isclose(a.sum(), b.sum(), rtol=1e-12)
    print(
        f"\nmass lumping, robust vs voronoi: max rel gap "
        f"{np.max(np.abs(a - b) / b):.3g}"
    )


# The thin rhombus has obtuse opposite angles, giving its shared edge negative weight.

RHOMBUS_VERTS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, 0.3, 0], [0, -0.3, 0]], dtype=np.float64
)
RHOMBUS_FACES = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32)  # (A,B,C), (A,D,B)
A, B = 0, 1


def test_plain_cotan_weight_negative_on_non_delaunay():
    # the PSD sign puts -w_e off-diagonal, so w_AB < 0 surfaces as a positive entry.
    m = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, laplacian="igl")
    assert m.L0[A, B] > 0


def test_max_principle_igl_violates_robust_holds():
    # A negative edge weight makes u(B) = -t L[B,A]/M_BB negative from delta_A.
    u0 = np.array([1.0, 0.0, 0.0, 0.0])

    def update(m, t):
        return u0 - t * (m.L0 @ u0) / m.M.diagonal()

    m_igl = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, laplacian="igl")
    assert update(m_igl, 1e-3)[B] < 0
    # Nonnegative weights preserve positivity below the explicit-step stability bound.
    m_rob = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES)
    t = 0.5 * (m_rob.M.diagonal() / m_rob.L0.diagonal()).min()
    assert (update(m_rob, t) >= 0).all()


def test_robust_weights_nonnegative_verts_untouched():
    m = Mesh(RHOMBUS_VERTS.copy(), RHOMBUS_FACES)
    off_diagonal = m.L0 - sparse.diags(m.L0.diagonal())
    assert off_diagonal.max() <= 1e-12
    assert np.linalg.eigvalsh(m.L0.toarray())[0] >= -1e-9
    # Intrinsic flips rewire the gluing, never the vertices.
    assert np.array_equal(m.verts, RHOMBUS_VERTS)


def test_laplacian_switch_reads_config():
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.dec.laplacian == "robust"
    # The config default selects the safeguarded assembly even on the bad mesh.
    m = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, laplacian=cfg.dec.laplacian)
    assert m.L0[A, B] <= 1e-12
    # "igl" routes L0/M to the plain pair; anything else is rejected.
    m_igl = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, laplacian="igl")
    assert m_igl.L0 is m_igl.L0_igl and m_igl.M is m_igl.M_igl
    with pytest.raises(ValueError, match="laplacian"):
        Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, laplacian="cotan")


# Dual mesh and the face Laplacian L2.

DUAL_TYPES = ["circumcentric", "barycentric"]


def face_adjacency(m):
    """Dual-graph adjacency from raw incidence alone (independent of the weights):
    faces share an interior edge iff d1 @ d1^T has a nonzero off-diagonal entry."""
    A = (m.d1 @ m.d1.T).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def bfs_rings(A, src):
    """Dual-graph ring index per face (graph distance from src), -1 if unreachable."""
    ring = np.full(A.shape[0], -1)
    ring[src] = 0
    frontier, d = [src], 0
    while frontier:
        d += 1
        nxt = []
        for f in frontier:
            for g in A.indices[A.indptr[f] : A.indptr[f + 1]]:
                if ring[g] < 0:
                    ring[g] = d
                    nxt.append(g)
        frontier = nxt
    return ring


@pytest.mark.parametrize("dual_type", DUAL_TYPES)
def test_L2_kills_constants(icosphere, dual_type):
    # Constants are harmonic on the closed dual graph.
    m = Mesh(*icosphere, dual_type=dual_type)
    assert np.abs(m.L2 @ np.ones(len(m.faces))).max() < 1e-10


@pytest.mark.parametrize("dual_type", DUAL_TYPES)
def test_L2_symmetric(icosphere, dual_type):
    m = Mesh(*icosphere, dual_type=dual_type)
    assert abs(m.L2 - m.L2.T).max() < 1e-12


@pytest.mark.parametrize("dual_type", DUAL_TYPES)
def test_L2_psd(icosphere, dual_type):
    # The Delaunay dual graph has positive weights and a constant zero mode.
    m = Mesh(*icosphere, dual_type=dual_type)
    assert np.linalg.eigvalsh(m.L2.toarray())[0] >= -1e-9


def test_L2_explicit_update_spreads_one_ring(icosphere):
    # Below the stability bound, each explicit heat step extends support by one dual ring.
    m = Mesh(*icosphere)
    ring = bfs_rings(face_adjacency(m), 0)
    areas = face_areas(m)
    Minv = sparse.diags(1.0 / areas)
    t = 0.5 * (areas / m.L2.diagonal()).min()
    u = np.zeros(len(m.faces))
    u[0] = 1.0
    for k in range(1, 4):
        u = u - t * (Minv @ (m.L2 @ u))
        support = set(np.flatnonzero(np.abs(u) > 1e-12))
        assert support == set(np.flatnonzero((ring >= 0) & (ring <= k)))


def test_L2_implicit_update_positive_and_decays(icosphere):
    # The implicit M-matrix solve gives a positive, distance-decaying Green's function.
    m = Mesh(*icosphere)
    ring = bfs_rings(face_adjacency(m), 0)
    areas = face_areas(m)
    rhs = np.zeros(len(m.faces))
    rhs[0] = areas[0]  # = M2 @ delta_0
    x = spsolve((sparse.diags(areas) + m.L2).tocsc(), rhs)
    assert (x > 0).all()
    means = [x[ring == r].mean() for r in range(6)]
    assert all(means[r] > means[r + 1] for r in range(5))


def test_L2_clamp_keeps_psd_on_non_delaunay():
    # Clamp the non-Delaunay primal star so its reciprocal dual weight stays positive.
    m = Mesh(RHOMBUS_VERTS, RHOMBUS_FACES)  # default circumcentric + clamp
    assert np.linalg.eigvalsh(m.L2.toarray())[0] >= -1e-9
    assert (m.L2.diagonal() > 0).all()
    # The reciprocal weight is capped at 1/clamp, not an unstable 1/eps.
    off = -m.L2[0, 1]
    assert off == pytest.approx(1.0 / m.dual_clamp)
    assert off <= 4.0


def test_dual_switch_reads_config():
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.dec.dual.type == "circumcentric"
    assert cfg.dec.dual.clamp == pytest.approx(0.5)
    m = Mesh(
        RHOMBUS_VERTS,
        RHOMBUS_FACES,
        dual_type=cfg.dec.dual.type,
        dual_clamp=cfg.dec.dual.clamp,
    )
    assert (m.L2.diagonal() > 0).all()
    with pytest.raises(ValueError, match="dual_type"):
        Mesh(RHOMBUS_VERTS, RHOMBUS_FACES, dual_type="voronoi")


def test_circumcenter_on_sphere_and_obtuse(icosphere):
    # On the unit sphere every face circumcenter lies just inside (radius < 1), no NaN.
    cc = circumcenters(Mesh(*icosphere))
    assert np.isfinite(cc).all()
    assert (np.linalg.norm(cc, axis=1) < 1.0).all()
    # An obtuse triangle's circumcentre stays equidistant from its vertices, outside the face.
    V = np.array([[0, 0, 0], [4, 0, 0], [1, 0.5, 0]], dtype=np.float64)
    Fz = np.array([[0, 1, 2]], dtype=np.int32)
    c = circumcenters(Mesh(V, Fz))[0]
    r = np.linalg.norm(V - c, axis=1)
    assert np.allclose(r, r[0])


def test_circumcenter_rejects_degenerate_face():
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)  # collinear
    Fz = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="degenerate"):
        circumcenters(Mesh(V, Fz))


def test_circumcenter_centroid_fallback_keeps_the_dual_finite():
    # Centroid fallback must handle zero-area faces without changing healthy ones.
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)  # collinear
    Fz = np.array([[0, 1, 2]], dtype=np.int32)
    cc = circumcenters(Mesh(V, Fz), on_degenerate="centroid")
    assert np.isfinite(cc).all() and np.allclose(cc[0], V.mean(axis=0))
    Vg = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
    )  # a right triangle
    mg = Mesh(Vg, Fz)
    assert np.array_equal(
        circumcenters(mg, "centroid"), circumcenters(mg)
    )  # untouched where finite


# The edit-time geodesic brush (dual_heat_diffuse).


def hairpin_strip(gap=0.1, arm=2.0, width=0.6, n_arm=24, n_bend=10, n_w=6):
    """A hairpin strip (two arms ``gap`` apart, joined by a bend) extruded along y.
    Mirror points across the gap are Euclidean-distance ``gap`` but geodesic-distance
    ``~arm``. Open manifold, consistently wound."""
    za = np.linspace(arm, 0.0, n_arm, endpoint=False)  # arm A at x = 0
    th = np.linspace(np.pi, 2 * np.pi, n_bend, endpoint=False)  # bend, dipping to z < 0
    zc = np.linspace(0.0, arm, n_arm + 1)  # arm B at x = gap
    path = np.vstack(
        [
            np.c_[np.zeros_like(za), za],
            np.c_[gap / 2 + gap / 2 * np.cos(th), gap / 2 * np.sin(th)],
            np.c_[np.full_like(zc, gap), zc],
        ]
    )  # (P, 2) = (x, z)
    y = np.linspace(0.0, width, n_w)
    V = np.array([[x, yy, z] for x, z in path for yy in y])  # vertex i*n_w + j
    P = len(path)
    quads = [
        (i * n_w + j, (i + 1) * n_w + j, (i + 1) * n_w + j + 1, i * n_w + j + 1)
        for i in range(P - 1)
        for j in range(n_w - 1)
    ]
    faces = np.array(
        [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int32
    )
    return V, faces


def test_dual_heat_conserves_mass(icosphere):
    # sum area_f c_f is invariant (1^T L2 = 0): the brush moves paint, never creates it.
    m = Mesh(*icosphere)
    area = face_areas(m)
    c = np.random.default_rng(0).standard_normal((len(m.faces), 3))
    m0 = (area[:, None] * c).sum(0)
    for mode in ("implicit", "explicit"):
        c1 = dual_heat_diffuse(m, c, 0.9, mode)
        assert np.allclose((area[:, None] * c1).sum(0), m0, atol=1e-9)
    with pytest.raises(ValueError, match="mode"):
        dual_heat_diffuse(m, c, 0.1, "bogus")


def test_dual_heat_implicit_is_positive_green_function(icosphere):
    # (M2 + t L2) x = M2 delta: positive, decaying with dual-graph (= geodesic here) distance.
    m = Mesh(*icosphere)
    F = len(m.faces)
    delta = np.zeros(F)
    delta[0] = 1.0
    x = dual_heat_diffuse(m, delta, 0.7, "implicit")
    ref = spsolve((m.M2 + 0.7 * m.L2).tocsc(), face_areas(m) * delta)
    assert np.allclose(x, ref, atol=1e-9)
    assert (x > 0).all()
    ring = bfs_rings(face_adjacency(m), 0)
    means = [x[ring == r].mean() for r in range(6)]
    assert all(means[r] > means[r + 1] for r in range(5))


def test_dual_heat_explicit_update_is_one_ring(icosphere):
    # one explicit step from a delta touches exactly the dual 1-ring.
    m = Mesh(*icosphere)
    F = len(m.faces)
    delta = np.zeros(F)
    delta[0] = 1.0
    t = 0.5 * (face_areas(m) / m.L2.diagonal()).min()
    u = dual_heat_diffuse(m, delta, t, "explicit")
    ring = bfs_rings(face_adjacency(m), 0)
    assert set(np.flatnonzero(np.abs(u) > 1e-12)) == set(
        np.flatnonzero((ring >= 0) & (ring <= 1))
    )


def test_dual_heat_diffuses_geodesically_not_euclidean():
    # At matched radius and mass, compare cross-gap leakage of DEC and Euclidean brushes.
    V, Fz = hairpin_strip()
    m = Mesh(V, Fz)
    C = V[Fz].mean(axis=1)  # face centroids
    src = int(np.argmin(np.linalg.norm(C - [0.0, 0.3, 1.0], axis=1)))
    d_euc = np.linalg.norm(C - C[src], axis=1)
    dv = pp3d.MeshHeatMethodDistanceSolver(V, Fz).compute_distance_multisource(
        Fz[src].tolist()
    )
    d_geo = dv[Fz].mean(axis=1)

    r = 0.5  # brush radius; t = r^2
    delta = np.zeros(len(Fz))
    delta[src] = 1.0
    w_dec = dual_heat_diffuse(m, delta, r * r, "implicit")
    w_dec /= w_dec.sum()
    w_euc = np.exp(-0.5 * (d_euc / r) ** 2)
    w_euc /= w_euc.sum()

    across = (d_euc < r) & (d_geo > 3 * r)  # the mirror arm
    assert across.sum() > 0
    leak_dec, leak_euc = w_dec[across].sum(), w_euc[across].sum()
    assert leak_dec < 0.05
    assert leak_euc > 10 * leak_dec
