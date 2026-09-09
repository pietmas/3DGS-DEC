"""The Sobolev metric on vertex positions: ``x <- x - eta (M + t L0)^-1 g``.

The claims here are exact where they can be. On an eigenpair ``L0 phi = lambda M phi`` the
preconditioner is a low-pass with a *written-down* transfer function ``1/(1 + t lambda)``, so the
smoothing is asserted mode by mode rather than through a tolerance; ``L0 @ 1 = 0`` makes uniform
translation an exact fixed point; ``t = 0`` is exactly mass-weighted descent. The one statistical
check - roughness of a synthetic gradient on a noisy sphere - is the DoD's own, and it is stated
as a factor, not a hope.
"""

import numpy as np
import pytest
import scipy.linalg
import torch
from scipy.sparse.linalg import spsolve

from dec3dgs.mesh import (
    Mesh,
    anisotropic_laplacian,
    bending_angles,
    dihedral_angles,
    gauss_map_angles,
    star1_kappa,
    theta_c_from_percentile,
)
from dec3dgs.metric import SobolevMetric, combinatorial_laplacian
from tests.conftest import icosphere_arrays
from tests.test_losses import roof_mesh


@pytest.fixture(scope="module")
def sphere():
    return Mesh(*icosphere_arrays(subdivisions=3))


def spectrum(mesh):
    """Generalised eigenpairs of ``L0 phi = lambda M phi``, M-orthonormal. Dense ``eigh`` on a
    small mesh - sparse solvers are flaky near the zero eigenvalue of a PSD Laplacian."""
    lam, phi = scipy.linalg.eigh(mesh.L0.toarray(), mesh.M.toarray())
    return lam, phi


def test_transfer_function_is_exact_on_eigenmodes(sphere):
    # Check (M + tL)^-1 M phi = phi / (1 + t*lambda) mode by mode.
    t = 0.05
    metric = SobolevMetric(sphere, t=t, laplacian="cotan")
    lam, phi = spectrum(sphere)
    M = sphere.M.toarray()
    for k in (0, 1, 7, 40, len(lam) - 1):
        g = (M @ phi[:, k])[:, None] * np.ones((1, 3))
        out = metric.apply(g)
        assert np.allclose(out, phi[:, k][:, None] / (1.0 + t * lam[k]), atol=1e-9)


def test_constants_pass_through_untouched(sphere):
    # The mass-weighted filter preserves constant fields because L0 @ 1 = 0.
    metric = SobolevMetric(sphere, t=10.0, laplacian="cotan")
    g = sphere.M @ np.tile(np.array([1.0, -2.0, 0.5]), (len(sphere.verts), 1))
    assert np.allclose(metric.apply(g), np.array([1.0, -2.0, 0.5]), atol=1e-9)


def test_zero_heat_time_is_mass_weighted_descent(sphere):
    # t -> 0 recovers M^-1 g exactly: the metric's floor is L2 descent, not Euclidean descent.
    metric = SobolevMetric(sphere, t=0.0, laplacian="cotan")
    rng = np.random.default_rng(0)
    g = rng.standard_normal((len(sphere.verts), 3))
    minv = 1.0 / sphere.M.diagonal()
    assert np.allclose(metric.apply(g), minv[:, None] * g, atol=1e-12)


def test_preconditioned_update_is_smooth_and_does_not_shrink(sphere):
    # Check roughness reduction and size on this example, not a general no-shrink guarantee.
    rng = np.random.default_rng(0)
    V, F = icosphere_arrays(subdivisions=3)
    V = V + 0.01 * rng.standard_normal(V.shape)
    mesh = Mesh(V, F)
    lam, phi = spectrum(mesh)
    cutoff = np.median(lam)

    g = rng.standard_normal(V.shape)  # white noise: energy at every frequency
    metric = SobolevMetric(mesh, t=0.05, laplacian="cotan")
    out = metric.apply(g)

    def rough_share(u):
        c = phi.T @ (mesh.M @ u)  # M-orthonormal expansion
        e = (c**2).sum(axis=1)
        return e[lam > cutoff].sum() / e.sum()

    assert rough_share(g) > 0.4  # the input really is rough (white: ~0.46)
    assert rough_share(out) < 0.2 * rough_share(
        g
    )  # measured 6.7x at t = 0.05 on this mesh

    def bbox_volume(x):
        return np.prod(x.max(axis=0) - x.min(axis=0))

    eta = 0.05 / np.abs(out).max()  # a visible step, not an infinitesimal one
    assert abs(bbox_volume(V - eta * out) / bbox_volume(V) - 1.0) < 0.05


def test_combinatorial_arm_is_topology_only(sphere):
    # PSD, annihilates constants, geometry-independent (so it factors once)
    L = combinatorial_laplacian(sphere.faces, len(sphere.verts))
    assert np.allclose(L @ np.ones(len(sphere.verts)), 0.0, atol=1e-12)
    assert abs((L - L.T)).max() == 0.0
    assert np.linalg.eigvalsh(L.toarray()).min() > -1e-10

    moved = Mesh(sphere.verts * 2.0, sphere.faces)
    assert (
        abs((combinatorial_laplacian(moved.faces, len(moved.verts)) - L)).max() == 0.0
    )
    assert (
        SobolevMetric(sphere, t=1.0, laplacian="combinatorial").geometry_dependent
        is False
    )
    assert SobolevMetric(sphere, t=1.0, laplacian="cotan").geometry_dependent is True


# Curvature-modulated metric: test the solve rather than descent on an energy.

THETA_C = 0.35


@pytest.fixture(scope="module")
def roof():
    """Clean and noisy roof, plus the edge classification the anisotropy claim turns on:
    ``line`` runs along the ridge, ``cut`` are the edges joining the two half-planes."""
    verts, faces = roof_mesh()
    clean = Mesh(verts, faces)
    line = dihedral_angles(clean) > 0.5
    ridge_v = np.unique(clean.edges[line])
    touches = np.isin(clean.edges, ridge_v).any(axis=1)
    cut = touches & ~line

    g = torch.Generator().manual_seed(0)  # test_losses' noise, bit for bit
    noise = torch.randn(len(verts), generator=g, dtype=torch.float64).numpy()
    nv = verts.copy()
    nv[:, 2] += 0.01 * noise
    return clean, Mesh(nv, faces), line, cut


def test_the_modulator_must_live_on_the_primal_edge(roof):
    # *1 weights the primal edge, so it needs phi (primal); the dihedral is dual and swapped here
    clean, _, line, cut = roof
    theta, phi = dihedral_angles(clean), gauss_map_angles(clean)
    far = ~line & ~cut

    assert theta[line].min() > 1.5  # the ridge: a 90 deg fold
    assert theta[cut].max() == 0.0  # the crossing edges: exactly flat
    assert phi[far].max() == 0.0  # planar: the two measures agree here
    assert phi[cut].min() > phi[line].max()  # and everywhere else they are swapped


def _leak(mesh, L, t, src, target):
    """Share of the response to a unit impulse at ``src`` that lands on ``target``."""
    g = np.zeros(len(mesh.verts))
    g[src] = 1.0
    u = np.abs(spsolve((mesh.M + t * L).tocsc(), g))
    return u[target].sum() / u.sum()


def test_anisotropic_metric_refuses_to_diffuse_across_the_crease(roof):
    # g(phi) should reduce cross-ridge leakage by about 10x; g(theta) is inert here.
    clean, _, _, _ = roof
    v, t = clean.verts, 0.05
    src = np.flatnonzero(
        (np.abs(v[:, 0] - 0.5) < 1e-9) & (np.abs(v[:, 1] - 0.5) < 1e-9)
    )[0]
    far = v[:, 0] < -1e-12

    iso = _leak(clean, clean.L0, t, src, far)
    gauss = _leak(clean, anisotropic_laplacian(clean, THETA_C, "gauss"), t, src, far)
    dihed = _leak(clean, anisotropic_laplacian(clean, THETA_C, "dihedral"), t, src, far)

    assert gauss < 0.15 * iso  # measured ~10x block
    assert abs(dihed / iso - 1.0) < 0.02  # the refuted arm: inert to 2 %


def _heat_update(mesh, L, t):
    """One implicit heat step on the positions, ``x <- (M + tL)^-1 M x``."""
    return np.column_stack(
        [spsolve((mesh.M + t * L).tocsc(), mesh.M @ mesh.verts[:, k]) for k in range(3)]
    )


def test_anisotropy_keeps_the_crease_and_fairs_the_noise_isotropy_erases_it(roof):
    # isotropic smoothing flattens the ridge; g(phi) holds it and still fairs the off-ridge noise
    clean, noisy, line, _ = roof
    faces, t = clean.faces, 0.05
    off = ~line

    def read(x):
        th = dihedral_angles(Mesh(x, faces))
        return float(th[line].mean()), float(np.sqrt((th[off] ** 2).mean()))

    crease0, rough0 = read(noisy.verts)
    crease_i, rough_i = read(_heat_update(noisy, noisy.L0, t))
    crease_a, rough_a = read(
        _heat_update(noisy, anisotropic_laplacian(noisy, THETA_C), t)
    )

    assert crease_i < 0.65 * crease0  # isotropic erases it (measured 0.56x)
    assert crease_a > 0.78 * crease0  # anisotropic holds it (0.83x)
    assert rough_a < 0.60 * rough0  # and fairs the off-ridge noise (0.51x)
    assert rough_a < 0.75 * rough_i  # better than isotropic manages (0.67x)


def test_switching_the_knee_off_recovers_L0_exactly(roof):
    # Without a knee, g=1 and L_D should recover the mesh's L0.
    clean, _, _, _ = roof
    assert abs(anisotropic_laplacian(clean, None) - clean.L0.tocsr()).max() < 1e-12
    assert abs(anisotropic_laplacian(clean, 1e8) - clean.L0.tocsr()).max() < 1e-8
    assert (
        abs(anisotropic_laplacian(clean, 1e8, "dihedral") - clean.L0.tocsr()).max()
        < 1e-8
    )

    # the robust weights need no clamp: nonnegative (only to roundoff here - roof is right-angled)
    _, _, w = star1_kappa(clean, None)
    assert w.min() > -1e-12


def test_anisotropic_laplacian_is_psd_and_the_system_spd(roof):
    # PSD for any nonnegative star, so M + t L_D stays SPD however hard g insulates
    clean, _, _, _ = roof
    for theta_c in (THETA_C, 0.02):  # 0.02: g ~ 0 on most of the ridge
        L = anisotropic_laplacian(clean, theta_c)
        assert abs((L - L.T)).max() < 1e-15
        assert np.allclose(L @ np.ones(len(clean.verts)), 0.0, atol=1e-12)
        assert np.linalg.eigvalsh(L.toarray()).min() > -1e-12
        lo = np.linalg.eigvalsh((clean.M + 1.0 * L).toarray()).min()
        assert lo > 0.0


def test_anisotropy_rejects_what_it_cannot_modulate(roof):
    # the combinatorial arm has no cotangent star to modulate, so the combination is refused
    clean, _, _, _ = roof
    with pytest.raises(ValueError, match="laplacian='cotan'"):
        SobolevMetric(clean, t=1e-3, laplacian="combinatorial", anisotropy="curvature")
    with pytest.raises(ValueError, match="modulator"):
        star1_kappa(clean, THETA_C, "dihedral_angle")
    m = SobolevMetric(clean, t=1e-3, anisotropy="curvature", theta_c=THETA_C)
    assert m.geometry_dependent is True  # the star moves with the mesh


def test_percentile_knee_fixes_the_insulated_fraction(sphere):
    # a percentile knee pins the insulated fraction (g < 1/2 on the top (100-p) % of edges)
    mesh = sphere  # a real, non-bimodal phi distribution
    _, _, _, phi = bending_angles(mesh)
    for p in (80.0, 90.0, 95.0):
        tc = theta_c_from_percentile(mesh, p)
        # g(phi) = 1/(1+(phi/tc)^2) < 1/2  <=>  phi > tc, so exactly the top (100-p) % sit sub-half
        assert abs(100.0 * np.mean(phi > tc) - (100.0 - p)) < 1.0
    # and the metric wires it: a knee derived here equals asking for it directly
    m = SobolevMetric(
        mesh, t=1e-3, anisotropy="curvature", theta_c=None, theta_c_percentile=90.0
    )
    assert abs(m.theta_c - theta_c_from_percentile(mesh, 90.0)) < 1e-12
    assert m.theta_c > 0.0


# the displacement roughness readout
def test_displacement_energy_is_the_rayleigh_quotient(sphere):
    """Roughness here has an exact meaning: on a generalised eigenpair ``L0 phi = lambda M phi``
    the Dirichlet number **is** ``lambda``, so the readout is the mean squared frequency of ``d``
    in the basis the metric attenuates. Asserted mode by mode, not through a tolerance."""
    from dec3dgs.metric import displacement_energy

    lam, phi = spectrum(sphere)
    # a uniform translation is the lambda = 0 mode: L0 @ 1 = 0, so it is EXACTLY zero-frequency
    d = np.tile([0.3, -0.2, 0.7], (len(sphere.verts), 1))
    assert displacement_energy(sphere, d)["dirichlet"] == pytest.approx(0.0, abs=1e-9)

    # and a single mode reports its own eigenvalue, low and high alike
    for k in (5, 40, 120):
        d = np.column_stack(
            [phi[:, k], np.zeros_like(phi[:, k]), np.zeros_like(phi[:, k])]
        )
        assert displacement_energy(sphere, d)["dirichlet"] == pytest.approx(
            lam[k], rel=1e-8
        )
    lo = displacement_energy(sphere, np.column_stack([phi[:, 5]] * 3))["dirichlet"]
    hi = displacement_energy(sphere, np.column_stack([phi[:, 120]] * 3))["dirichlet"]
    assert hi > lo  # a rougher field reports a larger number, which is the whole use


def test_displacement_energy_separates_normal_from_tangential(sphere):
    """A field can be perfectly smooth and still shear triangles, because shape is set by the
    field's *gradient*. The tangential fraction is what tells a surface that moved from a surface
    that slid. The split is against the **discrete** vertex normal, so it is exact there: a field
    along ``n`` reports 0 and any field orthogonal to ``n`` reports 1, both to roundoff."""
    from dec3dgs.mesh import vertex_normals
    from dec3dgs.metric import displacement_energy

    n = vertex_normals(sphere)
    assert displacement_energy(sphere, 0.01 * n)["tangential"] == pytest.approx(
        0.0, abs=1e-12
    )
    # cross with a fixed axis: orthogonal to n at every vertex, so purely tangential
    tang = np.cross(n, [0.0, 0.0, 1.0])
    keep = (
        np.linalg.norm(tang, axis=1) > 1e-6
    )  # drop the poles, where the cross vanishes
    assert displacement_energy(sphere, 0.01 * np.where(keep[:, None], tang, 0.0))[
        "tangential"
    ] == pytest.approx(1.0, abs=1e-12)

    # Radial and discrete normals differ by O(h^2), leaving a small tangential component.
    radial = sphere.verts / np.linalg.norm(sphere.verts, axis=1, keepdims=True)
    assert displacement_energy(sphere, 0.01 * radial)["tangential"] < 1e-3
