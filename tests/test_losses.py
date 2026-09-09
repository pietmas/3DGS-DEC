"""L_field tests (CPU torch on the icosphere): Dirichlet energy kernel/positivity, the
autograd gradient = L2 c, one-step support = the dual 1-ring, the mu term, and TV.
Plus the robust normal-consistency L_normal (Huber preserves a crease that plain
Dirichlet erases), and the curvature-weighted photometric loss L_curv (scale-invariant
weight, localised to where the surface bends, folded through a face-ID buffer)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from dec3dgs.losses import curv_weight, l_curv, l_distortion, l_field, l_normal, tv_dc
from dec3dgs.mesh import Mesh, mean_curvature
from dec3dgs.splats import (
    dual_edges,
    face_adjacency_weights,
    face_normals_torch,
    interior_face_pairs,
)

DT = torch.float64
CFG_DEFORM = OmegaConf.load(
    Path(__file__).parents[1] / "configs" / "default.yaml"
).deform


@pytest.fixture(scope="module")
def setup(icosphere):
    mesh = Mesh(*icosphere)
    L2 = mesh.L2.tocsr()
    L2t = torch.sparse_csr_tensor(
        L2.indptr, L2.indices, L2.data.astype(np.float32), size=L2.shape
    )
    return mesh, L2, L2t


def test_l_field_kernel_and_positivity(setup):
    _, _, L2t = setup
    F = L2t.shape[0]
    g = torch.Generator().manual_seed(0)
    e_rand = l_field(
        torch.randn(F, 16, 3, generator=g), torch.randn(F, 1, generator=g), L2t, mu=0.1
    )
    assert e_rand > 0
    # Each SH channel has a seperate constant value in ker L2.
    const_sh = torch.arange(48, dtype=torch.float32).reshape(1, 16, 3).expand(F, -1, -1)
    e_const = l_field(const_sh, torch.full((F, 1), 2.5), L2t, mu=0.1)
    assert abs(float(e_const)) < 1e-5 * float(e_rand)


def test_gradient_is_heat_update(setup):
    # Compare the Euclidean gradient L2 c against autograd; heat flow also needs mass.
    _, L2, L2t = setup
    F = L2.shape[0]
    g = torch.Generator().manual_seed(1)
    sh = torch.randn(F, 16, 3, generator=g).requires_grad_(True)
    (0.5 * l_field(sh, torch.zeros(F, 1), L2t, mu=0.0)).backward()
    want = L2 @ sh.detach().numpy().reshape(F, 48).astype(np.float64)
    got = sh.grad.numpy().reshape(F, 48)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())


def test_one_update_support_is_the_closed_one_ring(setup):
    # One explicit step has support on the closed dual 1-ring.
    mesh, _, L2t = setup
    F = L2t.shape[0]
    k = 17
    sh = torch.zeros(F, 16, 3)
    sh[k, 0, 0] = 1.0
    sh.requires_grad_(True)
    l_field(sh, torch.zeros(F, 1), L2t, mu=0.0).backward()
    touched = set(np.nonzero(sh.grad.numpy().reshape(F, 48)[:, 0])[0].tolist())
    pairs = dual_edges(mesh)
    ring = {k} | set(pairs[pairs[:, 0] == k, 1]) | set(pairs[pairs[:, 1] == k, 0])
    assert touched == ring


def test_mu_prices_the_opacity_term(setup):
    _, _, L2t = setup
    F = L2t.shape[0]
    g = torch.Generator().manual_seed(2)
    op = torch.randn(F, 1, generator=g)
    const_sh = torch.zeros(F, 16, 3)
    e1 = l_field(const_sh, op, L2t, mu=1.0)
    assert float(l_field(const_sh, op, L2t, mu=0.25)) == pytest.approx(
        0.25 * float(e1), rel=1e-6
    )
    assert float(l_field(const_sh, op, L2t, mu=0.0)) == 0.0


def test_tv_dc_constant_and_delta(setup):
    mesh, _, _ = setup
    pairs = torch.tensor(dual_edges(mesh), dtype=torch.long)
    F = int(pairs.max()) + 1
    assert float(tv_dc(torch.full((F, 1, 3), 0.7), pairs)) == 0.0
    # delta at face k: 3 dual edges on a closed mesh, |1| per colour channel each
    k = 5
    dc = torch.zeros(F, 1, 3)
    dc[k, 0] = 1.0
    deg = int(((pairs[:, 0] == k) | (pairs[:, 1] == k)).sum())
    assert deg == 3
    assert float(tv_dc(dc, pairs)) == pytest.approx(float(3 * deg))


# robust normal-consistency L_normal
def roof_mesh(s=6, ny=7, slope=1.0):
    """Two planar half-planes meeting at a ridge along ``x = 0`` (``z = slope|x|``): a single
    sharp crease of fixed dihedral, flat everywhere else. A column of vertices sits exactly on
    ``x = 0`` so the crease is an edge (no straddled face). ``slope = 1`` gives a 90 deg crease."""
    x = np.linspace(-1.0, 1.0, 2 * s + 1)  # includes 0 at index s
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    verts = np.stack([xx, yy, slope * np.abs(xx)], axis=-1).reshape(-1, 3)
    idx = np.arange((2 * s + 1) * ny).reshape(2 * s + 1, ny)
    faces = []
    for i in range(2 * s):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [[a, b, c], [a, c, d]]  # CCW, outward (+z when flat)
    return verts, np.asarray(faces, dtype=np.int64)


def _pair_angles(verts, faces, pairs):
    """Angle between adjacent face normals, per pair (the dihedral of a shared edge)."""
    n = face_normals_torch(verts, faces)
    ni, nj = n[pairs[:, 0]], n[pairs[:, 1]]
    sin = torch.linalg.cross(ni, nj).norm(dim=1)
    return torch.atan2(sin, (ni * nj).sum(1))


@pytest.fixture(scope="module")
def roof():
    verts, faces = roof_mesh()
    clean = torch.tensor(verts, dtype=DT)
    facest = torch.tensor(faces, dtype=torch.long)
    pairs = interior_face_pairs(facest)
    ang0 = _pair_angles(clean, facest, pairs)
    crease = ang0 > 0.5  # the roof is planar off the ridge
    assert crease.any() and float(ang0[~crease].max()) < 1e-9
    return clean, facest, pairs, crease, float(ang0[crease].mean())


def _descend(x0, facest, pairs, w, delta_huber, steps, lr):
    """Plain gradient descent on ``L_normal`` alone (no data term); returns the settled mesh."""
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=lr)
    for _ in range(steps):
        loss = l_normal(face_normals_torch(x, facest), pairs, w, delta_huber)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return x.detach()


def test_l_normal_huber_keeps_crease_dirichlet_erases_it(roof):
    # At a matched short budget, the Huber tail should preserve the crease better.
    clean, facest, pairs, crease, crease0 = roof
    g = torch.Generator().manual_seed(0)
    noisy = clean.clone()
    noisy[:, 2] += 0.01 * torch.randn(
        len(clean), generator=g, dtype=DT
    )  # vertical bumps
    w = face_adjacency_weights(clean, facest, pairs)  # fixed weights

    xh = _descend(noisy, facest, pairs, w, 0.17, steps=15, lr=1e-3)  # robust
    xd = _descend(noisy, facest, pairs, w, 1e3, steps=15, lr=1e-3)  # plain Dirichlet
    crease_h = float(_pair_angles(xh, facest, pairs)[crease].mean())
    crease_d = float(_pair_angles(xd, facest, pairs)[crease].mean())

    assert abs(crease_h - crease0) < 0.15  # crease preserved within ~9 deg (Huber)
    assert crease0 - crease_d > 0.50  # crease erased by > ~29 deg (Dirichlet)
    assert crease_d < crease_h - 0.30  # the two paths diverge sharply


def test_l_normal_denoises_a_flat_sheet():
    # Below the Huber knee, normal loss should smooth noise on a flat sheet.
    verts, faces = roof_mesh(slope=0.0)
    clean = torch.tensor(verts, dtype=DT)
    facest = torch.tensor(faces, dtype=torch.long)
    pairs = interior_face_pairs(facest)
    w = face_adjacency_weights(clean, facest, pairs)
    g = torch.Generator().manual_seed(1)
    noisy = clean.clone()
    noisy[:, 2] += 0.02 * torch.randn(len(clean), generator=g, dtype=DT)

    def rough(x):
        return float((_pair_angles(x, facest, pairs) ** 2).mean())

    settled = _descend(noisy, facest, pairs, w, 0.17, steps=400, lr=1e-3)
    assert rough(settled) < 0.35 * rough(noisy)  # normal field faired


def test_l_normal_nonneg_zero_on_plane_finite_grad(roof):
    clean, facest, pairs, crease, _ = roof
    # flat plane: all normals equal -> zero energy and a finite (zero) gradient, no acos NaN
    flat_v, flat_f = roof_mesh(slope=0.0)
    xf = torch.tensor(flat_v, dtype=DT, requires_grad=True)
    ff = torch.tensor(flat_f, dtype=torch.long)
    pf = interior_face_pairs(ff)
    wf = face_adjacency_weights(xf.detach(), ff, pf)
    lf = l_normal(face_normals_torch(xf, ff), pf, wf, 0.17)
    assert float(lf) < 1e-12
    lf.backward()
    assert torch.isfinite(xf.grad).all()
    # nonnegative on a genuinely creased mesh, with positive weights
    w = face_adjacency_weights(clean, facest, pairs)
    assert float(w.min()) > 0.0
    assert float(l_normal(face_normals_torch(clean, facest), pairs, w, 0.17)) >= 0.0


# The curvature density kappa, and the folding of it into the photometric residual.


def test_curv_weight_scale_invariant(icosphere):
    # kappa = |H| h + |K| h^2 is dimensionless, so rescaling must preserve it.
    v, f = icosphere
    k1 = curv_weight(Mesh(v.copy(), f), CFG_DEFORM)
    for s in (2.0, 0.5):
        ks = curv_weight(Mesh(s * v, f), CFG_DEFORM)
        assert (np.abs(ks - k1) / np.abs(k1).clip(1e-12)).max() < 1e-6


def test_curv_weight_localises_to_curvature(icosphere):
    # Uniform curvature -> near-constant weight: the sphere carries no localisation signal.
    v, f = icosphere
    ks = curv_weight(Mesh(v, f), CFG_DEFORM)
    assert ks.std() / ks.mean() < 0.15

    # Cube edges should have larger dimensionless curvature weights than flat panels.
    m = trimesh.creation.box(extents=(1, 1, 1))
    for _ in range(3):
        m = m.subdivide()
    cv, cf = (
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int32),
    )
    mesh = Mesh(cv, cf)
    k = curv_weight(mesh, CFG_DEFORM)
    Hf = np.abs(mean_curvature(mesh)[cf].mean(axis=1))
    flat = Hf < 1e-6
    assert flat.mean() > 0.4  # a cube is mostly flat panel
    assert k[flat].max() < 1e-9  # flat panels get ~zero weight
    assert k[~flat].mean() > 1e-2  # edges/corners spike
    assert spearmanr(k, Hf).correlation > 0.9  # the weight ranks by curvature


def test_l_curv_folds_weight_and_drops_background():
    # Weight foreground pixels by face curvature and ignore the background sentinel.
    kappa = torch.tensor([1.0, 5.0], dtype=DT)
    face_id = torch.tensor([[0, 1], [-1, 0]])
    pred = torch.zeros(3, 2, 2, dtype=DT, requires_grad=True)
    gt = torch.zeros(3, 2, 2, dtype=DT)
    gt[:, 0, 0], gt[:, 0, 1], gt[:, 1, 0], gt[:, 1, 1] = 0.1, 0.2, 999.0, 0.3
    out = l_curv(pred, gt, face_id, kappa)
    assert abs(float(out) - (1 * 0.1 + 5 * 0.2 + 1 * 0.3) / (1 + 5 + 1)) < 1e-12
    out.backward()
    assert torch.isfinite(pred.grad).all()
    assert (
        float(pred.grad[:, 1, 0].abs().sum()) == 0.0
    )  # background pixel gets no gradient


def test_l_curv_constant_weight_is_masked_mean_l1():
    # Constant curvature weights should recover mean L1 over valid pixels.
    face_id = torch.tensor([[0, 0], [-1, 0]])
    kappa = torch.full((1,), 3.0, dtype=DT)
    pred = torch.zeros(3, 2, 2, dtype=DT)
    gt = torch.zeros(3, 2, 2, dtype=DT)
    gt[:, 0, 0], gt[:, 0, 1], gt[:, 1, 1] = 0.1, 0.2, 0.6
    resid = (pred - gt).abs().mean(0)
    assert (
        abs(float(l_curv(pred, gt, face_id, kappa)) - float(resid[face_id >= 0].mean()))
        < 1e-12
    )


def test_face_normals_and_l_normal_survive_a_degenerate_face():
    """Regression for the undefined normal that ended two long runs at step ~2800.

    The cross-product norm was unfloored, so a collinear face produced ``0/0``, and
    a single NaN normal poisons the whole ``l_normal`` sum (and every gradient through it).

    ``l_normal`` already floors its *own* cross product for exactly this reason; the floor belongs
    one level up as well. The honest value for a face with no normal is the **zero vector** - there
    is no direction - which ``l_normal`` then reads at ``atan2(tiny, 0) = pi/2`` and penalises,
    rather than propagating a NaN. The fan below has one exactly-degenerate face (two coincident
    vertices), the case the float64 mesh-side guard can miss because the loss runs in float32."""
    verts = torch.tensor(
        [[1.0, 0, 0], [1.0, 0, 0], [0.5, 0.87, 0], [-0.5, 0.87, 0], [-1.0, 0, 0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=torch.long)

    n = face_normals_torch(verts, faces)
    assert torch.isfinite(n).all()
    assert torch.allclose(n[0], torch.zeros(3))  # no direction, not a NaN

    x = verts.clone().requires_grad_(True)
    pairs = interior_face_pairs(faces)
    w = face_adjacency_weights(x, faces, pairs).detach()
    loss = l_normal(face_normals_torch(x, faces), pairs, w, 0.17)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()


# The element-quality barrier is resolution-independent.


def test_l_quality_is_zero_on_an_equilateral_triangle_and_grows_with_distortion():
    """``E_conf = (a^2+b^2+c^2)/(4 sqrt3 A) >= 1`` with equality **iff** equilateral, so the loss
    ``mean(E_conf - 1)`` is a genuine shape measure: exactly zero on the optimum, positive
    otherwise, and unchanged by a global rescale (both numerator and denominator carry length^2)."""
    eq = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]], dtype=DT
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    # The area regularizer leaves a small O(eps^2/A^2) bias even on equilateral faces.
    assert float(l_distortion(eq, faces)) == pytest.approx(0.0, abs=1e-9)
    assert float(l_distortion(7.0 * eq, faces)) == pytest.approx(
        0.0, abs=1e-9
    )  # scale-invariant

    squashed = eq.clone()
    squashed[2, 1] *= 0.1  # same base, a tenth of the height
    assert float(l_distortion(squashed, faces)) > 1.0


def test_l_quality_diverges_as_a_face_flattens_but_stays_finite_and_differentiable_at_zero():
    """The barrier property, and the guard on it. ``E_conf ~ 1/A``, so flattening a triangle costs
    unboundedly more - that is what makes degeneracy something the descent avoids rather than
    damage read off the mesh afterwards. The area is **regularised**, not clamped
    (``A_eps = sqrt(A^2 + eps^2)``, derivative ``A/A_eps <= 1``): a clamp fixes the forward and
    leaves the backward carrying ``1/A_clamped``, which is the trap ``face_normals_torch`` was
    caught by. So a face of exactly zero area gives a large but finite value and a finite
    gradient."""
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    vals = []
    for h in (1e-1, 1e-2, 1e-3):
        v = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], dtype=DT)
        vals.append(float(l_distortion(v, faces)))
    assert vals[1] > 9.0 * vals[0] and vals[2] > 9.0 * vals[1]  # ~ 1/A, i.e. ~ 1/h

    flat = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = l_distortion(flat, faces)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flat.grad).all()


def test_l_quality_is_resolution_independent():
    """``E_conf`` is a dimensionless per-element shape statistic, so its **face mean** is the right
    summary and it does not move when the same surface is sampled finer - unlike a density being
    integrated, where a sum over elements is the right summary. Face-counting is also the only defensible
    weighting here: an area-weighted mean would let the smallest slivers, the ones that matter,
    weigh essentially nothing."""
    vals = []
    for sub in (2, 3, 4):
        m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
        vals.append(
            float(
                l_distortion(
                    torch.tensor(np.asarray(m.vertices), dtype=DT),
                    torch.tensor(np.asarray(m.faces), dtype=torch.long),
                )
            )
        )
    assert max(vals) - min(vals) < 0.01 * min(vals)


def test_the_normal_energy_is_a_sum_not_a_mean_and_only_the_sum_is_resolution_independent():
    """Why the loss divisor is a fixed calibration scale and not a live element count.

    ``L_normal = sum_e w_e psi(theta_e)`` discretises ``int |grad n|^2 dA``: on a smooth region the
    dihedral angle scales like ``kappa h`` and the interior-edge count like ``A/h^2``, so the **sum**
    is ``~ A kappa^2``, independent of resolution, while the **mean** is ``~ kappa^2 h^2`` and
    shrinks like ``h^2``. Sampling the unit sphere at three resolutions: the sum holds to 0.3 %
    (and lands on the continuum value ``1/2 int (kappa1^2 + kappa2^2) dA = 4 pi``), the mean falls by
    exactly a factor of 4 per subdivision.

    So the count divides the sum only to keep ``lambda2`` O(1) on a given mesh; it carries no
    discretisation meaning, and a divisor that moved with the mesh would silently re-weight the
    objective - invisible in the logs, too, since the reported value stays O(1) by construction."""
    sums, means = [], []
    for sub in (2, 3, 4):
        m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
        v = torch.tensor(np.asarray(m.vertices), dtype=DT)
        f = torch.tensor(np.asarray(m.faces), dtype=torch.long)
        pairs = interior_face_pairs(f)
        w = face_adjacency_weights(v, f, pairs)
        total = float(l_normal(face_normals_torch(v, f), pairs, w, 0.17))
        sums.append(total)
        means.append(total / len(pairs))
    assert max(sums) - min(sums) < 0.005 * min(sums)  # the sum is the invariant
    assert sums[0] == pytest.approx(
        4.0 * np.pi, rel=0.02
    )  # and it is the continuum value
    for a, b in zip(means, means[1:]):
        assert b == pytest.approx(a / 4.0, rel=0.05)  # the mean shrinks like h^2
