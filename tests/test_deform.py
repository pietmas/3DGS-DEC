"""The differentiable mesh. The in-graph cotangent ``L0_x(x)`` and the torch
splat derivation carry the correct gradient to the vertices (checked against central finite
differences), and the torch derivation reproduces the numpy reference to fp32. CPU, fp64;
torch enters here for the first time in the test suite, so this file - not the DEC tests -
is where it is imported.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import torch
from omegaconf import OmegaConf
from scipy.spatial import Delaunay

from dec3dgs.losses import l_lap
from dec3dgs.mesh import Mesh, cotan_laplacian_torch
from dec3dgs.splats import (
    _shape_operator_frame,
    derive_splats,
    derive_splats_torch,
    face_shape_operators,
)
from tests.test_splats import cylinder_arrays

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "default.yaml").splats
DT = torch.float64


def corrugated_lattice(n=13, amp=0.05):
    """Triangular lattice on ``[0,1] x [0,ay]`` lifted by ``z = amp sin(2 pi x)``:
    near-equilateral (acute) faces, and a *cylindrical* curvature (one principal curvature
    ~ 0) so the shape operator is strongly non-umbilic on the slopes - the clean regime the
    FD check needs (no obtuse pull, no umbilic fallback)."""
    dx = 1.0 / (n - 1)
    dy = dx * np.sqrt(3.0) / 2.0
    rows = [
        np.c_[np.arange(n) * dx + 0.5 * dx * (j % 2), np.full(n, j * dy)]
        for j in range(n)
    ]
    xy = np.vstack(rows)
    faces = Delaunay(xy).simplices
    u, v = xy[faces[:, 1]] - xy[faces[:, 0]], xy[faces[:, 2]] - xy[faces[:, 0]]
    cw = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] < 0
    faces[cw] = faces[cw][:, ::-1]  # CCW seen from +z
    verts = np.c_[xy, amp * np.sin(2.0 * np.pi * xy[:, 0])]
    return verts, faces.astype(np.int32)


def pick_clean_vertex(mesh):
    """An interior vertex all of whose faces are acute (circumcenter inside -> no pull) and
    non-umbilic (a real eigengap -> no fallback); return the one with the largest safety
    margin. Assert one exists so a degenerate test mesh fails loudly rather than silently."""
    p = mesh.verts[mesh.faces]
    obtuse = np.zeros(len(mesh.faces), bool)
    for k in range(3):
        a, b = p[:, (k + 1) % 3] - p[:, k], p[:, (k + 2) % 3] - p[:, k]
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        obtuse |= cos < 0
    gap = np.abs(
        np.diff(np.linalg.eigvalsh(face_shape_operators(mesh)), axis=1)
    ).ravel()
    good = (~obtuse) & (gap > 50.0 * CFG.eigengap_min)
    boundary = set(mesh.boundary_vertices.tolist())
    best = None
    for vtx in range(len(mesh.verts)):
        inc = np.flatnonzero((mesh.faces == vtx).any(axis=1))
        if vtx in boundary or inc.size == 0 or not good[inc].all():
            continue
        margin = gap[inc].min()
        if best is None or margin > best[1]:
            best = (vtx, margin)
    assert best is not None, "no clean interior vertex - fix the test mesh"
    return best[0]


def fd_grad(f, x, vtx, eps=1e-6):
    """Central finite-difference gradient of scalar ``f(x)`` w.r.t. the 3 coordinates of
    vertex ``vtx`` (fp64)."""
    g = np.empty(3)
    for d in range(3):
        xp, xm = x.clone(), x.clone()
        xp[vtx, d] += eps
        xm[vtx, d] -= eps
        g[d] = (f(xp) - f(xm)).item() / (2.0 * eps)
    return g


def autograd_row(f, x, vtx):
    x = x.clone().requires_grad_(True)
    f(x).backward()
    return x.grad[vtx].numpy()


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)


@pytest.fixture(scope="module")
def lattice():
    verts, faces = corrugated_lattice()
    return (
        torch.tensor(verts, dtype=DT),
        torch.tensor(faces, dtype=torch.long),
        pick_clean_vertex(Mesh(verts, faces)),
    )


def test_l0x_gradient_matches_fd(lattice):
    # Perturbing a vertex changes both the cotangent operator and the displacement.
    x, faces, vtx = lattice
    x0 = x.clone()
    x0[:, 2] = 0.0

    def f(v):
        L, M = cotan_laplacian_torch(v, faces)
        Ld = torch.sparse.mm(L, v - x0)
        return (M[:, None] * Ld**2).sum()

    assert rel_err(autograd_row(f, x, vtx), fd_grad(f, x, vtx)) < 1e-4


def test_sigma_gradient_matches_fd(lattice):
    # Check gradients through position, frame and scale in one scalar.
    x, faces, vtx = lattice

    def f(v):
        return derive_splats_torch(v, faces, CFG, "shape_operator").sigma.sum()

    assert rel_err(autograd_row(f, x, vtx), fd_grad(f, x, vtx)) < 1e-4


def test_normal_gradient_matches_fd(lattice):
    # A fixed off-axis direction isolates the normal gradient without cancelling it.
    x, faces, vtx = lattice
    c = torch.tensor([0.3, -0.2, 0.9], dtype=DT)

    def f(v):
        n = derive_splats_torch(v, faces, CFG, "shape_operator").frame[:, :, 2]
        return (n * c).sum()

    assert rel_err(autograd_row(f, x, vtx), fd_grad(f, x, vtx)) < 1e-4


def test_gram_schmidt_sigma_gradient_matches_fd(lattice):
    # The warmup frame path (no eigendecomposition) must be differentiable too.
    x, faces, vtx = lattice

    def f(v):
        return derive_splats_torch(v, faces, CFG, "gram_schmidt").sigma.sum()

    assert rel_err(autograd_row(f, x, vtx), fd_grad(f, x, vtx)) < 1e-4


def _agree(verts, faces):
    sp = derive_splats(Mesh(verts, faces), CFG)
    st = derive_splats_torch(
        torch.tensor(verts, dtype=DT),
        torch.tensor(faces, dtype=torch.long),
        CFG,
        "shape_operator",
    )
    assert np.allclose(sp.mu, st.mu.numpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(sp.sigma, st.sigma.numpy(), rtol=1e-6, atol=1e-6)
    for c in range(3):  # frame columns agree up to a per-column sign
        cos = np.abs(
            np.einsum("ij,ij->i", sp.frame[:, :, c], st.frame[:, :, c].numpy())
        )
        assert cos.min() > 1.0 - 1e-6


def test_agreement_cylinder():
    # eigenframe branch, large eigengap (kappa_theta = 1/r vs 0): no branch ambiguity
    _agree(*cylinder_arrays())


def test_agreement_torus(torus):
    # eigenframe branch, torus is nowhere umbilic
    _agree(*torus)


def test_agreement_flat_disk(flat_disk):
    # Gram-Schmidt fallback branch (S = 0 everywhere, gap 0 << eigengap_min)
    _agree(*flat_disk)


# displacement-Laplacian fairing L_lap
def highfreq_energy(mesh, delta, frac=0.5):
    """Spectral energy of the displacement ``delta`` (V, 3) in the upper ``frac`` of the
    L0 spectrum. Coefficients ``a = V^T M delta`` against the generalised eigenvectors
    ``L0 v = lam M v`` (M-orthonormal, so ``delta = V a``); energy summed over the high-lam
    tail. The *clean* mesh is a fixed frequency reference - the basis does not move with x."""
    L0 = mesh.L0.toarray()
    Md = mesh.M.diagonal()
    lam, V = scipy.linalg.eigh(L0, np.diag(Md))
    a = V.T @ (Md[:, None] * delta.numpy())
    cut = int((1.0 - frac) * len(lam))
    return float((a[cut:] ** 2).sum())


def test_l_lap_denoises_without_shrinking(icosphere):
    # Fair displacement, not positions, so high-frequency noise fades without deflating x0.
    verts, faces = icosphere
    mesh = Mesh(verts, faces)
    clean = torch.tensor(verts, dtype=DT)
    facest = torch.tensor(faces, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    radial = clean / clean.norm(dim=1, keepdim=True)
    x0 = clean.clone()
    x = (
        clean + 0.03 * torch.randn(len(clean), generator=g, dtype=DT)[:, None] * radial
    ).requires_grad_(True)

    e_hi0 = highfreq_energy(mesh, (x - x0).detach())
    r0 = float((x - x.mean(0)).detach().norm(dim=1).mean())

    A = mesh.L0.T @ mesh.M @ mesh.L0
    lr = 0.9 / float(scipy.linalg.eigh(A.toarray(), eigvals_only=True)[-1])
    opt = torch.optim.SGD([x], lr=lr)
    for _ in range(400):
        L0_x, M = cotan_laplacian_torch(x, facest)
        loss = l_lap(x, x0, L0_x, M)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    e_hi1 = highfreq_energy(mesh, (x - x0).detach())
    r1 = float((x - x.mean(0)).detach().norm(dim=1).mean())
    assert e_hi1 < 0.25 * e_hi0  # high-frequency displacement faired away
    assert abs(r1 / r0 - 1.0) < 0.03  # no shrinkage: mean radius preserved to a few %


def test_l_lap_kernel(icosphere):
    # delta = 0 and delta = const (a pure translation, in ker L0) both cost nothing.
    verts, faces = icosphere
    x0 = torch.tensor(verts, dtype=DT)
    facest = torch.tensor(faces, dtype=torch.long)
    L0_x, M = cotan_laplacian_torch(x0, facest)
    assert float(l_lap(x0, x0, L0_x, M)) < 1e-10
    shift = x0 + torch.tensor([0.3, -0.7, 0.2], dtype=DT)
    Ls, Ms = cotan_laplacian_torch(shift, facest)
    assert float(l_lap(shift, x0, Ls, Ms)) < 1e-10


# umbilic frame blend + eigengap clipping
EGM = float(CFG.eigengap_min)
EGC = 1e-3  # deform.eigengap_clip; >= EGM (the blend threshold)
_T1 = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
_T2 = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
_N = torch.tensor([0.0, 0.0, 1.0], dtype=DT)


def _basis(F):
    return _T1.repeat(F, 1), _T2.repeat(F, 1), _N.repeat(F, 1)


def _sym(off, dif, kappa=1.0):
    """A batch of symmetric 2x2 shape operators ``kappa*I + [[dif, off], [off, -dif]]``:
    eigengap ``2 sqrt(dif^2 + off^2)``, so tiny ``off, dif`` is a near-umbilic face."""
    F = off.shape[0]
    S = torch.zeros(F, 2, 2, dtype=DT)
    S[:, 0, 0], S[:, 1, 1] = kappa + dif, kappa - dif
    S[:, 0, 1] = S[:, 1, 0] = off
    return S


def _naive_eigframe(S, t1, t2):
    """The un-blended, un-clipped principal direction (larger |kappa|) straight from
    ``eigh`` - the naive path; it spins and NaNs where the surface is umbilic."""
    w, v = torch.linalg.eigh(S)
    order = torch.argsort(-w.abs(), dim=1)
    v = torch.gather(v, 2, order[:, None, :].expand(-1, 2, -1))
    return v[:, 0, 0, None] * t1 + v[:, 1, 0, None] * t2


def _unsigned_angle(a, b):
    return torch.arccos(
        (a * b).sum(-1).abs().clamp(max=1.0)
    )  # principal directions: mod pi


def test_umbilic_frame_stable_and_naive_spins():
    # Near an umbilic, the reference blend should resist the raw eigenframe's spin.
    F = 200
    g = torch.Generator().manual_seed(0)
    off0 = 1e-6 * torch.randn(F, generator=g, dtype=DT)
    dif0 = 1e-6 * torch.randn(F, generator=g, dtype=DT)
    t1, t2, n = _basis(F)
    ref = t1
    eb, en = [], []
    for _ in range(8):
        off = off0 + 1e-7 * torch.randn(F, generator=g, dtype=DT)
        dif = dif0 + 1e-7 * torch.randn(F, generator=g, dtype=DT)
        S = _sym(off, dif)
        eb.append(_shape_operator_frame(S, t1, t2, n, ref, EGM, EGC))
        en.append(_naive_eigframe(S, t1, t2))
    eb, en = torch.stack(eb), torch.stack(en)

    def dispersion(E):  # angular spread of e1 about the ensemble mean
        m = E.mean(0)
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return _unsigned_angle(E, m.expand_as(E))

    assert (
        float(dispersion(eb).max()) < 1e-2
    )  # blended: no spin (pinned to the reference)
    assert float(dispersion(en).max()) > 1.0  # naive: spins across the quarter-circle


def test_umbilic_backward_finite_naive_nans():
    # At zero eigengap the guarded blend must have a finite backward.
    F = 50
    t1, t2, n = _basis(F)
    zero = torch.zeros(F, dtype=DT)

    Sb = _sym(zero, zero).requires_grad_(True)
    _shape_operator_frame(Sb, t1, t2, n, t1, EGM, EGC).sum().backward()
    assert torch.isfinite(Sb.grad).all()

    Sn = _sym(zero, zero).requires_grad_(True)
    _naive_eigframe(Sn, t1, t2).sum().backward()
    assert not torch.isfinite(Sn.grad).all()


def test_blend_inert_on_anisotropic():
    # Away from umbilics, the blend should recover the ordinary eigenframe.
    F = 20
    t1, t2, n = _basis(F)
    S = _sym(
        torch.full((F,), 0.3, dtype=DT), torch.full((F,), 1.5, dtype=DT), kappa=0.5
    )
    e_blend = _shape_operator_frame(S, t1, t2, n, t1, EGM, EGC)
    e_eig = _naive_eigframe(S, t1, t2)
    e_eig = e_eig / e_eig.norm(dim=-1, keepdim=True)
    assert float(_unsigned_angle(e_blend, e_eig).max()) < 1e-6


def _warmup_cfg(warmup_steps=100, n_ramp=100):
    return OmegaConf.create({"warmup_steps": warmup_steps, "n_ramp": n_ramp})


def test_warmup_schedule_holds_then_ramps_linearly():
    # Warm up with alpha=0, switch frames, then ramp alpha to 1 without a jump.
    from dec3dgs.deform import WarmupSchedule

    W, R = 80, 100
    sched = WarmupSchedule(_warmup_cfg(warmup_steps=W, n_ramp=R))
    alphas = [sched.alpha(step) for step in range(1, 401)]
    for step in range(1, 401):  # frame tracks warming_up exactly
        assert sched.frame_mode(step) == (
            "gram_schmidt" if sched.warming_up(step) else "shape_operator"
        )
    assert sched.warming_up(W) and not sched.warming_up(
        W + 1
    )  # warmup is [1, W], ramp after
    assert sched.ramp_start == W
    assert all(a == 0.0 for a in alphas[:W])  # flat 0 through the warmup
    assert alphas[W - 1 + R // 2] == pytest.approx(
        0.5, abs=1e-9
    )  # linear, halfway up the ramp
    assert alphas[W - 1 + R + 20] == 1.0  # saturates at 1 past n_ramp
    assert all(
        b >= a - 1e-12 for a, b in zip(alphas, alphas[1:])
    )  # monotone non-decreasing


def test_warmup_schedule_length_and_disabled():
    from dec3dgs.deform import WarmupSchedule

    # The budget is exactly warmup_steps: warming through it, ramping the step after.
    sched = WarmupSchedule(_warmup_cfg(warmup_steps=200))
    assert sched.warming_up(200) and not sched.warming_up(201)
    assert sched.alpha(200) == 0.0 and sched.alpha(201) > 0.0
    # Disabling warmup activates the shape-operator frame and full loss immediately.
    off = WarmupSchedule(_warmup_cfg(), disabled=True)
    assert not off.warming_up(1) and off.frame_mode(1) == "shape_operator"
    assert off.alpha(1) == 1.0 and off.alpha(999) == 1.0


def test_shape_operator_frame_finite_backward_on_plane(flat_disk):
    # The shape-operator path must backpropagate safely on a fully flat mesh.
    verts, faces = flat_disk
    facest = torch.tensor(faces, dtype=torch.long)
    x = torch.tensor(verts, dtype=DT, requires_grad=True)
    st = derive_splats_torch(x, facest, CFG, "shape_operator", eigengap_clip=EGC)
    (st.frame[:, :, 0] * torch.tensor([0.3, -0.2, 0.9], dtype=DT)).sum().backward()
    assert torch.isfinite(x.grad).all()
    t1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    cos = np.abs(np.einsum("ij,ij->i", st.frame[:, :, 0].detach().numpy(), t1))
    assert cos.min() > 1.0 - 1e-9  # blend saturates to the reference on a plane


def test_repair_nonfinite_grads_zeroes_only_the_bad_rows():
    # Repair only bad gradient rows so a local defect does not freeze every vertex.
    from dec3dgs.deform import repair_nonfinite_grads

    good, bad = torch.zeros(4, 3), torch.zeros(5, 3)
    good.grad, bad.grad = torch.ones(4, 3), torch.ones(5, 3)
    assert repair_nonfinite_grads(verts=good, sh_dc=bad) == []  # clean: nothing touched
    bad.grad[2, 1] = float("nan")
    hit = repair_nonfinite_grads(verts=good, sh_dc=bad)
    # Check row IDs, not just counts; equal counts can refer to different vertices.
    assert len(hit) == 1 and hit[0][0] == "sh_dc" and hit[0][1].tolist() == [2]
    assert torch.equal(bad.grad[2], torch.zeros(3))  # the bad row is zeroed...
    assert torch.equal(bad.grad[[0, 1, 3, 4]], torch.ones(4, 3))  # ...and only that row
    assert torch.equal(good.grad, torch.ones(4, 3))  # a clean parameter is untouched
    # Report every affected parameter, including infinities, but ignore absent gradients.
    bad.grad[4, 0] = float("inf")
    none_grad, other = torch.zeros(2, 3), torch.zeros(3, 3)
    other.grad = torch.full((3, 3), float("nan"))
    hits = dict(
        (n, r.tolist())
        for n, r in repair_nonfinite_grads(
            opacity_logit=none_grad, sh_dc=bad, verts=other
        )
    )
    assert hits == {"sh_dc": [4], "verts": [0, 1, 2]}
    assert repair_nonfinite_grads(opacity_logit=none_grad) == []
