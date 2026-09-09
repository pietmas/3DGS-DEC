"""Conventions of the DTU Chamfer readout: the outlier rule discards rather than clamps, and
``pcu.chamfer_distance`` is *twice* the DTU convention (a 2x that hides inside the sanity gate's
own factor-of-2 tolerance). CPU, no DTU data required.
"""

import numpy as np
import point_cloud_utils as pcu


def _directed(a, b):
    """Mean nearest-neighbour distance a -> b, the DTU 'accuracy' when a is the reconstruction."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return d.min(axis=1)


def test_pcu_chamfer_is_twice_the_dtu_convention():
    # A 3-4-5 pair gives mean distance 5, sum 10, or squared distances 25 and 50.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0]])
    assert np.isclose(pcu.chamfer_distance(a, b), 10.0)  # sum of the two directed means
    dtu = 0.5 * (_directed(a, b).mean() + _directed(b, a).mean())
    assert np.isclose(dtu, 5.0)
    assert np.isclose(
        pcu.chamfer_distance(a, b), 2.0 * dtu
    )  # the trap, stated as an identity


def test_directed_means_are_asymmetric_and_pointwise():
    # Unequal point counts make the two directed means differ.
    a = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    b = np.array([[0.0, 0, 0]])
    assert np.isclose(_directed(a, b).mean(), 5.0)  # (0 + 10) / 2
    assert np.isclose(_directed(b, a).mean(), 0.0)  # b's only point sits on a[0]
    assert np.isclose(
        pcu.chamfer_distance(a, b), 5.0
    )  # their sum, again 2x the mean 2.5


def test_outlier_rule_drops_rather_than_clamps():
    # Drop distant pairs rather than clamping them to max_dist.
    d = np.array([1.0, 2.0, 3.0, 100.0])
    max_dist = 20.0
    dropped = d[d < max_dist].mean()
    clamped = np.minimum(d, max_dist).mean()
    assert np.isclose(dropped, 2.0)
    assert np.isclose(clamped, 6.5)
    assert dropped != clamped  # the two are not interchangeable

def test_reference_shuffle_seed_is_repeatable_and_scoped(monkeypatch):
    import sys
    from scripts import eval_chamfer

    draws = []
    factory = np.random.default_rng

    def fake_reference(path, run_name):
        draws.append(np.random.default_rng().permutation(50))

    monkeypatch.setattr(eval_chamfer.runpy, "run_path", fake_reference)
    for seed in (7, 7, 8):
        monkeypatch.setattr(sys, "argv", ["worker", str(seed), "eval.py"])
        eval_chamfer.seeded_reference()
        assert np.random.default_rng is factory
    assert np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
