import numpy as np

import cloud_entity_algorithm._entity_helpers as eh


def test_estimate_cloud_type():
    t, s, c, n = eh.estimate_cloud_type(
        np.array([1, 0.7, 1, 1, 1, 1.5, 1.5]), np.array([1.4, 2, 1.5, 1.5, 1.5, 1.6, 4])
    )
    assert t == 6
    assert np.all(s == np.array([5, 6], dtype="int64"))
    assert np.all(c == np.array([1], dtype="int64"))
    assert n == 4


def test_calculate_cbh_n_cth():
    b, t = eh.calculate_cbh_n_cth(np.array([3, 0.7]))
    assert b == np.array([0.7])
    assert t == np.array([3.0])
    b, t = eh.calculate_cbh_n_cth(np.array([np.nan, 0.7]), remove_nan=False)
    assert b == np.array([0.7])
    assert np.isnan(t)


def test_get_stsc_macrophysics():
    r = eh.get_stsc_macrophysics(
        np.array([1, 2]), np.array([0, 2, 2.3, 1.0]), np.array([2, 2.5, 2.5, 1.5])
    )
    assert np.all(np.round(r, 5) - np.array([0.2, 0.5, 0.35, 0.15]) == 0)


def test_get_nb_Cu_cores():
    assert eh.get_nb_Cu_cores([1, 2, 5, 6, 7, 8, 9, 11, 12, 13], 1) == 3.0


def test_get_precip():
    assert (
        eh.get_precip(
            np.array([[np.nan, np.nan, np.nan], [0, 10, 0], [-50, -40, -30]]),
            np.array([[0, 1], [1, 2]]),
        )
        == 1
    )
