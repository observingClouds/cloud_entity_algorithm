import numpy as np
from numpy.testing import assert_array_equal  # noqa: F401

import cloud_entity_algorithm.ccam as ccam


def test_add_boundary_to_cloud_slice():
    r = ccam.add_boundary_to_cloud_slice(np.array([[2, 2, 2], [2, 2, 2]]))
    assert_array_equal(
        r,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_f_cloud_edges():
    r = ccam.f_cloud_edges(
        np.array(
            [
                [0.0, 3.0, 0.0, 3.0],
                [0.0, 3.0, 0.0, 3.0],
                [3.0, 3.0, 0.0, 3.0],
                [3.0, 3.0, 0.0, 3.0],
                [0.0, 3.0, 0.0, 3.0],
            ]
        )
    )
    assert_array_equal(r, np.array([[3.0, 4.0, np.nan, 4.0], [2.0, 0.0, np.nan, 0.0]]))

    r = ccam.f_cloud_edges(
        np.array(
            [
                [0.0, 3.0, 0.0, 3.0],
                [0.0, 3.0, 0.0, 3.0],
                [3.0, 3.0, 0.0, 3.0],
                [3.0, 3.0, 0.0, 3.0],
                [0.0, 3.0, 0.0, 3.0],
            ]
        ),
        True,
    )
    assert_array_equal(r, np.array([[3.0, 4.0, np.nan, 4.0], [2.0, 0.0, np.nan, 0.0]]))
