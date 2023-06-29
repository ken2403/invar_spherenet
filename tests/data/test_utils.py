from __future__ import annotations

import numpy as np
import pytest
from numpy import ndarray

from invarsphere.data.utils import nearest_vec2rot_mat_3d

param_nearest_vec2rot_mat_3d = [
    (np.array([[1, 0, 0], [0, 0, 1]])),
    (np.array([[1, 0, 0], [0, 0, -1]])),
    (np.random.randn(3, 2)),
    (np.random.randint(-10, 10, (3, 2))),
    (np.random.randint(-100, 100, (3, 2)) / 10),
    # error test
    (np.array([1, 1, 1])),
]


@pytest.mark.parametrize("nearest_vec", param_nearest_vec2rot_mat_3d)
def test_nearest_vec2rot_mat_3d(nearest_vec: ndarray):
    if nearest_vec.shape != (3, 2):
        with pytest.raises(ValueError) as e:
            nearest_vec2rot_mat_3d(nearest_vec)
        assert f"nearest_vec must be (3, 2) shape, but got {nearest_vec.shape}" in str(e.value)
        return

    out = nearest_vec2rot_mat_3d(nearest_vec)
    assert out.shape == (3, 3)
    assert np.allclose(0.0, np.dot(out[:, 0], out[:, 1]))
    assert np.allclose(0.0, np.dot(out[:, 1], out[:, 2]))
    assert np.allclose(0.0, np.dot(out[:, 2], out[:, 0]))
    assert np.allclose(1.0, np.dot(out[:, 0], out[:, 0]))
    assert np.allclose(1.0, np.dot(out[:, 1], out[:, 1]))
    assert np.allclose(1.0, np.dot(out[:, 2], out[:, 2]))
