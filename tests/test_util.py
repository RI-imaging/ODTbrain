import numpy as np

import radontea.util as util


def test_even():
    angles = np.linspace(0, np.pi, 18, endpoint=False)
    res = util.compute_angle_weights_1d(angles)
    assert np.allclose(res, 1, rtol=0, atol=1e-14)


def test_same_end_value():
    angles = np.linspace(0, np.pi, 18, endpoint=True)
    res = util.compute_angle_weights_1d(angles)
    assert len(res) == len(angles)
    assert np.allclose(np.mean(res), 1, rtol=0, atol=1e-14)
    assert np.allclose(res[1:-1], 18 / 17, rtol=0, atol=1e-14)
    assert np.allclose(res[0], 18 / 17 / 2, rtol=0, atol=1e-14)
    assert np.allclose(res[-1], 18 / 17 / 2, rtol=0, atol=1e-14)


def test_multiple_identical_angles():
    angles_0 = np.linspace(0, np.pi, 14, endpoint=False)
    angles_1 = np.roll(angles_0, -3)
    angles_2 = np.concatenate((np.ones(4)*angles_1[0], angles_1))
    angles = np.roll(angles_2, 3)
    res = util.compute_angle_weights_1d(angles)
    assert len(res) == len(angles)
    assert np.allclose(np.mean(res), 1, rtol=0, atol=1e-14)
    assert np.allclose(res[:3], 18 / 14, rtol=0, atol=1e-14)
    assert np.allclose(res[3+4+1:], 18 / 14, rtol=0, atol=1e-14)
    assert np.allclose(res[3:8], 18 / 14 / 5, rtol=0, atol=1e-14)
