"""Test sinogram weighting"""
import numpy as np
import platform

import odtbrain

from common_methods import create_test_sino_3d, get_test_parameter_set


def test_3d_backprop_weights_even():
    """
    even weights
    """
    platform.system = lambda: "Windows"
    sino, angles = create_test_sino_3d()
    p = get_test_parameter_set(1)[0]
    f1 = odtbrain.backpropagate_3d(sino, angles, weight_angles=False, **p)
    f2 = odtbrain.backpropagate_3d(sino, angles, weight_angles=True, **p)
    data1 = np.array(f1).flatten().view(float)
    data2 = np.array(f2).flatten().view(float)
    assert np.allclose(data1, data2)


def test_3d_backprop_tilted_weights_even():
    """
    even weights
    """
    platform.system = lambda: "Windows"
    sino, angles = create_test_sino_3d()
    p = get_test_parameter_set(1)[0]
    f1 = odtbrain.backpropagate_3d_tilted(
        sino, angles, weight_angles=False, **p)
    f2 = odtbrain.backpropagate_3d_tilted(
        sino, angles, weight_angles=True, **p)
    data1 = np.array(f1).flatten().view(float)
    data2 = np.array(f2).flatten().view(float)
    assert np.allclose(data1, data2)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
