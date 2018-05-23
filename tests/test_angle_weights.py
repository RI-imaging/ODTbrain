"""Tests 1D angular weights"""
import numpy as np

import odtbrain

from common_methods import create_test_sino_2d, get_test_parameter_set


def test_angle_offset():
    """
    Tests if things are still correct when there is a 2PI offset in the angles.
    """
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(2)
    # reference
    r1 = []
    for p in parameters:
        f1 = odtbrain.backpropagate_2d(sino, angles, weight_angles=False, **p)
        r1.append(f1)
    # with offset
    angles[::2] += 2*np.pi*np.arange(angles[::2].shape[0])
    r2 = []
    for p in parameters:
        f2 = odtbrain.backpropagate_2d(sino, angles, weight_angles=False, **p)
        r2.append(f2)
    # with offset and weights
    r3 = []
    for p in parameters:
        f3 = odtbrain.backpropagate_2d(sino, angles, weight_angles=True, **p)
        r3.append(f3)
    assert np.allclose(np.array(r1).flatten().view(float),
                       np.array(r2).flatten().view(float))
    assert np.allclose(np.array(r2).flatten().view(float),
                       np.array(r3).flatten().view(float))


def test_angle_swap():
    """
    Test if everything still works, when angles are swapped.
    """
    sino, angles = create_test_sino_2d()
    # remove elements so that we can see that weighting works
    angles = angles[:-2]
    sino = sino[:-2, :]
    parameters = get_test_parameter_set(2)
    # reference
    r1 = []
    for p in parameters:
        f1 = odtbrain.backpropagate_2d(sino, angles, weight_angles=True, **p)
        r1.append(f1)
    # change order of angles
    order = np.argsort(angles % .5)
    angles = angles[order]
    sino = sino[order, :]
    r2 = []
    for p in parameters:
        f2 = odtbrain.backpropagate_2d(sino, angles, weight_angles=True, **p)
        r2.append(f2)
    assert np.allclose(np.array(r1).flatten().view(float),
                       np.array(r2).flatten().view(float))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
