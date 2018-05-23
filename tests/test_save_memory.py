"""Test save memory options"""
import numpy as np

import odtbrain

from common_methods import create_test_sino_3d, get_test_parameter_set


def test_back3d():
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    parameters = get_test_parameter_set(2)
    # complex
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                      dtype=np.float64,
                                      save_memory=False, **p)
        r.append(f)
    # real
    r2 = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                      dtype=np.float64,
                                      save_memory=True, **p)
        r2.append(f)
    assert np.allclose(np.array(r), np.array(r2))


def test_back3d_tilted():
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    parameters = get_test_parameter_set(2)
    # complex
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d_tilted(sino, angles, padval=0,
                                             dtype=np.float64,
                                             save_memory=False, **p)
        r.append(f)
    # real
    r2 = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d_tilted(sino, angles, padval=0,
                                             dtype=np.float64,
                                             save_memory=True, **p)
        r2.append(f)

    assert np.allclose(np.array(r), np.array(r2))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
