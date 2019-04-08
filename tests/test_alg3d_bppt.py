"""Test tilted backpropagation algorithm"""
import numpy as np

import odtbrain

from common_methods import create_test_sino_3d, create_test_sino_3d_tilted, \
    cutout, get_test_parameter_set


def test_3d_backprop_phase_real():
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    # reference
    rref = list()
    for p in parameters:
        fref = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                         dtype=np.float64, onlyreal=True, **p)
        rref.append(cutout(fref))
    dataref = np.array(rref).flatten().view(float)

    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d_tilted(sino, angles, padval=0,
                                             dtype=np.float64, onlyreal=True,
                                             **p)
        r.append(cutout(f))
    data = np.array(r).flatten().view(float)
    assert np.allclose(data, dataref)


def test_3d_backprop_pad():
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    # reference
    rref = list()
    for p in parameters:
        fref = odtbrain.backpropagate_3d(sino, angles, padval=None,
                                         dtype=np.float64, onlyreal=False, **p)
        rref.append(cutout(fref))
    dataref = np.array(rref).flatten().view(float)

    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d_tilted(sino, angles, padval=None,
                                             dtype=np.float64, onlyreal=False,
                                             **p)
        r.append(cutout(f))
    data = np.array(r).flatten().view(float)

    assert np.allclose(data, dataref)


def test_3d_backprop_plane_rotation():
    """
    A very soft test to check if planar rotation works fine
    in the reconstruction with tilted angles.
    """
    parameters = get_test_parameter_set(1)
    results = []

    # These are specially selected angles that don't give high results.
    # Probably due to phase-wrapping, errors >2 may appear. Hence, we
    # call it a soft test.
    tilts = [1.1, 0.0, 0.234, 2.80922, -.29, 9.87]

    for angz in tilts:
        sino, angles = create_test_sino_3d_tilted(tilt_plane=angz, A=21)
        rotmat = np.array([
            [np.cos(angz), -np.sin(angz), 0],
            [np.sin(angz),  np.cos(angz), 0],
            [0,             0, 1],
        ])
        # rotate `tilted_axis` onto the y-z plane.
        tilted_axis = np.dot(rotmat, [0, 1, 0])

        rref = list()
        for p in parameters:
            fref = odtbrain.backpropagate_3d_tilted(sino, angles,
                                                    padval=None,
                                                    tilted_axis=tilted_axis,
                                                    padding=(False, False),
                                                    dtype=np.float64,
                                                    onlyreal=False,
                                                    **p)
            rref.append(cutout(fref))
        data = np.array(rref).flatten().view(float)
        results.append(data)

    for ii in np.arange(len(results)):
        assert np.allclose(results[ii], results[ii-1], atol=.2, rtol=.2)


def test_3d_backprop_plane_alignment_along_axes():
    """
    Tests whether the reconstruction is always aligned with
    the rotational axis (and not antiparallel).
    """
    parameters = get_test_parameter_set(1)
    p = parameters[0]
    results = []

    # These are specially selected angles that don't give high results.
    # Probably due to phase-wrapping, errors >2 may appear. Hence, we
    # call it a soft test.
    tilts = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

    for angz in tilts:
        sino, angles = create_test_sino_3d_tilted(tilt_plane=angz, A=21)
        rotmat = np.array([
            [np.cos(angz), -np.sin(angz), 0],
            [np.sin(angz),  np.cos(angz), 0],
            [0,             0, 1],
        ])
        # rotate `tilted_axis` onto the y-z plane.
        tilted_axis = np.dot(rotmat, [0, 1, 0])
        fref = odtbrain.backpropagate_3d_tilted(sino, angles,
                                                padval=None,
                                                tilted_axis=tilted_axis,
                                                padding=(False, False),
                                                dtype=np.float64,
                                                onlyreal=True,
                                                **p)
        results.append(fref)

    for ii in np.arange(len(results)):
        assert np.allclose(results[ii], results[ii-1], atol=.2, rtol=.2)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
