"""Test apple core correction"""
import multiprocessing as mp
import sys

import numpy as np

import odtbrain

from common_methods import create_test_sino_3d, cutout, \
    get_test_parameter_set, write_results, get_results


WRITE_RES = False


def test_apple_core_3d_values():
    try:
        odtbrain.apple.apple_core_3d(shape=(10, 10, 5),
                                     res=.1,
                                     nm=1)
    except ValueError:
        pass
    else:
        assert False, "bad input shape should raise ValueError"


def test_correct_counter():
    count = mp.Value("I", lock=True)
    max_count = mp.Value("I", lock=True)

    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                  dtype=np.float64,
                                  copy=False, **p)
    odtbrain.apple.correct(ri=f,
                           res=p["res"],
                           nm=p["nm"],
                           ri_min=None,
                           ri_max=None,
                           enforce_envelope=.95,
                           max_iter=100,
                           min_diff=0.01,
                           count=count,
                           max_count=max_count)

    assert count.value == max_count.value


def test_correct_reproduce():
    myframe = sys._getframe()
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    sryt = odtbrain.sinogram_as_rytov(uSin=sino, u0=1, align=False)
    f = odtbrain.backpropagate_3d(sryt, angles, padval=0,
                                  dtype=np.float64,
                                  copy=False, **p)
    ri = odtbrain.odt_to_ri(f, res=p["res"], nm=p["nm"])
    rc = odtbrain.apple.correct(ri=ri,
                                res=p["res"],
                                nm=p["nm"],
                                ri_min=None,
                                ri_max=None,
                                enforce_envelope=.95,
                                max_iter=100,
                                min_diff=0.01)
    ro = cutout(rc)

    if WRITE_RES:
        write_results(myframe, ro)
    # convert to double precision for old test data
    data = np.array(ro, dtype=np.complex128).flatten().view(float)
    assert np.allclose(data, get_results(myframe))


def test_correct_values():
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                  dtype=np.float64,
                                  copy=False, **p)
    try:
        odtbrain.apple.correct(ri=f,
                               res=p["res"],
                               nm=p["nm"],
                               enforce_envelope=1.05,
                               )
    except ValueError:
        pass
    else:
        assert False, "`enforce_envelope` must be in [0, 1]"


def test_envelope_gauss_shape():
    """Make sure non-cubic input shape works"""
    # non-cubic reconstruction volume (1st and 3rd axis still have same length)
    shape = (60, 50, 60)
    ftdata = np.ones(shape)
    core = odtbrain.apple.apple_core_3d(shape=shape, res=.1, nm=1)
    envlp = odtbrain.apple.envelope_gauss(ftdata=ftdata, core=core)
    assert envlp.shape == shape


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
