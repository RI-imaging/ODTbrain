"""Test Fourier mapping algorithm"""
import sys

import numpy as np
import odtbrain
import pytest

from common_methods import create_test_sino_2d, cutout, \
    get_test_parameter_set, write_results, get_results

WRITE_RES = False


@pytest.mark.xfail(True, reason="Unexplained issue #13 and new algorithm #19")
@pytest.mark.filterwarnings(
    "ignore::odtbrain.warn.ImplementationAmbiguousWarning")
def test_2d_fmap():
    myframe = sys._getframe()
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(1)
    r = []
    for p in parameters:
        f = odtbrain.fourier_map_2d(sino, angles, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    assert np.allclose(np.array(r).flatten().view(float), get_results(myframe))
