#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests slow summing algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys


# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain._Back_2D

from common_methods import create_test_sino_2d, create_test_sino_3d, cutout, get_test_parameter_set, write_results, get_results

WRITE_RES = True


def test_2dsum():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(2)
    r = list()

    for p in parameters:
        f = odtbrain._Back_2D.sum_2d(sino, angles, **p)
        r.append(cutout(f))

    if WRITE_RES:
        write_results(myframe, r)
    assert np.allclose(np.array(r).flatten().view(float), get_results(myframe))

    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
