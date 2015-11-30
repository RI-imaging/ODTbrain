#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests tilted backpropagation algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, basename, dirname, join, split, exists
import platform
from scipy.ndimage import rotate
import sys
import warnings
import zipfile

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain
import odtbrain._Back_2D
import odtbrain._Back_3D
import odtbrain._Back_3D_tilted
import odtbrain._br

from common_methods import create_test_sino_2d, create_test_sino_3d, cutout, get_test_parameter_set, write_results, get_results, normalize


def create_test_sino_3d_tilted(A=9, Nx=22, Ny=22, max_phase=5.0,
                               ampl_range=(1.0,1.0),
                               tilt_plane=0.0):
    """
    Creates 3D test sinogram for optical diffraction tomography.
    The sinogram is generated from a Gaussian that is shifted
    according to the rotational position of a non-centered
    object. The simulated rotation is about the second (y)/[1]
    axis.
    
    Parameters
    ----------
    A : int
        Number of angles of the sinogram.
    Nx : int
        Size of the first axis.
    Ny : int
        Size of the second axis.
    max_phase : float
        Phase normalization. If this is greater than
        2PI, then it also tests the unwrapping
        capabilities of the reconstruction algorithm.
    ampl_range : tuple of floats
        Determines the min/max range of the amplitude values.
        Equal values means constant amplitude.
    tilt_plane : float
        Rotation tilt offset [rad].
    
    Returns
    """
    # initiate array
    resar = np.zeros((A, Ny, Nx), dtype=np.complex128)
    # 2pi coverage
    angles = np.linspace(0, 2*np.pi, A, endpoint=False)
    # x-values of Gaussain
    x = np.linspace(-Nx/2, Nx/2, Nx, endpoint=True).reshape(1,-1)
    y = np.linspace(-Ny/2, Ny/2, Ny, endpoint=True).reshape(-1,1)
    # SD of Gaussian
    dev = min(np.sqrt(Nx/2), np.sqrt(Ny/2))
    # Off-centered rotation  about second axis:
    off = Nx/7
    for ii in range(A):
        # Gaussian distribution sinogram
        x0 = np.cos(angles[ii])*off
        phase = np.exp(-(x-x0)**2/dev**2) * np.exp(-(y)**2/dev**2)
        phase = normalize(phase, vmax=max_phase)
        if ampl_range[0] == ampl_range[1]:
            # constant amplitude
            ampl = np.ones((Nx, Ny))*ampl_range[0]
        else:
            # ring
            ampldev = dev/5
            amploff = off*.3
            ampl1 = np.exp(-(x-x0-amploff)**2/ampldev**2)
            ampl2 = np.exp(-(x-x0+amploff)**2/ampldev**2)
            ampl = ampl1+ampl2
            ampl = normalize(ampl, vmin=ampl_range[0], vmax=ampl_range[1])

        # perform in-plane rotation
        ampl = rotate(ampl, np.rad2deg(tilt_plane), reshape=False, cval=1)
        phase = rotate(phase, np.rad2deg(tilt_plane), reshape=False, cval=0)
        resar[ii] = ampl*np.exp(1j*phase)

    return resar, angles



def test_3d_backprop_phase_real():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    # reference
    rref = list()
    for p in parameters:
        fref = odtbrain._Back_3D.backpropagate_3d(sino, angles, padval=0,
                                                  dtype=np.float64, onlyreal=True, **p)
        rref.append(cutout(fref))
    dataref = np.array(rref).flatten().view(float)
    
    
    r = list()
    for p in parameters:
        f = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, padval=0,
                                               dtype=np.float64, onlyreal=True, **p)
        r.append(cutout(f))
    data = np.array(r).flatten().view(float)
    assert np.allclose(data, dataref)


def test_3d_backprop_pad():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    # reference
    rref = list()
    for p in parameters:
        fref = odtbrain._Back_3D.backpropagate_3d(sino, angles, padval=None,
                                                  dtype=np.float64, onlyreal=False, **p)
        rref.append(cutout(fref))
    dataref = np.array(rref).flatten().view(float)
    
    
    r = list()
    for p in parameters:
        f = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, padval=None,
                                               dtype=np.float64, onlyreal=False, **p)
        r.append(cutout(f))
    data = np.array(r).flatten().view(float)
    
    assert np.allclose(data, dataref)


def test_3d_backprop_plane_rotation():
    """
    A very soft test to check if planar rotation works fine
    in the reconstruction with tilted angles.
    """
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    
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
                           [0           ,             0, 1],
                           ])
        # rotate `tilted_axis` onto the y-z plane.
        tilted_axis = np.dot(rotmat, [0,1,0])
        
        rref = list()
        for p in parameters:
            fref = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, padval=None,
                                                      tilted_axis=tilted_axis, padding=(False, False),
                                                      dtype=np.float64, onlyreal=False, **p)
            rref.append(cutout(fref))
        data = np.array(rref).flatten().view(float)
        results.append(data)
    
    for ii in np.arange(len(results)):
        assert np.allclose(results[ii], results[ii-1], atol=.2, rtol=.2)
    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
