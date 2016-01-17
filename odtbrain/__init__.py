#!/usr/bin/env python
# -*- coding: utf-8 -*-
u""" Algorithms for scalar diffraction tomography.

This package provides reconstruction algorithms for diffraction
tomography in two and three dimensions.

Installation
------------
ODTbrain is a python library that is compatible to Python 2 and Python 3.
To install via the `Python package manager (PyPi)`_, simply issue the
following command.

    pip install odtbrain


The `FFTW3 library`_ and the scientific python packages
:py:mod:`numpy` and :py:mod:`scipy` are required by ODTbrain.
If the above command does not work, please refer to the 
installation instructions at the `GitHub repository`_.

.. _`FFTW3 library`: http://fftw.org
.. _`GitHub repository`: https://github.com/RI-imaging/ODTbrain
.. _`Python package manager (PyPi)`: https://pypi.python.org/pypi/odtbrain/


Theoretical background
----------------------
A detailed summary of the underlying theory is available in [1]_.

The Fourier diffraction theorem states, that the Fourier transform
:math:`\widehat{U}_{\mathrm{B},\phi_0}(\mathbf{k_\mathrm{D}})` of 
the scattered field :math:`u_\mathrm{B}(\mathbf{r_D})`, measured at 
a certain angle :math:`\phi_0`, is distributed along a circular arc 
(2D) or along a semi-spherical surface (3D) in Fourier space,
synthesizing the Fourier transform 
:math:`\widehat{F}(\mathbf{k})` of the object function 
:math:`f(\mathbf{r})` ([4]_, [5]_).

.. math::

   \widehat{F}(k_\mathrm{m}(\mathbf{s - s_0}))= 
        - \sqrt{\\frac{2}{\pi}}  \\frac{i k_\mathrm{m}}{a_0} 
        M \widehat{U}_{\mathrm{B},\phi_0}(\mathbf{k_\mathrm{D}}) 
        \exp \! \\left(-i k_\mathrm{m} M l_\mathrm{D} \\right)
    
In this notation, 
:math:`k_\mathrm{m}` is the wave number,
:math:`\mathbf{s_0}` is the norm vector pointing at :math:`\phi_0`,
:math:`M=\sqrt{1-s_\mathrm{x}^2}` (2D) and
:math:`M=\sqrt{1-s_\mathrm{x}^2-s_\mathrm{y}^2}` (3D)
enforces the spherical constraint, and
:math:`l_\mathrm{D}` is the distance from the center of the object
function :math:`f(\mathbf{r})` to the detector plane
:math:`\mathbf{r_D}`.


Fields of Application
---------------------
The algorithms presented here are based on the (scalar) Helmholtz
equation. Furthermore, the Born and Rytov approximations to the
scattered wave :math:`u(\mathbf{r})` are used to linearize the
problem for a straight-forward inversion.

The package is intended for optical diffraction
tomography to determine the refractive index of biological cells.
Because the Helmholtz equation is only an approximation to the
Maxwell equations, describing the propagation of light, 
:abbr:`FDTD (Finite Difference Time Domain)` simulations were performed
to test the reconstruction algorithms within this package.
The algorithms present in this package should also be valid for the
following cases, but have not been tested appropriately:

* tomographic measurements of absorbing materials (complex refractive 
  index :math:`n(\mathbf{r})`)

* ultrasonic diffraction tomography, which is correctly described by
  the Helmholtz equation

How to cite
-----------
If you use ODTbrain in a scientific publication, please cite at least
[2]_. 


"""
from ._Back_2D import backpropagate_2d, fourier_map_2d, sum_2d
from ._Back_3D import backpropagate_3d
from ._Back_3D_tilted import backpropagate_3d_tilted
from ._br import odt_to_ri, opt_to_ri, sinogram_as_radon, sinogram_as_rytov
from ._version import version as __version__
from ._version import longversion as __version_full__

__author__ = u"Paul Müller"
__license__ = "BSD (3 clause)"


# Shared variable used by 3D backpropagation
_shared_array = None