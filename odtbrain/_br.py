#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Pre- and post-processing for diffraction tomography

Tomographic data sets consist of detector images for different
rotational positions :math:`\phi_0` of the objects. The methods
described here include pre-processing filters that are applied 
to the measured field :math:`u(\mathbf{r})` to achieve the Radon or the
Rytov approximation. To obtain the refractive index map :math:`n(\mathbf{r})`
from an object function :math:`f(\mathbf{r})`, this submodule 
also provides post-processing methods 
(for backprojection/Radon or backpropagation/Born,Rytov).
"""
import numpy as np
from scipy.stats import mode

import unwrap

__all__ = ["odt_to_ri", "opt_to_ri",
           "sinogram_as_rytov", "sinogram_as_radon"]


def align_unwrapped(sino):
    """tries to align an unwrapped phase array
    
    all operations are performed in-place
    """
    samples = list()
    if len(sino.shape) == 2:
        # 2D
        # take 1D samples at beginning and end of array
        samples.append(sino[:,0])
        samples.append(sino[:,1])
        samples.append(sino[:,2])
        samples.append(sino[:,-1])
        samples.append(sino[:,-2])
        
    elif len(sino.shape) == 3:
        # 3D
        # take 1D samples at beginning and end of array
        samples.append(sino[:,0, 0])
        samples.append(sino[:,0,-1])
        samples.append(sino[:,-1,0])
        samples.append(sino[:,-1,-1])
        samples.append(sino[:,0,1])
        
    # find discontinuities in the samples
    steps = np.zeros((len(samples), samples[0].shape[0]))
    for i in range(len(samples)):
        t = np.unwrap(samples[i])
        steps[i] = samples[i] - t
    
    # if the majority believes so, add a step of PI
    remove = mode(steps, axis=0)[0][0]

    # obtain divmod min
    twopi = 2*np.pi

    minimum = divmod_neg(np.min(sino), twopi)[0]
    remove += minimum*twopi
    
    for i in range(len(sino)):
        sino[i] -= remove[i]


def divmod_neg(a,b):
    """returns divmod with closest result to zero"""
    q, r = divmod(a,b)
    # make sure r is close to zero
    if np.abs(r)>b/2:
        r -= b * np.sign(r)
        q -= np.sign(r)
    return q, r


def odt_to_ri(f, res, nm):
    u""" Converts the ODT object function to refractive index.

    In :abbr:`ODT (Optical Diffraction Tomography)`, the object function
    is defined by the Helmholtz equation

    .. math::

        f(\mathbf{r})  =  k_\mathrm{m}^2 \\left[
            \\left( \\frac{n(\mathbf{r})}{n_\mathrm{m}} \\right)^2 - 1
            \\right]

    with :math:`k_\mathrm{m} = \\frac{2\pi n_\mathrm{m}}{\lambda}`.
    By inverting this equation, we obtain the refractive index
    :math:`n(\mathbf{r})`.

    .. math::

        n(\mathbf{r})  = n_\mathrm{m} 
            \sqrt{\\frac{f(\mathbf{r})}{k_\mathrm{m}^2} + 1 }

    Parameters
    ----------
    f : n-dimensional ndarray
        The reconstructed object function :math:`f(\mathbf{r})`.
    res : float
        The size of the vacuum wave length :math:`\lambda` in pixels.
    nm : float
        The refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`f(\mathbf{r})`.

    Returns
    -------
    ri : n-dimensional ndarray
        The complex refractive index :math:`n(\mathbf{r})`.


    Notes
    -----
    Because this function computes the root of a complex number, there
    are several solutions to the refractive index. Always the positive
    (real) root of the refractive index is used.

    """
    km = (2 * np.pi * nm) / res
    ri = nm * np.sqrt(f / km**2 + 1)
    # Always take the positive root as the refractive index.
    # Because f can be imaginary, numpy cannot return the correct
    # positive root of f. However, we know that *ri* must be postive and
    # thus we take the absolute value of ri.
    # This also is what happens in Slaneys
    # diffract/Src/back.c in line 414.
    negrootcoord = np.where(ri.real < 0)
    ri[negrootcoord] *= -1
    return ri


def opt_to_ri(f, res, nm):
    u""" Converts the OPT object function to refractive index.

    In :abbr:`OPT (Optical Projection Tomography)`, the object function
    is computed from the raw phase data. This method converts phase data
    to refractive index data.

    .. math::

        n(\mathbf{r})  = n_\mathrm{m} + 
            \\frac{f(\mathbf{r}) \cdot \lambda}{2 \pi}

    Parameters
    ----------
    f : n-dimensional ndarray
        The reconstructed object function :math:`f(\mathbf{r})`.
    res : float
        The size of the vacuum wave length :math:`\lambda` in pixels.
    nm : float
        The refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`f(\mathbf{r})`.

    Returns
    -------
    ri : n-dimensional ndarray
        The complex refractive index :math:`n(\mathbf{r})`.

    Notes
    -----
    This function is not meant to be used with diffraction tomography
    data. For ODT, use :py:func:`odt_to_ri` instead.

    """
    ri = nm + f / (2 * np.pi) * res
    return ri


def sinogram_as_radon(uSin, align=True):
    u""" Computes the phase from a complex wave field sinogram.

    This step is essential when using the ray approximation before
    computation of the refractive index with the inverse Radon
    transform.

    Parameters
    ----------
    uSin : 2d or 3d complex ndarray
        The background-corrected sinogram of the complex scattered wave
        :math:`u(\mathbf{r})/u_0(\mathbf{r})`. The first axis iterates
        through the angles :math:`\phi_0`.
    align : bool
        Tries to correct for a phase offset in the phase sinogram.

    Returns
    -------
    phase : 2d or 3d real ndarray
        The unwrapped phase array corresponding to ``uSin``.

    Notes
    -----
    The phase-unwrapping is performed with the `unwrap`_ package.

    .. _unwrap: https://pypi.python.org/pypi/unwrap
    """
    ndims = len(uSin.shape)

    if ndims == 2:
        # unwrapping is very important
        phiR = np.unwrap(np.angle(uSin), axis=-1)
    else:
        # Unwrap gets the dimension of the problem from the input
        # data. Since we have a sinogram, we need to pass it the
        # slices one by one.
        phiR = np.angle(uSin)
        for i in range(len(phiR)):
            phiR[i] = unwrap.unwrap(phiR[i])

    if align:
        align_unwrapped(phiR)

    return phiR


def sinogram_as_rytov(uSin, u0=1, align=True):
    u""" Converts the complex wave field sinogram to Rytov data

    This method applies the Rytov approximation to the
    recorded complex wave sinogram. To achieve this, the following
    filter is applied:

    .. math::
        u_\mathrm{B}(\mathbf{r}) = u_\mathrm{0}(\mathbf{r})
            \ln\!\\left(  
            \\frac{u_\mathrm{R}(\mathbf{r})}{u_\mathrm{0}(\mathbf{r})}
             +1 \\right)

    This filter step effectively replaces the Born approximation
    :math:`u_\mathrm{B}(\mathbf{r})` with the Rytov approximation
    :math:`u_\mathrm{R}(\mathbf{r})`, assuming that the scattered
    field is equal to
    :math:`u(\mathbf{r})\\approx u_\mathrm{R}(\mathbf{r})+
    u_\mathrm{0}(\mathbf{r})`.
    

    Parameters
    ----------
    uSin : 2d or 3d complex ndarray
        The sinogram of the complex wave 
        :math:`u_\mathrm{R}(\mathbf{r}) + u_\mathrm{0}(\mathbf{r})`.
        The first axis iterates through the angles :math:`\phi_0`.
    u0 : ndarray of dimension as ``uSin`` or less, or int.
        The incident plane wave 
        :math:`u_\mathrm{0}(\mathbf{r})` at the detector.
        If ``u0`` is "1", it is assumed that the data is already
        background-corrected (
        ``uSin`` :math:`= \\frac{u_\mathrm{R}(\mathbf{r})}{
        u_\mathrm{0}(\mathbf{r})} + 1`
        ). Note that if the reconstruction distance :math:`l_\mathrm{D}`
        of the original experiment is non-zero and `u0` is set to 1,
        then the reconstruction will be wrong; the field is not focused
        to the center of the reconstruction volume.
    align : bool
        Tries to correct for a phase offset in the phase sinogram.

    Returns
    -------
    uB : 2d or 3d real ndarray
        The Rytov-filtered complex sinogram 
        :math:`u_\mathrm{B}(\mathbf{r})`.

    Notes
    -----
    The phase-unwrapping is performed with the `unwrap`_ package.

    .. _unwrap: https://pypi.python.org/pypi/unwrap    """
    ndims = len(uSin.shape)

    # imaginary part of the complex Rytov phase
    phiR = np.angle(uSin / u0)

    # real part of the complex Rytov phase
    lna = np.log(np.absolute(uSin / u0))

    if ndims == 2:
        # unwrapping is very important
        phiR[:] = np.unwrap(phiR, axis=-1)
    else:
        # Unwrap gets the dimension of the problem from the input
        # data. Since we have a sinogram, we need to pass it the
        # slices one by one.
        for i in range(len(phiR)):
            phiR[i] = unwrap.unwrap(phiR[i])

    if align:
        align_unwrapped(phiR)

    #rytovSin = u0*(np.log(a/a0) + 1j*phiR)
    # u0 is one - we already did background correction

    # complex rytov phase:
    rytovSin = 1j * phiR + lna
    return u0 * rytovSin
