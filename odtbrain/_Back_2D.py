#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 2D reconstruction in optical tomography with the Born approximation

The first Born approximation for a 2D scattering problem with a plane
wave 
:math:`u_0(\mathbf{r}) = a_0 \exp(-ik_\mathrm{m}\mathbf{s_0r})`
reads:

.. math::
    u_\mathrm{B}(\mathbf{r}) = \iint \!\! d^2r' 
        G(\mathbf{r-r'}) f(\mathbf{r'}) u_0(\mathbf{r'})

The Green's function in 2D is the zero-order Hankel function
of the first kind:

.. math::
    G(\mathbf{r-r'}) = \\frac{i}{4} 
        H_0^\mathrm{(1)}(k_\mathrm{m} \\left| \mathbf{r-r'} \\right|) 

Solving for :math:`f(\mathbf{r})` yields the Fourier diffraction theorem
in 2D

.. math::
    \widehat{F}(k_\mathrm{m}(\mathbf{s-s_0})) = 
        - \sqrt{\\frac{2}{\pi}} 
        \\frac{i k_\mathrm{m}}{a_0} M
        \widehat{U}_{\mathrm{B},\phi_0}(k_\mathrm{Dx})
        \exp \! \\left(-i k_\mathrm{m} M l_\mathrm{D} \\right)
    
where 
:math:`\widehat{F}(k_\mathrm{x}, k_\mathrm{z})`
is the Fourier transformed object function and 
:math:`\widehat{U}_{\mathrm{B}, \phi_0}(k_\mathrm{Dx})` is the
Fourier transformed complex wave that travels along :math:`\mathbf{s_0}`
(in the direction of :math:`\phi_0`) measured at the detector
:math:`\mathbf{r_D}`.


The following identities are used:

.. math::
    k_\mathrm{m} (\mathbf{s-s_0}) &= k_\mathrm{Dx} \, \mathbf{t_\perp} +
    k_\mathrm{m}(M - 1) \, \mathbf{s_0}
    
    \mathbf{s_0} &= \\left(p_0 , \, M_0 \\right) = 
    (-\sin\phi_0, \, \cos\phi_0)
    
    \mathbf{t_\\perp} &= \\left(- M_0 , \, p_0 \\right) = 
    (\cos\phi_0, \, \sin\phi_0)


"""
from __future__ import division, print_function

import numpy as np
import scipy.interpolate as intp
import scipy.ndimage

from . import util

__all__ = ["backpropagate_2d", "fourier_map_2d", "sum_2d"]
_verbose = 1


def backpropagate_2d(uSin, angles, res, nm, lD=0, coords=None,
                     weight_angles=True,
                     onlyreal=False, padding=True, padval=0,
                     jmc=None, jmm=None, verbose=_verbose):
    u""" 2D backpropagation with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)` 
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This method implements the 2D backpropagation algorithm [1]_

    .. math::
        f(\mathbf{r}) = 
            -\\frac{i k_\mathrm{m}}{2\pi}
            \\sum_{j=1}^{N} \! \Delta \phi_0 D_{-\phi_j} \!\!
            \\left \{
            \\text{FFT}^{-1}_{\mathrm{1D}}
            \\left \{
            \\left| k_\mathrm{Dx} \\right|  
            \\frac{\\text{FFT}_{\mathrm{1D}} \\left \{
            u_{\mathrm{B},\phi_j}(x_\mathrm{D}) \\right \}
            }{u_0(l_\mathrm{D})}
            \exp \! \\left[i k_\mathrm{m}(M - 1) \cdot 
            (z_{\phi_j}-l_\mathrm{D}) \\right]
            \\right \} 
            \\right \}

    with the forward :math:`\\text{FFT}_{\mathrm{1D}}` and inverse 
    :math:`\\text{FFT}^{-1}_{\mathrm{1D}}` 1D fast Fourier transform, the
    rotational operator :math:`D_{-\phi_j}`, the angular distance between the
    projections :math:`\Delta \phi_0`, the ramp filter in Fourier space
    :math:`|k_\mathrm{Dx}|`, and the propagation distance 
    :math:`(z_{\phi_j}-l_\mathrm{D})`.

    Parameters
    ----------
    uSin : (A,N) ndarray
        Two-dimensional sinogram of line recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D})`
        divided by the incident plane wave :math:`u_0(l_\mathrm{D})`
        measured at the detector.
    angles : (A,) ndarray
        Angular positions :math:`\phi_j` of ``uSin`` in radians.
    res : float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm : float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD : float
        Distance from center of rotation to detector plane 
        :math:`l_\mathrm{D}` in pixels.
    coords : None [(2,M) ndarray]
        Computes only the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    weight_angles : bool
        If ``True``, weights each backpropagated projection with a factor
        proportional to the angular distance between the neighboring
        projections. 
        
        .. math::
            \Delta \phi_0 \\longmapsto \Delta \phi_j = \\frac{\phi_{j+1} - \phi_{j-1}}{2}
        
        .. versionadded:: 0.1.1
    onlyreal : bool
        If ``True``, only the real part of the reconstructed image
        will be returned. This saves computation time.
    padding : bool
        Pad the input data to the second next power of 2 before
        Fourier transforming. This reduces artifacts and speeds up
        the process for input image sizes that are not powers of 2.
    padval : float
        The value used for padding. This is important for the Rytov
        approximation, where an approximate zero in the phase might
        translate to 2πi due to the unwrapping algorithm. In that
        case, this value should be a multiple of 2πi. 
        If ``padval`` is ``None``, then the edge values are used for
        padding (see documentation of :func:`numpy.pad`).
    jmc, jmm : instance of :func:`multiprocessing.Value` or ``None``
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step ``jmc.value`` is
        incremented ``jmm.value`` times. ``jmm.value`` is set at the 
        beginning.
    verbose : int
        Increment to increase verbosity.


    Returns
    -------
    f : ndarray of shape (N,N), complex if `onlyreal` is `False`
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) = 
        k_m^2 \\left(\\left(\\frac{n(x,z)}{n_m}\\right)^2 -1\\right)`


    See Also
    --------
    odt_to_ri : conversion of the object function :math:`f(\mathbf{r})` 
        to refractive index :math:`n(\mathbf{r})`.

    radontea.backproject : backprojection based on the Fourier slice
        theorem.
    
    Notes
    -----
    Do not use the parameter `lD` in combination with the Rytov
    approximation - the propagation is not correctly described.
    Instead, numerically refocus the sinogram prior to converting
    it to Rytov data (using e.g. :func:`odtbrain.sinogram_as_rytov`)
    with a numerical focusing algorithm (available in the Python
    package :py:mod:`nrefocus`).
    """
    ##
    ##
    # TODO:
    # - combine the 2nd filter and the rotation in the for loop
    # to save memory. However, memory is not a big issue in 2D.
    ##
    ##
    A = angles.shape[0]
    if jmm is not None:
        jmm.value = A + 2
    # Check input data
    assert len(uSin.shape) == 2, "Input data `uB` must have shape (A,N)!"
    assert len(uSin) == A, "`len(angles)` must be  equal to `len(uSin)`!"
    
    if coords is not None:
        raise NotImplementedError("Output coordinates cannot yet" +
                                  " be set for the 2D backrpopagation algorithm.")
    # Cut-Off frequency
    # km [1/px]
    km = (2 * np.pi * nm) / res
    # Here, the notation defines 
    # a wave propagating to the right as:
    #
    #    u0(x) = exp(ikx)
    #
    # However, in physics usually we use the other sign convention:
    #
    #    u0(x) = exp(-ikx)
    #
    # In order to be consistent with programs like Meep or our
    # scattering script for a dielectric cylinder, we want to use the
    # latter sign convention.
    # This is not a big problem. We only need to multiply the imaginary
    # part of the scattered wave by -1.

    # Perform weighting
    if weight_angles:
        weights = util.compute_angle_weights_1d(angles).reshape(-1,1)
        sinogram = uSin * weights
    else:
        sinogram = uSin

    # Size of the input data
    ln = sinogram.shape[1]

    # We perform padding before performing the Fourier transform.
    # This gets rid of artifacts due to false periodicity and also
    # speeds up Fourier transforms of the input image size is not
    # a power of 2.
    order = max(64., 2**np.ceil(np.log(ln * 2.1) / np.log(2)))

    if padding:
        pad = order - ln
    else:
        pad = 0

    padl = np.int(np.ceil(pad / 2))
    padr = np.int(pad - padl)

    if padval is None:
        sino = np.pad(sinogram, ((0, 0), (padl, padr)),
                      mode="edge")
        if verbose > 0:
            print("......Padding with edge values.")
    else:
        sino = np.pad(sinogram, ((0, 0), (padl, padr)),
                      mode="linear_ramp",
                      end_values=(padval,))
        if verbose > 0:
            print("......Verifying padding value: {}".format(padval))

    # zero-padded length of sinogram.
    lN = sino.shape[1]

    # Ask for the filter. Do not include zero (first element).
    #
    # Integrals over ϕ₀ [0,2π]; kx [-kₘ,kₘ]
    #   - double coverage factor 1/2 already included
    #   - unitary angular frequency to unitary ordinary frequency
    #     conversion performed in calculation of UB=FT(uB).
    #
    # f(r) = -i kₘ / ((2π)^(3/2) a₀)            (prefactor)
    #      * iint dϕ₀ dkx                       (prefactor)
    #      * |kx|                               (prefactor)
    #      * exp(-i kₘ M lD )                   (prefactor)
    #      * UBϕ₀(kx)                             (dependent on ϕ₀)
    #      * exp( i (kx t⊥ + kₘ (M - 1) s₀) r )   (dependent on ϕ₀ and r)
    #
    # (r and s₀ are vectors. In the last term we perform the dot-product)
    #
    # kₘM = sqrt( kₘ² - kx² )
    # t⊥  = (  cos(ϕ₀), sin(ϕ₀) )
    # s₀  = ( -sin(ϕ₀), cos(ϕ₀) )
    #
    # The filter can be split into two parts
    #
    # 1) part without dependence on the z-coordinate
    #
    #        -i kₘ / ((2π)^(3/2) a₀)
    #      * iint dϕ₀ dkx
    #      * |kx|
    #      * exp(-i kₘ M lD )
    #
    # 2) part with dependence of the z-coordinate
    #
    #        exp( i (kx t⊥ + kₘ (M - 1) s₀) r )
    #
    # The filter (1) can be performed using the classical filter process
    # as in the backprojection algorithm.
    #
    #
    if jmc is not None:
        jmc.value += 1

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(lN)  # 1D array
    # kx is a 1D array.
    kx = 2 * np.pi * fx
    # Differentials for integral
    dphi0 = 2 * np.pi / A
    # We will later multiply with phi0.
    #               a, x
    kx = kx.reshape(1, -1)
    # Low-pass filter:
    # less-than-or-equal would give us zero division error.
    filter_klp = (kx**2 < km**2)

    # Filter M so there are no nans from the root
    M = 1. / km * np.sqrt((km**2 - kx**2) * filter_klp)

    prefactor = -1j * km / (2 * np.pi)
    prefactor *= dphi0
    prefactor *= np.abs(kx) * filter_klp
    # new in version 0.1.4:
    # We multiply by the factor (M-1) instead of just (M)
    # to take into account that we have a scattered
    # wave that is normalized by u0.
    prefactor *= np.exp(-1j * km * (M-1) * lD)
    # Perform filtering of the sinogram
    projection = np.fft.fft(sino, axis=-1) * prefactor

    #
    # filter (2) must be applied before rotation as well
    # exp( i (kx t⊥ + kₘ (M - 1) s₀) r )
    #
    # t⊥  = (  cos(ϕ₀), sin(ϕ₀) )
    # s₀  = ( -sin(ϕ₀), cos(ϕ₀) )
    #
    # This filter is effectively an inverse Fourier transform
    #
    # exp(i kx xD) exp(i kₘ (M - 1) yD )
    #
    # xD =   x cos(ϕ₀) + y sin(ϕ₀)
    # yD = - x sin(ϕ₀) + y cos(ϕ₀)

    # Everything is in pixels
    center = ln / 2.0
    x = np.arange(lN) - center + .5
    # Meshgrid for output array
    yv = x.reshape(-1, 1)

    Mp = M.reshape(1, -1)
    filter2 = np.exp(1j * yv * km * (Mp - 1))  # .reshape(1,lN,lN)

    projection = projection.reshape(A, 1, lN)  # * filter2


    # Prepare complex output image
    if onlyreal:
        outarr = np.zeros((ln, ln))
    else:
        outarr = np.zeros((ln, ln), dtype=np.dtype(complex))

    if jmc is not None:
        jmc.value += 1

    # Calculate backpropagations
    for i in np.arange(A):
        # Create an interpolation object of the projection.

        # interpolation of the rotated fourier transformed projection
        # this is already tiled onto the entire image.
        sino_filtered = np.fft.ifft(projection[i] * filter2, axis=-1)

        # Resize filtered sinogram back to original size
        sino = sino_filtered[:ln, padl:padl + ln]

        rotated_projr = scipy.ndimage.interpolation.rotate(
            sino.real, -angles[i] * 180 / np.pi,
            reshape=False, mode="constant", cval=0)
        # Append results

        outarr += rotated_projr

        if not onlyreal:
            outarr += 1j * scipy.ndimage.interpolation.rotate(
                sino.imag, -angles[i] * 180 / np.pi,
                reshape=False, mode="constant", cval=0)

        if jmc is not None:
            jmc.value += 1

    return outarr


def fourier_map_2d(uSin, angles, res, nm, lD=0, semi_coverage=False,
                   coords=None, jmc=None, jmm=None, verbose=_verbose):
    u""" 2D Fourier mapping with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)` 
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This function implements the solution by interpolation in
    Fourier space.

    Parameters
    ----------
    uSin : (A,N) ndarray
        Two-dimensional sinogram of line recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D})`
        divided by the incident plane wave :math:`u_0(l_\mathrm{D})`
        measured at the detector.
    angles : (A,) ndarray
        Angular positions :math:`\phi_j` of ``uSin`` in radians.
    res : float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm : float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD : float
        Distance from center of rotation to detector plane 
        :math:`l_\mathrm{D}` in pixels.
    semi_coverage : bool
        If set to ``True``, it is assumed that the sinogram does not 
        necessarily cover the full angular range from 0 to 2π, but an
        equidistant coverage over 2π can be achieved by inferring point
        (anti)symmetry of the (imaginary) real parts of the Fourier 
        transform of f. Valid for any set of angles {X} that result in
        a 2π coverage with the union set {X}U{X+π}.
    coords : None [(2,M) ndarray]
        Computes only the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    jmc, jmm : instance of :func:`multiprocessing.Value` or ``None``
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step ``jmc.value`` is
        incremented ``jmm.value`` times. ``jmm.value`` is set at the 
        beginning.
    verbose : int
        Increment to increase verbosity.


    Returns
    -------
    f : ndarray of shape (N,N), complex if `onlyreal` is `False`
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) = 
        k_m^2 \\left(\\left(\\frac{n(x,z)}{n_m}\\right)^2 -1\\right)`


    See Also
    --------
    backpropagate_2d : implementation by backpropagation
    odt_to_ri : conversion of the object function :math:`f(\mathbf{r})` 
        to refractive index :math:`n(\mathbf{r})`.

    Notes
    -----
    Do not use the parameter `lD` in combination with the Rytov
    approximation - the propagation is not correctly described.
    Instead, numerically refocus the sinogram prior to converting
    it to Rytov data (using e.g. :func:`odtbrain.sinogram_as_rytov`)
    with a numerical focusing algorithm (available in the Python
    package :py:mod:`nrefocus`).
    """
    ##
    ##
    # TODO:
    # - zero-padding as for backpropagate_2D - However this is not
    # necessary as Fourier interpolation is not parallelizable with
    # multiprocessing and thus unattractive. Could be interesting for
    # specific environments without the Python GIL.
    # - Deal with oversampled data. Maybe issue a warning.
    ##
    ##
    A = angles.shape[0]
    if jmm is not None:
        jmm.value = 4
    # Check input data
    assert len(uSin.shape) == 2, "Input data `uSin` must have shape (A,N)!"
    assert len(uSin) == A, "`len(angles)` must be  equal to `len(uSin)`!"
    
    if coords is not None:
        raise NotImplementedError("Output coordinates cannot yet" +
                                  " be set for the 2D backrpopagation algorithm.")
    # Cut-Off frequency
    # km [1/px]
    km = (2 * np.pi * nm) / res

    # Fourier transform of all uB's
    # In the script we used the unitary angular frequency (uaf) Fourier
    # Transform. The discrete Fourier transform is equivalent to the
    # unitary ordinary frequency (uof) Fourier transform.
    #
    # uof: f₁(ξ) = int f(x) exp(-2πi xξ)
    #
    # uaf: f₃(ω) = (2π)^(-n/2) int f(x) exp(-i ωx)
    #
    # f₁(ω/(2π)) = (2π)^(n/2) f₃(ω)
    # ω = 2πξ
    #
    # Our Backpropagation Formula is with uaf convention of the Form
    #
    # F(k) = 1/sqrt(2π) U(kD)
    #
    # If we convert now to uof convention, we get
    #
    # F(k) = U(kD)
    #
    # This means that if we divide the Fourier transform of the input
    # data by sqrt(2π) to convert f₃(ω) to f₁(ω/(2π)), the resulting
    # value for F is off by a factor of 2π.
    #
    # Instead, we can just multiply *UB* by sqrt(2π) and calculate
    # everything in uof.
    #UB =  np.fft.fft(np.fft.ifftshift(uSin, axes=-1))/np.sqrt(2*np.pi)
    #
    #
    # Furthermore, we define 
    # a wave propagating to the right as:
    #
    #    u0(x) = exp(ikx)
    #
    # However, in physics usually we use the other sign convention:
    #
    #    u0(x) = exp(-ikx)
    #
    # In order to be consistent with programs like Meep or our
    # scattering script for a dielectric cylinder, we want to use the
    # latter sign convention.
    # This is not a big problem. We only need to multiply the imaginary
    # part of the scattered wave by -1.

    # The radon transform introduced a shift in the data, which is not
    # conform with the FFT algorithm.
    UB = np.fft.fft(np.fft.ifftshift(uSin, axes=-1)) * np.sqrt(2 * np.pi)

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(len(uSin[0]))  # 1D array

    # kx is an 1D array.
    kx = 2 * np.pi * fx

    if jmc is not None:
        jmc.value += 1

    # Undersampling/oversampling?
    # Determine if the resolution of the image is too low by looking
    # at the maximum value for kx. This is no comparison between
    # Nyquist and Rayleigh frequency.
    if verbose and np.max(kx**2) <= km**2:
        # Detector is not set up properly. Higher resolution
        # can be achieved.
        print("......Measurement data is undersampled.")
    else:
        print("......Measurement data is oversampled.")
        # raise NotImplementedError("Oversampled data not yet supported."+
        #                   " Please rescale xD-axis of the input data.")
        # DEAL WITH OVERSAMPLED DATA?
        #import IPython
        # IPython.embed()
        #lenk = len(kx)
        #kx = np.fft.ifftshift(np.linspace(-np.sqrt(km),np.sqrt(km),len(fx), endpoint=False))

    #
    # F(kD-kₘs₀) = - i kₘ sqrt(2/π) / a₀ * M exp(-i kₘ M lD) * UB(kD)
    # kₘM = sqrt( kₘ² - kx² )
    # s₀  = ( -sin(ϕ₀), cos(ϕ₀) )
    #
    # We create the 2D interpolation object F
    #   - We compute the real coordinates (krx,kry) = kD-kₘs₀
    #   - We set as grid points the right side of the equation
    #
    # The interpolated griddata may go up to sqrt(2)*kₘ for kx and ky.

    kx = kx.reshape(1, -1)
    # a0 should have same shape as kx and UB
    #a0 = np.atleast_1d(a0)
    #a0 = a0.reshape(1,-1)

    filter_klp = (kx**2 < km**2)
    M = 1. / km * np.sqrt(km**2 - kx**2)
    #Fsin =  -1j * km * np.sqrt(2/np.pi) / a0 * M * np.exp(-1j*km*M*lD)
    # new in version 0.1.4:
    # We multiply by the factor (M-1) instead of just (M)
    # to take into account that we have a scattered
    # wave that is normalized by u0.
    Fsin = -1j * km * np.sqrt(2 / np.pi) * M * np.exp(-1j * km * (M-1) * lD)

    # UB has same shape (len(angles), len(kx))
    Fsin = Fsin * UB * filter_klp

    ang = angles.reshape(-1, 1)

    if semi_coverage:
        Fsin = np.vstack((Fsin, np.conj(Fsin)))
        ang = np.vstack((ang, ang + np.pi))


    if jmc is not None:
        jmc.value += 1

    # Compute kxl and kyl (in rotated system ϕ₀)
    kxl = kx
    kyl = np.sqrt((km**2 - kx**2) * filter_klp) - km
    # rotate kxl and kyl to where they belong
    krx = np.cos(ang) * kxl + np.sin(ang) * kyl
    kry = - np.sin(ang) * kxl + np.cos(ang) * kyl

    Xf = krx.flatten()
    Yf = kry.flatten()
    Zf = Fsin.flatten()

    # DEBUG: plot kry vs krx
    #from matplotlib import pylab as plt
    # plt.figure()
    # for i in range(len(krx)):
    #    plt.plot(krx[i],kry[i],"x")
    # plt.axes().set_aspect('equal')
    # plt.show()

    # interpolation on grid with same resolution as input data
    kintp = np.fft.fftshift(kx.reshape(-1))

    Fcomp = intp.griddata((Xf, Yf), Zf, (kintp[None, :], kintp[:, None]))

    if jmc is not None:
        jmc.value += 1

    # removed nans
    Fcomp[np.where(np.isnan(Fcomp))] = 0

    # Filter data
    kinx, kiny = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(kx))
    Fcomp[np.where((kinx**2 + kiny**2) > np.sqrt(2) * km)] = 0

    # DEBUG: Output filtered fourier image
    #proc_arr2im(Fcomp, scale=True).save("Fourier.bmp")

    #Fcomp[np.where(kinx**2+kiny**2<km)] = 0

    # Fcomp is centered at K = 0 due to the way we chose kintp/coords
    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fcomp)))

    if jmc is not None:
        jmc.value += 1

    return f[::-1]


def sum_2d(uSin, angles, res, nm, lD=0, coords=None,
           jmc=None, jmm=None, verbose=_verbose):
    u""" 2D sum-reconstruction with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)` 
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This function implements the solution by summation in real
    space, which is extremely slow.

    Parameters
    ----------
    uSin : (A,N) ndarray
        Two-dimensional sinogram of line recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D})`
        divided by the incident plane wave :math:`u_0(l_\mathrm{D})`
        measured at the detector.
    angles : (A,) ndarray
        Angular positions :math:`\phi_j` of ``uSin`` in radians.
    res : float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm : float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD : float
        Distance from center of rotation to detector plane 
        :math:`l_\mathrm{D}` in pixels.
    coords : None or (2,M) ndarray]
        Computes only the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    jmc, jmm : instance of :func:`multiprocessing.Value` or ``None``
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step ``jmc.value`` is
        incremented ``jmm.value`` times. ``jmm.value`` is set at the 
        beginning.
    verbose : int
        Increment to increase verbosity.

    Returns
    -------
    f : ndarray of shape (N,N), complex if `onlyreal` is `False`
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) = 
        k_m^2 \\left(\\left(\\frac{n(x,z)}{n_m}\\right)^2 -1\\right)`


    See Also
    --------
    backpropagate_2d : implementation by backprojection
    fourier_map_2d : implementation by Fourier interpolation
    odt_to_ri : conversion of the object function :math:`f(\mathbf{r})` 
        to refractive index :math:`n(\mathbf{r})`.


    Notes
    -----
    This method is not meant for production use. The computation time
    is very long and the reconstruction quality is bad. This function
    is included in the package, because of its educational value,
    exemplifying the backpropagation algorithm.

    Do not use the parameter `lD` in combination with the Rytov
    approximation - the propagation is not correctly described.
    Instead, numerically refocus the sinogram prior to converting
    it to Rytov data (using e.g. :func:`odtbrain.sinogram_as_rytov`)
    with a numerical focusing algorithm (available in the Python
    package :py:mod:`nrefocus`).
    """
    if coords is None:
        lx = uSin.shape[1]
        x = np.linspace(-lx/2, lx/2, lx, endpoint=False)
        xv, yv = np.meshgrid(x,x)
        coords = np.zeros((2, lx**2))
        coords[0,:] = xv.flat
        coords[1,:] = yv.flat
    
    
    if jmm is not None:
        jmm.value = coords.shape[1] + 1
    # Cut-Off frequency
    km = (2 * np.pi * nm) / res

    # Fourier transform of all uB's
    # In the script we used the unitary angular frequency (uaf) Fourier
    # Transform. The discrete Fourier transform is equivalent to the
    # unitary ordinary frequency (uof) Fourier transform.
    #
    # uof: f₁(ξ) = int f(x) exp(-2πi xξ)
    #
    # uaf: f₃(ω) = (2π)^(-n/2) int f(x) exp(-i ωx)
    #
    # f₁(ω/(2π)) = (2π)^(n/2) f₃(ω)
    # ω = 2πξ
    #
    # We have a one-dimensional (n=1) Fourier transform and UB in the
    # script is equivalent to f₃(ω). Because we are working with the
    # uaf, we divide by sqrt(2π) after computing the fft with the uof.
    #
    # We calculate the fourier transform of uB further below. This is
    # necessary for memory control.

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(uSin[0].shape[0])  # 1D array
    # kx is a 1D array.
    kx = 2 * np.pi * fx

    # Undersampling/oversampling?
    # Determine if the resolution of the image is too low by looking
    # at the maximum value for kx. This is no comparison between
    # Nyquist and Rayleigh frequency.
    if np.max(kx**2) <= 2 * km**2:
        # Detector is not set up properly. Higher resolution
        # can be achieved.
        if verbose:
            print("......Measurement data is undersampled.")
    else:
        if verbose:
            print("......Measurement data is oversampled.")
        raise NotImplementedError("Oversampled data not yet supported." +
                                  " Please rescale input data")

    # Differentials for integral
    dphi0 = 2 * np.pi / len(angles)
    dkx = kx[1] - kx[0]

    # We will later multiply with phi0.
    # Make sure we are using correct shapes
    kx = kx.reshape(1, kx.shape[0])

    # Low-pass filter:
    # less-than-or-equal would give us zero division error.
    filter_klp = (kx**2 < km**2)

    # a0 will be multiplied with kx
    #a0 = np.atleast_1d(a0)
    #a0 = a0.reshape(1,-1)

    # Create the integrand
    # Integrals over ϕ₀ [0,2π]; kx [-kₘ,kₘ]
    #   - double coverage factor 1/2 already included
    #   - unitary angular frequency to unitary ordinary frequency
    #     conversion performed in calculation of UB=FT(uB).
    #
    # f(r) = -i kₘ / ((2π)^(3/2) a₀)            (prefactor)
    #      * iint dϕ₀ dkx                       (prefactor)
    #      * |kx|                               (prefactor)
    #      * exp(-i kₘ M lD )                   (prefactor)
    #      * UBϕ₀(kx)                             (dependent on ϕ₀)
    #      * exp( i (kx t⊥ + kₘ (M - 1) s₀) r )   (dependent on ϕ₀ and r)
    #
    # (r and s₀ are vectors. In the last term we perform the dot-product)
    #
    # kₘM = sqrt( kₘ² - kx² )
    # t⊥  = (  cos(ϕ₀), sin(ϕ₀) )
    # s₀  = ( -sin(ϕ₀), cos(ϕ₀) )
    #
    #
    # everything that is not dependent on phi0:
    #
    # Filter M so there are no nans from the root
    M = 1. / km * np.sqrt((km**2 - kx**2) * filter_klp)
    prefactor = -1j * km / ((2 * np.pi)**(3. / 2))
    prefactor *= dphi0 * dkx
    # Also filter the prefactor, so nothing outside the required
    # low-pass contributes to the sum.
    prefactor *= np.abs(kx) * filter_klp
    # new in version 0.1.4:
    # We multiply by the factor (M-1) instead of just (M)
    # to take into account that we have a scattered
    # wave that is normalized by u0.
    prefactor *= np.exp(-1j * km * (M-1) * lD)

    # Initiate function f
    f = np.zeros(len(coords[0]), dtype=np.complex128)
    lenf = len(f)
    lenu0 = len(uSin[0])  # lenu0 = len(kx[0])

    # Initiate vector r that corresponds to calculating a value of f.
    r = np.zeros((2, 1, 1))

    # Everything is normal.
    # Get the angles ϕ₀.
    phi0 = angles.reshape(-1, 1)
    # Compute the Fourier transform of uB.
    # This is true: np.fft.fft(UB)[0] == np.fft.fft(UB[0])
    # because axis -1 is always used.
    #
    #
    # Furthermore, The notation in the our optical tomography script for
    # a wave propagating to the right is:
    #
    #    u0(x) = exp(ikx)
    #
    # However, in physics usually usethe other sign convention:
    #
    #    u0(x) = exp(-ikx)
    #
    # In order to be consisten with programs like Meep or our scattering
    # script for a dielectric cylinder, we want to use the latter sign
    # convention.
    # This is not a big problem. We only need to multiply the imaginary
    # part of the scattered wave by -1.
    UB = np.fft.fft(np.fft.ifftshift(uSin, axes=-1)) / np.sqrt(2 * np.pi)
    UBi = UB.reshape(len(angles), lenu0)

    if jmc is not None:
        jmc.value += 1

    for j in range(lenf):
        # Get r (We compute f(r) in this for-loop)
        r[0][:] = coords[0, j]  # x
        r[1][:] = coords[1, j]  # y

        # Integrand changes with r, so we have to create a new
        # array:
        integrand = prefactor * UBi

        # We save memory by directly applying the following to
        # the integrand:
        #
        # Vector along which we measured
        #s0 = np.zeros((2, phi0.shape[0], kx.shape[0]))
        #s0[0] = -np.sin(phi0)
        #s0[1] = +np.cos(phi0)

        # Vector perpendicular to s0
        #t_perp_kx = np.zeros((2, phi0.shape[0], kx.shape[1]))
        #
        #t_perp_kx[0] = kx*np.cos(phi0)
        #t_perp_kx[1] = kx*np.sin(phi0)

        #
        #term3 = np.exp(1j*np.sum(r*( t_perp_kx + (gamma-km)*s0 ), axis=0))
        # integrand* = term3
        #
        # Reminder:
        # f(r) = -i kₘ / ((2π)^(3/2) a₀)            (prefactor)
        #      * iint dϕ₀ dkx                       (prefactor)
        #      * |kx|                               (prefactor)
        #      * exp(-i kₘ M lD )                   (prefactor)
        #      * UB(kx)                             (dependent on ϕ₀)
        #      * exp( i (kx t⊥ + kₘ(M - 1) s₀) r )   (dependent on ϕ₀ and r)
        #
        # (r and s₀ are vectors. In the last term we perform the dot-product)
        #
        # kₘM = sqrt( kₘ² - kx² )
        # t⊥  = (  cos(ϕ₀), sin(ϕ₀) )
        # s₀  = ( -sin(ϕ₀), cos(ϕ₀) )
        integrand *= np.exp(1j * (
            r[0] * (kx * np.cos(phi0) - km * (M - 1) * np.sin(phi0)) +
            r[1] * (kx * np.sin(phi0) + km * (M - 1) * np.cos(phi0))))

        # Calculate the integral for the position r
        # integrand.sort()
        f[j] = np.sum(integrand)

        # free memory
        del integrand

        if jmc is not None:
            jmc.value += 1

    return f.reshape(lx,lx)
