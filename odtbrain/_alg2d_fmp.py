"""2D Fourier mapping"""
import numpy as np
import scipy.interpolate as intp


def fourier_map_2d(uSin, angles, res, nm, lD=0, semi_coverage=False,
                   coords=None, count=None, max_count=None, verbose=0):
    r"""2D Fourier mapping with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)`
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This function implements the solution by interpolation in
    Fourier space.

    Parameters
    ----------
    uSin: (A,N) ndarray
        Two-dimensional sinogram of line recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D})`
        divided by the incident plane wave :math:`u_0(l_\mathrm{D})`
        measured at the detector.
    angles: (A,) ndarray
        Angular positions :math:`\phi_j` of `uSin` in radians.
    res: float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm: float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD: float
        Distance from center of rotation to detector plane
        :math:`l_\mathrm{D}` in pixels.
    semi_coverage: bool
        If set to `True`, it is assumed that the sinogram does not
        necessarily cover the full angular range from 0 to 2π, but an
        equidistant coverage over 2π can be achieved by inferring point
        (anti)symmetry of the (imaginary) real parts of the Fourier
        transform of f. Valid for any set of angles {X} that result in
        a 2π coverage with the union set {X}U{X+π}.
    coords: None [(2,M) ndarray]
        Computes only the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.
    verbose: int
        Increment to increase verbosity.


    Returns
    -------
    f: ndarray of shape (N,N), complex if `onlyreal` is `False`
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) =
        k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 -1\right)`


    See Also
    --------
    backpropagate_2d: implementation by backpropagation
    odt_to_ri: conversion of the object function :math:`f(\mathbf{r})`
        to refractive index :math:`n(\mathbf{r})`

    Notes
    -----
    Do not use the parameter `lD` in combination with the Rytov
    approximation - the propagation is not correctly described.
    Instead, numerically refocus the sinogram prior to converting
    it to Rytov data (using e.g. :func:`odtbrain.sinogram_as_rytov`)
    with a numerical focusing algorithm (available in the Python
    package :py:mod:`nrefocus`).

    The interpolation in Fourier space (which is done with
    :func:`scipy.interpolate.griddata`) may be unstable and lead to
    artifacts if the data to interpolate contains sharp spikes. This
    issue is not handled at all by this method (in fact, a test has
    been removed in version 0.2.6 because ``griddata`` gave different
    results on Windows and Linux).
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
    if max_count is not None:
        max_count.value += 4
    # Check input data
    assert len(uSin.shape) == 2, "Input data `uSin` must have shape (A,N)!"
    assert len(uSin) == A, "`len(angles)` must be  equal to `len(uSin)`!"

    if coords is not None:
        raise NotImplementedError("Output coordinates cannot yet be set"
                                  + "for the 2D backrpopagation algorithm.")
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
    # UB =  np.fft.fft(np.fft.ifftshift(uSin, axes=-1))/np.sqrt(2*np.pi)
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

    UB = np.fft.fft(np.fft.ifftshift(uSin, axes=-1)) * np.sqrt(2 * np.pi)

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(len(uSin[0]))  # 1D array

    # kx is a 1D array.
    kx = 2 * np.pi * fx

    if count is not None:
        count.value += 1

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
        # lenk = len(kx)
        # kx = np.fft.ifftshift(np.linspace(-np.sqrt(km),
        #                                   np.sqrt(km),
        #                                   len(fx), endpoint=False))

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
    # a0 = np.atleast_1d(a0)
    # a0 = a0.reshape(1,-1)

    filter_klp = (kx**2 < km**2)
    M = 1. / km * np.sqrt(km**2 - kx**2)
    # Fsin =  -1j * km * np.sqrt(2/np.pi) / a0 * M * np.exp(-1j*km*M*lD)
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

    if count is not None:
        count.value += 1

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
    # from matplotlib import pylab as plt
    # plt.figure()
    # for i in range(len(krx)):
    #    plt.plot(krx[i],kry[i],"x")
    # plt.axes().set_aspect('equal')
    # plt.show()

    # interpolation on grid with same resolution as input data
    kintp = np.fft.fftshift(kx.reshape(-1))

    Fcomp = intp.griddata((Xf, Yf), Zf, (kintp[None, :], kintp[:, None]))

    if count is not None:
        count.value += 1

    # removed nans
    Fcomp[np.where(np.isnan(Fcomp))] = 0

    # Filter data
    kinx, kiny = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(kx))
    Fcomp[np.where((kinx**2 + kiny**2) > np.sqrt(2) * km)] = 0
    # Fcomp[np.where(kinx**2+kiny**2<km)] = 0

    # Fcomp is centered at K = 0 due to the way we chose kintp/coords
    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fcomp)))

    if count is not None:
        count.value += 1

    return f[::-1]
