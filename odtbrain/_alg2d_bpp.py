"""2D backpropagation algorithm"""
import numpy as np
import scipy.ndimage

from . import util


def backpropagate_2d(uSin, angles, res, nm, lD=0, coords=None,
                     weight_angles=True,
                     onlyreal=False, padding=True, padval=0,
                     count=None, max_count=None, verbose=0):
    r"""2D backpropagation with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)`
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This method implements the 2D backpropagation algorithm
    :cite:`Mueller2015arxiv`.

    .. math::
        f(\mathbf{r}) =
            -\frac{i k_\mathrm{m}}{2\pi}
            \sum_{j=1}^{N} \! \Delta \phi_0 D_{-\phi_j} \!\!
            \left \{
            \text{FFT}^{-1}_{\mathrm{1D}}
            \left \{
            \left| k_\mathrm{Dx} \right|
            \frac{\text{FFT}_{\mathrm{1D}} \left \{
            u_{\mathrm{B},\phi_j}(x_\mathrm{D}) \right \}
            }{u_0(l_\mathrm{D})}
            \exp \! \left[i k_\mathrm{m}(M - 1) \cdot
            (z_{\phi_j}-l_\mathrm{D}) \right]
            \right \}
            \right \}

    with the forward :math:`\text{FFT}_{\mathrm{1D}}` and inverse
    :math:`\text{FFT}^{-1}_{\mathrm{1D}}` 1D fast Fourier transform, the
    rotational operator :math:`D_{-\phi_j}`, the angular distance between the
    projections :math:`\Delta \phi_0`, the ramp filter in Fourier space
    :math:`|k_\mathrm{Dx}|`, and the propagation distance
    :math:`(z_{\phi_j}-l_\mathrm{D})`.

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
    coords: None [(2,M) ndarray]
        Computes only the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    weight_angles: bool
        If `True`, weights each backpropagated projection with a factor
        proportional to the angular distance between the neighboring
        projections.

        .. math::
            \Delta \phi_0 \longmapsto \Delta \phi_j =
                \frac{\phi_{j+1} - \phi_{j-1}}{2}

        .. versionadded:: 0.1.1
    onlyreal: bool
        If `True`, only the real part of the reconstructed image
        will be returned. This saves computation time.
    padding: bool
        Pad the input data to the second next power of 2 before
        Fourier transforming. This reduces artifacts and speeds up
        the process for input image sizes that are not powers of 2.
    padval: float
        The value used for padding. This is important for the Rytov
        approximation, where an approximate zero in the phase might
        translate to 2πi due to the unwrapping algorithm. In that
        case, this value should be a multiple of 2πi.
        If `padval` is `None`, then the edge values are used for
        padding (see documentation of :func:`numpy.pad`).
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
    odt_to_ri: conversion of the object function :math:`f(\mathbf{r})`
        to refractive index :math:`n(\mathbf{r})`

    radontea.backproject: backprojection based on the Fourier slice
        theorem

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
    if max_count is not None:
        max_count.value += A + 2
    # Check input data
    assert len(uSin.shape) == 2, "Input data `uB` must have shape (A,N)!"
    assert len(uSin) == A, "`len(angles)` must be  equal to `len(uSin)`!"

    if coords is not None:
        raise NotImplementedError("Output coordinates cannot yet be set " +
                                  + "for the 2D backrpopagation algorithm.")
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
        weights = util.compute_angle_weights_1d(angles).reshape(-1, 1)
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
    if count is not None:
        count.value += 1

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

    if count is not None:
        count.value += 1

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

        if count is not None:
            count.value += 1

    return outarr
