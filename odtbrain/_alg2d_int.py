"""2D slow integration"""
import numpy as np


def integrate_2d(uSin, angles, res, nm, lD=0, coords=None,
                 count=None, max_count=None, verbose=0):
    r"""(slow) 2D reconstruction with the Fourier diffraction theorem

    Two-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,z)`
    by a dielectric object with refractive index
    :math:`n(x,z)`.

    This function implements the solution by summation in real
    space, which is extremely slow.

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
    coords: None or (2,M) ndarray]
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
    backpropagate_2d: implementation by backprojection
    fourier_map_2d: implementation by Fourier interpolation
    odt_to_ri: conversion of the object function :math:`f(\mathbf{r})`
        to refractive index :math:`n(\mathbf{r})`


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
        xv, yv = np.meshgrid(x, x)
        coords = np.zeros((2, lx**2))
        coords[0, :] = xv.flat
        coords[1, :] = yv.flat

    if max_count is not None:
        max_count.value += coords.shape[1] + 1

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
    # a0 = np.atleast_1d(a0)
    # a0 = a0.reshape(1,-1)

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

    if count is not None:
        count.value += 1

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
        # s0 = np.zeros((2, phi0.shape[0], kx.shape[0]))
        # s0[0] = -np.sin(phi0)
        # s0[1] = +np.cos(phi0)

        # Vector perpendicular to s0
        # t_perp_kx = np.zeros((2, phi0.shape[0], kx.shape[1]))
        #
        # t_perp_kx[0] = kx*np.cos(phi0)
        # t_perp_kx[1] = kx*np.sin(phi0)

        #
        # term3 = np.exp(1j*np.sum(r*( t_perp_kx + (gamma-km)*s0 ), axis=0))
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

        if count is not None:
            count.value += 1

    return f.reshape(lx, lx)
