"""3D backpropagation algorithm"""
import ctypes
import multiprocessing as mp

import numexpr as ne
import numpy as np
import pyfftw
import scipy.ndimage

from . import util


ncores = mp.cpu_count()
mprotate_dict = {}


def _cleanup_worker():
    if "X" in mprotate_dict:
        mprotate_dict.pop("X")
    if "X_shape" in mprotate_dict:
        mprotate_dict.pop("X_shape")
    if "X_dtype" in mprotate_dict:
        mprotate_dict.pop("X_dtype")


def _init_worker(X, X_shape, X_dtype):
    """Initializer for pool for _mprotate"""
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    mprotate_dict["X"] = X
    mprotate_dict["X_shape"] = X_shape
    mprotate_dict["X_dtype"] = X_dtype


def _mprotate(ang, lny, pool, order):
    """Uses multiprocessing to wrap around _rotate

    4x speedup on an intel i7-3820 CPU @ 3.60GHz with 8 cores.

    The function calls _rotate which accesses the `mprotate_dict`.
    Data is rotated in-place.

    Parameters
    ----------
    ang: float
        rotation angle in degrees
    lny: int
        total number of rotations to perform
    pool: instance of multiprocessing.pool.Pool
        the pool object used for the computation
    order: int
        interpolation order
    """
    targ_args = list()

    slsize = np.int(np.floor(lny / ncores))

    for t in range(ncores):
        ymin = t * slsize
        ymax = (t + 1) * slsize
        if t == ncores - 1:
            ymax = lny
        targ_args.append((ymin, ymax, ang, order))

    pool.map(_rotate, targ_args)


def _rotate(d):
    arr = np.frombuffer(mprotate_dict["X"],
                        dtype=mprotate_dict["X_dtype"]).reshape(
                            mprotate_dict["X_shape"])
    (ymin, ymax, ang, order) = d
    return scipy.ndimage.interpolation.rotate(
        arr[:, ymin:ymax, :],  # input
        angle=-ang,  # angle
        axes=(0, 2),  # axes
        reshape=False,  # reshape
        output=arr[:, ymin:ymax, :],  # output
        order=order,  # order
        mode="constant",  # mode
        cval=0)


def backpropagate_3d(uSin, angles, res, nm, lD=0, coords=None,
                     weight_angles=True, onlyreal=False,
                     padding=(True, True), padfac=1.75, padval=None,
                     intp_order=2, dtype=None,
                     num_cores=ncores,
                     save_memory=False,
                     copy=True,
                     count=None, max_count=None,
                     verbose=0):
    r"""3D backpropagation

    Three-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,y,z)`
    by a dielectric object with refractive index
    :math:`n(x,y,z)`.

    This method implements the 3D backpropagation algorithm
    :cite:`Mueller2015arxiv`.


    .. math::
        f(\mathbf{r}) =
            -\frac{i k_\mathrm{m}}{2\pi}
            \sum_{j=1}^{N} \! \Delta \phi_0 D_{-\phi_j} \!\!
            \left \{
            \text{FFT}^{-1}_{\mathrm{2D}}
            \left \{
            \left| k_\mathrm{Dx} \right|
            \frac{\text{FFT}_{\mathrm{2D}} \left \{
            u_{\mathrm{B},\phi_j}(x_\mathrm{D}, y_\mathrm{D}) \right \}}
            {u_0(l_\mathrm{D})}
            \exp \! \left[i k_\mathrm{m}(M - 1) \cdot
            (z_{\phi_j}-l_\mathrm{D}) \right]
            \right \}
            \right \}

    with the forward :math:`\text{FFT}_{\mathrm{2D}}` and inverse
    :math:`\text{FFT}^{-1}_{\mathrm{2D}}` 2D fast Fourier transform, the
    rotational operator :math:`D_{-\phi_j}`, the angular distance between the
    projections :math:`\Delta \phi_0`, the ramp filter in Fourier space
    :math:`|k_\mathrm{Dx}|`, and the propagation distance
    :math:`(z_{\phi_j}-l_\mathrm{D})`.

    Parameters
    ----------
    uSin: (A, Ny, Nx) ndarray
        Three-dimensional sinogram of plane recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D}, y_\mathrm{D})`
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
    coords: None [(3, M) ndarray]
        Only compute the output image at these coordinates. This
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
    padding: tuple of bool
        Pad the input data to the second next power of 2 before
        Fourier transforming. This reduces artifacts and speeds up
        the process for input image sizes that are not powers of 2.
        The default is padding in x and y: `padding=(True, True)`.
        For padding only in x-direction (e.g. for cylindrical
        symmetries), set `padding` to `(True, False)`. To turn off
        padding, set it to `(False, False)`.
    padfac: float
        Increase padding size of the input data. A value greater
        than one will trigger padding to the second-next power of
        two. For example, a value of 1.75 will lead to a padded
        size of 256 for an initial size of 144, whereas it will
        lead to a padded size of 512 for an initial size of 150.
        Values geater than 2 are allowed. This parameter may
        greatly increase memory usage!
    padval: float
        The value used for padding. This is important for the Rytov
        approximation, where an approximat zero in the phase might
        translate to 2πi due to the unwrapping algorithm. In that
        case, this value should be a multiple of 2πi.
        If `padval` is `None`, then the edge values are used for
        padding (see documentation of :func:`numpy.pad`).
    intp_order: int between 0 and 5
        Order of the interpolation for rotation.
        See :func:`scipy.ndimage.interpolation.rotate` for details.
    dtype: dtype object or argument for :func:`numpy.dtype`
        The data type that is used for calculations (float or double).
        Defaults to `numpy.float_`.
    num_cores: int
        The number of cores to use for parallel operations. This value
        defaults to the number of cores on the system.
    save_memory: bool
        Saves memory at the cost of longer computation time.

        .. versionadded:: 0.1.5

    copy: bool
        Copy input sinogram `uSin` for data processing. If `copy`
        is set to `False`, then `uSin` will be overridden.

        .. versionadded:: 0.1.5

    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.
    verbose: int
        Increment to increase verbosity.


    Returns
    -------
    f: ndarray of shape (Nx, Ny, Nx), complex if `onlyreal==False`
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) =
        k_m^2 \left(\left(\frac{n(x,z)}{n_m}\right)^2 -1\right)`


    See Also
    --------
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
    """
    A = angles.size

    if len(uSin.shape) != 3:
        raise ValueError("Input data `uSin` must have shape (A,Ny,Nx).")
    if len(uSin) != A:
        raise ValueError("`len(angles)` must be  equal to `len(uSin)`.")
    if len(list(padding)) != 2:
        raise ValueError("`padding` must be boolean tuple of length 2!")
    if np.array(padding).dtype is not np.dtype(bool):
        raise ValueError("Parameter `padding` must be boolean tuple.")
    if coords is not None:
        raise NotImplementedError("Setting coordinates is not yet supported.")
    if num_cores > ncores:
        raise ValueError("`num_cores` must not exceed number "
                         + "of physical cores: {}".format(ncores))

    # setup dtype
    if dtype is None:
        dtype = np.float_
    dtype = np.dtype(dtype)
    if dtype.name not in ["float32", "float64"]:
        raise ValueError("dtype must be float32 or float64!")
    dtype_complex = np.dtype("complex{}".format(
        2 * np.int(dtype.name.strip("float"))))
    # set ctype
    ct_dt_map = {np.dtype(np.float32): ctypes.c_float,
                 np.dtype(np.float64): ctypes.c_double
                 }

    # progress
    if max_count is not None:
        max_count.value += A + 2

    ne.set_num_threads(num_cores)

    uSin = np.array(uSin, copy=copy)

    # lengths of the input data
    lny, lnx = uSin.shape[1], uSin.shape[2]
    # The z-size of the output array must match the x-size.
    # The rotation is performed about the y-axis (lny).
    ln = lnx

    # We perform zero-padding before performing the Fourier transform.
    # This gets rid of artifacts due to false periodicity and also
    # speeds up Fourier transforms of the input image size is not
    # a power of 2.
    orderx = np.int(max(64., 2**np.ceil(np.log(lnx * padfac) / np.log(2))))
    ordery = np.int(max(64., 2**np.ceil(np.log(lny * padfac) / np.log(2))))

    if padding[0]:
        padx = orderx - lnx
    else:
        padx = 0
    if padding[1]:
        pady = ordery - lny
    else:
        pady = 0

    padyl = np.int(np.ceil(pady / 2))
    padyr = pady - padyl
    padxl = np.int(np.ceil(padx / 2))
    padxr = padx - padxl

    # zero-padded length of sinogram.
    lNx, lNy = lnx + padx, lny + pady
    lNz = ln

    if verbose > 0:
        print("......Image size (x,y): {}x{}, padded: {}x{}".format(
            lnx, lny, lNx, lNy))

    # Perform weighting
    if weight_angles:
        weights = util.compute_angle_weights_1d(angles).reshape(-1, 1, 1)
        uSin *= weights

    # Cut-Off frequency
    # km [1/px]
    km = (2 * np.pi * nm) / res
    # Here, the notation for
    # a wave propagating to the right is:
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

    # Ask for the filter. Do not include zero (first element).
    #
    # Integrals over ϕ₀ [0,2π]; kx [-kₘ,kₘ]
    #   - double coverage factor 1/2 already included
    #   - unitary angular frequency to unitary ordinary frequency
    #     conversion performed in calculation of UB=FT(uB).
    #
    # f(r) = -i kₘ / ((2π)² a₀)                 (prefactor)
    #      * iiint dϕ₀ dkx dky                  (prefactor)
    #      * |kx|                               (prefactor)
    #      * exp(-i kₘ M lD )                   (prefactor)
    #      * UBϕ₀(kx)                           (dependent on ϕ₀)
    #      * exp( i (kx t⊥ + kₘ (M - 1) s₀) r ) (dependent on ϕ₀ and r)
    # (r and s₀ are vectors. The last term contains a dot-product)
    #
    # kₘM = sqrt( kₘ² - kx² - ky² )
    # t⊥  = (  cos(ϕ₀), ky/kx, sin(ϕ₀) )
    # s₀  = ( -sin(ϕ₀), 0    , cos(ϕ₀) )
    #
    # The filter can be split into two parts
    #
    # 1) part without dependence on the z-coordinate
    #
    #        -i kₘ / ((2π)² a₀)
    #      * iiint dϕ₀ dkx dky
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

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(lNx)  # 1D array
    fy = np.fft.fftfreq(lNy)  # 1D array
    # kx is a 1D array.
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    # Differentials for integral
    dphi0 = 2 * np.pi / A
    # We will later multiply with phi0.
    #               y, x
    kx = kx.reshape(1, -1)
    ky = ky.reshape(-1, 1)
    # Low-pass filter:
    # less-than-or-equal would give us zero division error.
    filter_klp = (kx**2 + ky**2 < km**2)

    # Filter M so there are no nans from the root
    M = 1. / km * np.sqrt((km**2 - kx**2 - ky**2) * filter_klp)

    prefactor = -1j * km / (2 * np.pi)
    prefactor *= dphi0
    # Also filter the prefactor, so nothing outside the required
    # low-pass contributes to the sum.
    prefactor *= np.abs(kx) * filter_klp
    # prefactor *= np.sqrt(((kx**2+ky**2)) * filter_klp )
    # new in version 0.1.4:
    # We multiply by the factor (M-1) instead of just (M)
    # to take into account that we have a scattered
    # wave that is normalized by u0.
    prefactor *= np.exp(-1j * km * (M-1) * lD)

    if count is not None:
        count.value += 1

    # filter (2) must be applied before rotation as well
    # exp( i (kx t⊥ + kₘ (M - 1) s₀) r )
    #
    # kₘM = sqrt( kₘ² - kx² - ky² )
    # t⊥  = (  cos(ϕ₀), ky/kx, sin(ϕ₀) )
    # s₀  = ( -sin(ϕ₀), 0    , cos(ϕ₀) )
    #
    # This filter is effectively an inverse Fourier transform
    #
    # exp(i kx xD) exp(i ky yD) exp(i kₘ (M - 1) zD )
    #
    # xD =   x cos(ϕ₀) + z sin(ϕ₀)
    # zD = - x sin(ϕ₀) + z cos(ϕ₀)

    # Everything is in pixels
    center = lNz / 2.0

    z = np.linspace(-center, center, lNz, endpoint=False)
    zv = z.reshape(-1, 1, 1)

    #              z, y, x
    Mp = M.reshape(lNy, lNx)

    # filter2 = np.exp(1j * zv * km * (Mp - 1))
    f2_exp_fac = 1j * km * (Mp - 1)
    if save_memory:
        # compute filter2 later
        pass
    else:
        # compute filter2 now
        filter2 = ne.evaluate("exp(factor * zv)",
                              local_dict={"factor": f2_exp_fac,
                                          "zv": zv})
        # occupies some amount of ram, but yields faster
        # computation later

    if count is not None:
        count.value += 1

    # Prepare complex output image
    if onlyreal:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype)
    else:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    # Create plan for FFTW
    # save memory by in-place operations
    # projection = np.fft.fft2(sino, axes=(-1,-2)) * prefactor
    # FFTW-flag is "estimate":
    #   specifies that, instead of actual measurements of different
    #   algorithms, a simple heuristic is used to pick a (probably
    #   sub-optimal) plan quickly. With this flag, the input/output
    #   arrays are not overwritten during planning.

    # Byte-aligned arrays
    oneslice = pyfftw.empty_aligned((lNy, lNx), dtype_complex)

    myfftw_plan = pyfftw.FFTW(oneslice, oneslice, threads=num_cores,
                              flags=["FFTW_ESTIMATE"], axes=(0, 1))

    # Create plan for IFFTW:
    inarr = pyfftw.empty_aligned((lNy, lNx), dtype_complex)
    # inarr[:] = (projection[0]*filter2)[0,:,:]
    # plan is "patient":
    #    FFTW_PATIENT is like FFTW_MEASURE, but considers a wider range
    #    of algorithms and often produces a “more optimal” plan
    #    (especially for large transforms), but at the expense of
    #    several times longer planning time (especially for large
    #    transforms).
    # print(inarr.flags)

    myifftw_plan = pyfftw.FFTW(inarr, inarr, threads=num_cores,
                               axes=(0, 1),
                               direction="FFTW_BACKWARD",
                               flags=["FFTW_MEASURE"])

    # Setup a shared array
    shared_array = mp.RawArray(ct_dt_map[dtype], ln * lny * lnx)
    arr = np.frombuffer(shared_array, dtype=dtype).reshape(ln, lny, lnx)

    # Initialize the pool with the shared array
    pool4loop = mp.Pool(processes=num_cores,
                        initializer=_init_worker,
                        initargs=(shared_array, (ln, lny, lnx), dtype))

    # filtered projections in loop
    filtered_proj = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    for aa in np.arange(A):
        if padval is None:
            oneslice[:] = np.pad(uSin[aa],
                                 ((padyl, padyr), (padxl, padxr)),
                                 mode="edge")
        else:
            oneslice[:] = np.pad(uSin[aa],
                                 ((padyl, padyr), (padxl, padxr)),
                                 mode="linear_ramp",
                                 end_values=(padval,))
        myfftw_plan.execute()
        # normalize to (lNx * lNy) for FFTW and multiply with prefactor
        oneslice *= prefactor / (lNx * lNy)
        # 14x Speedup with fftw3 compared to numpy fft and
        # memory reduction by a factor of 2!
        # ifft will be computed in-place

        for p in range(len(zv)):
            if save_memory:
                # compute filter2 here;
                # this is comparatively slower than the other case
                ne.evaluate("exp(factor * zvp) * projectioni",
                            local_dict={"zvp": zv[p],
                                        "projectioni": oneslice,
                                        "factor": f2_exp_fac},
                            out=inarr)
            else:
                # use universal functions
                np.multiply(filter2[p], oneslice, out=inarr)
            myifftw_plan.execute()
            filtered_proj[p, :, :] = inarr[padyl:lny+padyl, padxl:lnx+padxl]

        # resize image to original size
        # The copy is necessary to prevent memory leakage.
        arr[:] = filtered_proj.real

        phi0 = np.rad2deg(angles[aa])

        if not onlyreal:
            filtered_proj_imag = filtered_proj.imag

        _mprotate(phi0, lny, pool4loop, intp_order)

        outarr.real += arr

        if not onlyreal:
            arr[:] = filtered_proj_imag
            _mprotate(phi0, lny, pool4loop, intp_order)
            outarr.imag += arr

        if count is not None:
            count.value += 1

    pool4loop.terminate()
    pool4loop.join()
    _cleanup_worker()

    return outarr
