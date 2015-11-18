#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import ctypes
import gc
import multiprocessing as mp
import numpy as np
from numpy import cos, sin

import pyfftw
import scipy.ndimage
import warnings

from ._Back_3D import _ncores, _np_float64, _verbose, _filter2_func
from . import util
import odtbrain



def compute_trafo_jacobian(loc, km, kDx, kDy, M):
    """ jacobian/filter in Fourier space for backpropagation along arbitrary axis
    
    Parameters
    ----------
    loc : list/array of length 3
        The vector (x,y,z) on the unit sphere describing the 
        direction of the rotational axis.
    km : float
        Wave number in surrounding medium.
    kDx, kDy : 2d ndarrays
        kx/ky values of the detector in Fourier space.
    """
    # numpy definitions
    #sqrt = np.sqrt
    # substitutions
    #M = sqrt(km**2-kDx**2-kDy**2)
    #M1sq = 1+kDx**2+kDy**2
    u, v, _w = loc
    # return abs
    return np.abs((kDx*v-kDy*u)/M)


def estimate_major_rotation_axis(loc):
    """ 
    For a list of points on the unit sphere, estimate the main
    rotational axis and return a list of angles that correspond
    to the rotational position for each point.
    
    
    """
    #TODO:
    raise NotImplementedError("estimation of rotational axis not implemented.") 


def norm_vec(vector):
    """
    Normalizes a vector to one.
    """
    assert len(vector)==3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))


def rotate_points_to_axis(points, axis):
    """ Rotate all points of a list such that axis=[0,1,0] the shortest way.
    
    This is accomplished by rotating in the x-z-plane by phi into the
    y-z-plane, then rotation in the y-z-plane by theta up to [0,1,0],
    and finally rotating back in the x-z-plane by -phi.
    
    Parameters
    ----------
    points : list-like with elements of length 3
        The Cartesian points. These should be in the same format as
        produced by `sphere_points_from_angles_and_tilt`.
    axis : list-like, length 3
        The reference axis that will be used to determine the
        rotation angle of the points. The points will be rotated
        about the origin such that `axis` matches [0,1,0].
    
    Returns
    -------
    rotated_points : np.ndarray of shape (N,3)
        The rotated points.
    """
    axis = norm_vec(axis)
    u, v, w = axis
    points = np.array(points)
    # Determine the rotational angle in the x-z plane
    phi = np.arctan2(u, w)
    # Determine the tilt angle w.r.t. the y-axis
    theta = np.arccos(v)
    
    # Negative rotation about y-axis
    Rphi = np.array([
                     [np.cos(phi), 0, -np.sin(phi)],
                     [0          , 1,            0],
                     [np.sin(phi), 0,  np.cos(phi)],
                     ])

    # Negative rotation about x-axis
    Rtheta = np.array([
                       [1,             0,              0],
                       [0, np.cos(theta),  np.sin(theta)],
                       [0,-np.sin(theta),  np.cos(theta)],
                       ])

    DR1 = np.dot(Rtheta, Rphi)
    # Rotate back by -phi such that effective rotation was only
    # towards [0,1,0].
    DR = np.dot(Rphi.T, DR1)
    rotpoints = np.zeros((len(points), 3))
    for ii, pnt in enumerate(points):
        rotpoints[ii] = np.dot(DR, pnt)
    
    return rotpoints


def rotation_matrix_from_point(point):
    """ Compute rotation matrix to go from [0,0,1] to `point`.
    
    First, the matrix rotates to in the polar direction. Then,
    a rotation about the y-axis is performed to match the
    azimuthal angle in the x-z-plane.
    
    This rotation matrix is required for the correct 3D orientation
    of the backpropagated projections.

    Parameters
    ----------
    points : list-like, length 3
        The coordinates of the point in 3D.
    
    
    Returns
    -------
    Rmat : 3x3 ndarray
        The rotation matrix that rotates [0,0,1] to `point`.
    """
    x, y, z = point
    # azimuthal angle
    phi = np.arctan2(x, z)
    # angle in polar direction
    theta = np.arctan2(y, np.sqrt(x**2+z**2))
    
    # Rotation in polar direction (negative)
    Rtheta = np.array([
                       [1,             0,              0],
                       [0, np.cos(theta),  np.sin(theta)],
                       [0,-np.sin(theta),  np.cos(theta)],
                       ])

    # rotation in x-z-plane
    Rphi = np.array([
                     [np.cos(phi),  0, np.sin(phi)],
                     [0           , 1,           0],
                     [-np.sin(phi), 0, np.cos(phi)],
                     ])

    D = np.dot(Rphi, Rtheta)
    return D



def sphere_points_from_angles_and_tilt(angles, tilted_axis):
    """
    For a given tilt of the rotational axis `tilted_axis`, compute
    the points on a unit sphere that correspond to the distribution
    `angles` along the great circle about this axis.
    
    Parameters
    ----------
    angles : 1d ndarray
        The angles that will be distributed on the great circle.
    tilted_axis : list of length 3
        The tilted axis of rotation that determines the great
        circle.
    
    Notes
    -----
    The reference axis is always [0,1,0].

    """
    ## Normalize tilted axis.
    tilted_axis = norm_vec(tilted_axis)
    [u, v, w] = tilted_axis
    ## Initial distribution of points about great circle (x-z).
    newang = np.zeros((angles.shape[0], 3), dtype=float)
    # We subtract angles[0], because in step (a) we want that
    # newang[0]==[0,0,1]. This only works if we actually start
    # at that point.
    newang[:,0] = np.sin(angles-angles[0])
    newang[:,2] = np.cos(angles-angles[0])
    
    ## Compute rotational angles w.r.t. [0,1,0].
    # - Draw a unit sphere with the y-axis pointing up and the
    #   z-axis pointing right
    # - The rotation of `tilted_axis` can be described by two
    #   separate rotations. We will use these two angles:
    #   (a) Rotation from y=1 within the y-z plane:
    #       This is the rotation that is critical for data
    #       reconstruction. If this angle is zero, then we
    #       have a rotational axis in the imaging plane. If
    #       this angle is PI/2, then our sinogram consists
    #       of a rotating image and 3D reconstruction is
    #       impossible. This angle is counted from the y-axis
    #       onto the x-z plane.
    #   (b) Rotation in the x-z plane:
    #       This angle is responsible for matching up the angles
    #       with the correct sinogram images. If this angle is zero,
    #       then the projection of the rotational axis onto the
    #       x-y plane is aligned with the y-axis. If this angle is
    #       PI/2, then the axis and its projection onto the x-y
    #       plane are identical. This angle is counted from the
    #       positive z-axis towards the positive x-axis. By default,
    #       angles[0] is the point that touches the great circle
    #       that lies in the x-z plane. angles[1] is the next point
    #       towards the x-axis if phi==0.
    
    # (a) This angle is the polar angle theta measured from the
    #     y-axis.
    theta = np.arccos(v)
    
    # (b) This is the angle measured in the x-z plane starting
    #     at the x-axis and measured towards the positive z-axis.
    phi = np.arctan2(u, w)
    
    ## Determine the projection points on the unit sphere.
    # The resulting circle meets the x-z-plane at phi, and
    # is tilted by theta w.r.t. the y-axis.
    
    # (a) Create a tilted data set. This is achieved in 3 steps.

    # a1) Determine radius of tilted circle and get the centered
    #     circle with a smaller radius.
    rtilt = np.cos(theta)
    newang *= rtilt

    # a2) Rotate this circle about the x-axis by theta
    #     (right-handed/counter-clockwise/basic/elemental rotation)
    Rx = np.array([  
               [1,          0,           0],
               [0, cos(theta), -sin(theta)],
               [0, sin(theta),  cos(theta)]
               ])
    for ii in range(newang.shape[0]):
        newang[ii] = np.dot(Rx, newang[ii])

    # a3) Shift newang such that newang[0] is located at (0,0,1)
    newang = newang - (newang[0] - np.array([0,0,1])).reshape(1,3)

    # (b) Rotate the entire thing with phi about the y-axis
    #     (right-handed/counter-clockwise/basic/elemental rotation)
    Ry = np.array([  
                   [ cos(phi), 0, sin(phi)],
                   [        0, 1,        0],
                   [-sin(phi), 0, cos(phi)]
                   ])
    
    for jj in range(newang.shape[0]):
        newang[jj] = np.dot(Ry, newang[jj])

    return newang



    

def backpropagate_3d_tilted(uSin, angles, res, nm, lD,
                     tilted_axis=[0, 1, 0],
                     coords=None, weight_angles=True, onlyreal=False,
                     offset_alpha=0, offset_beta=0,
                     padding=(True, True), padfac=1.75, padval=None,
                     intp_order=2, dtype=_np_float64,
                     num_cores=_ncores, 
                     jmc=None, jmm=None,
                     verbose=_verbose):
    u""" 3D backpropagation with the Fourier diffraction theorem

    Three-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,y,z)` 
    by a dielectric object with refractive index
    :math:`n(x,y,z)`.

    This method implements the 3D backpropagation algorithm with
    a rotational axis that is tilted w.r.t. the imaging plane.
    
    Parameters
    ----------
    uSin : (A, Ny, Nx) ndarray
        Three-dimensional sinogram of plane wave recordings
        :math:`u_{\mathrm{B}, \phi_0}(x_\mathrm{D}, y_\mathrm{D},
        z_\mathrm{D})`
        normalized by the amplitude of the unscattered wave :math:`a_0`
        measured at the detector.
    angles : ndarray of shape (A,3) or 1D array of length A
        If the shape is (A,3), then `angles` consists of vectors
        on the unit sphere that correspond to the direction
        of illumination and acquisition (s₀). If the shape is (A,),
        then `angles` is  a one-dimensional array of angles [rad]
        that determines the rotational position in a plane.
        In both cases, `tilted_axis` must be set according to the
        tilt of the rotational axis.
    res : float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm : float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD : float
        Distance from center of rotation to detector plane 
        :math:`l_\mathrm{D}` in pixels.
    tilted_axis : list of floats
        The coordinates [u, v, w] on a unit sphere representing the
        tilted axis of rotation. The w-component is used to check
        if the vector is normalized to 1. The default is (0,1,0),
        which corresponds to a rotation about the y-axis and
        follows the behavior of `odtbrain.backrpoject_3d`.
    coords : None [(3, M) ndarray]
        Only compute the output image at these coordinates. This
        keyword is reserved for future versions and is not
        implemented yet.
    weight_angles : bool, optional
        If `True` weight each backpropagated projection with a factor
        proportional to the angular distance between the neighboring
        projections.
    onlyreal : bool
        If `True`, only the real part of the reconstructed image
        will be returned. This saves computation time.
    padding : tuple of bool
        Pad the input data to the second next power of 2 before
        Fourier transforming. This reduces artifacts and speeds up
        the process for input image sizes that are not powers of 2.
        The default is padding in x and y: `padding=(True, True)`.
        For padding only in x-direction (e.g. for cylindrical
        symmetries), set `padding` to `(True, False)`. To turn off
        padding, set it to `(False, False)`.
    padfac : float
        Increase padding size of the input data. A value greater
        than one will trigger padding to the second-next power of
        two. For example, a value of 1.75 will lead to a padded
        size of 256 for an initial size of 144, whereas for it will
        lead to a padded size of 512 for an initial size of 150.
        Values geater than 2 are allowed. This parameter may
        greatly increase memory usage!
    padval : float
        The value used for padding. This is important for the Rytov
        approximation, where an approximat zero in the phase might
        translate to 2πi due to the unwrapping algorithm. In that
        case, this value should be a multiple of 2πi. 
        If `padval` is `None`, then the edge values are used for
        padding (see documentation of `numpy.pad`).
    order : int between 0 and 5
        Order of the interpolation for rotation.
        See `scipy.ndimage.interpolation.rotate` for details.
    dtype : dtype object or argument for np.dtype
        The data type that is used for calculations (float or double).
        Defaults to np.float.
    num_cores : int
        The number of cores to use for parallel operations. This value
        defaults to the number of cores on the system.
    jmc, jmm : instance of :func:`multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.
    verbose : int
        Increment to increase verbosity.


    Returns
    -------
    f : ndarray of shape (Nx, Ny, Nx), complex if ``onlyreal==False``
        Reconstructed object function :math:`f(\mathbf{r})` as defined
        by the Helmholtz equation.
        :math:`f(x,z) = 
        k_m^2 \\left(\\left(\\frac{n(x,z)}{n_m}\\right)^2 -1\\right)`


    See Also
    --------
    odt_to_ri : conversion of the object function :math:`f(\mathbf{r})` 
        to refractive index :math:`n(\mathbf{r})`.


    .. versionadded:: 0.1.2
    """
    A = angles.shape[0]
    assert angles.shape in [(A,), (A,3)], "`angles` must have shape (A,) or (A,3)!"
    # jobmanager
    if jmm is not None:
        jmm.value = A + 2
    
    # normalize titled axis
    tilted_axis = norm_vec(tilted_axis)
    
    if len(angles.shape) != 2:
        if weight_angles:
            weights = util.compute_angle_weights_1d(angles).reshape(-1,1,1)
        else:
            weights = 1
        # compute the 3D points from tilted_axis
        angles = sphere_points_from_angles_and_tilt(angles, tilted_axis)
    else:
        if weight_angles:
            warnings.warn("3D angular weighting not yet supported!")
        weights = 1
    
    # check for dtype
    dtype = np.dtype(dtype)
    if not dtype.name in ["float32", "float64"]:
        raise ValueError("dtype must be float32 or float64.")

    assert num_cores <= _ncores, "`num_cores` must not exceed number " +\
                                 "of physical cores: {}".format(_ncores)

    assert uSin.dtype == np.complex128, "uSin dtype must be complex128."

    dtype_complex = np.dtype("complex{}".format(
        2 * int(dtype.name.strip("float"))))

    # set ctype
    ct_dt_map = {np.dtype(np.float32): ctypes.c_float,
                 np.dtype(np.float64): ctypes.c_double
                 }

    assert len(uSin.shape) == 3, "Input data `uSin` must have shape (A,Ny,Nx)."
    assert len(uSin) == A, "`len(angles)` must be  equal to `len(uSin)`."
    assert len(list(padding)) == 2, "Parameter `padding` must be boolean tuple of length 2!"
    assert np.array(padding).dtype is np.dtype(bool), "Parameter `padding` must be boolean tuple."
    assert coords is None, "Setting coordinates is not yet supported."

    # Make sure that angles are normalized
    angles = np.array(angles)
    shouldbeone = np.sum(angles**2, axis=1)
    if not np.allclose(shouldbeone, np.ones_like(shouldbeone)):
        print("...Angles are not normalized; Normalizing.")
        angles /= np.sqrt(shouldbeone).reshape(-1,1)

    # Cut-Off frequency
    # km [1/px]
    km = (2 * np.pi * nm) / res
    # The notation in the our optical tomography script for
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

    # save memory
    sinogram = uSin*weights

    # lengths of the input data
    (la, lny, lnx) = sinogram.shape
    ln = max(lnx, lny)

    # We do a zero-padding before performing the Fourier transform.
    # This gets rid of artifacts due to false periodicity and also
    # speeds up Fourier transforms of the input image size is not
    # a power of 2.
    # transpose so we can call resize correctly

    orderx = max(64., 2**np.ceil(np.log(lnx * padfac) / np.log(2)))
    ordery = max(64., 2**np.ceil(np.log(lny * padfac) / np.log(2)))

    if padding[0]:
        padx = orderx - lnx
    else:
        padx = 0
    if padding[1]:
        pady = ordery - lny
    else:
        pady = 0

    # Apply a Fourier filter before projecting the sinogram slices.
    # Resize image to next power of two for fourier analysis
    # Reduces artifacts

    padyl = np.int(np.ceil(pady / 2))
    padyr = np.int(pady - padyl)
    padxl = np.int(np.ceil(padx / 2))
    padxr = np.int(padx - padyl)


    #TODO: This padding takes up a lot of memory. Move it to a separate
    # for loop or to the main for-loop.
    if padval is None:
        sino = np.pad(sinogram, ((0, 0), (padyl, padyr), (padxl, padxr)),
                      mode="edge")
        if verbose > 0:
            print("......Padding with edge values.")
    else:
        sino = np.pad(sinogram, ((0, 0), (padyl, padyr), (padxl, padxr)),
                      mode="linear_ramp",
                      end_values=(padval,))
        if verbose > 0:
            print("......Verifying padding value: {}".format(padval))

    # save memory
    del sinogram
    if verbose > 0:
        print("......Image size (x,y): {}x{}, padded: {}x{}".format(
            lnx, lny, sino.shape[2], sino.shape[1]))

    # zero-padded length of sinogram.
    (lA, lNy, lNx) = sino.shape  # @UnusedVariable
    lNz = ln


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

    # if lNx != lNy:
    #    raise NotImplementedError("Input data must be square shaped!")

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(lNx)  # 1D array
    fy = np.fft.fftfreq(lNy)  # 1D array
    # kx is a 1D array.
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    # Differentials for integral
    dphi0 = 2 * np.pi / A
    # We will later multiply with phi0.
    #               a, y, x
    kx = kx.reshape(1, 1, -1)
    ky = ky.reshape(1, -1, 1)
    # Low-pass filter:
    # less-than-or-equal would give us zero division error.
    filter_klp = (kx**2 + ky**2 < km**2)

    # Filter M so there are no nans from the root
    M = 1. / km * np.sqrt((km**2 - kx**2 - ky**2) * filter_klp)
    # The input data is already divided by a0
    #prefactor  = -1j * km / ( 2 * np.pi * a0 )
    prefactor = -1j * km / (2 * np.pi)
    prefactor *= dphi0
    # Also filter the prefactor, so nothing outside the required
    # low-pass contributes to the sum.
    # The filter is now dependent on the rotational position of the
    # specimen. We have to include information from the angles.
    # We want to estimate the rotational axis for every frame. We 
    # do that by computing the cross-product of the vectors in
    # angles from the current and previous image.
    
    #filterabs = np.abs(kx*v-ky*u) * filter_klp
    u, v, w = tilted_axis
    filterabs = np.abs(kx*v+ky*u) * filter_klp
    #prefactor *= np.sqrt(((kx**2+ky**2)) * filter_klp )
    prefactor *= np.exp(-1j * km * M * lD)
    # Perform filtering of the sinogram,
    # save memory by in-place operations
    #projection = np.fft.fft2(sino, axes=(-1,-2)) * prefactor
    # Flag is "estimate":
    #   specifies that, instead of actual measurements of different
    #   algorithms, a simple heuristic is used to pick a (probably
    #   sub-optimal) plan quickly. With this flag, the input/output
    #   arrays are not overwritten during planning.

    # Byte-aligned arrays
    temp_array = pyfftw.n_byte_align_empty(sino[0].shape, 16, dtype_complex)

    myfftw_plan = pyfftw.FFTW(temp_array, temp_array, threads=num_cores,
                              flags=["FFTW_ESTIMATE"], axes=(0,1))


    if jmc is not None:
        jmc.value += 1

    for p in range(len(sino)):
        # this overwrites sino
        temp_array[:] = sino[p, :, :]
        myfftw_plan.execute()
        sino[p, :, :] = temp_array[:]

    temp_array, myfftw_plan

    projection = sino
    projection[:] *= prefactor
    projection[:] *= filterabs

    # save memory
    del prefactor, filter_klp
    #
    #
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
    #x = np.linspace(-centerx, centerx, lNx, endpoint=False)
    #x = np.arange(lNx) - center + .5
    # Meshgrid for output array
    #zv, yv, xv = np.meshgrid(x,x,x)
    #               z, y, x
    #xv = x.reshape( 1, 1,-1)
    #yv = x.reshape( 1,-1, 1)

    #z = np.arange(ln) - center + .5
    z = np.linspace(-center, center, lNz, endpoint=False)
    zv = z.reshape(-1, 1, 1)

    #              z, y, x
    Mp = M.reshape(lNy, lNx)


    # Compute the filter in Fourier space in parallel on-by one
    # This saves an enormous amount of memory when compared to
    # simply executing:
    # filter2 = np.exp(1j * zv * km * (Mp - 1))
    
    Mpm1 = km * (Mp - 1)
    args=zip(zv.flatten(), [Mpm1]*zv.shape[0])
    filter2_pool = mp.Pool(processes=num_cores)
    filter2 = filter2_pool.map(_filter2_func, args)
    filter2_pool.terminate()
    filter2_pool.terminate()
    del filter2_pool, args, Mpm1

    # occupies some amount of ram
    #filter2[0].size*len(filter2)*128/8/1024**3

    if jmc is not None:
        jmc.value += 1

    #                               a, z, y,  x
    #projection = projection.reshape(la, 1, lNy, lNx)
    projection = projection.reshape(la, lNy, lNx)


    # This frees comparatively few data
    del M
    #del Mp

    # Prepare complex output image
    if onlyreal:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype)
    else:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    # Create plan for fftw:
    inarr = pyfftw.n_byte_align_empty((lNy, lNx), 16, dtype_complex)

    #inarr[:] = (projection[0]*filter2)[0,:,:]
    # plan is "patient":
    #    FFTW_PATIENT is like FFTW_MEASURE, but considers a wider range
    #    of algorithms and often produces a “more optimal” plan
    #    (especially for large transforms), but at the expense of
    #    several times longer planning time (especially for large
    #    transforms).
    # print(inarr.flags)


    myifftw_plan = pyfftw.FFTW(inarr, inarr, threads=num_cores,
                               axes=(0,1),
                               direction="FFTW_BACKWARD",
                               flags=["FFTW_MEASURE"])


    #assert shared_array.base.base is shared_array_base.get_obj()
    shared_array_base = mp.Array(ct_dt_map[dtype], ln * lny * lnx)
    _shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    _shared_array = _shared_array.reshape(ln, lny, lnx)

    # Initialize the pool with the shared array
    odtbrain._shared_array = _shared_array
    pool4loop = mp.Pool(processes=num_cores)

    # filtered projections in loop
    filtered_proj = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    # Rotate all points such that we are effectively rotating everything
    # about the y-axis.
    angles = rotate_points_to_axis(points=angles, axis=tilted_axis)

    #TODO:
    # - remove the `angles*`-stuff below
    # - write method to determine rotation matrix for each point in angles
    # - apply rotation matrix in the affine transform
    # - apply same to imaginary part.

    angles_yz = np.arctan2(angles[:,1], angles[:,2])
    angles_yz -= angles_yz[0] # start at zero for comparison
    angles_yz = np.unwrap(angles_yz)
    
    angles_xz = np.arcsin(angles[:,0])
    #angles_xz -= angles_xz[0]

    # correct for general offset from linear array with multiples of 2PI
    mmax = ((np.max(angles_yz)+np.pi)//(2*np.pi))*2*np.pi
    angles_ref = np.linspace(0, mmax, angles_yz.shape[0], endpoint=False)
    angles_yz += np.average(angles_ref-angles_yz)
    
    # offset
    #offset_yz = np.rad2deg(offset_alpha)
    #offset_xz = np.rad2deg(offset_beta)
    #angles_yz += offset_yz
    #angles_xz += offset_xz
    
    #angles_xz -= np.median(angles_xz[0])
    #angles_xz -= np.average(angles_xz)

    #import matplotlib.pylab as plt
    #plt.plot(angles_yz)
    #plt.show()
    #import IPython
    #IPython.embed()

    for i in np.arange(A):
        # A == la
        # projection.shape == (A, lNx, lNy)
        # filter2.shape == (ln, lNx, lNy)
        
        for p in range(len(zv)):
            inarr[:] = filter2[p] * projection[i]
            myifftw_plan.execute()
            filtered_proj[p, :, :] = inarr[
                                           padyl:padyl + lny,
                                           padxl:padxl + lnx
                                          ] / (lNx * lNy)
        
        #phi_xz = np.rad2deg(angles_xz[i])
        #phi_yz = np.rad2deg(angles_yz[i])

        warnings.warn("Only real backpropagation implemented")

        # perform an affine transform
        #print(angles_yz[i])
        a = angles_xz[i]
        b = angles_yz[i]
        g = 0
        
        Rx = np.array([  
                       [1, 0,      0      ],
                       [0, cos(a), -sin(a)],
                       [0, sin(a),  cos(a)]
                       ])
        
        Ry = np.array([  
                       [cos(b),  0, sin(b)],
                       [0,       1, 0     ],
                       [-sin(b), 0, cos(b)]
                       ])

        
        DR = np.dot(Rx, Ry)
        
        c = 0.5*np.array(filtered_proj.shape)
        offset=c-c.dot(DR.T)
        
        rotxzr = scipy.ndimage.interpolation.affine_transform(
            filtered_proj.real, DR,
            offset=offset,
            mode="constant", cval=0, order=intp_order)
        
        outarr += rotxzr

        #if False:
        #    print("PETER")
        #    rotxzr = scipy.ndimage.interpolation.rotate(
        #        filtered_proj.real, phi_yz+offset_yz, reshape=False,
        #        #                                                 x,z
        #        mode="constant", cval=0, axes=(0, 2), order=intp_order,
        #        prefilter=False)

        
        #outarr += weights[i]*scipy.ndimage.interpolation.rotate(
        #    rotxzr, phi_xz+offset_xz, reshape=False,
        #    #                                                 x,y
        #    mode="constant", cval=0, axes=(1, 0), order=0,
        #    prefilter=False)


        #sino_filtered = projection[i] * filter2
        #for p in range(len(sino_filtered)):
        #    inarr[:] = sino_filtered[p, :, :]
        #    myifftw_plan.execute()
        #    sino_filtered[p, :, :] = inarr[:]


        # resize image to original size
        # The copy is necessary to prevent memory leakage.
        # The fftw did not normalize the data.
        #_shared_array[:] = sino_filtered.real[:ln, :lny, :lnx] / (lNx * lNy)
        # By performing the "/" operation here, we magically use less
        # memory and we gain speed...
#        _shared_array[:] = filtered_proj.real[:]
#        #_shared_array[:] = sino_filtered.real[ :ln, padyl:padyl + lny, padxl:padxl + lnx] / (lNx * lNy)
#
#        phi0 = np.rad2deg(angles[i])
#
#        if not onlyreal:
#            filtered_proj_imag = filtered_proj.imag
#        else:
#            filtered_proj_imag = None
#            # Free memory
#            del filtered_proj
#
#        _mprotate(phi0, lny, pool4loop, intp_order)
#
#        outarr.real += _shared_array
#
#        if not onlyreal:
#            _shared_array[:] = filtered_proj_imag
#            #_shared_array[:] = sino_filtered_imag[
#            #    :ln, :lny, :lnx] / (lNx * lNy)
#            del filtered_proj_imag
#            _mprotate(phi0, lny, pool4loop, intp_order)
#            outarr.imag += _shared_array

        # if False:
        #    # ~(0.9^num_cores)x speedup
        #    ang = np.rad2deg(angles[i])
        #
        #    N = int(ln)
        #
        #    slsize = int(np.floor(ln/N))
        #
        #    targ_args=list()
        #    for t in range(N):
        #        ymin = t*slsize
        #        ymax = (t+1)*slsize
        #        if t == N - 1:
        #            ymax = ln
        #        #print(ymin,ymax,sino_fin.shape)
        #        targ_args.append((sino_fin[:,ymin:ymax,:],ang))
        #
        #        if t%num_cores == num_cores-1:
        #            out = pool4loop.map(_rotate2, targ_args)
        #            um = len(out)
        #
        #            for u in range(1,um+1):
        #                #print(slsize, um, u, t)
        #                sino_fin[:,(t-um+u)*slsize:(t-um+u+1)*slsize,:] = out[u-1]
        #            targ_args=list()
        #            del out
        #
        #    #print(len(targ_args))
        #    #targ_args=sino_fin.real.reshape(2,ln,-1,ln)
        #
        #
        #    #for t in range(N):
        #    #    ymin = t*slsize
        #    #    ymax = (t+1)*slsize
        #    #    if t == N - 1:
        #    #        ymax = ln
        #    #    #print(ymin,ymax,sino_fin.shape)
        #    #    sino_fin.real[:,ymin:ymax,:] = out[t]
        #
        # if False:
        #    # This is very time-consuming
        #    rot_data = scipy.ndimage.interpolation.rotate(
        #                  sino_fin.real[:],
        #                  np.rad2deg(angles[i]),
        #                  (0,2),
        #                  False,
        #                  sino_fin.real[:],
        #                  3,
        #                  "constant",
        #                  0)

        #outarr += np.array(results).reshape(ln,ln,ln)

        # if not onlyreal:
        #    outarr += 1j*scipy.ndimage.interpolation.rotate(
        #              sino_fin.imag, angles[i]*180/np.pi, reshape=False,
        # z,x
        #              mode="constant", cval=0, axes=(0,2))

        if jmc is not None:
            jmc.value += 1

    pool4loop.terminate()
    pool4loop.join()

    #del _shared_array, inarr, odtbrain._shared_array
    #del shared_array_base

    gc.collect()

    return outarr
