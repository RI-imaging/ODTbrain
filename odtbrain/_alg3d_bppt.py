"""3D backpropagation algorithm with a tilted axis of rotation"""
import multiprocessing as mp
import warnings

import numexpr as ne
import numpy as np
import pyfftw
import scipy.ndimage

from . import util


_ncores = mp.cpu_count()


def estimate_major_rotation_axis(loc):
    """
    For a list of points on the unit sphere, estimate the main
    rotational axis and return a list of angles that correspond
    to the rotational position for each point.
    """
    # TODO:
    raise NotImplementedError("estimation of rotational axis not implemented.")


def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))


def rotate_points_to_axis(points, axis):
    """Rotate all points of a list, such that `axis==[0,1,0]`

    This is accomplished by rotating in the x-z-plane by phi into the
    y-z-plane, then rotation in the y-z-plane by theta up to [0,1,0],
    and finally rotating back in the x-z-plane by -phi.

    Parameters
    ----------
    points: list-like with elements of length 3
        The Cartesian points. These should be in the same format as
        produced by `sphere_points_from_angles_and_tilt`.
    axis: list-like, length 3
        The reference axis that will be used to determine the
        rotation angle of the points. The points will be rotated
        about the origin such that `axis` matches [0,1,0].

    Returns
    -------
    rotated_points: np.ndarray of shape (N,3)
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
        [0, 1,            0],
        [np.sin(phi), 0,  np.cos(phi)],
    ])

    # Negative rotation about x-axis
    Rtheta = np.array([
        [1,             0,              0],
        [0, np.cos(theta),  np.sin(theta)],
        [0, -np.sin(theta),  np.cos(theta)],
    ])

    DR1 = np.dot(Rtheta, Rphi)
    # Rotate back by -phi such that effective rotation was only
    # towards [0,1,0].
    DR = np.dot(Rphi.T, DR1)
    rotpoints = np.zeros((len(points), 3))
    for ii, pnt in enumerate(points):
        rotpoints[ii] = np.dot(DR, pnt)

    # For visualiztaion:
    # import matplotlib.pylab as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.patches import FancyArrowPatch
    # from mpl_toolkits.mplot3d import proj3d
    #
    # class Arrow3D(FancyArrowPatch):
    #     def __init__(self, xs, ys, zs, *args, **kwargs):
    #         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
    #         self._verts3d = xs, ys, zs
    #
    #     def draw(self, renderer):
    #         xs3d, ys3d, zs3d = self._verts3d
    #         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    #         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
    #         FancyArrowPatch.draw(self, renderer)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # for vec in rotpoints:
    #     u,v,w = vec
    #     a = Arrow3D([0,u],[0,v],[0,w],
    #                 mutation_scale=20, lw=1, arrowstyle="-|>")
    #     ax.add_artist(a)
    #
    # radius=1
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(-radius*1.5, radius*1.5)
    # ax.set_ylim(-radius*1.5, radius*1.5)
    # ax.set_zlim(-radius*1.5, radius*1.5)
    # plt.tight_layout()
    # plt.show()

    return rotpoints


def rotation_matrix_from_point(point, ret_inv=False):
    """Compute rotation matrix to go from [0,0,1] to `point`.

    First, the matrix rotates to in the polar direction. Then,
    a rotation about the y-axis is performed to match the
    azimuthal angle in the x-z-plane.

    This rotation matrix is required for the correct 3D orientation
    of the backpropagated projections.

    Parameters
    ----------
    points: list-like, length 3
        The coordinates of the point in 3D.
    ret_inv: bool
        Also return the inverse of the rotation matrix. The inverse
        is required for :func:`scipy.ndimage.interpolation.affine_transform`
        which maps the output coordinates to the input coordinates.

    Returns
    -------
    Rmat [, Rmat_inv]: 3x3 ndarrays
        The rotation matrix that rotates [0,0,1] to `point` and
        optionally its inverse.
    """
    x, y, z = point
    # azimuthal angle
    phi = np.arctan2(x, z)
    # angle in polar direction (negative)
    theta = -np.arctan2(y, np.sqrt(x**2+z**2))

    # Rotation in polar direction
    Rtheta = np.array([
        [1,             0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
    ])

    # rotation in x-z-plane
    Rphi = np.array([
        [np.cos(phi), 0, -np.sin(phi)],
        [0, 1,            0],
        [np.sin(phi), 0, np.cos(phi)],
    ])

    D = np.dot(Rphi, Rtheta)
    # The inverse of D
    Dinv = np.dot(Rtheta.T, Rphi.T)

    if ret_inv:
        return D, Dinv
    else:
        return D


def rotation_matrix_from_point_planerot(point, plane_angle, ret_inv=False):
    """
    Compute rotation matrix to go from [0,0,1] to `point`,
    while taking into account the tilted axis of rotation.

    First, the matrix rotates to in the polar direction. Then,
    a rotation about the y-axis is performed to match the
    azimuthal angle in the x-z-plane.

    This rotation matrix is required for the correct 3D orientation
    of the backpropagated projections.

    Parameters
    ----------
    points: list-like, length 3
        The coordinates of the point in 3D.
    axis: list-like, length 3
        The coordinates of the point in 3D.
    ret_inv: bool
        Also return the inverse of the rotation matrix. The inverse
        is required for :func:`scipy.ndimage.interpolation.affine_transform`
        which maps the output coordinates to the input coordinates.

    Returns
    -------
    Rmat [, Rmat_inv]: 3x3 ndarrays
        The rotation matrix that rotates [0,0,1] to `point` and
        optionally its inverse.
    """
    # These matrices are correct if there is no tilt of the
    # rotational axis within the detector plane (x-y).
    D, Dinv = rotation_matrix_from_point(point, ret_inv=True)

    # We need an additional rotation about the z-axis to correct
    # for the tilt for all the the other cases.
    angz = plane_angle

    Rz = np.array([
        [np.cos(angz), -np.sin(angz), 0],
        [np.sin(angz),  np.cos(angz), 0],
        [0,             0,            1],
    ])

    DR = np.dot(D, Rz)
    DRinv = np.dot(Rz.T, Dinv)

    if ret_inv:
        return DR, DRinv
    else:
        return DR


def sphere_points_from_angles_and_tilt(angles, tilted_axis):
    """
    For a given tilt of the rotational axis `tilted_axis`, compute
    the points on a unit sphere that correspond to the distribution
    `angles` along the great circle about this axis.

    Parameters
    ----------
    angles: 1d ndarray
        The angles that will be distributed on the great circle.
    tilted_axis: list of length 3
        The tilted axis of rotation that determines the great
        circle.

    Notes
    -----
    The reference axis is always [0,1,0].
    `theta` is the azimuthal angle measured down from the y-axis.
    `phi` is the polar angle in the x-z plane measured from z towards x.

    """
    assert len(angles.shape) == 1
    # Normalize tilted axis.
    tilted_axis = norm_vec(tilted_axis)
    [u, v, w] = tilted_axis
    # Initial distribution of points about great circle (x-z).
    newang = np.zeros((angles.shape[0], 3), dtype=float)
    # We subtract angles[0], because in step (a) we want that
    # newang[0]==[0,0,1]. This only works if we actually start
    # at that point.
    newang[:, 0] = np.sin(angles-angles[0])
    newang[:, 2] = np.cos(angles-angles[0])

    # Compute rotational angles w.r.t. [0,1,0].
    # - Draw a unit sphere with the y-axis pointing up and the
    #   z-axis pointing right
    # - The rotation of `tilted_axis` can be described by two
    #   separate rotations. We will use these two angles:
    #   (a) Rotation from y=1 within the y-z plane: theta
    #       This is the rotation that is critical for data
    #       reconstruction. If this angle is zero, then we
    #       have a rotational axis in the imaging plane. If
    #       this angle is PI/2, then our sinogram consists
    #       of a rotating image and 3D reconstruction is
    #       impossible. This angle is counted from the y-axis
    #       onto the x-z plane.
    #   (b) Rotation in the x-z plane: phi
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

    # (a) This angle is the azimuthal angle theta measured from the
    #     y-axis.
    theta = np.arccos(v)

    # (b) This is the polar angle measured in the x-z plane starting
    #     at the x-axis and measured towards the positive z-axis.
    if np.allclose(u, 0) and np.allclose(w, 0):
        # Avoid flipping the axis of rotation due to numerical
        # errors during its computation.
        phi = 0
    else:
        phi = np.arctan2(u, w)

    # Determine the projection points on the unit sphere.
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
        [1,             0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    for ii in range(newang.shape[0]):
        newang[ii] = np.dot(Rx, newang[ii])

    # a3) Shift newang such that newang[0] is located at (0,0,1)
    newang = newang - (newang[0] - np.array([0, 0, 1])).reshape(1, 3)

    # (b) Rotate the entire thing with phi about the y-axis
    #     (right-handed/counter-clockwise/basic/elemental rotation)
    Ry = np.array([
        [+np.cos(phi), 0, np.sin(phi)],
        [0,            1,           0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    for jj in range(newang.shape[0]):
        newang[jj] = np.dot(Ry, newang[jj])

    # For visualiztaion:
    # import matplotlib.pylab as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.patches import FancyArrowPatch
    # from mpl_toolkits.mplot3d import proj3d
    #
    # class Arrow3D(FancyArrowPatch):
    #     def __init__(self, xs, ys, zs, *args, **kwargs):
    #         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
    #         self._verts3d = xs, ys, zs
    #
    #     def draw(self, renderer):
    #         xs3d, ys3d, zs3d = self._verts3d
    #         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    #         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
    #         FancyArrowPatch.draw(self, renderer)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # for vec in newang:
    #     u,v,w = vec
    #     a = Arrow3D([0,u],[0,v],[0,w],
    #                 mutation_scale=20, lw=1, arrowstyle="-|>")
    #     ax.add_artist(a)
    #
    # radius=1
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(-radius*1.5, radius*1.5)
    # ax.set_ylim(-radius*1.5, radius*1.5)
    # ax.set_zlim(-radius*1.5, radius*1.5)
    # plt.tight_layout()
    # plt.show()

    return newang


def backpropagate_3d_tilted(uSin, angles, res, nm, lD=0,
                            tilted_axis=[0, 1, 0],
                            coords=None, weight_angles=True, onlyreal=False,
                            padding=(True, True), padfac=1.75, padval=None,
                            intp_order=2, dtype=None,
                            num_cores=_ncores,
                            save_memory=False,
                            copy=True,
                            count=None, max_count=None,
                            verbose=0):
    r"""3D backpropagation with a tilted axis of rotation

    Three-dimensional diffraction tomography reconstruction
    algorithm for scattering of a plane wave
    :math:`u_0(\mathbf{r}) = u_0(x,y,z)`
    by a dielectric object with refractive index
    :math:`n(x,y,z)`.

    This method implements the 3D backpropagation algorithm with
    a rotational axis that is tilted by :math:`\theta_\mathrm{tilt}`
    w.r.t. the imaging plane :cite:`Mueller2015tilted`.

    .. math::
        f(\mathbf{r}) =
            -\frac{i k_\mathrm{m}}{2\pi}
            \sum_{j=1}^{N} \! \Delta \phi_0 D_{-\phi_j}^\mathrm{tilt} \!\!
            \left \{
            \text{FFT}^{-1}_{\mathrm{2D}}
            \left \{
            \left| k_\mathrm{Dx} \cdot \cos \theta_\mathrm{tilt}\right|
            \frac{\text{FFT}_{\mathrm{2D}} \left \{
            u_{\mathrm{B},\phi_j}(x_\mathrm{D}, y_\mathrm{D}) \right \}}
            {u_0(l_\mathrm{D})}
            \exp \! \left[i k_\mathrm{m}(M - 1) \cdot
            (z_{\phi_j}-l_\mathrm{D}) \right]
            \right \}
            \right \}

    with a modified rotational operator :math:`D_{-\phi_j}^\mathrm{tilt}`
    and a different filter in Fourier space
    :math:`|k_\mathrm{Dx} \cdot \cos \theta_\mathrm{tilt}|` when compared
    to :func:`backpropagate_3d`.


    .. versionadded:: 0.1.2

    Parameters
    ----------
    uSin: (A, Ny, Nx) ndarray
        Three-dimensional sinogram of plane recordings
        :math:`u_{\mathrm{B}, \phi_j}(x_\mathrm{D}, y_\mathrm{D})`
        divided by the incident plane wave :math:`u_0(l_\mathrm{D})`
        measured at the detector.
    angles: ndarray of shape (A,3) or 1D array of length A
        If the shape is (A,3), then `angles` consists of vectors
        on the unit sphere that correspond to the direction
        of illumination and acquisition (s₀). If the shape is (A,),
        then `angles` is  a one-dimensional array of angles in radians
        that determines the angular position :math:`\phi_j`.
        In both cases, `tilted_axis` must be set according to the
        tilt of the rotational axis.
    res: float
        Vacuum wavelength of the light :math:`\lambda` in pixels.
    nm: float
        Refractive index of the surrounding medium :math:`n_\mathrm{m}`.
    lD: float
        Distance from center of rotation to detector plane
        :math:`l_\mathrm{D}` in pixels.
    tilted_axis: list of floats
        The coordinates [x, y, z] on a unit sphere representing the
        tilted axis of rotation. The default is (0,1,0),
        which corresponds to a rotation about the y-axis and
        follows the behavior of :func:`odtbrain.backpropagate_3d`.
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

        This currently only works when `angles` has the shape (A,).
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
        size of 256 for an initial size of 144, whereas for it will
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
        See :func:`scipy.ndimage.interpolation.affine_transform` for details.
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
    This implementation can deal with projection angles that are not
    distributed along a circle  about the rotational axis. If there are
    slight deviations from this circle, simply pass the 3D rotational
    positions instead of the 1D angles to the `angles` argument. In
    principle, this should improve the reconstruction. The general
    problem here is that the backpropagation algorithm requires a
    ramp filter in Fourier space that is oriented perpendicular to the
    rotational axis. If the sample does not rotate about a single axis,
    then a 1D parametric representation of this rotation must be found
    to correctly determine the filter in Fourier space. Such a
    parametric representation could e.g. be a spiral between the poles
    of the unit sphere (but this kind of rotation is probably difficult
    to implement experimentally).

    If you have input images with rectangular shape, e.g. Nx!=Ny and
    the rotational axis deviates by approximately PI/2 from the axis
    (0,1,0), then data might get cropped in the reconstruction volume.
    You can avoid that by rotating your input data and the rotational
    axis by PI/2. For instance, change`tilted_axis` from [1,0,0] to
    [0,1,0] and `np.rot90` the sinogram images.

    Do not use the parameter `lD` in combination with the Rytov
    approximation - the propagation is not correctly described.
    Instead, numerically refocus the sinogram prior to converting
    it to Rytov data (using e.g. :func:`odtbrain.sinogram_as_rytov`)
    with a numerical focusing algorithm (available in the Python
    package :py:mod:`nrefocus`).
    """
    A = angles.shape[0]

    if angles.shape not in [(A,), (A, 1), (A, 3)]:
        raise ValueError("`angles` must have shape (A,) or (A,3)!")
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
    if num_cores > _ncores:
        raise ValueError("`num_cores` must not exceed number "
                         + "of physical cores: {}".format(_ncores))

    # setup dtype
    if dtype is None:
        dtype = np.float_
    dtype = np.dtype(dtype)
    if dtype.name not in ["float32", "float64"]:
        raise ValueError("dtype must be float32 or float64!")
    dtype_complex = np.dtype("complex{}".format(
        2 * int(dtype.name.strip("float"))))

    # progess monitoring
    if max_count is not None:
        max_count.value += A + 2

    ne.set_num_threads(num_cores)

    uSin = np.array(uSin, copy=copy)
    angles = np.array(angles, copy=copy)
    angles = np.squeeze(angles)  # support shape (A,1)

    # lengths of the input data
    lny, lnx = uSin.shape[1], uSin.shape[2]
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

    # `tilted_axis` is required for several things:
    # 1. the filter |kDx*v + kDy*u| with (u,v,w)==tilted_axis
    # 2. the alignment of the rotational axis with the y-axis
    # 3. the determination of the point coordinates if only
    #    angles in radians are given.

    # For (1) we need the exact axis that corresponds to our input data.
    # For (2) and (3) we need `tilted_axis_yz` (see below) which is the
    # axis `tilted_axis` rotated in the detector plane such that its
    # projection onto the detector coincides with the y-axis.

    # Normalize input axis
    tilted_axis = norm_vec(tilted_axis)

    # `tilted_axis_yz` is computed by performing the inverse rotation in
    # the x-y plane with `angz`. We will again use `angz` in the transform
    # within the for-loop to rotate each projection according to its
    # acquisition angle.
    angz = np.arctan2(tilted_axis[0], tilted_axis[1])
    rotmat = np.array([
        [np.cos(angz), -np.sin(angz), 0],
        [np.sin(angz),  np.cos(angz), 0],
        [0,             0, 1],
    ])
    # rotate `tilted_axis` onto the y-z plane.
    tilted_axis_yz = norm_vec(np.dot(rotmat, tilted_axis))

    if len(angles.shape) == 1:
        if weight_angles:
            weights = util.compute_angle_weights_1d(angles).reshape(-1, 1, 1)
        # compute the 3D points from tilted axis
        angles = sphere_points_from_angles_and_tilt(angles, tilted_axis_yz)
    else:
        if weight_angles:
            warnings.warn("3D angular weighting not yet supported!")
            weights = 1

        # normalize and rotate angles
        for ii in range(angles.shape[0]):
            # angles[ii] = norm_vec(angles[ii]) #-> not correct
            # instead rotate like `tilted_axis` onto the y-z plane.
            angles[ii] = norm_vec(np.dot(rotmat, angles[ii]))

    if weight_angles:
        uSin *= weights

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
    # The filter is now dependent on the rotational position of the
    # specimen. We have to include information from the angles.
    # We want to estimate the rotational axis for every frame. We
    # do that by computing the cross-product of the vectors in
    # angles from the current and previous image.

    u, v, _w = tilted_axis
    filterabs = np.abs(kx*v+ky*u) * filter_klp
    # new in version 0.1.4:
    # We multiply by the factor (M-1) instead of just (M)
    # to take into account that we have a scattered
    # wave that is normalized by u0.
    prefactor *= np.exp(-1j * km * (M-1) * lD)

    if count is not None:
        count.value += 1

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
    # x = np.linspace(-centerx, centerx, lNx, endpoint=False)
    # x = np.arange(lNx) - center + .5
    # Meshgrid for output array
    # zv, yv, xv = np.meshgrid(x,x,x)
    #               z, y, x
    # xv = x.reshape( 1, 1,-1)
    # yv = x.reshape( 1,-1, 1)

    # z = np.arange(ln) - center + .5
    z = np.linspace(-center, center, lNz, endpoint=False)
    zv = z.reshape(-1, 1, 1)

    #              y,   x
    Mp = M.reshape(lNy, lNx)

    # filter2 = np.exp(1j * zv * km * (Mp - 1))
    f2_exp_fac = 1j * km * (Mp - 1)
    if save_memory:
        # compute filter2 later
        pass
    else:
        # compute filter2 now
        # (this requires more RAM but is faster)
        filter2 = ne.evaluate("exp(factor * zv)",
                              local_dict={"factor": f2_exp_fac,
                                          "zv": zv})

    if count is not None:
        count.value += 1

    # Prepare complex output image
    if onlyreal:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype)
    else:
        outarr = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    # Create plan for FFTW:
    # Flag is "estimate":
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

    # filtered projections in loop
    filtered_proj = np.zeros((ln, lny, lnx), dtype=dtype_complex)

    # Rotate all points such that we are effectively rotating everything
    # about the y-axis.
    angles = rotate_points_to_axis(points=angles, axis=tilted_axis_yz)

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
        # normalize to (lNx * lNy) for FFTW and multiply with prefactor, filter
        oneslice *= filterabs * prefactor / (lNx * lNy)

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
            filtered_proj[p, :, :] = inarr[padyl:padyl+lny, padxl:padxl+lnx]

        # The Cartesian axes in our array are ordered like this: [z,y,x]
        # However, the rotation matrix requires [x,y,z]. Therefore, we
        # need to np.transpose the first and last axis and also invert the
        # y-axis.
        fil_p_t = filtered_proj.transpose(2, 1, 0)[:, ::-1, :]

        # get rotation matrix for this point and also rotate in plane
        _drot, drotinv = rotation_matrix_from_point_planerot(angles[aa],
                                                             plane_angle=angz,
                                                             ret_inv=True)

        # apply offset required by affine_transform
        # The offset is only required for the rotation in
        # the x-z-plane.
        # This could be achieved like so:
        # The offset "-.5" assures that we are rotating about
        # the center of the image and not the value at the center
        # of the array (this is also what `scipy.ndimage.rotate` does.
        c = 0.5 * np.array(fil_p_t.shape) - .5
        offset = c - np.dot(drotinv, c)

        # Perform rotation
        # We cannot split the inplace-rotation into multiple subrotations
        # as we did in _Back_3d_tilted.backpropagate_3d, because the rotation
        # axis is arbitrarily placed in the 3d array. Rotating single
        # slices does not yield the same result as rotating the entire
        # array. Instead of using affine_transform, map_coordinates might
        # be faster for multiple cores.

        # Also undo the axis transposition that we performed previously.

        outarr.real += scipy.ndimage.interpolation.affine_transform(
            fil_p_t.real, drotinv,
            offset=offset,
            mode="constant",
            cval=0,
            order=intp_order).transpose(2, 1, 0)[:, ::-1, :]

        if not onlyreal:
            outarr.imag += scipy.ndimage.interpolation.affine_transform(
                fil_p_t.imag, drotinv,
                offset=offset,
                mode="constant",
                cval=0,
                order=intp_order).transpose(2, 1, 0)[:, ::-1, :]

        if count is not None:
            count.value += 1

    return outarr
