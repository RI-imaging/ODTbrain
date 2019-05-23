import multiprocessing as mp

import numpy as np
import pyfftw
import scipy.ndimage as ndi


def apple_core_3d(shape, res, nm):
    r"""Return a binary array with the apple core in 3D

    Parameters
    ----------
    shape: list-like, length 3
        Shape of the reconstruction volume for which to compute
        the apple core; The second (y-) axis is assumed to be the
        axis of symmetry (according to ODTbrain standard notation)
    res: float
        Size of the vacuum wave length :math:`\lambda` in pixels
    nm: float
        Refractive index of the medium :math:`n_\mathrm{m}`

    Returns
    -------
    core: 3D ndarray
        The mask is `True` for positions within the apple core
    """
    km = (2 * np.pi * nm) / res

    lNx, lNy, lNz = shape

    if lNx != lNz:
        raise ValueError("`shape[0]` and `shape[2]` must be identical!")

    fx = np.fft.fftfreq(lNx).reshape(-1, 1, 1)
    fy = np.fft.fftfreq(lNy).reshape(1, -1, 1)
    fz = np.fft.fftfreq(lNz).reshape(1, 1, -1)

    ky = 2*np.pi * fy
    kxz = 2*np.pi * np.sqrt(fx**2 + fz**2)
    kr = 2*np.pi * np.sqrt(fx**2 + fy**2 + fz**2)

    # 1. initiate empy array
    core = np.zeros(shape, dtype=bool)

    # 2. fill core
    root = 2*km*kxz - kxz**2
    root[root < 0] = 0
    core[np.abs(ky) > np.sqrt(root)] = True

    # 3. remove enveloping sphere (resolution limit)
    core[kr > np.sqrt(2) * km] = False

    return core


def constraint_nn(data, mask=None, bg_shell=None):
    """Non-negativity constraint"""
    # No imaginary RI (no absorption)
    if np.iscomplexobj(data):
        data.imag[:] = 0
    # background medium shell
    if bg_shell is not None:
        data.real[bg_shell] = 0
    # Also remove outer shell
    spov = spillover_region(data.shape)
    data.real[spov] = 0

    lowri = data.real < 0
    if mask is not None:
        # honor given mask
        lowri *= mask
    data.real[lowri] = 0


def constraint_sh(data, mask=None, bg_shell=None):
    """Symmetric histogram background data constraint"""
    # No imaginary RI (no absorption)
    if np.iscomplexobj(data):
        data.imag[:] = 0

    # determine range of medium RI (using background support)
    spov = spillover_region(data.shape)
    if bg_shell is not None:
        spov |= bg_shell
    fmin = np.min(data.real[spov])
    fmax = np.max(data.real[spov])

    # center
    full_hist, full_edge = np.histogram(
        data.real, bins=100, range=(fmin, fmax))
    de = full_edge[1] - full_edge[0]
    full_f = full_edge[1:] - de/2
    # center index (actually we would expect f_c==0)
    idx_c = np.argmax(full_hist)
    # half-maximum indices
    idx_start = idx_c - count_to_half(full_hist[:idx_c][::-1])
    idx_end = idx_c + count_to_half(full_hist[idx_c:])
    # RI values outside
    below = (data.real > fmin) * (data.real < full_f[idx_start])
    above = (data.real > full_f[idx_end]) * (data.real < fmax)
    out = below | above
    if mask is not None:
        # honor given mask
        out *= mask
    # push RI values to zero
    data.real[out] *= .5

    if bg_shell is not None:
        # push known background data to zero
        data.real[bg_shell] *= .5


def correct(f, res, nm, method="nn", mask=None, bg_shell_width=None,
            enforce_envelope=0.95, max_iter=100, min_diff=.01,
            count=None, max_count=None):
    r"""Fill the missing apple core of the object function

    Parameters
    ----------
    f: 3D ndarray
        Complex objec function :math:`f(\mathbf{r})`
    res: float
        Size of the vacuum wave length :math:`\lambda` in pixels
    nm: float
        Refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`n(\mathbf{r})`
    method: str
        One of:

        - "nn": non-negativity constraint (`f > 0`). This method
          resembles classic missing apple core correction.
        - "sh": symmetric histogram constraint (background data in
          `f`). This method works well for sparse-gradient data (e.g.
          works better than "nn" for simulated data), but might result
          in stripe-like artifacts when applied to experimental data.

        The imaginary part of the refractive index is suppressed
        in both cases.
        Note that these constraints are soft, i.e. after the final
        inverse Fourier transform, the conditions might not be met.

    mask: 3D boolean ndarray, or None
        Optional, defines background region(s) used for enforcing
        `method`. If a boolean ndarray, the values set to `True` define
        the used background regions.
    bg_shell_width: float
        Optional, defines the width of an ellipsoid shell (outer radii
        matching image shape) that is used additionally for enforcing
        `method`.
    enforce_envelope: float in interval [0,1] or False
        Set the suppression factor for frequencies that are above
        the envelope function; disabled if set to False or 0
    max_iter: int
        Maximum number of iterations to perform
    min_diff: float
        Stopping criterion computed as the relative difference
        (relative to the first iteration `norm`) of the changes applied
        during the current iteration `cur_diff`:
        ``np.abs(cur_diff/norm) < min_diff``
    count: multiprocessing.Value
        May be used for tracking progress. At each iteration
        `count.value` is incremented by one.
    max_count: multiprocessing.Value
        May be used for tracking progress; is incremented initially.

    Notes
    -----
    Internally, the Fourier transform is performed with single-precision
    floating point values (complex64).
    """
    if enforce_envelope < 0 or enforce_envelope > 1:
        raise ValueError("`enforce_envelope` must be in interval [0, 1]")

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += max_iter + 2

    # Location of the apple core
    core = apple_core_3d(shape=f.shape, res=res, nm=nm)

    if count is not None:
        with count.get_lock():
            count.value += 1

    data = pyfftw.empty_aligned(f.shape, dtype='complex64')
    ftdata = pyfftw.empty_aligned(f.shape, dtype='complex64')
    fftw_forw = pyfftw.FFTW(data, ftdata,
                            axes=(0, 1, 2),
                            direction="FFTW_FORWARD",
                            flags=["FFTW_MEASURE"],
                            threads=mp.cpu_count())
    # Note: input array `ftdata` is destroyed when invoking `fftw_back`
    fftw_back = pyfftw.FFTW(ftdata, data,
                            axes=(0, 1, 2),
                            direction="FFTW_BACKWARD",
                            flags=["FFTW_MEASURE"],
                            threads=mp.cpu_count())

    data.real[:] = f.real
    data.imag[:] = 0

    fftw_forw.execute()
    ftdata_orig = ftdata.copy()

    if count is not None:
        with count.get_lock():
            count.value += 1

    if enforce_envelope:
        # Envelope function of Fourier amplitude
        ftevlp = envelope_gauss(ftdata_orig, core)

    init_state = np.sum(np.abs(ftdata_orig[core])) / data.size
    prev_state = init_state

    if bg_shell_width is not None:
        bg_shell = ellipsoid_shell(data.shape, width=bg_shell_width)
    else:
        bg_shell = None

    for ii in range(max_iter):
        if method == "nn":
            # non-negativity
            constraint_nn(data=data, mask=mask, bg_shell=bg_shell)
        elif method == "sh":
            # symmetric histogram
            constraint_sh(data=data, mask=mask, bg_shell=bg_shell)

        # Go into Fourier domain
        fftw_forw.execute()
        if enforce_envelope:
            # Suppress large frequencies with the envelope
            high = np.abs(ftdata) > ftevlp
            ftdata[high] *= enforce_envelope

        if method == "sh":
            # update dc term
            ftdata_orig[0, 0, 0] = (ftdata_orig[0, 0, 0] + ftdata[0, 0, 0])/2
        # Enforce original data
        ftdata[~core] = ftdata_orig[~core]

        fftw_back.execute()
        data[:] /= fftw_forw.N

        if count is not None:
            with count.get_lock():
                count.value += 1

        cur_state = np.sum(np.abs(ftdata[core])) / data.size
        cur_diff = cur_state - prev_state
        if ii == 0:
            norm = cur_diff
        else:
            if np.abs(cur_diff/norm) < min_diff:
                break
        prev_state = cur_state

    if count is not None:
        with count.get_lock():
            # add skipped counts (due to stopping criterion)
            count.value += max_iter - ii - 1

    return data


def count_to_half(array):
    """Determination of half-initial value index

    Return first index at which array values decrease below 1/2 of
    the initial initial value `array[0]`.
    """
    num = 0
    for item in array[1:]:
        if item < array[0] / 2:
            break
        else:
            num += 1
    return num


def ellipsoid_shell(shape, width=20):
    """Return background ellipsoid shell"""
    spov_outer = spillover_region(shape, shell=0)
    spov_inner = spillover_region(shape, shell=width)
    reg = spov_outer ^ spov_inner
    return reg


def envelope_gauss(ftdata, core):
    r"""Compute a gaussian-filtered envelope, without apple core

    Parameters
    ----------
    ftdata: 3D ndarray
        Fourier transform of the object function data
        (zero frequency not shifted to center of array)
    core: 3D ndarray (same shape as ftdata)
        Apple core (as defined by :func:`apple_core_3d`)

    Returns
    -------
    envelope: 3D ndarray
        Envelope function in Fourier space
    """
    hull = np.abs(ftdata)
    hull[core] = np.nan  # label core data with nans
    # Fill the apple core region with data from known regions from
    # the other axes (we only need an estimate if the envelope, so
    # this is a very good estimation of the Fourier amplitudes).
    shx, shy, _ = hull.shape
    maxsh = max(shx, shy)
    dsh = abs(shy - shx) // 2

    # Determine the slice
    if shx > shy:
        theslice = (slice(0, shx),
                    slice(dsh, shy+dsh),
                    slice(0, shx))
    else:
        theslice = (slice(dsh, shx+dsh),
                    slice(0, shy),
                    slice(dsh, shx+dsh),
                    )
    # 1. Create padded versions of the arrays, because shx and shy
    # can be different and inserting a transposed array will not work.
    hull_pad = np.zeros((maxsh, maxsh, maxsh), dtype=float)
    hull_pad[theslice] = np.fft.fftshift(hull)
    core_pad = np.zeros((maxsh, maxsh, maxsh), dtype=bool)
    core_pad[theslice] = np.fft.fftshift(core)
    # 2. Fill values from other axes were data are missing.
    hull_pad[core_pad] = np.transpose(hull_pad, (1, 0, 2))[core_pad]
    # 3. Fill any remaining nan-values (due to different shape or tilt)
    # with nearest neighbors. Use a distance transform for nearest
    # neighbor interpolation.
    invalid = np.isnan(hull_pad)
    ind = ndi.distance_transform_edt(invalid,
                                     return_distances=False,
                                     return_indices=True)
    hull_pad[:] = hull_pad[tuple(ind)]
    # 4. Write the data back to the original array.
    hull[:] = np.fft.ifftshift(hull_pad[theslice])
    # Perform gaussian blurring (shift data to make it smooth)
    gauss = ndi.gaussian_filter(input=np.fft.fftshift(hull),
                                sigma=np.max(ftdata.shape)/100,
                                mode="constant",
                                cval=0,
                                truncate=4.0)
    # Shift back gauss
    shifted_gauss = np.fft.ifftshift(gauss)
    return shifted_gauss


def spillover_region(shape, shell=0):
    """Return boolean array for region outside ellipsoid"""
    x = np.arange(shape[0]).reshape(-1, 1, 1)
    y = np.arange(shape[1]).reshape(1, -1, 1)
    z = np.arange(shape[2]).reshape(1, 1, -1)

    cx, cy, cz = np.array(shape) / 2
    spov = (((x-cx)/(cx-shell))**2
            + ((y-cy)/(cy-shell))**2
            + ((z-cz)/(cz-shell))**2) > 1
    return spov
