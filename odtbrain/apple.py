import numpy as np
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


def correct(ri, res, nm, bg_mask=None, ri_min=None, ri_max=None,
            enforce_envelope=0.95, max_iter=100, min_diff=.01,
            count=None, max_count=None):
    r"""Fill the missing apple core of the object function

    Enforces `ri.imag==0` and `ri.real>=ri_min`; This enforcement
    is soft, i.e. after the final inverse Fourier transform, these
    conditions might not be met.

    Parameters
    ----------
    ri: 3D ndarray
        Complex refractive index :math:`n(\mathbf{r})`
    res: float
        Size of the vacuum wave length :math:`\lambda` in pixels
    nm: float
        Refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`n(\mathbf{r})`
    bg_mask: 3D boolean ndarray
        Defines background region(s) used for enforcing `ri_min`
    ri_min: float
        Minimum refractive index value (condition set while iterating);
        If set to `None`, then `ri_min` is set to `nm`; the region
        used for enforcing this condition can be defined with `bg_mask`
    ri_max: float
        Maximum refractive index value (condition set during iteration)
    enforce_envelope: float (>0) or False
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
        May be used for tracking progress. Is incremented initially.
    """
    if enforce_envelope < 0 or enforce_envelope > 1:
        raise ValueError("`enforce_envelope` must be in interval [0, 1]")

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += max_iter + 2

    core = apple_core_3d(shape=ri.shape, res=res, nm=nm)

    if count is not None:
        with count.get_lock():
            count.value += 1

    if ri_min is None:
        ri_min = nm

    data = ri.real.copy()
    ftdata = np.fft.fftn(data)

    if count is not None:
        with count.get_lock():
            count.value += 1

    if enforce_envelope:
        ftevlp = envelope_gauss(ftdata, core)

    init_state = np.sum(np.abs(ftdata[core])) / data.size
    prev_state = init_state

    for ii in range(max_iter):
        # No imaginary RI (no absorption)
        data = data.real
        # refractive index is higher than air/water/medium
        lowri = data < ri_min
        if bg_mask is not None:
            # if given, onnly do this in the masked data regions
            lowri *= ~bg_mask
        data[lowri] = ri_min
        if ri_max is not None:
            data[data > ri_max] = ri_max
        # go into Fourier domain
        ftdata2 = np.fft.fftn(data)
        if enforce_envelope:
            # suppress large frequencies
            high = np.abs(ftdata2) > ftevlp
            ftdata2[high] *= enforce_envelope

        # enforce original data
        ftdata2[~core] = ftdata[~core]

        data = np.fft.ifftn(ftdata2)

        if count is not None:
            with count.get_lock():
                count.value += 1

        cur_state = np.sum(np.abs(ftdata2[core])) / data.size
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


def envelope_gauss(ftdata, core):
    r"""Compute a gaussian-filtered envelope, without apple core

    Parameters
    ----------
    ftdata: 3D ndarray
        Fourier transform of the refractive index data
        (zero frequency not shifted to center of array)
    core: 3D ndarray (same shape as ftdata)
        Apple core (as defined by :func:`apple_core_3d`)

    Returns
    -------
    envelope: 3D ndarray
        Envelope function in Fourier space
    """
    hull = np.abs(ftdata)
    hull[core] = np.nan
    # fill unknown values
    hull2 = np.transpose(hull, (1, 0, 2))
    hull[core] = hull2[core]

    if np.sum(np.isnan(hull)):
        raise NotImplementedError("Probably tilted axis!")

    # gaussian blur
    gauss = ndi.gaussian_filter(input=np.fft.fftshift(hull),
                                sigma=np.max(ftdata.shape)/100,
                                mode="constant",
                                cval=0,
                                truncate=4.0)

    shifted_gauss = np.fft.ifftshift(gauss)
    return shifted_gauss
