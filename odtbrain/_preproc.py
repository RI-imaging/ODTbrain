"""Data pre-processing in optical tomography"""
import numpy as np
from scipy.stats import mode
from skimage.restoration import unwrap_phase


def align_unwrapped(sino):
    """Align an unwrapped phase array to zero-phase

    All operations are performed in-place.
    """
    samples = []
    if len(sino.shape) == 2:
        # 2D
        # take 1D samples at beginning and end of array
        samples.append(sino[:, 0])
        samples.append(sino[:, 1])
        samples.append(sino[:, 2])
        samples.append(sino[:, -1])
        samples.append(sino[:, -2])

    elif len(sino.shape) == 3:
        # 3D
        # take 1D samples at beginning and end of array
        samples.append(sino[:, 0, 0])
        samples.append(sino[:, 0, -1])
        samples.append(sino[:, -1, 0])
        samples.append(sino[:, -1, -1])
        samples.append(sino[:, 0, 1])

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


def divmod_neg(a, b):
    """Return divmod with closest result to zero"""
    q, r = divmod(a, b)
    # make sure r is close to zero
    sr = np.sign(r)
    if np.abs(r) > b/2:
        q += sr
        r -= b * sr
    return q, r


def sinogram_as_radon(uSin, align=True):
    r"""Compute the phase from a complex wave field sinogram

    This step is essential when using the ray approximation before
    computation of the refractive index with the inverse Radon
    transform.

    Parameters
    ----------
    uSin: 2d or 3d complex ndarray
        The background-corrected sinogram of the complex scattered wave
        :math:`u(\mathbf{r})/u_0(\mathbf{r})`. The first axis iterates
        through the angles :math:`\phi_0`.
    align: bool
        Tries to correct for a phase offset in the phase sinogram.

    Returns
    -------
    phase: 2d or 3d real ndarray
        The unwrapped phase array corresponding to `uSin`.

    See Also
    --------
    skimage.restoration.unwrap_phase: phase unwrapping
    radontea.backproject_3d: e.g. reconstruction via backprojection
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
        for ii in range(len(phiR)):
            phiR[ii] = unwrap_phase(phiR[ii], seed=47)

    if align:
        align_unwrapped(phiR)

    return phiR


def sinogram_as_rytov(uSin, u0=1, align=True):
    r"""Convert the complex wave field sinogram to the Rytov phase

    This method applies the Rytov approximation to the
    recorded complex wave sinogram. To achieve this, the following
    filter is applied:

    .. math::
        u_\mathrm{B}(\mathbf{r}) = u_\mathrm{0}(\mathbf{r})
            \ln\!\left(
            \frac{u_\mathrm{R}(\mathbf{r})}{u_\mathrm{0}(\mathbf{r})}
             +1 \right)

    This filter step effectively replaces the Born approximation
    :math:`u_\mathrm{B}(\mathbf{r})` with the Rytov approximation
    :math:`u_\mathrm{R}(\mathbf{r})`, assuming that the scattered
    field is equal to
    :math:`u(\mathbf{r})\approx u_\mathrm{R}(\mathbf{r})+
    u_\mathrm{0}(\mathbf{r})`.


    Parameters
    ----------
    uSin: 2d or 3d complex ndarray
        The sinogram of the complex wave
        :math:`u_\mathrm{R}(\mathbf{r}) + u_\mathrm{0}(\mathbf{r})`.
        The first axis iterates through the angles :math:`\phi_0`.
    u0: ndarray of dimension as `uSin` or less, or int.
        The incident plane wave
        :math:`u_\mathrm{0}(\mathbf{r})` at the detector.
        If `u0` is "1", it is assumed that the data is already
        background-corrected (
        `uSin` :math:`= \frac{u_\mathrm{R}(\mathbf{r})}{
        u_\mathrm{0}(\mathbf{r})} + 1`
        ). Note that if the reconstruction distance :math:`l_\mathrm{D}`
        of the original experiment is non-zero and `u0` is set to 1,
        then the reconstruction will be wrong; the field is not focused
        to the center of the reconstruction volume.
    align: bool
        Tries to correct for a phase offset in the phase sinogram.

    Returns
    -------
    uB: 2d or 3d real ndarray
        The Rytov-filtered complex sinogram
        :math:`u_\mathrm{B}(\mathbf{r})`.

    See Also
    --------
    skimage.restoration.unwrap_phase: phase unwrapping
    """
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
        for ii in range(len(phiR)):
            phiR[ii] = unwrap_phase(phiR[ii], seed=47)

    if align:
        align_unwrapped(phiR)

    # rytovSin = u0*(np.log(a/a0) + 1j*phiR)
    # u0 is one - we already did background correction

    # complex rytov phase:
    rytovSin = 1j * phiR + lna
    return u0 * rytovSin
