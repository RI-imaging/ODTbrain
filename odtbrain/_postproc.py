"""Data post-processing in optical tomography"""
import numpy as np


def odt_to_ri(f, res, nm):
    r"""Convert the ODT object function to refractive index

    In :abbr:`ODT (Optical Diffraction Tomography)`, the object function
    is defined by the Helmholtz equation

    .. math::

        f(\mathbf{r})  =  k_\mathrm{m}^2 \left[
            \left( \frac{n(\mathbf{r})}{n_\mathrm{m}} \right)^2 - 1
            \right]

    with :math:`k_\mathrm{m} = \frac{2\pi n_\mathrm{m}}{\lambda}`.
    By inverting this equation, we obtain the refractive index
    :math:`n(\mathbf{r})`.

    .. math::

        n(\mathbf{r})  = n_\mathrm{m}
            \sqrt{\frac{f(\mathbf{r})}{k_\mathrm{m}^2} + 1 }

    Parameters
    ----------
    f: n-dimensional ndarray
        The reconstructed object function :math:`f(\mathbf{r})`.
    res: float
        The size of the vacuum wave length :math:`\lambda` in pixels.
    nm: float
        The refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`f(\mathbf{r})`.

    Returns
    -------
    ri: n-dimensional ndarray
        The complex refractive index :math:`n(\mathbf{r})`.

    Notes
    -----
    Because this function computes the root of a complex number, there
    are several solutions to the refractive index. Always the positive
    (real) root of the refractive index is used.
    """
    km = (2 * np.pi * nm) / res
    ri = nm * np.sqrt(f / km**2 + 1)
    # Always take the positive root as the refractive index.
    # Because f can be imaginary, numpy cannot return the correct
    # positive root of f. However, we know that *ri* must be postive and
    # thus we take the absolute value of ri.
    # This also is what happens in Slaneys
    # diffract/Src/back.c in line 414.
    negrootcoord = np.where(ri.real < 0)
    ri[negrootcoord] *= -1
    return ri


def opt_to_ri(f, res, nm):
    r"""Convert the OPT object function to refractive index

    In :abbr:`OPT (Optical Projection Tomography)`, the object function
    is computed from the raw phase data. This method converts phase data
    to refractive index data.

    .. math::

        n(\mathbf{r})  = n_\mathrm{m} +
            \frac{f(\mathbf{r}) \cdot \lambda}{2 \pi}

    Parameters
    ----------
    f: n-dimensional ndarray
        The reconstructed object function :math:`f(\mathbf{r})`.
    res: float
        The size of the vacuum wave length :math:`\lambda` in pixels.
    nm: float
        The refractive index of the medium :math:`n_\mathrm{m}` that
        surrounds the object in :math:`f(\mathbf{r})`.

    Returns
    -------
    ri: n-dimensional ndarray
        The complex refractive index :math:`n(\mathbf{r})`.

    Notes
    -----
    This function is not meant to be used with diffraction tomography
    data. For ODT, use :py:func:`odt_to_ri` instead.
    """
    ri = nm + f / (2 * np.pi) * res
    return ri
