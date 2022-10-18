import numpy as np


def compute_angle_weights_1d(angles: np.ndarray) -> np.ndarray:
    """Compute angular weights for tomographic reconstruction

    This method aims to address the issue of unevenly distributed angles
    in tomographic reconstruction. For algorithms, such as filtered
    backprojection, weighting each backprojection with a factor
    proportional to the angular distance to its neighbors dramatically
    improves the reconstructed image.

    Weights computation also takes into account these special cases:

    - Same angle present multiple times
    - Angular coverage larger than 180° (PI)

    Parameters
    ----------
    angles: 1d ndarray of length A
        Angles corresponding to the projections [rad]

    Returns
    -------
    weights: 1d ndarray of length A
        Weights for each angle; the mean of `weights` is one.

    Notes
    -----
    If one angle is passed multiple times `N` (e.g. `N=2`, 0° and 180°
    via `np.linspace(0, np.pi, 10, endpoint=True)`), then this angle will
    have a weight smaller than the other angles by a factor of `1/N`.

    This method is dupicated in the :ref:`radontea <radontea:index>`
    package. Even though, for ODT you normally need a coverage of 2 PI
    (instead of one PI in OPT), it makes sense here to wrap the coverage
    at PI. The idea is that opposite projections contribute similarly to
    the reconstruction and can be treated as PI-wrapped.
    """
    # copy and modulo np.pi
    # This is an array with values in [0, np.pi)
    angles = (angles.flatten() - np.min(angles)) % np.pi

    # If we have duplicate entries, we need to take them into account
    unq_angles, unq_reverse, unq_counts = np.unique(
        angles, return_inverse=True, return_counts=True)

    # sort the array
    srt_idx = np.argsort(unq_angles)
    srt_angles = unq_angles[srt_idx]
    srt_count = unq_counts[srt_idx]

    # compute weights for sorted angles
    da = (np.roll(srt_angles, -1) - np.roll(srt_angles, 1)) % np.pi
    sum_da = np.sum(da)

    # normalize with number of occurrences
    da /= srt_count
    srt_weights = da / sum_da * angles.size

    # Sort everything back where it belongs
    unq_weights = np.zeros_like(srt_weights)
    unq_weights[srt_idx] = srt_weights

    # Set the weights for each item in the original angles
    weights = unq_weights[unq_reverse]
    return weights
