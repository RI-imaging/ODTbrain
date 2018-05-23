import numpy as np


def compute_angle_weights_1d(angles):
    """
    Compute the weight for each angle according to the distance between its
    neighbors.
    Parameters
    ----------
    angles: 1d ndarray of length A
        Angles in radians
    Returns
    -------
    weights: 1d ndarray of length A
        The weights for each angle
    Notes
    -----
    To compute the weights, the angles are set modulo PI, not modulo 2PI.
    This reduces artifacts when the angular coverage is between PI and 2PI
    but does not affect the result when the angles cover the full 2PI interval.
    """
    # copy and modulo np.pi
    # This is an array with values in [0, np.pi)
    angles = (angles.flatten() - angles.min()) % (np.pi)
    # sort the array
    sortargs = np.argsort(angles)
    sortangl = angles[sortargs]
    # compute weights for sorted angles
    da = (np.roll(sortangl, -1) - np.roll(sortangl, 1)) % (np.pi)
    weights = da/np.sum(da)*da.shape[0]

    unsortweights = np.zeros_like(weights)
    # Sort everything back where it belongs
    unsortweights[sortargs] = weights
    return unsortweights
