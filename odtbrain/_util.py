#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for ODTbrain
"""
from __future__ import division, print_function

import numpy as np


def compute_angle_weights_1d(angles):
    """
    Compute the weight for each angle according to the distance between its
    neighbors.
    
    Parameters
    ----------
    angles : 1d ndarray of length A
        Angles in radians
        
    Returns
    -------
    weights : 1d ndarray of length A
        The weights for each angle
    """
    # copy and modulo 2*np.pi
    # This is an array with values in [0, 2*np.pi)
    angles = angles.flatten() % (np.pi) 
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