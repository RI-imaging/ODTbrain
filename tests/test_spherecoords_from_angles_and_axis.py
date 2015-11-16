#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests backpropagation algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, basename, dirname, join, split, exists
import platform
import sys
import warnings
import zipfile

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain
import odtbrain._Back_3D_tilted



if __name__ == "__main__":
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    axes = [[0,1,0], [0,1,0.1], [0,1,-1]]
    colors = ["k", "blue", "red"]
    angles = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')    
    for i in range(len(axes)):
        tilted_axis = axes[i]
        color = colors[i]
        tilted_axis = np.array(tilted_axis)
        tilted_axis = tilted_axis/np.sqrt(np.sum(tilted_axis**2)) 
        
        angle_coords = odtbrain._Back_3D_tilted.sphere_points_from_angles_and_tilt(angles, tilted_axis)
        
        u,v,w = tilted_axis
        a = Arrow3D([0,u],[0,v],[0,w], mutation_scale=20, lw=1, arrowstyle="-|>", color=color)
        ax.add_artist(a)
        ax.scatter(angle_coords[:,0], angle_coords[:,1], angle_coords[:,2], c=color, marker='o')
    
    radius=1
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-radius*1.5, radius*1.5)
    ax.set_ylim(-radius*1.5, radius*1.5)
    ax.set_zlim(-radius*1.5, radius*1.5)
    plt.tight_layout()

    plt.show()