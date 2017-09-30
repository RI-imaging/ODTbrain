#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mie off-center cylinder

The *in silico* data set was created with the
softare `miefield  <https://github.com/RI-imaging/miefield>`_.
The data are 1D projections of an off-center cylinder of constant
refractive index. The Born approximation is error-prone due to
a relatively large radius of the cylinder (30 wavelengths) and
a refractive index difference of 0.006 between cylinder and
surrounding medium. The reconstruction of the refractive index
with the Rytov approximation is in good agreement with the
input data. When only 50 projections are used for the reconstruction,
artifacts appear. These vanish when more projections are used for
the reconstruction.

The figure shows different reconstruction approaches for a
cylinder from data computed with Mie theory.
"""
from __future__ import division, print_function

import zipfile

import matplotlib.pylab as plt
import numpy as np
import unwrap

import odtbrain as odt

from example_helper import get_file

datazip = get_file("mie_2d_noncentered_cylinder_A250_R2.zip")

# Get simulation data
arc = zipfile.ZipFile(datazip)

angles = np.loadtxt(arc.open("mie_angles.txt"))

# sinogram computed with Mie theory
# miefield.GetSinogramCylinderRotation(radius, nmed, ncyl, lD, lC, size, A, res)
sino_real = np.loadtxt(arc.open("sino_real.txt"))
sino_imag = np.loadtxt(arc.open("sino_imag.txt"))
sino = sino_real + 1j * sino_imag
A, size = sino_real.shape

# background sinogram computed with Mie theory
# miefield.GetSinogramCylinderRotation(radius, nmed, nmed, lD, lC, size, A, res)
u0_real = np.loadtxt(arc.open("u0_real.txt"))
u0_imag = np.loadtxt(arc.open("u0_imag.txt"))
u0 = u0_real + 1j * u0_imag
# create 2d array
u0 = np.tile(u0, size).reshape(A, size).transpose()

# background field necessary to compute initial born field
# u0_single = mie.GetFieldCylinder(radius, nmed, nmed, lD, size, res)
u0_single_real = np.loadtxt(arc.open("u0_single_real.txt"))
u0_single_imag = np.loadtxt(arc.open("u0_single_real.txt"))
u0_single = u0_single_real + 1j * u0_single_imag

with arc.open("mie_info.txt") as info:
    cfg = {}
    for l in info.readlines():
        l = l.decode()
        if l.count("=") == 1:
            l = l.decode()
            key, val = l.split("=")
            cfg[key.strip()] = float(val.strip())

print("Example: Backpropagation from 2d Mie simulations")
print("Refractive index of medium:", cfg["nmed"])
print("Measurement position from object center:", cfg["lD"])
print("Wavelength sampling:", cfg["res"])
print("Performing backpropagation.")

# Set measurement parameters
# Compute scattered field from cylinder
radius = cfg["radius"]  # wavelengths
nmed = cfg["nmed"]
ncyl = cfg["ncyl"]

lD = cfg["lD"]  # measurement distance in wavelengths
lC = cfg["lC"]  # displacement from center of image
size = cfg["size"]
res = cfg["res"]  # px/wavelengths
A = cfg["A"]  # number of projections

#phantom = np.loadtxt(arc.open("mie_phantom.txt"))
x = np.arange(size) - size / 2
X, Y = np.meshgrid(x, x)
rad_px = radius * res
phantom = np.array(((Y - lC * res)**2 + X**2) < rad_px**2,
                   dtype=np.float) * (ncyl - nmed) + nmed

# Born
u_sinB = (sino / u0 * u0_single - u0_single)  # fake born
fB = odt.backpropagate_2d(u_sinB, angles, res, nmed, lD * res)
nB = odt.odt_to_ri(fB, res, nmed)

# Rytov
u_sinR = odt.sinogram_as_rytov(sino / u0)
fR = odt.backpropagate_2d(u_sinR, angles, res, nmed, lD * res)
nR = odt.odt_to_ri(fR, res, nmed)

# Rytov 50
u_sinR50 = odt.sinogram_as_rytov((sino / u0)[::5, :])
fR50 = odt.backpropagate_2d(u_sinR50, angles[::5], res, nmed, lD * res)
nR50 = odt.odt_to_ri(fR50, res, nmed)

# Plot sinogram phase and amplitude
ph = unwrap.unwrap(np.angle(sino / u0))

am = np.abs(sino / u0)

# prepare plot
vmin = np.min(np.array([phantom, nB.real, nR50.real, nR.real]))
vmax = np.max(np.array([phantom, nB.real, nR50.real, nR.real]))

fig, axes = plt.subplots(2, 3, figsize=(8, 5))
axes = np.array(axes).flatten()

phantommap = axes[0].imshow(phantom, vmin=vmin, vmax=vmax)
axes[0].set_title("phantom \n(non-centered cylinder)")

amplmap = axes[1].imshow(am, cmap=plt.cm.gray)  # @UndefinedVariable
axes[1].set_title("amplitude sinogram \n(background-corrected)")

phasemap = axes[2].imshow(ph, cmap=plt.cm.coolwarm)  # @UndefinedVariable
axes[2].set_title("phase sinogram [rad] \n(background-corrected)")

axes[3].imshow(nB.real, vmin=vmin, vmax=vmax)
axes[3].set_title("reconstruction (Born) \n(250 projections)")

axes[4].imshow(nR50.real, vmin=vmin, vmax=vmax)
axes[4].set_title("reconstruction (Rytov) \n(50 projections)")

axes[5].imshow(nR.real, vmin=vmin, vmax=vmax)
axes[5].set_title("reconstruction (Rytov) \n(250 projections)")

# color bars
cbkwargs = {"fraction": 0.045}
plt.colorbar(phantommap, ax=axes[0], **cbkwargs)
plt.colorbar(amplmap, ax=axes[1], **cbkwargs)
plt.colorbar(phasemap, ax=axes[2], **cbkwargs)
plt.colorbar(phantommap, ax=axes[3], **cbkwargs)
plt.colorbar(phantommap, ax=axes[4], **cbkwargs)
plt.colorbar(phantommap, ax=axes[5], **cbkwargs)

plt.tight_layout()
plt.show()
