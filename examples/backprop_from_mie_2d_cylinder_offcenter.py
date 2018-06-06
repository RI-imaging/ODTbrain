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
"""
import matplotlib.pylab as plt
import numpy as np

import odtbrain as odt

from example_helper import load_data


# simulation data
sino, angles, cfg = load_data("mie_2d_noncentered_cylinder_A250_R2.zip",
                              f_sino_imag="sino_imag.txt",
                              f_sino_real="sino_real.txt",
                              f_angles="mie_angles.txt",
                              f_info="mie_info.txt")
A, size = sino.shape

# background sinogram computed with Mie theory
# miefield.GetSinogramCylinderRotation(radius, nmed, nmed, lD, lC, size, A,res)
u0 = load_data("mie_2d_noncentered_cylinder_A250_R2.zip",
               f_sino_imag="u0_imag.txt",
               f_sino_real="u0_real.txt")
# create 2d array
u0 = np.tile(u0, size).reshape(A, size).transpose()

# background field necessary to compute initial born field
# u0_single = mie.GetFieldCylinder(radius, nmed, nmed, lD, size, res)
u0_single = load_data("mie_2d_noncentered_cylinder_A250_R2.zip",
                      f_sino_imag="u0_single_imag.txt",
                      f_sino_real="u0_single_real.txt")

print("Example: Backpropagation from 2D Mie simulations")
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
ph = odt.sinogram_as_radon(sino / u0)

am = np.abs(sino / u0)

# prepare plot
vmin = np.min(np.array([phantom, nB.real, nR50.real, nR.real]))
vmax = np.max(np.array([phantom, nB.real, nR50.real, nR.real]))

fig, axes = plt.subplots(2, 3, figsize=(8, 5))
axes = np.array(axes).flatten()

phantommap = axes[0].imshow(phantom, vmin=vmin, vmax=vmax)
axes[0].set_title("phantom \n(non-centered cylinder)")

amplmap = axes[1].imshow(am, cmap="gray")
axes[1].set_title("amplitude sinogram \n(background-corrected)")

phasemap = axes[2].imshow(ph, cmap="coolwarm")
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
