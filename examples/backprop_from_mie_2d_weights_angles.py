"""Mie cylinder with unevenly spaced angles

Angular weighting can significantly improve reconstruction quality
when the angular projections are sampled at non-equidistant
intervals :cite:`Tam1981`. The *in silico* data set was created with
the softare `miefield  <https://github.com/RI-imaging/miefield>`_.
The data are 1D projections of a non-centered cylinder of constant
refractive index 1.339 embedded in water with refractive index 1.333.
The first column shows the used sinograms (missing angles are displayed
as zeros) that were created from the original sinogram with 250
projections. The second column shows the reconstruction without angular
weights and the third column shows the reconstruction with angular
weights. The keyword argument `weight_angles` was introduced in version
0.1.1.
"""
import matplotlib.pylab as plt
import numpy as np
import unwrap

import odtbrain as odt

from example_helper import load_data


sino, angles, cfg = load_data("mie_2d_noncentered_cylinder_A250_R2.zip",
                              f_angles="mie_angles.txt",
                              f_sino_real="sino_real.txt",
                              f_sino_imag="sino_imag.txt",
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


print("Example: Backpropagation from 2D FDTD simulations")
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

x = np.arange(size) - size / 2.0
X, Y = np.meshgrid(x, x)
rad_px = radius * res
phantom = np.array(((Y - lC * res)**2 + X**2) < rad_px **
                   2, dtype=np.float) * (ncyl - nmed) + nmed

u_sinR = odt.sinogram_as_rytov(sino / u0)

# Rytov 200 projections
# remove 50 projections from total of 250 projections
remove200 = np.argsort(angles % .0002)[:50]
angles200 = np.delete(angles, remove200, axis=0)
u_sinR200 = np.delete(u_sinR, remove200, axis=0)
ph200 = unwrap.unwrap(np.angle(sino / u0))
ph200[remove200] = 0

fR200 = odt.backpropagate_2d(u_sinR200, angles200, res, nmed, lD*res)
nR200 = odt.odt_to_ri(fR200, res, nmed)
fR200nw = odt.backpropagate_2d(u_sinR200, angles200, res, nmed, lD*res,
                               weight_angles=False)
nR200nw = odt.odt_to_ri(fR200nw, res, nmed)

# Rytov 50 projections
remove50 = np.argsort(angles % .0002)[:200]
angles50 = np.delete(angles, remove50, axis=0)
u_sinR50 = np.delete(u_sinR, remove50, axis=0)
ph50 = unwrap.unwrap(np.angle(sino / u0))
ph50[remove50] = 0

fR50 = odt.backpropagate_2d(u_sinR50, angles50, res, nmed, lD*res)
nR50 = odt.odt_to_ri(fR50, res, nmed)
fR50nw = odt.backpropagate_2d(u_sinR50, angles50, res, nmed, lD*res,
                              weight_angles=False)
nR50nw = odt.odt_to_ri(fR50nw, res, nmed)

# prepare plot
kw_ri = {"vmin": 1.330,
         "vmax": 1.340}

kw_ph = {"vmin": np.min(np.array([ph200, ph50])),
         "vmax": np.max(np.array([ph200, ph50])),
         "cmap": "coolwarm"}

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
axes = np.array(axes).flatten()

phmap = axes[0].imshow(ph200, **kw_ph)
axes[0].set_title("Phase sinogram (200 proj.)")

rimap = axes[1].imshow(nR200nw.real, **kw_ri)
axes[1].set_title("RI without angular weights")

axes[2].imshow(nR200.real, **kw_ri)
axes[2].set_title("RI with angular weights")

axes[3].imshow(ph50, **kw_ph)
axes[3].set_title("Phase sinogram (50 proj.)")

axes[4].imshow(nR50nw.real, **kw_ri)
axes[4].set_title("RI without angular weights")

axes[5].imshow(nR50.real, **kw_ri)
axes[5].set_title("RI with angular weights")

# color bars
cbkwargs = {"fraction": 0.045,
            "format": "%.3f"}
plt.colorbar(phmap, ax=axes[0], **cbkwargs)
plt.colorbar(phmap, ax=axes[3], **cbkwargs)
plt.colorbar(rimap, ax=axes[1], **cbkwargs)
plt.colorbar(rimap, ax=axes[2], **cbkwargs)
plt.colorbar(rimap, ax=axes[5], **cbkwargs)
plt.colorbar(rimap, ax=axes[4], **cbkwargs)

plt.tight_layout()
plt.show()
