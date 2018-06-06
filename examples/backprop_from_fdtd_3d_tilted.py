"""FDTD cell phantom with tilted axis of rotation

The *in silico* data set was created with the
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 2D projections of a 3D refractive index phantom that is rotated
about an axis which is tilted by 0.2 rad (11.5 degrees) with respect
to the imaging plane. The example showcases the method
:func:`odtbrain.backpropagate_3d_tilted` which takes into account
such a tilted axis of rotation. The data are downsampled by a factor
of two. A total of 220 projections are used for the reconstruction.
Note that the information required for reconstruction decreases as the
tilt angle increases. If the tilt angle is 90 degrees w.r.t. the
imaging plane, then we get a rotating image of a cell (not images of a
rotating cell) and tomographic reconstruction is impossible. A brief
description of this algorithm is given in :cite:`Mueller2015tilted`.


The first column shows the measured phase, visualizing the
tilt (compare to other examples). The second column shows a
reconstruction that does not take into account the tilted axis of
rotation; the result is a blurry reconstruction. The third column
shows the improved reconstruction; the known tilted axis of rotation
is used in the reconstruction process.

.. _`meep`: http://ab-initio.mit.edu/wiki/index.php/Meep
"""
import matplotlib.pylab as plt
import numpy as np

import odtbrain as odt

from example_helper import load_data

sino, angles, phantom, cfg = \
    load_data("fdtd_3d_sino_A220_R6.500_tiltyz0.2.tar.lzma")

A = angles.shape[0]

print("Example: Backpropagation from 3D FDTD simulations")
print("Refractive index of medium:", cfg["nm"])
print("Measurement position from object center:", cfg["lD"])
print("Wavelength sampling:", cfg["res"])
print("Axis tilt in y-z direction:", cfg["tilt_yz"])
print("Number of projections:", A)

print("Performing normal backpropagation.")
# Apply the Rytov approximation
sinoRytov = odt.sinogram_as_rytov(sino)

# Perform naive backpropagation
f_naiv = odt.backpropagate_3d(uSin=sinoRytov,
                              angles=angles,
                              res=cfg["res"],
                              nm=cfg["nm"],
                              lD=cfg["lD"]
                              )

print("Performing tilted backpropagation.")
# Determine tilted axis
tilted_axis = [0, np.cos(cfg["tilt_yz"]), np.sin(cfg["tilt_yz"])]

# Perform tilted backpropagation
f_tilt = odt.backpropagate_3d_tilted(uSin=sinoRytov,
                                     angles=angles,
                                     res=cfg["res"],
                                     nm=cfg["nm"],
                                     lD=cfg["lD"],
                                     tilted_axis=tilted_axis,
                                     )

# compute refractive index n from object function
n_naiv = odt.odt_to_ri(f_naiv, res=cfg["res"], nm=cfg["nm"])
n_tilt = odt.odt_to_ri(f_tilt, res=cfg["res"], nm=cfg["nm"])

sx, sy, sz = n_tilt.shape
px, py, pz = phantom.shape

sino_phase = np.angle(sino)

# compare phantom and reconstruction in plot
fig, axes = plt.subplots(2, 3, figsize=(8, 4.5))
kwri = {"vmin": n_tilt.real.min(), "vmax": n_tilt.real.max()}
kwph = {"vmin": sino_phase.min(), "vmax": sino_phase.max(),
        "cmap": "coolwarm"}

# Sinogram
axes[0, 0].set_title("phase projection")
phmap = axes[0, 0].imshow(sino_phase[A // 2, :, :], **kwph)
axes[0, 0].set_xlabel("detector x")
axes[0, 0].set_ylabel("detector y")

axes[1, 0].set_title("sinogram slice")
axes[1, 0].imshow(sino_phase[:, :, sino.shape[2] // 2],
                  aspect=sino.shape[1] / sino.shape[0], **kwph)
axes[1, 0].set_xlabel("detector y")
axes[1, 0].set_ylabel("angle [rad]")
# set y ticks for sinogram
labels = np.linspace(0, 2 * np.pi, len(axes[1, 1].get_yticks()))
labels = ["{:.2f}".format(i) for i in labels]
axes[1, 0].set_yticks(np.linspace(0, len(angles), len(labels)))
axes[1, 0].set_yticklabels(labels)

axes[0, 1].set_title("normal (center)")
rimap = axes[0, 1].imshow(n_naiv[sx // 2].real, **kwri)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")

axes[1, 1].set_title("normal (nucleolus)")
axes[1, 1].imshow(n_naiv[int(sx / 2 + 2 * cfg["res"])].real, **kwri)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

axes[0, 2].set_title("tilt correction (center)")
axes[0, 2].imshow(n_tilt[sx // 2].real, **kwri)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

axes[1, 2].set_title("tilt correction (nucleolus)")
axes[1, 2].imshow(n_tilt[int(sx / 2 + 2 * cfg["res"])].real, **kwri)
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")

# color bars
cbkwargs = {"fraction": 0.045,
            "format": "%.3f"}
plt.colorbar(phmap, ax=axes[0, 0], **cbkwargs)
plt.colorbar(phmap, ax=axes[1, 0], **cbkwargs)
plt.colorbar(rimap, ax=axes[0, 1], **cbkwargs)
plt.colorbar(rimap, ax=axes[1, 1], **cbkwargs)
plt.colorbar(rimap, ax=axes[0, 2], **cbkwargs)
plt.colorbar(rimap, ax=axes[1, 2], **cbkwargs)

plt.tight_layout()
plt.show()
