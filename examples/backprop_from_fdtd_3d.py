"""FDTD cell phantom
The *in silico* data set was created with the
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 2D projections of a 3D refractive index phantom. The reconstruction
of the refractive index with the Rytov approximation is in good
agreement with the phantom that was used in the simulation. The data
are downsampled by a factor of two. The rotational axis is the `y`-axis.
A total of 180 projections are used for the reconstruction. A detailed
description of this phantom is given in :cite:`Mueller2015`.

.. _`meep`: http://ab-initio.mit.edu/wiki/index.php/Meep
"""
import matplotlib.pylab as plt
import numpy as np

import odtbrain as odt

from example_helper import load_data


sino, angles, phantom, cfg = \
    load_data("fdtd_3d_sino_A180_R6.500.tar.lzma")

A = angles.shape[0]

print("Example: Backpropagation from 3D FDTD simulations")
print("Refractive index of medium:", cfg["nm"])
print("Measurement position from object center:", cfg["lD"])
print("Wavelength sampling:", cfg["res"])
print("Number of projections:", A)
print("Performing backpropagation.")

# Apply the Rytov approximation
sinoRytov = odt.sinogram_as_rytov(sino)

# perform backpropagation to obtain object function f
f = odt.backpropagate_3d(uSin=sinoRytov,
                         angles=angles,
                         res=cfg["res"],
                         nm=cfg["nm"],
                         lD=cfg["lD"]
                         )

# compute refractive index n from object function
n = odt.odt_to_ri(f, res=cfg["res"], nm=cfg["nm"])

sx, sy, sz = n.shape
px, py, pz = phantom.shape

sino_phase = np.angle(sino)

# compare phantom and reconstruction in plot
fig, axes = plt.subplots(2, 3, figsize=(8, 4))
kwri = {"vmin": n.real.min(), "vmax": n.real.max()}
kwph = {"vmin": sino_phase.min(), "vmax": sino_phase.max(),
        "cmap": "coolwarm"}

# Phantom
axes[0, 0].set_title("FDTD phantom center")
rimap = axes[0, 0].imshow(phantom[px // 2], **kwri)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")

axes[1, 0].set_title("FDTD phantom nucleolus")
axes[1, 0].imshow(phantom[int(px / 2 + 2 * cfg["res"])], **kwri)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")

# Sinogram
axes[0, 1].set_title("phase projection")
phmap = axes[0, 1].imshow(sino_phase[A // 2, :, :], **kwph)
axes[0, 1].set_xlabel("detector x")
axes[0, 1].set_ylabel("detector y")

axes[1, 1].set_title("sinogram slice")
axes[1, 1].imshow(sino_phase[:, :, sino.shape[2] // 2],
                  aspect=sino.shape[1] / sino.shape[0], **kwph)
axes[1, 1].set_xlabel("detector y")
axes[1, 1].set_ylabel("angle [rad]")
# set y ticks for sinogram
labels = np.linspace(0, 2 * np.pi, len(axes[1, 1].get_yticks()))
labels = ["{:.2f}".format(i) for i in labels]
axes[1, 1].set_yticks(np.linspace(0, len(angles), len(labels)))
axes[1, 1].set_yticklabels(labels)

axes[0, 2].set_title("reconstruction center")
axes[0, 2].imshow(n[sx // 2].real, **kwri)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

axes[1, 2].set_title("reconstruction nucleolus")
axes[1, 2].imshow(n[int(sx / 2 + 2 * cfg["res"])].real, **kwri)
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")

# color bars
cbkwargs = {"fraction": 0.045,
            "format": "%.3f"}
plt.colorbar(phmap, ax=axes[0, 1], **cbkwargs)
plt.colorbar(phmap, ax=axes[1, 1], **cbkwargs)
plt.colorbar(rimap, ax=axes[0, 0], **cbkwargs)
plt.colorbar(rimap, ax=axes[1, 0], **cbkwargs)
plt.colorbar(rimap, ax=axes[0, 2], **cbkwargs)
plt.colorbar(rimap, ax=axes[1, 2], **cbkwargs)

plt.tight_layout()
plt.show()
