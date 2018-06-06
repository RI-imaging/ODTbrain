"""FDTD cell phantom with tilted and rolled axis of rotation

The *in silico* data set was created with the
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 2D projections of a 3D refractive index phantom that is rotated
about an axis which is tilted by 0.2 rad (11.5 degrees) with respect to
the imaging plane and rolled by -.42 rad (-24.1 degrees) within the
imaging plane. The data are the same as were used in the previous
example. A brief description of this algorithm is given in
:cite:`Mueller2015tilted`.

.. _`meep`: http://ab-initio.mit.edu/wiki/index.php/Meep
"""
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import rotate

import odtbrain as odt

from example_helper import load_data


sino, angles, phantom, cfg = \
    load_data("fdtd_3d_sino_A220_R6.500_tiltyz0.2.tar.lzma")

# Perform titlt by -.42 rad in detector plane
rotang = -0.42
rotkwargs = {"mode": "constant",
             "order": 2,
             "reshape": False,
             }
for ii in range(len(sino)):
    sino[ii].real = rotate(
        sino[ii].real, np.rad2deg(rotang), cval=1, **rotkwargs)
    sino[ii].imag = rotate(
        sino[ii].imag, np.rad2deg(rotang), cval=0, **rotkwargs)

A = angles.shape[0]

print("Example: Backpropagation from 3D FDTD simulations")
print("Refractive index of medium:", cfg["nm"])
print("Measurement position from object center:", cfg["lD"])
print("Wavelength sampling:", cfg["res"])
print("Axis tilt in y-z direction:", cfg["tilt_yz"])
print("Number of projections:", A)

# Apply the Rytov approximation
sinoRytov = odt.sinogram_as_rytov(sino)

# Determine tilted axis
tilted_axis = [0, np.cos(cfg["tilt_yz"]), np.sin(cfg["tilt_yz"])]
rotmat = np.array([
    [np.cos(rotang), -np.sin(rotang), 0],
    [np.sin(rotang), np.cos(rotang), 0],
    [0, 0, 1],
])
tilted_axis = np.dot(rotmat, tilted_axis)

print("Performing tilted backpropagation.")
# Perform tilted backpropagation
f_tilt = odt.backpropagate_3d_tilted(uSin=sinoRytov,
                                     angles=angles,
                                     res=cfg["res"],
                                     nm=cfg["nm"],
                                     lD=cfg["lD"],
                                     tilted_axis=tilted_axis,
                                     )

# compute refractive index n from object function
n_tilt = odt.odt_to_ri(f_tilt, res=cfg["res"], nm=cfg["nm"])

sx, sy, sz = n_tilt.shape
px, py, pz = phantom.shape

sino_phase = np.angle(sino)

# compare phantom and reconstruction in plot
fig, axes = plt.subplots(1, 3, figsize=(8, 2.4))
kwri = {"vmin": n_tilt.real.min(), "vmax": n_tilt.real.max()}
kwph = {"vmin": sino_phase.min(), "vmax": sino_phase.max(),
        "cmap": "coolwarm"}

# Sinogram
axes[0].set_title("phase projection")
phmap = axes[0].imshow(sino_phase[A // 2, :, :], **kwph)
axes[0].set_xlabel("detector x")
axes[0].set_ylabel("detector y")

axes[1].set_title("sinogram slice")
axes[1].imshow(sino_phase[:, :, sino.shape[2] // 2],
               aspect=sino.shape[1] / sino.shape[0], **kwph)
axes[1].set_xlabel("detector y")
axes[1].set_ylabel("angle [rad]")
# set y ticks for sinogram
labels = np.linspace(0, 2 * np.pi, len(axes[1].get_yticks()))
labels = ["{:.2f}".format(i) for i in labels]
axes[1].set_yticks(np.linspace(0, len(angles), len(labels)))
axes[1].set_yticklabels(labels)

axes[2].set_title("tilt correction (nucleolus)")
rimap = axes[2].imshow(n_tilt[int(sx / 2 + 2 * cfg["res"])].real, **kwri)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

# color bars
cbkwargs = {"fraction": 0.045,
            "format": "%.3f"}
plt.colorbar(phmap, ax=axes[0], **cbkwargs)
plt.colorbar(phmap, ax=axes[1], **cbkwargs)
plt.colorbar(rimap, ax=axes[2], **cbkwargs)

plt.tight_layout()
plt.show()
