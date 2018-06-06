"""FDTD cell phantom

The *in silico* data set was created with the
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 1D projections of a 2D refractive index phantom. The
reconstruction of the refractive index with the Rytov approximation
is in good agreement with the phantom that was used in the
simulation.

.. _`meep`: http://ab-initio.mit.edu/wiki/index.php/Meep
"""
import matplotlib.pylab as plt
import numpy as np
import odtbrain as odt

from example_helper import load_data


sino, angles, phantom, cfg = load_data("fdtd_2d_sino_A100_R13.zip",
                                       f_angles="fdtd_angles.txt",
                                       f_sino_imag="fdtd_imag.txt",
                                       f_sino_real="fdtd_real.txt",
                                       f_info="fdtd_info.txt",
                                       f_phantom="fdtd_phantom.txt",
                                       )

print("Example: Backpropagation from 2D FDTD simulations")
print("Refractive index of medium:", cfg["nm"])
print("Measurement position from object center:", cfg["lD"])
print("Wavelength sampling:", cfg["res"])
print("Performing backpropagation.")

# Apply the Rytov approximation
sino_rytov = odt.sinogram_as_rytov(sino)

# perform backpropagation to obtain object function f
f = odt.backpropagate_2d(uSin=sino_rytov,
                         angles=angles,
                         res=cfg["res"],
                         nm=cfg["nm"],
                         lD=cfg["lD"] * cfg["res"]
                         )

# compute refractive index n from object function
n = odt.odt_to_ri(f, res=cfg["res"], nm=cfg["nm"])

# compare phantom and reconstruction in plot
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

axes[0].set_title("FDTD phantom")
axes[0].imshow(phantom, vmin=phantom.min(), vmax=phantom.max())
sino_phase = np.unwrap(np.angle(sino), axis=1)

axes[1].set_title("phase sinogram")
axes[1].imshow(sino_phase, vmin=sino_phase.min(), vmax=sino_phase.max(),
               aspect=sino.shape[1] / sino.shape[0],
               cmap="coolwarm")
axes[1].set_xlabel("detector")
axes[1].set_ylabel("angle [rad]")

axes[2].set_title("reconstructed image")
axes[2].imshow(n.real, vmin=phantom.min(), vmax=phantom.max())

# set y ticks for sinogram
labels = np.linspace(0, 2 * np.pi, len(axes[1].get_yticks()))
labels = ["{:.2f}".format(i) for i in labels]
axes[1].set_yticks(np.linspace(0, len(angles), len(labels)))
axes[1].set_yticklabels(labels)

plt.tight_layout()
plt.show()
