"""Missing apple core correction

The missing apple core :cite:`Vertu2009` is a phenomenon in diffraction
tomography that is a result of the fact the the Fourier space is not
filled completely when the sample is rotated only about a single axis.
The resulting artifacts include ringing and blurring in the
reconstruction parallel to the original rotation axis. By enforcing
constraints (refractive index real-valued and larger than the
surrounding medium), these artifacts can be attenuated.

This example generates an artificial sinogram using the Python
library :ref:`cellsino <cellsino:index>` (The example parameters
are reused from :ref:`this example <cellsino:example_simple_cell>`).
The sinogram is then reconstructed with the backpropagation algorithm
and the missing apple core correction is applied.

.. note::
    The missing apple core correction :func:`odtbrain.apple.correct`
    was implemented in version 0.3.0 and is thus not used in the
    older examples.
"""
import matplotlib.pylab as plt
import numpy as np

import cellsino
import odtbrain as odt

# number of sinogram angles
num_ang = 160
# sinogram acquisition angles
angles = np.linspace(0, 2*np.pi, num_ang, endpoint=False)
# detector grid size
grid_size = (250, 250)
# vacuum wavelength [m]
wavelength = 550e-9
# pixel size [m]
pixel_size = 0.08e-6
# refractive index of the surrounding medium
medium_index = 1.335

# initialize cell phantom
phantom = cellsino.phantoms.SimpleCell()

# initialize sinogram with geometric parameters
sino = cellsino.Sinogram(phantom=phantom,
                         wavelength=wavelength,
                         pixel_size=pixel_size,
                         grid_size=grid_size)

# compute sinogram (field according to Rytov approximation and fluorescence)
sino = sino.compute(angles=angles, propagator="rytov", mode="field")

# reconstruction of refractive index
sino_rytov = odt.sinogram_as_rytov(sino)
f = odt.backpropagate_3d(uSin=sino_rytov,
                         angles=angles,
                         res=wavelength/pixel_size,
                         nm=medium_index)
ri = odt.odt_to_ri(f=f,
                   res=wavelength/pixel_size,
                   nm=medium_index)

# apple core correction
ric = odt.apple.correct(ri=ri,
                        res=wavelength/pixel_size,
                        nm=medium_index)

# plotting
idx = ri.shape[2] // 2

# log-scaled power spectra
ft = np.log(1 + np.abs(np.fft.fftshift(np.fft.fftn(ri))))
ftc = np.log(1 + np.abs(np.fft.fftshift(np.fft.fftn(ric))))

plt.figure(figsize=(7, 5.5))

plotkwri = {"vmax": ri.real.max(),
            "vmin": ri.real.min(),
            "interpolation": "none",
            }

plotkwft = {"vmax": ft.max(),
            "vmin": 0,
            "interpolation": "none",
            }

ax1 = plt.subplot(221, title="plain refractive index")
mapper = ax1.imshow(ri[:, :, idx].real, **plotkwri)
plt.colorbar(mappable=mapper, ax=ax1)

ax2 = plt.subplot(222, title="corrected refractive index")
mapper = ax2.imshow(ric[:, :, idx].real, **plotkwri)
plt.colorbar(mappable=mapper, ax=ax2)

ax3 = plt.subplot(223, title="Fourier space (visible apple core)")
mapper = ax3.imshow(ft[:, :, idx], **plotkwft)
plt.colorbar(mappable=mapper, ax=ax3)

ax4 = plt.subplot(224, title="Fourier space (with correction)")
mapper = ax4.imshow(ftc[:, :, idx], **plotkwft)
plt.colorbar(mappable=mapper, ax=ax4)

plt.tight_layout()
plt.show()
