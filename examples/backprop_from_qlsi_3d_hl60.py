"""HL60 cell

The quantitative phase data of an HL60 S/4 cell were recorded using
:abbr:`QLSI (quadri-wave lateral shearing interferometry)`.
The original dataset was used in a previous publication
:cite:`Schuermann2017` to illustrate the capabilities of combined
fluorescence and refractive index tomography.

The example data set is already aligned and background-corrected as
described in the original publication and the fluorescence data are
not included. The lzma-archive contains the sinogram data stored in
the :ref:`qpimage <qpimage:index>` file format and the rotational
positions of each sinogram image as a text file.

The figure reproduces parts of figure 4 of the original manuscript.
Note that minor deviations from the original figure can be attributed
to the strong compression (scale offset filter) and due to the fact
that the original sinogram images were cropped from 196x196 px to
140x140 px (which in particular affects the background-part of the
refractive index histogram).

The raw data is available
`on figshare <https://doi.org/10.6084/m9.figshare.8055407.v1>`
(hl60_sinogram_qpi.h5).
"""
import pathlib
import tarfile
import tempfile

import matplotlib.pylab as plt
import numpy as np
import odtbrain as odt
import qpimage

from example_helper import get_file, extract_lzma


# ascertain the data
path = get_file("qlsi_3d_hl60-cell_A140.tar.lzma")
tarf = extract_lzma(path)
tdir = tempfile.mkdtemp(prefix="odtbrain_example_")

with tarfile.open(tarf) as tf:
    tf.extract("series.h5", path=tdir)
    angles = np.loadtxt(tf.extractfile("angles.txt"))

# extract the complex field sinogram from the qpimage series data
h5file = pathlib.Path(tdir) / "series.h5"
with qpimage.QPSeries(h5file=h5file, h5mode="r") as qps:
    qp0 = qps[0]
    meta = qp0.meta
    sino = np.zeros((len(qps), qp0.shape[0], qp0.shape[1]), dtype=np.complex)
    for ii in range(len(qps)):
        sino[ii] = qps[ii].field

# perform backgpropagation
u_sinR = odt.sinogram_as_rytov(sino)
res = meta["wavelength"] / meta["pixel size"]
nm = meta["medium index"]

fR = odt.backpropagate_3d(uSin=u_sinR,
                          angles=angles,
                          res=res,
                          nm=nm)

ri = odt.odt_to_ri(fR, res, nm)

# plot results
ext = meta["pixel size"] * 1e6 * 70
kw = {"vmin": ri.real.min(),
      "vmax": ri.real.max(),
      "extent": [-ext, ext, -ext, ext]}
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
axes[0].imshow(ri[70, :, :].real, **kw)
axes[0].set_xlabel("x [µm]")
axes[0].set_ylabel("y [µm]")

x = np.linspace(-ext, ext, 140)
axes[1].plot(x, ri[70, :, 70], label="line plot x=0")
axes[1].plot(x, ri[70, 70, :], label="line plot y=0")
axes[1].set_xlabel("distance from center [µm]")
axes[1].set_ylabel("refractive index")
axes[1].legend()


hist, xh = np.histogram(ri.real, bins=100)
axes[2].plot(xh[1:], hist)
axes[2].set_yscale('log')
axes[2].set_xlabel("refractive index")
axes[2].set_ylabel("histogram")

plt.tight_layout()
plt.show()
