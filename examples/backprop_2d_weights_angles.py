#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Unevenly spaced angles (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Angular weighting can significantly improve reconstruction quality
when the angular projections are sampled at non-equidistant
intervals [6]_. The *in silico* data set was created with the 
softare `miefield  <https://github.com/RI-imaging/miefield>`_.
The data are 1D projections of a non-centered cylinder of constant
refractive index 1.339 embedded in water with refractive index 1.333. 
The first column shows the used sinograms (missing angles are displayed
as zeros) that were created from the original sinogram with 250
projections. The second column shows the reconstruction without angular
weights and the third column shows the reconstruction with angular
weights. The keyword argument `weight_angle` was introduced in version
0.1.1. 

.. figure::  ../examples/backprop_2d_weights_angles_repo.png
   :align:   center

   Impact of angular weighting on backpropagation with the Rytov
   approximation. 

Download the :download:`full example <../examples/backprop_2d_weights_angles.py>`.
If you are not running the example from the git repository, make sure the
file :download:`example_helper.py <../examples/example_helper.py>` is present
in the current directory.

"""
from __future__ import division, print_function

# All imports are moved to "__main__", because
# sphinx might complain about some imports (e.g. mpl).

if __name__ == "__main__":
    try:
        from example_helper import get_file
    except ImportError:
        print("Please make sure example_helper.py is available.")
        raise
    import matplotlib.pylab as plt
    import numpy as np
    from os.path import abspath, dirname, split, join
    import sys
    import zipfile
    import unwrap
    
    # Add parent directory to beginning of path variable
    DIR = dirname(abspath(__file__))
    sys.path.insert(0, split(DIR)[0])
    
    import odtbrain as odt
    
    # use jobmanager if available
    try:
        import jobmanager as jm
        jm.decorators.decorate_module_ProgressBar(odt, 
                            decorator=jm.decorators.ProgressBarOverrideCount,
                            interval=.1)
    except:
        pass

    datazip = get_file("mie_2d_noncentered_cylinder_A250_R2.zip")
    
    # Get simulation data
    arc = zipfile.ZipFile(datazip)

    angles = np.loadtxt(arc.open("mie_angles.txt"))

    # sinogram computed with mie
    # computed with
    # miefield.GetSinogramCylinderRotation(radius, nmed, ncyl, lD, lC, size, A, res)
    sino_real = np.loadtxt(arc.open("sino_real.txt"))
    sino_imag = np.loadtxt(arc.open("sino_imag.txt"))
    sino = sino_real + 1j*sino_imag
    A, size = sino_real.shape

    # background sinogram computed with mie
    # computed with
    # miefield.GetSinogramCylinderRotation(radius, nmed, nmed, lD, lC, size, A, res)
    u0_real = np.loadtxt(arc.open("u0_real.txt"))
    u0_imag = np.loadtxt(arc.open("u0_imag.txt"))
    u0 = u0_real + 1j*u0_imag
    # create 2d array
    u0 = np.tile(u0, size).reshape(A,size).transpose()

    # background field necessary to compute initial born field
    # computed with
    # u0_single = mie.GetFieldCylinder(radius, nmed, nmed, lD, size, res)
    u0_single_real = np.loadtxt(arc.open("u0_single_real.txt"))
    u0_single_imag = np.loadtxt(arc.open("u0_single_real.txt"))
    u0_single = u0_single_real + 1j*u0_single_imag

    with arc.open("mie_info.txt") as info:
        cfg = {}
        for l in info.readlines():
            l = l.decode()
            if l.count("=") == 1:
                key, val = l.split("=")
                cfg[key.strip()] = float(val.strip())

    print("Example: Backpropagation from 2d FDTD simulations")
    print("Refractive index of medium:", cfg["nmed"])
    print("Measurement position from object center:", cfg["lD"])
    print("Wavelength sampling:", cfg["res"])
    print("Performing backpropagation.")

    
    # Set measurement parameters
    # Compute scattered field from cylinder
    radius = cfg["radius"] # wavelengths
    nmed = cfg["nmed"]
    ncyl = cfg["ncyl"]
    
    lD = cfg["lD"] # measurement distance in wavelengths
    lC = cfg["lC"] # displacement from center of image
    size = cfg["size"]
    res = cfg["res"] # px/wavelengths
    A = cfg["A"] # number of projections

    #phantom = np.loadtxt(arc.open("mie_phantom.txt"))
    x = np.arange(size)-size/2.0
    X,Y = np.meshgrid(x,x)
    rad_px = radius*res
    phantom = np.array(((Y-lC*res)**2+X**2)<rad_px**2, dtype=np.float)*(ncyl-nmed)+nmed

    u_sinR = odt.sinogram_as_rytov(sino/u0)
    
    # Rytov 200 projections    
    # remove 50 projections from total of 250 projections
    remove200 = np.argsort(angles % .002)[:50]
    angles200 = np.delete(angles, remove200, axis=0)
    u_sinR200 = np.delete(u_sinR, remove200, axis=0)
    ph200 = unwrap.unwrap(np.angle(sino/u0))
    ph200[remove200] = 0
    
    fR200 = odt.backpropagate_2d(u_sinR200, angles200, res, nmed, lD*res)
    nR200 = odt.odt_to_ri(fR200, res, nmed)
    fR200nw = odt.backpropagate_2d(u_sinR200, angles200, res, nmed, lD*res, weight_angles=False)
    nR200nw = odt.odt_to_ri(fR200nw, res, nmed)
    
    
    # Rytov 50 projections
    remove50 = np.argsort(angles % .002)[:200]
    angles50 = np.delete(angles, remove50, axis=0)
    u_sinR50 = np.delete(u_sinR, remove50, axis=0)
    ph50 = unwrap.unwrap(np.angle(sino/u0))
    ph50[remove50] = 0
    
    fR50 = odt.backpropagate_2d(u_sinR50, angles50, res, nmed, lD*res)
    nR50 = odt.odt_to_ri(fR50, res, nmed)    
    fR50nw = odt.backpropagate_2d(u_sinR50, angles50, res, nmed, lD*res, weight_angles=False)
    nR50nw = odt.odt_to_ri(fR50nw, res, nmed)
    
    # prepare plot
    
    kw_ri = {"vmin": np.min(np.array([phantom, nR50.real, nR200.real])),
             "vmax": np.max(np.array([phantom, nR50.real, nR200.real]))
             }

    kw_ph = {"vmin": np.min(np.array([ph200, ph50])),
             "vmax": np.max(np.array([ph200, ph50])),
             "cmap": plt.cm.coolwarm  # @UndefinedVariable
             }
    
    
    fig, axes = plt.subplots(2,3, figsize=(12,7), dpi=300)
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
                "format":"%.3f"}
    plt.colorbar(phmap, ax=axes[0], **cbkwargs)
    plt.colorbar(phmap, ax=axes[3], **cbkwargs)
    plt.colorbar(rimap, ax=axes[1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[2], **cbkwargs)
    plt.colorbar(rimap, ax=axes[5], **cbkwargs)
    plt.colorbar(rimap, ax=axes[4], **cbkwargs)

    
    plt.tight_layout()
    
    outname = join(DIR, "backprop_2d_weights_angles.png")
    print("Creating output file:", outname)
    plt.savefig(outname)
