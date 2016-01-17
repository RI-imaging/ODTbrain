#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Incomplete angular coverage (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example illustrates how the backpropagation algorithm of ODTbrain
handles incomplete angular coverage. All examples use 100 projections
at 100%, 60%, and 40% total angular coverage. The keyword argument
`weight_angles` that invokes angular weighting is set to `True` by default.
The *in silico* data set was created with the 
softare `miefield  <https://github.com/RI-imaging/miefield>`_.
The data are 1D projections of a non-centered cylinder of constant
refractive index 1.339 embedded in water with refractive index 1.333. 
The first column shows the used sinograms (missing angles are displayed
as zeros) that were created from the original sinogram with 250
projections. The second column shows the reconstruction without angular
weights and the third column shows the reconstruction with angular
weights. The keyword argument `weight_angle` was introduced in version
0.1.1. 

.. figure::  ../examples/backprop_2d_incomplete_coverage_repo.png
   :align:   center

   A 180 degree coverage results in a good reconstruction of the object.
   Angular weighting as implemented in the backpropagation algorithm
   of ODTbrain automatically addresses uneven and incomplete angular coverage.

Download the :download:`full example <../examples/backprop_2d_incomplete_coverage.py>`.
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
    
    # Rytov 100 projections evenly distributed    
    removeeven = np.argsort(angles % .002)[:150]
    angleseven = np.delete(angles, removeeven, axis=0)
    u_sinReven = np.delete(u_sinR, removeeven, axis=0)
    pheven = unwrap.unwrap(np.angle(sino/u0))
    pheven[removeeven] = 0
    
    fReven = odt.backpropagate_2d(u_sinReven, angleseven, res, nmed, lD*res)
    nReven = odt.odt_to_ri(fReven, res, nmed)
    fRevennw = odt.backpropagate_2d(u_sinReven, angleseven, res, nmed, lD*res, weight_angles=False)
    nRevennw = odt.odt_to_ri(fRevennw, res, nmed)
    
    
    # Rytov 100 projections more than 180
    removemiss = 249 - np.concatenate((np.arange(100), 100+np.arange(150)[::3]))
    anglesmiss = np.delete(angles, removemiss, axis=0)
    u_sinRmiss = np.delete(u_sinR, removemiss, axis=0)
    phmiss = unwrap.unwrap(np.angle(sino/u0))
    phmiss[removemiss] = 0
    
    fRmiss = odt.backpropagate_2d(u_sinRmiss, anglesmiss, res, nmed, lD*res)
    nRmiss = odt.odt_to_ri(fRmiss, res, nmed)    
    fRmissnw = odt.backpropagate_2d(u_sinRmiss, anglesmiss, res, nmed, lD*res, weight_angles=False)
    nRmissnw = odt.odt_to_ri(fRmissnw, res, nmed)
    

    # Rytov 100 projections less than 180
    removebad = 249 - np.arange(150)
    anglesbad = np.delete(angles, removebad, axis=0)
    u_sinRbad = np.delete(u_sinR, removebad, axis=0)
    phbad = unwrap.unwrap(np.angle(sino/u0))
    phbad[removebad] = 0
    
    fRbad = odt.backpropagate_2d(u_sinRbad, anglesbad, res, nmed, lD*res)
    nRbad = odt.odt_to_ri(fRbad, res, nmed)    
    fRbadnw = odt.backpropagate_2d(u_sinRbad, anglesbad, res, nmed, lD*res, weight_angles=False)
    nRbadnw = odt.odt_to_ri(fRbadnw, res, nmed)    
    
    # prepare plot
    
    kw_ri = {"vmin": np.min(np.array([phantom, nRmiss.real, nReven.real])),
             "vmax": np.max(np.array([phantom, nRmiss.real, nReven.real]))
             }

    kw_ph = {"vmin": np.min(np.array([pheven, phmiss])),
             "vmax": np.max(np.array([pheven, phmiss])),
             "cmap": plt.cm.coolwarm  # @UndefinedVariable
             }
    
    
    fig, axes = plt.subplots(3,3, figsize=(12,10), dpi=300)
    
    axes[0,0].set_title("100% coverage ({} proj.)".format(angleseven.shape[0]))
    phmap = axes[0,0].imshow(pheven, **kw_ph)

    axes[0,1].set_title("RI without angular weights")
    rimap = axes[0,1].imshow(nRevennw.real, **kw_ri)

    axes[0,2].set_title("RI with angular weights")
    rimap = axes[0,2].imshow(nReven.real, **kw_ri)
    
    axes[1,0].set_title("60% coverage ({} proj.)".format(anglesmiss.shape[0]))
    axes[1,0].imshow(phmiss, **kw_ph)
    
    axes[1,1].set_title("RI without angular weights")
    axes[1,1].imshow(nRmissnw.real, **kw_ri)
    
    axes[1,2].set_title("RI with angular weights")    
    axes[1,2].imshow(nRmiss.real, **kw_ri)

    axes[2,0].set_title("40% coverage ({} proj.)".format(anglesbad.shape[0]))
    axes[2,0].imshow(phbad, **kw_ph)
    
    axes[2,1].set_title("RI without angular weights")
    axes[2,1].imshow(nRbadnw.real, **kw_ri)
    
    axes[2,2].set_title("RI with angular weights")
    axes[2,2].imshow(nRbad.real, **kw_ri)
    
    
    # color bars
    cbkwargs = {"fraction": 0.045,
                "format":"%.3f"}
    plt.colorbar(phmap, ax=axes[0,0], **cbkwargs)
    plt.colorbar(phmap, ax=axes[1,0], **cbkwargs)
    plt.colorbar(phmap, ax=axes[2,0], **cbkwargs)
    plt.colorbar(rimap, ax=axes[0,1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[1,1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[2,1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[0,2], **cbkwargs)
    plt.colorbar(rimap, ax=axes[1,2], **cbkwargs)
    plt.colorbar(rimap, ax=axes[2,2], **cbkwargs)


    
    plt.tight_layout()
    
    outname = join(DIR, "backprop_2d_incomplete_coverage.png")
    print("Creating output file:", outname)
    plt.savefig(outname)
