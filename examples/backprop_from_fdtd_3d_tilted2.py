#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
3D cell phantom with tilted and rolled axis of rotation (FDTD simulation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *in silico* data set was created with the 
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 2D projections of a 3D refractive index phantom that is rotated about
an axis which is tilted by 0.2 rad (11.5 degrees) with respect to the imaging
plane and rolled by -.42 rad (-24.1 degrees) within the imaging plane. The data
are the same as were used in the previous example. A brief description of this
algorithm is given in [3]_.

.. figure::  ../examples/backprop_from_fdtd_3d_tilted2_repo.png
   :align:   center

   3D reconstruction from :abbr:`FDTD (Finite Difference Time Domain)`
   data created by `meep`_ simulations. The known tilted axis of
   rotation is used in the reconstruction process.

Download the :download:`full example <../examples/backprop_from_fdtd_3d_tilted2.py>`.
If you are not running the example from the git repository, make sure the
file :download:`example_helper.py <../examples/example_helper.py>` is present
in the current directory.

.. _`meep`: http://ab-initio.mit.edu/wiki/index.php/Meep

This example requires Python3 because the data are lzma-compressed.
"""
from __future__ import division, print_function


def load_tar_lzma_data(afile):
    """
    Load FDTD data from a .tar.lzma file.
    """
    # open lzma file
    with lzma.open(afile, "rb") as l:
        data = l.read()
    # write tar file
    with open(afile[:-5], "wb") as t:
        t.write(data)
    # open tar file
    fields_real = []
    fields_imag = []
    phantom = []
    parms = {}
    
    with tarfile.open(afile[:-5], "r") as t:
        members = t.getmembers()
        members.sort(key=lambda x: x.name)
        
        for m in members:
            n = m.name
            f = t.extractfile(m)
            if n.startswith("fdtd_info"):
                for l in f.readlines():
                    l = l.decode()
                    if l.count("=") == 1:
                        key, val = l.split("=")
                        parms[key.strip()] = float(val.strip())
            elif n.startswith("phantom"):
                phantom.append(np.loadtxt(f))
            elif n.startswith("field"):
                if n.endswith("imag.txt"):
                    fields_imag.append(np.loadtxt(f))
                elif n.endswith("real.txt"):
                    fields_real.append(np.loadtxt(f))

    phantom = np.array(phantom)
    sino = np.array(fields_real)+1j*np.array(fields_imag)
    angles = np.linspace(0, 2*np.pi, sino.shape[0], endpoint=False)
    
    return sino, angles, phantom, parms
    

# All imports are moved to "__main__", because
# sphinx might complain about some imports (e.g. mpl).
if __name__ == "__main__":
    try:
        from example_helper import get_file
    except ImportError:
        print("Please make sure example_helper.py is available.")
        raise
    import lzma  # @UnresolvedImport    # Only Python3
    import tarfile
    import matplotlib.pylab as plt
    import numpy as np
    import os
    from os.path import abspath, dirname, split, join
    import sys
    
    # Add parent directory to beginning of path variable
    DIR = dirname(abspath(__file__))
    sys.path.insert(0, split(DIR)[0])
    
    import odtbrain as odt
    


    try:
        # use jobmanager if available
        import jobmanager as jm
        jm.decorators.decorate_module_ProgressBar(odt, 
                            decorator=jm.decorators.ProgressBarOverrideCount,
                            interval=.1)
    except:
        pass


    lzmafile = get_file("fdtd_3d_sino_A220_R6.500_tiltyz0.2.tar.lzma")
    
    sino, angles, phantom, cfg = load_tar_lzma_data(lzmafile)

    # Perform titlt by -.42 rad in detector plane    
    rotang = -0.42
    from scipy.ndimage import rotate
    rotkwargs= {"mode":"constant",
                "order":2,
                "reshape":False,
                }
    for ii in range(len(sino)):
        sino[ii].real = rotate(sino[ii].real, np.rad2deg(rotang), cval=1, **rotkwargs)
        sino[ii].imag = rotate(sino[ii].imag, np.rad2deg(rotang), cval=0, **rotkwargs)

    A = angles.shape[0]
    
    print("Example: Backpropagation from 3d FDTD simulations")
    print("Refractive index of medium:", cfg["nm"])
    print("Measurement position from object center:", cfg["lD"])
    print("Wavelength sampling:", cfg["res"])
    print("Axis tilt in y-z direction:", cfg["tilt_yz"])
    print("Number of projections:", A)
    
    ## Apply the Rytov approximation
    sinoRytov = odt.sinogram_as_rytov(sino)

    ## Determine tilted axis
    tilted_axis = [0, np.cos(cfg["tilt_yz"]), np.sin(cfg["tilt_yz"])]
    rotmat = np.array([ 
                       [np.cos(rotang), -np.sin(rotang),0],
                       [np.sin(rotang), np.cos(rotang),0],
                       [0,0,1],
                       ])
    tilted_axis = np.dot(rotmat, tilted_axis)

    print("Performing tilted backpropagation.")
    ## Perform tilted backpropagation
    f_name_tilt = "f_tilt2.npy"
    if not os.path.exists(f_name_tilt):
        f_tilt = odt.backpropagate_3d_tilted( uSin=sinoRytov,
                                              angles=angles,
                                              res=cfg["res"],
                                              nm=cfg["nm"],
                                              lD=cfg["lD"],
                                              tilted_axis=tilted_axis,
                                              )
        np.save(f_name_tilt, f_tilt)
    else:
        f_tilt = np.load(f_name_tilt)

    ## compute refractive index n from object function
    n_tilt = odt.odt_to_ri(f_tilt, res=cfg["res"], nm=cfg["nm"])

    sx, sy, sz = n_tilt.shape
    px, py, pz = phantom.shape


    sino_phase = np.angle(sino)    
    
    ## compare phantom and reconstruction in plot
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5), dpi=300)
    kwri = {"vmin": n_tilt.real.min(), "vmax": n_tilt.real.max()}
    kwph = {"vmin": sino_phase.min(), "vmax": sino_phase.max(), "cmap":plt.cm.coolwarm}  # @UndefinedVariable
    
    
    # Sinorgam
    axes[0].set_title("phase projection")    
    phmap=axes[0].imshow(sino_phase[int(A/2),:,:], **kwph)
    axes[0].set_xlabel("detector x")
    axes[0].set_ylabel("detector y")

    axes[1].set_title("sinogram slice")    
    axes[1].imshow(sino_phase[:,:,int(sino.shape[2]/2)], aspect=sino.shape[1]/sino.shape[0], **kwph)
    axes[1].set_xlabel("detector y")
    axes[1].set_ylabel("angle [rad]")
    # set y ticks for sinogram
    labels = np.linspace(0, 2*np.pi, len(axes[1].get_yticks()))
    labels = [ "{:.2f}".format(i) for i in labels ]
    axes[1].set_yticks(np.linspace(0, len(angles), len(labels)))
    axes[1].set_yticklabels(labels)

    axes[2].set_title("tilt correction (nucleolus)")
    rimap=axes[2].imshow(n_tilt[int(sx/2)+2*cfg["res"]].real, **kwri)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # color bars
    cbkwargs = {"fraction": 0.045,
                "format":"%.3f"}
    plt.colorbar(phmap, ax=axes[0], **cbkwargs)
    plt.colorbar(phmap, ax=axes[1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[2], **cbkwargs)

    plt.tight_layout()
    
    outname = join(DIR, "backprop_from_fdtd_3d_tilted2.png")
    print("Creating output file:", outname)
    plt.savefig(outname)

