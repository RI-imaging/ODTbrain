#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
3D cell phantom (FDTD simulation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *in silico* data set was created with the 
:abbr:`FDTD (Finite Difference Time Domain)` software `meep`_. The data
are 2D projections of a 3D refractive index phantom. The reconstruction 
of the refractive index with the Rytov approximation is in good agreement
with the phantom that was used in the simulation. The data are downsampled
by a factor of two. The rotational axis is the `y`-axis. A total of 180
projections are used for the reconstruction. A detailed description of
this phantom is given in [2]_.

.. figure::  ../examples/backprop_from_fdtd_3d_repo.png
   :align:   center

   3D reconstruction from :abbr:`FDTD (Finite Difference Time Domain)`
   data created by `meep`_ simulations.

Download the :download:`full example <../examples/backprop_from_fdtd_3d.py>`.
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
    from os.path import abspath, dirname, split, join
    import sys
    
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


    lzmafile = get_file("fdtd_3d_sino_A180_R6.500.tar.lzma")
    sino, angles, phantom, cfg = load_tar_lzma_data(lzmafile)

    A = angles.shape[0]
    
    print("Example: Backpropagation from 3d FDTD simulations")
    print("Refractive index of medium:", cfg["nm"])
    print("Measurement position from object center:", cfg["lD"])
    print("Wavelength sampling:", cfg["res"])
    print("Number of projections:", A)
    print("Performing backpropagation.")

    ## Apply the Rytov approximation
    sinoRytov = odt.sinogram_as_rytov(sino)


    ## perform backpropagation to obtain object function f
    f = odt.backpropagate_3d( uSin=sinoRytov,
                              angles=angles,
                              res=cfg["res"],
                              nm=cfg["nm"],
                              lD=cfg["lD"]
                              )

    ## compute refractive index n from object function
    n = odt.odt_to_ri(f, res=cfg["res"], nm=cfg["nm"])

    sx, sy, sz = n.shape
    px, py, pz = phantom.shape


    sino_phase = np.angle(sino)    
    
    ## compare phantom and reconstruction in plot
    fig, axes = plt.subplots(2, 3, figsize=(12,7), dpi=300)
    kwri = {"vmin": n.real.min(), "vmax": n.real.max()}
    kwph = {"vmin": sino_phase.min(), "vmax": sino_phase.max(), "cmap":plt.cm.coolwarm}  # @UndefinedVariable
    
    
    # Phantom
    axes[0,0].set_title("FDTD phantom center")
    rimap=axes[0,0].imshow(phantom[int(px/2)], **kwri)
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")

    axes[1,0].set_title("FDTD phantom nucleolus")
    axes[1,0].imshow(phantom[int(px/2)+2*cfg["res"]], **kwri)
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("y")    
    
    # Sinorgam
    axes[0,1].set_title("phase projection")    
    phmap=axes[0,1].imshow(sino_phase[int(A/2)], aspect=sino.shape[1]/sino.shape[0], **kwph)
    axes[0,1].set_xlabel("detector x")
    axes[0,1].set_ylabel("detector y")

    axes[1,1].set_title("sinogram slice")    
    axes[1,1].imshow(sino_phase[:,:,int(A/2)], aspect=sino.shape[1]/sino.shape[0], **kwph)
    axes[1,1].set_xlabel("detector y")
    axes[1,1].set_ylabel("angle [rad]")
    # set y ticks for sinogram
    labels = np.linspace(0, 2*np.pi, len(axes[1,1].get_yticks()))
    labels = [ "{:.2f}".format(i) for i in labels ]
    axes[1,1].set_yticks(np.linspace(0, len(angles), len(labels)))
    axes[1,1].set_yticklabels(labels)

    axes[0,2].set_title("reconstruction center")
    axes[0,2].imshow(n[int(sx/2)].real, **kwri)
    axes[0,2].set_xlabel("x")    
    axes[0,2].set_ylabel("y")

    axes[1,2].set_title("reconstruction nucleolus")
    axes[1,2].imshow(n[int(sx/2)+2*cfg["res"]].real, **kwri)
    axes[1,2].set_xlabel("x")    
    axes[1,2].set_ylabel("y")

    # color bars
    cbkwargs = {"fraction": 0.045,
                "format":"%.3f"}
    plt.colorbar(phmap, ax=axes[0,1], **cbkwargs)
    plt.colorbar(phmap, ax=axes[1,1], **cbkwargs)
    plt.colorbar(rimap, ax=axes[0,0], **cbkwargs)
    plt.colorbar(rimap, ax=axes[1,0], **cbkwargs)
    plt.colorbar(rimap, ax=axes[0,2], **cbkwargs)
    plt.colorbar(rimap, ax=axes[1,2], **cbkwargs)

    plt.tight_layout()
   
    outname = join(DIR, "backprop_from_fdtd_3d.png")
    print("Creating output file:", outname)
    plt.savefig(outname)
