============
Introduction
============

This package provides reconstruction algorithms for diffraction
tomography in two and three dimensions.

Installation
------------
To install via the `Python Package Index (PyPI)`_, run:

    pip install odtbrain


On some systems, the `FFTW3 library`_ might have to be
installed manually before installing ODTbrain. All other
dependencies are installed automatically.
If the above command does not work, please refer to the 
installation instructions at the `GitHub repository`_ or
`create an issue`_

.. _`FFTW3 library`: http://fftw.org
.. _`GitHub repository`: https://github.com/RI-imaging/ODTbrain
.. _`Python Package Index (PyPI)`: https://pypi.python.org/pypi/odtbrain/
.. _`create an issue`: https://github.com/RI-imaging/ODTbrain/issues


Theoretical background
----------------------
A detailed summary of the underlying theory is available
in :cite:`Mueller2015arxiv`.

The Fourier diffraction theorem states, that the Fourier transform
:math:`\widehat{U}_{\mathrm{B},\phi_0}(\mathbf{k_\mathrm{D}})` of 
the scattered field :math:`u_\mathrm{B}(\mathbf{r_D})`, measured at 
a certain angle :math:`\phi_0`, is distributed along a circular arc 
(2D) or along a semi-spherical surface (3D) in Fourier space,
synthesizing the Fourier transform 
:math:`\widehat{F}(\mathbf{k})` of the object function 
:math:`f(\mathbf{r})` :cite:`Kak2001`, :cite:`Wolf1969`.

.. math::

   \widehat{F}(k_\mathrm{m}(\mathbf{s - s_0}))= 
        - \sqrt{\frac{2}{\pi}}  \frac{i k_\mathrm{m}}{a_0} 
        M \widehat{U}_{\mathrm{B},\phi_0}(\mathbf{k_\mathrm{D}}) 
        \exp \! \left(-i k_\mathrm{m} M l_\mathrm{D} \right)
    
In this notation, 
:math:`k_\mathrm{m}` is the wave number,
:math:`\mathbf{s_0}` is the norm vector pointing at :math:`\phi_0`,
:math:`M=\sqrt{1-s_\mathrm{x}^2}` (2D) and
:math:`M=\sqrt{1-s_\mathrm{x}^2-s_\mathrm{y}^2}` (3D)
enforces the spherical constraint, and
:math:`l_\mathrm{D}` is the distance from the center of the object
function :math:`f(\mathbf{r})` to the detector plane
:math:`\mathbf{r_D}`.


Fields of Application
---------------------
The algorithms presented here are based on the (scalar) Helmholtz
equation. Furthermore, the Born and Rytov approximations to the
scattered wave :math:`u(\mathbf{r})` are used to linearize the
problem for a straight-forward inversion.

The package is intended for optical diffraction
tomography to determine the refractive index of biological cells.
Because the Helmholtz equation is only an approximation to the
Maxwell equations, describing the propagation of light, 
:abbr:`FDTD (Finite Difference Time Domain)` simulations were performed
to test the reconstruction algorithms within this package.
The algorithms present in this package should also be valid for the
following cases, but have not been tested appropriately:

* tomographic measurements of absorbing materials (complex refractive 
  index :math:`n(\mathbf{r})`)

* ultrasonic diffraction tomography, which is correctly described by
  the Helmholtz equation

How to cite
-----------
If you use ODTbrain in a scientific publication, please cite
Müller et al., *BMC Bioinformatics* (2015) :cite:`Mueller2015`. 

