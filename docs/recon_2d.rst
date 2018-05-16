2D inversion
------------
The first Born approximation for a 2D scattering problem with a plane
wave 
:math:`u_0(\mathbf{r}) = a_0 \exp(-ik_\mathrm{m}\mathbf{s_0r})`
reads:

.. math::
    u_\mathrm{B}(\mathbf{r}) = \iint \!\! d^2r' 
        G(\mathbf{r-r'}) f(\mathbf{r'}) u_0(\mathbf{r'})

The Green's function in 2D is the zero-order Hankel function
of the first kind:

.. math::
    G(\mathbf{r-r'}) = \frac{i}{4} 
        H_0^\mathrm{(1)}(k_\mathrm{m} \left| \mathbf{r-r'} \right|) 

Solving for :math:`f(\mathbf{r})` yields the Fourier diffraction theorem
in 2D

.. math::
    \widehat{F}(k_\mathrm{m}(\mathbf{s-s_0})) = 
        - \sqrt{\frac{2}{\pi}} 
        \frac{i k_\mathrm{m}}{a_0} M
        \widehat{U}_{\mathrm{B},\phi_0}(k_\mathrm{Dx})
        \exp \! \left(-i k_\mathrm{m} M l_\mathrm{D} \right)
    
where 
:math:`\widehat{F}(k_\mathrm{x}, k_\mathrm{z})`
is the Fourier transformed object function and 
:math:`\widehat{U}_{\mathrm{B}, \phi_0}(k_\mathrm{Dx})` is the
Fourier transformed complex wave that travels along :math:`\mathbf{s_0}`
(in the direction of :math:`\phi_0`) measured at the detector
:math:`\mathbf{r_D}`.


The following identities are used:

.. math::
    k_\mathrm{m} (\mathbf{s-s_0}) &= k_\mathrm{Dx} \, \mathbf{t_\perp} +
    k_\mathrm{m}(M - 1) \, \mathbf{s_0}
    
    \mathbf{s_0} &= \left(p_0 , \, M_0 \right) = 
    (-\sin\phi_0, \, \cos\phi_0)
    
    \mathbf{t_\perp} &= \left(- M_0 , \, p_0 \right) = 
    (\cos\phi_0, \, \sin\phi_0)

.. currentmodule:: odtbrain


Method summary
~~~~~~~~~~~~~~
.. autosummary:: 
    backpropagate_2d
    fourier_map_2d
    integrate_2d


Backpropagation
~~~~~~~~~~~~~~~
.. autofunction:: backpropagate_2d

Fourier mapping
~~~~~~~~~~~~~~~
.. autofunction:: fourier_map_2d

Direct sum
~~~~~~~~~~
.. autofunction:: integrate_2d
