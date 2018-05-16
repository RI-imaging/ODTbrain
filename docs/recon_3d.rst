3D inversion
------------
.. currentmodule:: odtbrain


The first Born approximation for a 3D scattering problem with a plane
wave 
:math:`u_0(\mathbf{r}) = a_0 \exp(-ik_\mathrm{m}\mathbf{s_0r})`
reads:


.. math::
    u_\mathrm{B}(\mathbf{r}) = \iiint \!\! d^3r' 
        G(\mathbf{r-r'}) f(\mathbf{r'}) u_0(\mathbf{r'})

The Green's function in 3D can be written as:

.. math::
    G(\mathbf{r-r'}) = \frac{ik_\mathrm{m}}{8\pi^2} \iint \!\! dpdq 
        \frac{1}{M} \exp\! \left \lbrace i k_\mathrm{m} \left[ 
        p(x-x') + q(y-y') + M(z-z') \right] \right \rbrace

with

.. math::
    
    M = \sqrt{1-p^2-q^2}
    
Solving for :math:`f(\mathbf{r})` yields the Fourier diffraction theorem
in 3D

.. math::
    \widehat{F}(k_\mathrm{m}(\mathbf{s-s_0})) = 
        - \sqrt{\frac{2}{\pi}} 
        \frac{i k_\mathrm{m}}{a_0} M
        \widehat{U}_{\mathrm{B},\phi_0}(k_\mathrm{Dx}, k_\mathrm{Dy})
        \exp \! \left(-i k_\mathrm{m} M l_\mathrm{D} \right)
    
where 
:math:`\widehat{F}(k_\mathrm{x}, k_\mathrm{y}, k_\mathrm{z})`
is the Fourier transformed object function and 
:math:`\widehat{U}_{\mathrm{B}, \phi_0}(k_\mathrm{Dx}, k_\mathrm{Dy})` 
is the Fourier transformed complex wave that travels along 
:math:`\mathbf{s_0}`
(in the direction of :math:`\phi_0`) measured at the detector
:math:`\mathbf{r_D}`.


The following identities are used:

.. math::
    k_\mathrm{m} (\mathbf{s-s_0}) &= k_\mathrm{Dx} \, \mathbf{t_\perp} +
    k_\mathrm{m}(M - 1) \, \mathbf{s_0}
    
    \mathbf{s} &= (p, q, M)

    \mathbf{s_0} &= (p_0, q_0, M_0) = (-\sin\phi_0, \, 0, \, \cos\phi_0)

    \mathbf{t_\perp} &= \left(\cos\phi_0, \,
                \frac{k_\mathrm{Dy}}{k_\mathrm{Dx}}, \,
                \sin\phi_0 \right)^\top 

.. currentmodule:: odtbrain


Method summary
~~~~~~~~~~~~~~

.. autosummary:: 
    backpropagate_3d
    backpropagate_3d_tilted


Backpropagation
~~~~~~~~~~~~~~~
.. autofunction:: backpropagate_3d


Backpropagation with tilted axis of rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: backpropagate_3d_tilted

