Pre- and post-processing
------------------------
.. currentmodule:: odtbrain

.. autosummary:: 
    odt_to_ri
    opt_to_ri
    sinogram_as_radon
    sinogram_as_rytov


Pre-processing models
~~~~~~~~~~~~~~~~~~~~~
Tomographic data sets consist of detector images for different
rotational positions :math:`\phi_0` of the object. Pre-processing
in this context means that the measured field :math:`u(\mathbf{r})`
is transformed to either the Rytov approximation (diffraction tomography)
or the Radon phase (classical tomography).

.. autofunction:: sinogram_as_radon
.. autofunction:: sinogram_as_rytov


Post-processing (Refractive index retrieval)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To obtain the refractive index map :math:`n(\mathbf{r})`
from an object function :math:`f(\mathbf{r})` returned
by e.g. :func:`backpropagate_3d`, an additional conversion
step is necessary. For diffraction based models, :func:`odt_to_ri`
must be used whereas for Radon-based models :func:`opt_to_ri`
must be used.

.. autofunction:: odt_to_ri
.. autofunction:: opt_to_ri
