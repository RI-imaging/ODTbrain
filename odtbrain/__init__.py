"""Algorithms for scalar diffraction tomography"""
from ._alg2d_bpp import backpropagate_2d
from ._alg2d_fmp import fourier_map_2d
from ._alg2d_int import integrate_2d

from ._alg3d_bpp import backpropagate_3d
from ._alg3d_bppt import backpropagate_3d_tilted

from ._br import odt_to_ri, opt_to_ri, sinogram_as_radon, sinogram_as_rytov
from ._version import version as __version__
from ._version import longversion as __version_full__

__author__ = "Paul Müller"
__license__ = "BSD (3 clause)"


# Shared variable used by 3D backpropagation
_shared_array = None