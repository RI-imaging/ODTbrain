3D examples
===========
These examples require raw data which are automatically
downloaded from the source repository by the script
:download:`example_helper.py <../examples/example_helper.py>`.
Please make sure that this script is present in the example
script folder.

.. note:: Windows users:
    If you cannot run these examples directly, because you are getting
    a `RuntimeError` with the text `"An attempt has been made to start
    a new process before the current process has finished its
    bootstrapping phase"`, please add the line
    ``if __name__ == "__main__":`` at the beginning of the file
    and indent the rest of the file by four spaces, i.e.

    .. code:: python

        if __name__ == "__main__":
            import matplotlib.pylab as plt
            import nrefocus
            import numpy as np
    
            import odtbrain as odt
            
            # etc.

.. fancy_include:: backprop_from_rytov_3d_phantom_apple.py

.. fancy_include:: backprop_from_qlsi_3d_hl60.py

.. fancy_include:: backprop_from_fdtd_3d.py

.. fancy_include:: backprop_from_fdtd_3d_tilted.py

.. fancy_include:: backprop_from_fdtd_3d_tilted2.py

.. fancy_include:: backprop_from_mie_3d_sphere.py
