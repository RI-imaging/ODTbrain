3D examples
===========
These examples require raw data which are automatically
downloaded from the source repository by the script
:download:`example_helper.py <../examples/example_helper.py>`.
Please make sure that this script is present in the example
script folder.

.. note::
    The ``if __name__ == "__main__"`` guard is necessary on Windows and macOS
    which *spawn* new processes instead of *forking* the current process.
    The 3D backpropagation algorithm makes use of ``multiprocessing.Pool``.


.. fancy_include:: backprop_from_rytov_3d_phantom_apple.py

.. fancy_include:: backprop_from_qlsi_3d_hl60.py

.. fancy_include:: backprop_from_fdtd_3d.py

.. fancy_include:: backprop_from_fdtd_3d_tilted.py

.. fancy_include:: backprop_from_fdtd_3d_tilted2.py

.. fancy_include:: backprop_from_mie_3d_sphere.py
