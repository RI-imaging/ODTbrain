ODTbrain
========

|PyPI Version| |Tests Status| |Coverage Status| |Docs Status|


**ODTbrain** provides image reconstruction algorithms for **O**\ ptical **D**\ iffraction **T**\ omography with a **B**\ orn and **R**\ ytov
**A**\ pproximation-based **I**\ nversion to compute the refractive index (**n**\ ) in 2D and in 3D.


Documentation
-------------

The documentation, including the reference and examples, is available at `odtbrain.readthedocs.io <https://odtbrain.readthedocs.io/en/stable/>`__.


Installation
------------
::

    pip install odtbrain



Testing
-------

After cloning into odtbrain, create a virtual environment::

    virtualenv --system-site-packages env
    source env/bin/activate

Install ODTbrain in editable mode::

    pip install -e .
    
Running an example::

    python examples/backprop_from_fdtd_2d.py
   
Running tests::

    pip install pytest
    pytest tests


.. |PyPI Version| image:: https://img.shields.io/pypi/v/odtbrain.svg
   :target: https://pypi.python.org/pypi/odtbrain
.. |Tests Status| image:: https://img.shields.io/github/actions/workflow/status/RI-Imaging/ODTbrain/check.yml
   :target: https://github.com/RI-Imaging/ODTbrain/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/RI-imaging/ODTbrain/master.svg
   :target: https://codecov.io/gh/RI-imaging/ODTbrain
.. |Docs Status| image:: https://readthedocs.org/projects/odtbrain/badge/?version=latest
   :target: https://readthedocs.org/projects/odtbrain/builds/
