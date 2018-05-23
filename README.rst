ODTbrain
========

|PyPI Version| |Tests Status| |Coverage Status| |Docs Status|


**ODTbrain** provides image reconstruction algorithms for **O**\ ptical **D**\ iffraction **T**\ omography with a **B**\ orn and **R**\ ytov
**A**\ pproximation-based **I**\ nversion to compute the refractive index (**n**\ ) in 2D and in 3D.


Documentation
-------------

The documentation, including the reference and examples, is available on `readthedocs.io <https://odtbrain.readthedocs.io/en/stable/>`__.


Installation
------------

Dependencies
~~~~~~~~~~~~

- Python 3.4 or higher
- The FFTW3 library
- These Python packages: 

  - `PyFFTW <https://github.com/pyFFTW/pyFFTW>`__ (not `PyFFTW3`)
  - `numpy <https://github.com/numpy/numpy>`__
  - `scikit-image <https://github.com/scikit-image/scikit-image/>`__
  - `scipy <https://github.com/scipy/scipy>`__


Mac OS X
~~~~~~~~

`MacPorts <https://www.macports.org/>`__
________________________________________

Install the FFTW3 and Python libraries. For Python 3.6, run

::

    sudo port selfupdate  
    sudo port install fftw-3 py36-numpy py36-scipy py36-pyfftw pip
    sudo easy_install pip
    sudo pip install odtbrain


`Homebrew <http://brew.sh/>`__
______________________________

Install the FFTW3 and Python libraries. For Python 3.6, run

::

    sudo brew tap homebrew/python
    sudo brew update && brew upgrade
    sudo brew install python --framework
    sudo brew install fftw numpy scipy
    sudo easy_install pip
    sudo pip install odtbrain


Windows
~~~~~~~

- Install `Anaconda <http://continuum.io/downloads#all>`__ with Python3 for your architecture (32-bit or 64-bit).
- Run:
  ::
  
      pip install odtbrain


Debian/Ubuntu
~~~~~~~~~~~~~

Install the following packages:

- Packages from the repository:
  ::
  
      sudo apt-get install libfftw3-3 libfftw3-dev python3-cffi python3-numpy python3-pip python3-scipy``
- Install the `PyFFTW` package. Depending on your distribution, the package name is
  ``python3-fftw3`` (old), ``python3-pyfftw`` (new), or non-existent.
  Attention: The package ``python-fftw`` provides a different FFTW library that is not used by ODTbrain.
  Alternatively, install from PyPI:
  ::
  
      pip install pyfftw

- Finally:
  ::
  
      pip install odtbrain


Testing
-------

After cloning into odtbrain, create a virtual environment

::

    virtualenv --system-site-packages env
    source env/bin/activate

Install all dependencies

::

    pip install -e .
    
Running an example

::

    python examples/backprop_from_fdtd_2d.py
   
Running tests

::

    python setup.py test

    

.. |PyPI Version| image:: http://img.shields.io/pypi/v/odtbrain.svg
   :target: https://pypi.python.org/pypi/odtbrain
.. |Tests Status| image:: http://img.shields.io/travis/RI-imaging/ODTbrain.svg?label=tests
   :target: https://travis-ci.org/RI-imaging/ODTbrain
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/RI-imaging/ODTbrain/master.svg
   :target: https://codecov.io/gh/RI-imaging/ODTbrain
.. |Docs Status| image:: https://readthedocs.org/projects/odtbrain/badge/?version=latest
   :target: https://readthedocs.org/projects/odtbrain/builds/
