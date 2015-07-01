ODTbrain
==========
[![PyPI](http://img.shields.io/pypi/v/odtbrain.svg)](https://pypi.python.org/pypi/odtbrain)
[![Travis](http://img.shields.io/travis/paulmueller/ODTbrain.svg)](https://travis-ci.org/paulmueller/ODTbrain)
[![Coveralls](https://img.shields.io/coveralls/paulmueller/ODTbrain.svg)](https://coveralls.io/r/paulmueller/ODTbrain)


This package provides image reconstruction algorithms for diffraction tomography in two and three dimensions.


### Install requirements

#### General requirements
 - Python 2.7
 - The FFTW3 library
 - These Python packages: 
    - [PyFFTW](https://github.com/hgomersall/pyFFTW) (not `PyFFTW3`)
    - [numpy](https://github.com/numpy/numpy)
    - [scipy](https://github.com/scipy/scipy)
    - [unwrap](https://github.com/geggo/phase-unwrap)


#### Mac OS X

##### [MacPorts](https://www.macports.org/)
 - Install the FFTW3 and Python libraries. For Python 2.7, run
   
         sudo port selfupdate  
         sudo port install fftw-3 py27-numpy py27-scipy py27-pyfftw pip
         sudo easy_install pip
         sudo pip install odtbrain
      
##### [Homebrew](http://brew.sh/)
 - Install the FFTW3 and Python libraries. For Python 2.7, run
    
         sudo brew tap homebrew/python
         sudo brew update && brew upgrade
         sudo brew install python --framework
         sudo brew install fftw numpy scipy
         sudo easy_install pip
         sudo pip install odtbrain


#### Windows
 - Install the [Anaconda](http://continuum.io/downloads#all) version matching your architecture (32-bit or 64-bit).
 - Install `PyFFTW` using the corresponding installer at PyPI:
   https://pypi.python.org/pypi/pyFFTW
 - Finally: `pip install odtbrain`


#### Debian/Ubuntu
Install the following packages:
 - for Python 2.7: `sudo apt-get install libfftw3-3 libfftw3-dev python-cffi python-numpy python-pip python-scipy`
 - for Python 3.4: `sudo apt-get install libfftw3-3 libfftw3-dev python3-cffi python3-numpy python3-pip python3-scipy`
 - Install the `PyFFTW` package. Depending on your distribution, the package name is
   either `python-fftw3`, `python3-fftw3` (old) or `python-pyfftw`, `python3-pyfftw` (new).
   Attention: The package `python-fftw` provides a different FFTW library that is not used by ODTbrain.
   Alternatively, install from PyPI: `pip install pyfftw`.
 - Finally: `pip install odtbrain`


### Testing
After cloning into odtbrain, create a virtual environment

    virtualenv --system-site-packages ve
    source ve/bin/activate

Install all dependencies

    python setup.py develop
    
Running an example

    python examples/simple_example.py
   
Running tests

    python setup.py test
