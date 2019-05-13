from os.path import dirname, realpath, exists
from setuptools import setup
import sys


author = "Paul Müller"
authors = [author]
description = 'Algorithms for diffraction tomography'
name = 'odtbrain'
year = "2015"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version


if __name__ == "__main__":  # required by Windows/pytest (multiprocessing)
    setup(
        name=name,
        author=author,
        author_email='dev@craban.de',
        url='https://github.com/RI-imaging/ODTbrain',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=open('README.rst').read() if exists('README.rst') else '',
        install_requires=["numexpr",  # 3D backpropagation
                          "numpy>=1.7.0",
                          "pyfftw>=0.9.2",  # 3D backpropagation
                          "scikit-image>=0.11.0",  # phase-unwrapping
                          "scipy>=0.10.0",  # initial requirement
                          ],
        setup_requires=['pytest-runner'],
        tests_require=["pytest"],
        python_requires='>=3.4, <4',
        keywords=["odt", "opt", "diffraction", "born", "rytov", "radon",
                  "backprojection", "backpropagation", "inverse problem",
                  "Fourier diffraction theorem", "Fourier slice theorem"],
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        )
