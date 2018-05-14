#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import dirname, realpath, exists
from setuptools import setup
import sys


author = u"Paul Müller"
authors = [author]
description = 'Algorithms for diffraction tomography'
name = 'odtbrain'
year = "2015"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version

if __name__ == "__main__":
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
        install_requires=["numexpr",
                          "numpy>=1.7.0",
                          "pyfftw>=0.9.2", 
                          "scipy>=0.10.0",
                          "unwrap>=0.1.1"],
        setup_requires=['pytest-runner'],
        tests_require=["pytest"],
        keywords=["odt", "opt", "diffraction", "born", "rytov", "radon",
                  "backprojection", "backpropagation", "inverse problem",
                  "Fourier diffraction theorem", "Fourier slice theorem"],
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        )
