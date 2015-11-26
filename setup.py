#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import dirname, realpath
from setuptools import setup, Command
import subprocess
import sys


author = u"Paul Müller"
authors = [author]
name = 'odtbrain'
description = 'Algorithms for diffraction tomography'
year = "2015"

long_description = """
This package provides inverse scattering algorithms in 2D and 3D
for diffraction tomogrpahy. Visit the home page for more information.
"""


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import shortversion as version


class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='paul.mueller at biotec.tu-dresden.de',
        url='http://paulmueller.github.io/ODTbrain/',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=long_description,
        install_requires=["unwrap>=0.1.1", "NumPy>=1.7.0", "SciPy>=0.10.0",
                          "PyFFTW>=0.9.2"],
        keywords=["odt", "opt", "diffraction", "born", "rytov", "radon",
                  "backprojection", "backpropagation", "inverse problem",
                  "Fourier diffraction theorem", "Fourier slice theorem"],
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        cmdclass = {'test': PyTest,
                    },
        )
