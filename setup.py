#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from os.path import abspath, dirname, join, realpath
from setuptools import setup, find_packages, Command
import subprocess
import sys
from warnings import warn


author = u"Paul Müller"
authors = [author]
name = 'odtbrain'
description = 'Algorithms for diffraction tomography'
year = "2015"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version
except:
    version = "unknown"


class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='paul.mueller at biotec.tu-dresden.de',
        url='https://github.com/paulmueller/odtbrain',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description="""Python algorithm for diffraction tomography""",
        install_requires=["unwrap>=0.1.1", "NumPy>=1.5.1", "SciPy>=0.9.0",
                          "PyFFTW>=0.9.2"],
        #tests_require=["psutil"],
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
