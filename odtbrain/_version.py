#!/usr/bin/env python
"""
Determine package version for git repositories.

Each time this file is imported it checks if the ".git" folder is
present and if so, obtains the version from the git history using
`git describe`. This information is then stored in the file
`_version_save.py` which is not versioned by git, but distributed
along with e.g. pypi.
"""
from __future__ import print_function

# Put the entire script into a `True` statement and add the hint
# `pragma: no cover` to ignore code coverage here.
if True:  # pragma: no cover
    import imp
    import os
    from os.path import join, abspath, dirname
    import subprocess
    import sys
    import time
    import traceback
    import warnings

    def git_describe():
        """
        Returns a string describing the version returned by the
        command `git describe --tags HEAD`.
        If it is not possible to determine the correct version,
        then an empty string is returned.
        """
        # make sure we are in a directory that belongs to the correct
        # repository.
        ourdir = dirname(abspath(__file__))

        def _minimal_ext_cmd(cmd):
            # construct minimal environment
            env = {}
            for k in ['SYSTEMROOT', 'PATH']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            pop = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=env)
            out = pop.communicate()[0]
            return out

        # change directory
        olddir = abspath(os.curdir)
        os.chdir(ourdir)

        try:
            out = _minimal_ext_cmd(['git', 'describe', '--tags', 'HEAD'])
            git_revision = out.strip().decode('ascii')
        except OSError:
            git_revision = ""

        # go back to original directory
        os.chdir(olddir)

        return git_revision

    def load_version(versionfile):
        """ load version from version_save.py
        """
        longversion = ""
        try:
            _version_save = imp.load_source("_version_save", versionfile)
            longversion = _version_save.longversion
        except BaseException:
            try:
                from ._version_save import longversion
            except BaseException:
                try:
                    from _version_save import longversion
                except BaseException:
                    pass

        return longversion

    def save_version(version, versionfile):
        """ save version to version_save.py
        """
        data = "#!/usr/bin/env python\n" \
            + "# This file was created automatically\n" \
            + "longversion = '{VERSION}'\n"
        try:
            with open(versionfile, "w") as fd:
                fd.write(data.format(VERSION=version))
        except BaseException:
            msg = "Could not write package version to {}.".format(versionfile)
            warnings.warn(msg)

    versionfile = join(dirname(abspath(__file__)), "_version_save.py")

    # Determine the accurate version
    longversion = ""

    # 1. git describe
    try:
        # Get the version using `git describe`
        longversion = git_describe()
    except BaseException:
        pass

    # 2. previously created version file
    if longversion == "":
        # Either this is this is not a git repository or we are in the
        # wrong git repository.
        # Get the version from the previously generated `_version_save.py`
        longversion = load_version(versionfile)

    # 3. last resort: date
    if longversion == "":
        print("Could not determine version. Reason:")
        print(traceback.format_exc())
        ctime = os.stat(__file__)[8]
        longversion = time.strftime("%Y.%m.%d-%H-%M-%S", time.gmtime(ctime))
        print("Using creation time as version: {}".format(longversion))

    if not hasattr(sys, 'frozen'):
        # Save the version to `_version_save.py` to allow distribution using
        # `python setup.py sdist`.
        # This is only done if the program is not frozen (with e.g.
        # pyinstaller),
        if longversion != load_version(versionfile):
            save_version(longversion, versionfile)

    # PEP 440-conform development version:
    version = ".dev".join(longversion.split("-")[:2])
