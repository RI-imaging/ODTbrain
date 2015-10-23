#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains miscellaneous methods that are used by
example scripts.
"""
from __future__ import print_function
import os
from os.path import dirname, join, exists, isdir

import warnings

datadir = "data"
webloc = "https://github.com/paulmueller/ODTbrain/raw/master/examples/data"


def get_file(fname):
    """
    Return the full path to a basename. If the file does not exist
    in the current directory or in subdirectory `datadir`, try to 
    download it from the public GitHub repository.
    """
    # download location
    dlloc = join(dirname(__file__), datadir)
    if exists(dlloc):
        if isdir(dlloc):
            pass
        else:
            raise OSError("Must be directory: "+dlloc)
    else:
        os.mkdir(dlloc)
        
    
    # find the file
    foundloc = None
    
    # Search possible file locations
    possloc = [dirname(__file__), dlloc]  

    for pl in possloc:
        if exists(join(pl, fname)):
            foundloc = join(pl, fname)
            break
                
    if foundloc is None:
        from urllib2 import urlopen
        # Download file with urllib2.urlopen
        print("Attempting to download file {} from {} to {}.".
              format(fname, webloc, dlloc))
        try:
            f = urlopen(join(webloc, fname))
            # Open our local file for writing
            with open(join(dlloc, fname), "wb") as local_file:
                local_file.write(f.read())
        except:
            warnings.warn("Download failed: "+fname)
            raise
        else:
            foundloc = join(dlloc, fname)
    
    if foundloc is None:
        raise OSError("Could not obtain file: "+fname)
    
    return foundloc