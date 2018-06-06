"""Miscellaneous methods for example data handling"""
import lzma
import os
import pathlib
import tarfile
import tempfile
import warnings
import zipfile

import numpy as np


datapath = pathlib.Path(__file__).parent / "data"
webloc = "https://github.com/RI-imaging/ODTbrain/raw/master/examples/data/"


def dl_file(url, dest, chunk_size=6553):
    """Download `url` to `dest`"""
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with dest.open('wb') as out:
        while True:
            data = r.read(chunk_size)
            if data is None or len(data) == 0:
                break
            out.write(data)
    r.release_conn()


def extract_lzma(path):
    """Extract an lzma file and return the temporary file name"""
    tlfile = pathlib.Path(path)
    # open lzma file
    with tlfile.open("rb") as td:
        data = lzma.decompress(td.read())
    # write temporary tar file
    fd, tmpname = tempfile.mkstemp(prefix="odt_ex_", suffix=".tar")
    with open(fd, "wb") as fo:
        fo.write(data)
    return tmpname


def get_file(fname, datapath=datapath):
    """Return path of an example data file

    Return the full path to an example data file name.
    If the file does not exist in the `datapath` directory,
    tries to download it from the ODTbrain GitHub repository.
    """
    # download location
    datapath = pathlib.Path(datapath)
    datapath.mkdir(parents=True, exist_ok=True)

    dlfile = datapath / fname
    if not dlfile.exists():
        print("Attempting to download file {} from {} to {}.".
              format(fname, webloc, datapath))
        try:
            dl_file(url=webloc+fname, dest=dlfile)
        except BaseException:
            warnings.warn("Download failed: {}".format(fname))
            raise
    return dlfile


def load_data(fname, **kwargs):
    """Load example data"""
    fname = get_file(fname)
    if fname.suffix == ".lzma":
        return load_tar_lzma_data(fname)
    elif fname.suffix == ".zip":
        return load_zip_data(fname, **kwargs)


def load_tar_lzma_data(tlfile):
    """Load example sinogram data from a .tar.lzma file"""
    tmpname = extract_lzma(tlfile)

    # open tar file
    fields_real = []
    fields_imag = []
    phantom = []
    parms = {}

    with tarfile.open(tmpname, "r") as t:
        members = t.getmembers()
        members.sort(key=lambda x: x.name)

        for m in members:
            n = m.name
            f = t.extractfile(m)
            if n.startswith("fdtd_info"):
                for ln in f.readlines():
                    ln = ln.decode()
                    if ln.count("=") == 1:
                        key, val = ln.split("=")
                        parms[key.strip()] = float(val.strip())
            elif n.startswith("phantom"):
                phantom.append(np.loadtxt(f))
            elif n.startswith("field"):
                if n.endswith("imag.txt"):
                    fields_imag.append(np.loadtxt(f))
                elif n.endswith("real.txt"):
                    fields_real.append(np.loadtxt(f))

    try:
        os.remove(tmpname)
    except OSError:
        pass

    phantom = np.array(phantom)
    sino = np.array(fields_real) + 1j * np.array(fields_imag)
    angles = np.linspace(0, 2 * np.pi, sino.shape[0], endpoint=False)

    return sino, angles, phantom, parms


def load_zip_data(zipname, f_sino_real, f_sino_imag,
                  f_angles=None, f_phantom=None, f_info=None):
    """Load example sinogram data from a .zip file"""
    ret = []
    with zipfile.ZipFile(str(zipname)) as arc:
        sino_real = np.loadtxt(arc.open(f_sino_real))
        sino_imag = np.loadtxt(arc.open(f_sino_imag))
        sino = sino_real + 1j * sino_imag
        ret.append(sino)
        if f_angles:
            angles = np.loadtxt(arc.open(f_angles))
            ret.append(angles)
        if f_phantom:
            phantom = np.loadtxt(arc.open(f_phantom))
            ret.append(phantom)
        if f_info:
            with arc.open(f_info) as info:
                cfg = {}
                for li in info.readlines():
                    li = li.decode()
                    if li.count("=") == 1:
                        key, val = li.split("=")
                        cfg[key.strip()] = float(val.strip())
            ret.append(cfg)
    return ret
