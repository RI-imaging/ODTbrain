import os
import os.path as op
import sys

import matplotlib.pylab as plt

thisdir = op.dirname(op.abspath(__file__))
sys.path.insert(0, op.dirname(thisdir))

DPI = 80


if __name__ == "__main__":
    # Do not display example plots
    plt.show = lambda: None
    files = os.listdir(thisdir)
    files = [f for f in files if f.endswith(".py")]
    files = [f for f in files if not f == op.basename(__file__)]
    files = sorted([op.join(thisdir, f) for f in files])

    for f in files:
        fname = f[:-3] + ".jpg"
        if not op.exists(fname):
            exec_str = open(f).read()
            if exec_str.count("plt.show()"):
                exec(exec_str)
                plt.savefig(fname, dpi=DPI)
                print("Image created: '{}'".format(fname))
            else:
                print("No image: '{}'".format(fname))
        else:
            print("Image skipped (already exists): '{}'".format(fname))
        plt.close()
