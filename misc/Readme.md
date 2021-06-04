## C++ Meep simulation files 
I wrote these [meep](http://ab-initio.mit.edu/wiki/index.php?title=Meep)
simulation files to obtain the in-silico sinogram data used in the
[original manuscript](https://dx.doi.org/10.1186/s12859-015-0764-0).

### Before you start
Note that at the time of writing this, there is a Python interface for meep
available for Ubuntu/Debian. If you are starting from scratch, it might be
worth working with Python instead of wrapping these C++ scripts.

### Simulation workflow
You will need a working meep installation (I recommend the parallel version)
and a C++ compiler.

I designed the files such that it is easy to write a wrapper and modify
certain aspects. For instance, you can modify the incident angle of
illumination simply by replacing the line starting with `#define ACQUISITION_PHI`.
You can then create multiple subdirectories, one for each angle, and compile
and run the simulation:

```
# compile (using GNU C++ Compiler on linux)
g++ -malign-double meep_phantom_2d.cpp -o meep_phantom_2d.bin -lmeep_mpi -lhdf5 -lz -lgsl -lharminv -llapack -lcblas -latlas -lfftw3 -lm
# run with 4 cores (requires meep compiled with Open MPI support)
mpirun -n 4 ./meep_phantom_2d.bin
```
