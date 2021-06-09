# The NAS Parallel Benchmark suite in C++

The original repository can be found [here](https://github.com/GMAP/NPB-CPP).

## Implementations

**NPB-SER** - This directory contains the sequential version of the benchmarks.

**NPB-OMP** - This directory contains the parallel OpenMP version of the benchmarks.

**NPB-DSM** - This directory contains the parallel ArgoDSM/OpenMP version of the benchmarks.

**NPB-OSS** - This directory contains the parallel OmpSs-2(@Cluster) version of the benchmarks.

## The Five Kernels and Three Pseudo-applications

Each directory contains its own implemented version of the kernels and pseudo-applications.

### Kernels

```sh
EP - Embarrassingly Parallel, floating-point operation capacity
MG - Multi-Grid, non-local memory accesses, short- and long-distance communication
CG - Conjugate Gradient, irregular memory accesses and communication
FT - discrete 3D fast Fourier Transform, intensive long-distance communication
IS - Integer Sort, integer computation and communication
```

### Pseudo-applications

```sh
BT - Block Tri-diagonal solver
SP - Scalar Penta-diagonal solver
LU - Lower-Upper Gauss-Seidel solver
```

## Software Requirements

**NPB-DSM** - Assumes you have already installed [argodsm](https://github.com/etascale/argodsm).

**NPB-OSS** - Assumes you have already installed [mercurium](https://github.com/bsc-pm/mcxx) and [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster).

## Building 

Enter the directory from the version desired and execute:
```sh
$ make _BENCHMARK CLASS=_WORKLOAD
```

_BENCHMARKs are: 
```sh
EP, CG, MG, IS, FT, BT, SP and LU 
```

_WORKLOADs are: 
```sh
Class S: small for quick test purposes
Class W: workstation size (a 90s workstation; now likely too small)	
Classes A, B, C: standard test problems; ~4X size increase going from one class to the next	
Classes D, E, F: large test problems; ~16X size increase from each of the previous classes
```

Command example:
```sh
$ make ep CLASS=A
```

## Executing

Binaries are generated inside the **bin** folder.

Command example:
```sh
$ (mpirun $OMPIFLAGS) ./bin/ep.A
```

## Compiler and Parallel Configurations

Each folder contains a default compiler configuration that can be modified in the `config/make.def` file.
You must use this file if you want to modify the target compiler, flags or links that will be used to compile the applications.
