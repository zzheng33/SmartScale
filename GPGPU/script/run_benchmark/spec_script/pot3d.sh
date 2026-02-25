#!/bin/bash

# POT3D Benchmark Run Script
# Usage: ./run_pot3d.sh [MPI_RANKS]
# Example: ./run_pot3d.sh 4

MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified

cd /home/ac.zzheng/benchmark/spec/POT3D/testsuite/validation/run

# Copy the executable
cp ../../../bin/pot3d .

# Configure input parameters
# Problem size options (grid points: nr × nt × np):
#   Tiny:    NR=67,  NT=181,  NP=451   (~5M grid points)
#   Small:   NR=133, NT=361,  NP=901   (~43M grid points)
#   Medium:  NR=267, NT=721,  NP=1801  (~347M grid points)
#   Large:   NR=534, NT=1441, NP=3601  (~2.8B grid points)

NR=133    # Radial grid points
NT=361    # Theta grid points
NP=901   # Phi grid points (even smaller for testing)

IFPREC=1        # 1 = Jacobi preconditioner, 2 = cuSparse (use 1 for compatibility)
NCGMAX=10000
EPSCG=1.e-99
NCGHIST=1000    # Print residual every N iterations (default: 10)

# Generate pot3d.dat input file in the run directory
cat > pot3d.dat << EOF
 &topology
  nr=${NR}
  nt=${NT}
  np=${NP}
 /
 &inputvars
  ifprec=${IFPREC}
  option='ss'
  r1=2.5
  rfrac=0.0,1.0
  drratio=2.5
  nfrmesh=0
  tfrac=0.00
  dtratio=1.0
  nftmesh=0
  pfrac=0.00
  dpratio=1.0
  nfpmesh=0
  phishift=0.
  br0file='br_input_tiny.h5'
  phifile=''
  brfile=''
  btfile=''
  bpfile=''
  br_photo_file=''
  ncghist=${NCGHIST}
  ncgmax=${NCGMAX}
  epscg=${EPSCG}
  idebug=0
 /
EOF

# Run with MPI
mpirun -np ${MPI_RANKS} --mca btl_openib_warn_no_device_params_found 0 --mca btl tcp,self,vader ./pot3d
