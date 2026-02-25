#!/bin/bash

# Minisweep Benchmark Run Script
# Usage: ./run_minisweep.sh [MPI_RANKS]
# Example: ./run_minisweep.sh 4

cd /home/ac.zzheng/benchmark/spec/minisweep/build

# Configure parameters
NCELL_X=128
NCELL_Y=128
NCELL_Z=128
NE=16
NA=32
NBLOCK_Z=128
NITERATIONS=10

# MPI configuration
MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified

# Automatically calculate NPROC_X and NPROC_Y based on MPI_RANKS
# Uses square decomposition when possible, otherwise linear
case ${MPI_RANKS} in
    1)
        NPROC_X=1
        NPROC_Y=1
        ;;
    2)
        NPROC_X=2
        NPROC_Y=1
        ;;
    3)
        NPROC_X=3
        NPROC_Y=1
        ;;
    4)
        NPROC_X=2
        NPROC_Y=2
        ;;
    *)
        # Default: linear decomposition
        NPROC_X=${MPI_RANKS}
        NPROC_Y=1
        ;;
esac


# Run with MPI and CUDA
mpirun -np ${MPI_RANKS} --mca btl_openib_warn_no_device_params_found 0 --mca btl tcp,self,vader \
  ./sweep \
  --ncell_x ${NCELL_X} --ncell_y ${NCELL_Y} --ncell_z ${NCELL_Z} \
  --ne ${NE} --na ${NA} \
  --nproc_x ${NPROC_X} --nproc_y ${NPROC_Y} \
  --nblock_z ${NBLOCK_Z} \
  --is_using_device 1 \
  --nthread_octant 8 --nthread_e 16 \
  --niterations ${NITERATIONS}
