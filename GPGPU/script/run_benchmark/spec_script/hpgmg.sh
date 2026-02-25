#!/bin/bash

# HPGMG Benchmark Run Script
# Usage: ./run_hpgmg.sh [MPI_RANKS] [LOG2_BOX_DIM] [BOXES_PER_RANK]
# Example: ./run_hpgmg.sh 4 9 2

MPI_RANKS=${1:-4}         # Default to 4 GPUs if not specified
LOG2_BOX_DIM=${2:-8}      # Default to 512³ boxes (2^9)
BOXES_PER_RANK=${3:-8}    # Default to 2 boxes per rank

cd /home/ac.zzheng/benchmark/spec/hpgmg

# Run HPGMG
mpirun -np ${MPI_RANKS} ./build/bin/hpgmg-fv ${LOG2_BOX_DIM} ${BOXES_PER_RANK}
