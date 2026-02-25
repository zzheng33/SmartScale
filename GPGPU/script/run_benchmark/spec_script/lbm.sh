#!/bin/bash

# LBM Benchmark Run Script
# Usage: ./lbm.sh [MPI_RANKS]
# Example: ./lbm.sh 4

MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified

cd /home/ac.zzheng/benchmark/spec/LBM/run

# Configure problem size
GSIZEX=1200
GSIZEY=4800
NITER=5000
SEED=13948

# Generate control input file
cat > control << EOF
${GSIZEX} # GSIZEX
${GSIZEY} # GSIZEY
${NITER}    # NITER
${SEED} # SEED
EOF

mpirun -np ${MPI_RANKS} --mca btl_openib_warn_no_device_params_found 0 --mca btl tcp,self,vader ./lbm_cuda