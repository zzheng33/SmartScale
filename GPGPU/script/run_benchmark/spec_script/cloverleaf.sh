#!/bin/bash

# CloverLeaf Benchmark Run Script
# Usage: ./run_cloverleaf.sh [MPI_RANKS]
# Example: ./run_cloverleaf.sh 4

MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified

cd /home/ac.zzheng/benchmark/spec/CloverLeaf

# Configure parameters
X_CELLS=7680
Y_CELLS=7680
END_STEP=1000
END_TIME=15.6
TILES_PER_CHUNK=1

# Generate clover.in input file
cat > clover.in << EOF
*clover

 state 1 density=0.2 energy 1.0
 state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0

 x_cells=${X_CELLS}
 y_cells=${Y_CELLS}

 xmin=0.0
 ymin=0.0
 xmax=10.0
 ymax=10.0

 initial_timestep=0.04
 timestep_rise=1.5
 max_timestep=0.04
 end_time=${END_TIME}
 end_step=${END_STEP}
 test_problem 5
 tiles_per_chunk ${TILES_PER_CHUNK}

 use_cuda_kernels
*endclover
EOF

mpirun -np ${MPI_RANKS} --mca btl_openib_warn_no_device_params_found 0 --mca btl tcp,self,vader ./clover_leaf
