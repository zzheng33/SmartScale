#!/bin/bash

# TeaLeaf Benchmark Run Script
# Usage: ./run_tealeaf.sh [MPI_RANKS]
# Example: ./run_tealeaf.sh 4

MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified

cd /home/ac.zzheng/benchmark/spec/TeaLeaf

# Configure parameters
X_CELLS=10000
Y_CELLS=10000
END_STEP=1
MAX_ITERS=10000
INITIAL_TIMESTEP=0.004
EPS=1.0e-15

# Generate tea.in input file
cat > tea.in << EOF
*tea
state 1 density=100.0 energy=0.0001
state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0
state 3 density=0.1 energy=0.1 geometry=rectangle xmin=1.0 xmax=6.0 ymin=1.0 ymax=2.0
state 4 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=6.0 ymin=1.0 ymax=8.0
state 5 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=10.0 ymin=7.0 ymax=8.0
x_cells=${X_CELLS}
y_cells=${Y_CELLS}
xmin=0.0
ymin=0.0
xmax=10.0
ymax=10.0
initial_timestep=${INITIAL_TIMESTEP}
end_step=${END_STEP}
max_iters=${MAX_ITERS}
use_cg
eps ${EPS}
test_problem 5
profiler_on
use_c_kernels
*endtea
EOF

mpirun -np ${MPI_RANKS} --mca btl_openib_warn_no_device_params_found 0 --mca btl tcp,self,vader ./build/cuda-tealeaf
