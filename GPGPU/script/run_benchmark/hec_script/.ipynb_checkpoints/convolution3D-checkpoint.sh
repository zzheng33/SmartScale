#!/bin/bash

home_dir=$HOME
benchmark_dir="${home_dir}/benchmark/Hec/convolution3D/"

"$benchmark_dir/main" 32 64 128 56 56 3 100
