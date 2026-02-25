#!/bin/bash

home_dir=$HOME
benchmark_dir="${home_dir}/benchmark/Hec/relu/"

"$benchmark_dir/main" ./main 10000000 100000
