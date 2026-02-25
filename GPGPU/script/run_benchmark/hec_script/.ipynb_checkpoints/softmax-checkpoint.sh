#!/bin/bash

home_dir=$HOME
benchmark_dir="${home_dir}/benchmark/Hec/softmax/"

"$benchmark_dir/main" 10000 100000 1
