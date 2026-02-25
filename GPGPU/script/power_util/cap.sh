#!/bin/bash
pkill -f geopmwrite
CPU_W=$1
GPU_W=$2

# Convert CPU Watts → microWatts (GEOPM expects µW)
CPU_CAP=$(( CPU_W * 5 / 10 ))

# Apply CPU caps
geopmwrite POWERCAP::CPU_POWER_LIMIT package 0 $CPU_CAP
geopmwrite POWERCAP::CPU_POWER_LIMIT package 1 $CPU_CAP

# Apply GPU caps
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 0 $GPU_W
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 1 $GPU_W
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 2 $GPU_W
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 3 $GPU_W
