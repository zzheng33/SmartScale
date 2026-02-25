#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <freq_mhz> [gpu_index]"
    echo "  freq_mhz: GPU max core frequency in MHz (e.g., 1200)"
    echo "  gpu_index: optional single GPU index (0-3). Default: apply to all GPUs 0-3"
    exit 1
fi

FREQ_MHZ="$1"
if ! [[ "$FREQ_MHZ" =~ ^[0-9]+$ ]]; then
    echo "Error: freq_mhz must be a positive integer in MHz."
    exit 1
fi

FREQ_HZ=$((FREQ_MHZ * 1000000))

if [[ $# -eq 2 ]]; then
    TARGET_GPUS=("$2")
else
    TARGET_GPUS=(0 1 2 3)
fi

for GPU_IDX in "${TARGET_GPUS[@]}"; do
    geopmwrite NVML::GPU_CORE_FREQUENCY_MAX_CONTROL gpu "$GPU_IDX" "$FREQ_HZ"
    echo "Set GPU ${GPU_IDX} max frequency control to ${FREQ_MHZ} MHz (${FREQ_HZ} Hz)"
done
