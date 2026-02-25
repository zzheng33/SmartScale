#!/bin/bash

PowerCap=$1
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 0 $PowerCap
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 1 $PowerCap
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 2 $PowerCap
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 3 $PowerCap

