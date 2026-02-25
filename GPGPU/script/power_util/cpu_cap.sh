#!/bin/bash


# Convert cap from Watts to microWatts and apply it

cap=$(( $1 * 5 / 10 ))


geopmwrite POWERCAP::CPU_POWER_LIMIT package 0 $cap
geopmwrite POWERCAP::CPU_POWER_LIMIT package 1 $cap


