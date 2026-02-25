#!/bin/bash


# Convert watts to microwatts
cap_uw=$(($1 * 500000))

# Path to e_smi_tool
ESMI_TOOL="/home/cc/esmi_ib_library/build/e_smi_tool"

# Set power limit for socket 0 and 1
sudo $ESMI_TOOL --setpowerlimit 0 $cap_uw
sudo $ESMI_TOOL --setpowerlimit 1 $cap_uw
