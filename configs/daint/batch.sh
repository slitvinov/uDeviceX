#!/bin/bash

set -eu

# run this script from `configs' directory
# run all jobs in cart.dl.daint
#gitroot=${HOME}/ctc-stretching
gitroot=${HOME}/uDeviceX/litvinov/couette/wip

./alcartesio.awk cart.daint | \
    sh -x ./aldriver.sh "$gitroot"
