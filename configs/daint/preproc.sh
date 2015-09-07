#!/bin/bash

# setup one RBC
cp configs/initial-conditions/rbc1-ic.txt mpi-dpd/rbcs-ic.txt

# generate an sdf file
device-gen/sdf-tiny/tsdf.sh \
    configs/ywall2.tsdf \
    mpi-dpd/sdf.dat mpi-dpd/sdf.vtk
