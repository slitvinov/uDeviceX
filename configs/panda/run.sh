#!/bin/bash

cd mpi-dpd
./test 1 1 1 -walls -wall_creation_stepid=1000 \
       -rbcs -hdf5part_dumps -steps_per_dump=2000 \
       -tend=500 -hdf5field_dumps -xyz_dumps
    
