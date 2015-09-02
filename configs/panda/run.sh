#!/bin/bash

mpi-dpd/test \
    1 1 1 -walls \
    -wall_creation_stepid 1 \
    -hdf5part_dumps -xyz_dumps

