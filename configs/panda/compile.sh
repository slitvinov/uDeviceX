#!/bin/bash

cp configs/panda/Makefile mpi-dpd/.cache.Makefile
make cleanall         -C mpi-dpd
make -j slevel="-2" -C mpi-dpd


