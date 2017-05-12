#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh
#
# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

#### RBC with wall
# cTEST: rbc.t1
# export PATH=../tools:$PATH
# cp sdf/wall1/wall.dat sdf.dat
# cp .rbcs-ic.txt rbcs-ic.txt
# cp .rbc.rbc.h params/rbc.inc0.h
# cp .conf.rbc.h .conf.h
# rm -rf ply h5 diag.txt
# (make clean && make -j && make -C ../tools) > /dev/null
# srun ./test 
# ply2punto ply/rbcs-00009.ply | uscale 1 > ply.out.txt
