#!/bin/bash

# Run from this directory:
#  > atest test.rbc.sh
#
# To update the test change nTEST to cTEST and run
#  > atest test.rbc.sh
# add files from test_data/* to git

#### RBC with wall
# TEST: rbc.t1
# export PATH=../tools:$PATH
# cp sdf/wall1/wall.dat sdf.dat
# x=6 y=6 z=6; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# cp .rbc.rbc.h params/rbc.inc0.h
# cp .conf.rbc.h .conf.h
# rm -rf ply h5 diag.txt
# (make clean && make -j && make -C ../tools) > /dev/null
#  ./test 
# ply2punto ply/rbcs-00009.ply | uscale 1 > ply.out.txt
