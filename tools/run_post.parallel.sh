#!/bin/bash

d=$1
m1=$2 # M for shear 1

(
cd $d
dt=`awk '$1 == "dt"             {print $2}' params.txt`
sh=`awk '$1 == "gamma_dot"      {print $2}' params.txt`
st=`awk '$1 == "steps_per_dump" {print $2}' params.txt`
gc=`awk '$1 == "RBCgammaC"      {print $2}' params.txt`
m2=`sh get_sc.sh diag.txt`

if [ -z $m1 ]; then
    shm=$sh
else
    shm=`awk -v sh=$sh -v m1=$m1 -v m2=$m2 'BEGIN {print m2/m1}'`
fi

dsh=`awk -v s1=$sh -v s2=$shm 'BEGIN {print (s1-s2)/s1*100}'`
echo "Processing run $d with parameters $gc $shm ($dsh%)"
post.py --dt=$dt --ply=ply --sh=$shm --st=$st
)
