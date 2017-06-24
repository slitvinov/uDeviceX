#!/bin/bash

d=$1
m1=$2 # M for shear 1

(
cd $d
dt=`awk '$1 == "dt"             {print $2}' params.txt`
sh=`awk '$1 == "gamma_dot"      {print $2}' params.txt`
st=`awk '$1 == "steps_per_dump" {print $2}' params.txt`
dtf=`awk -v dt=$dt -v st=$st 'BEGIN {print dt*st}'`

if [ -z $m1 ]; then
    shm=$sh
else
    m2=`sh get_sc.sh diag.txt`
    shm=`awk -v sh=$sh -v m1=$m1 -v m2=$m2 'BEGIN {print m2/m1}'`
fi

dsh=`awk -v s1=$sh -v s2=$shm 'BEGIN {print (s1-s2)/s1*100}'`
echo "Processing run $d with shear $shm ($dsh%)"
post.py --dtf=$dtf --ply=ply --sh=$shm
)
