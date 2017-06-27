#### strached RBC without solvent
# nTEST: rbc.t1
# rm -rf ply h5 diag.txt
# :
# export PATH=../tools:$PATH
# :
# x=6 y=6 z=6; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# :
# cp rbc.r1.dat        rbc.dat
# cp .rbc.rbc.h        params/rbc.inc0.h
# cp conf/rbc.h        .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
#  ./test 
# ply2punto ply/rbcs-00009.ply | uscale 1 > ply.out.txt
#
