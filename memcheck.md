# memory check

if `MEM` is set `udx` is ran with cuda-memcheck
and `MEM` is used as a list of parameters

    MEM= u.test test/*
	MEM='--leakcheck --blocking'              u.test test/*
	MEM='--tool initcheck'                    u.test test/*


if `VAL` is set `udx` is ran with valgrind and `VAL` is used as a list
of parameters.

    VAL= u.test test/*
    VAL="--leak-check=full --show-leak-kinds=all"  u.test test/*

`u.*` respect `DRYRUN` : only show the commands do not execute them

    DRYRUN= VAL=--option u.run ./udx
	
	module: load cray-hdf5-parallel cudatoolkit daint-gpu
    cmd: srun -n 1 -u valgrind --option ./udx 1 1 1

see also [poc/memcheck](poc/memcheck)
