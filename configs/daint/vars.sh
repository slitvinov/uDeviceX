# parameters of the simulations

# this is linked to mpi-dpd/common.h
# and should be changed simultaneously
XSIZE_SUBDOMAIN=32
YSIZE_SUBDOMAIN=32
ZSIZE_SUBDOMAIN=32

xranks=1
yranks=1
zranks=1
tend=250

args_wall="-walls -wall_creation_stepid=1000"
args_dumps="-steps_per_dump=2000 -hdf5field_dumps  -hdf5part_dumps"
args="$xranks $yranks $zranks -rbcs -tend=$tend $args_wall $args_dumps"

echo "args: $args"
