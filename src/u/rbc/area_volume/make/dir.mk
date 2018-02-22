D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords                && \
    d $B/d                     && \
    d $B/io/bop                && \
    d $B/io/com                && \
    d $B/io/diag/mesh          && \
    d $B/io/diag/part          && \
    d $B/io/field              && \
    d $B/io/field/h5           && \
    d $B/io/field/xmf          && \
    d $B/io/mesh               && \
    d $B/io/mesh/write         && \
    d $B/io/mesh_read          && \
    d $B/io/restart            && \
    d $B/io/rig                && \
    d $B/io/txt                && \
    d $B/math/linal            && \
    d $B/math/rnd              && \
    d $B/math/tform            && \
    d $B/mpi                   && \
    d $B/parser                && \
    d $B/rbc                   && \
    d $B/rbc/adj               && \
    d $B/rbc/adj/edg           && \
    d $B/rbc/com               && \
    d $B/rbc/force             && \
    d $B/rbc/force/area_volume && \
    d $B/rbc/force/rnd         && \
    d $B/rbc/force/rnd/api     && \
    d $B/rbc/gen               && \
    d $B/rbc/params            && \
    d $B/rbc/shape             && \
    d $B/rbc/stretch           && \
    d $B/u/rbc/area_volume     && \
    d $B/utils                
