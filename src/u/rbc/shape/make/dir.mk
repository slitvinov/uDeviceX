D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/conf             && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/math/linal       && \
    d $B/math/rnd         && \
    d $B/math/tform       && \
    d $B/math/tri         && \
    d $B/mpi              && \
    d $B/rbc/adj          && \
    d $B/rbc/shape        && \
    d $B/u/rbc/shape      && \
    d $B/utils            && \
    d $B/utils/nvtx       && \
    d $B/utils/string    
