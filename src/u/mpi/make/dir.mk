D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d        && \
    d $B/glb      && \
    d $B/glb/gdot && \
    d $B/mpi      && \
    d $B/u/mpi    && \
    d $B/utils   