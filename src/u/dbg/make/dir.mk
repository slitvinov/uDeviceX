D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d     && \
    d $B/dbg   && \
    d $B/mpi   && \
    d $B/u/dbg && \
    d $B/utils