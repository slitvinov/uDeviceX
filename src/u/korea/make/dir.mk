D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/minmax              && \
    d $B/algo/scan                && \
    d $B/clist                    && \
    d $B/cnt                      && \
    d $B/color                    && \
    d $B/comm                     && \
    d $B/control/den              && \
    d $B/control/inflow           && \
    d $B/control/outflow          && \
    d $B/control/vel              && \
    d $B/d                        && \
    d $B/dbg                      && \
    d $B/distr/flu                && \
    d $B/distr/rbc                && \
    d $B/distr/rig                && \
    d $B/exch/flu                 && \
    d $B/exch/mesh                && \
    d $B/exch/obj                 && \
    d $B/flu                      && \
    d $B/fluforces                && \
    d $B/fluforces/bulk           && \
    d $B/fluforces/bulk/transpose && \
    d $B/fluforces/halo           && \
    d $B/frag                     && \
    d $B/fsi                      && \
    d $B/generate/rig             && \
    d $B/glob                     && \
    d $B/inter                    && \
    d $B/io                       && \
    d $B/io/bop                   && \
    d $B/io/field                 && \
    d $B/io/field/h5              && \
    d $B/io/field/xmf             && \
    d $B/io/mesh                  && \
    d $B/io/mesh/write            && \
    d $B/math/linal               && \
    d $B/math/rnd                 && \
    d $B/mesh                     && \
    d $B/meshbb                   && \
    d $B/mpi                      && \
    d $B/parser                   && \
    d $B/rbc/adj                  && \
    d $B/rbc/com                  && \
    d $B/rbc/edg                  && \
    d $B/rbc/force                && \
    d $B/rbc/force/area_volume    && \
    d $B/rbc/gen                  && \
    d $B/rbc/main                 && \
    d $B/rbc/main/anti            && \
    d $B/rbc/rnd                  && \
    d $B/rbc/rnd/api              && \
    d $B/rbc/stretch              && \
    d $B/rig                      && \
    d $B/rigid                    && \
    d $B/scheme/force             && \
    d $B/scheme/move              && \
    d $B/scheme/restrain          && \
    d $B/scheme/restrain/sub      && \
    d $B/scheme/restrain/sub/stat && \
    d $B/scheme/restrain/sub/sum  && \
    d $B/sdf                      && \
    d $B/sdf/array3d              && \
    d $B/sdf/bounce               && \
    d $B/sdf/field                && \
    d $B/sdf/label                && \
    d $B/sim                      && \
    d $B/utils                    && \
    d $B/wall                     && \
    d $B/wall/exch                && \
    d $B/wall/force               && \
    d $B/wvel                    
