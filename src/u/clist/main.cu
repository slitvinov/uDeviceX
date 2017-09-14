#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h" /* mini-MPI and -device */
#include "d/api.h"

#include "glb.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "utils/cc.h"

#include "algo/scan/int.h"
#include "clistx/imp.h"

enum {X,Y,Z};

#define MAXN 10000

void ini_1ppc(int3 d, int *n, Particle *pp) {
    int i, ix, iy, iz;
    Particle p;
    *n = d.x * d.y * d.z;

    for (iz = 0; iz < d.z; ++iz)
        for (iy = 0; iy < d.y; ++iy)
            for (ix = 0; ix < d.x; ++ix) {
                p.r[X] = (-d.x + ix) * 0.5f + 0.5f;
                p.r[Y] = (-d.y + iy) * 0.5f + 0.5f;
                p.r[Z] = (-d.z + iz) * 0.5f + 0.5f;
                p.v[X] = p.v[Y] = p.v[Z] = 0.f;
                i = ix + d.x * (iy + d.z * iz);
                pp[i] = p;
            }
}

void verify(int3 dims, const int *starts, const int *counts, const Particle *pp, int n) {
    
}

int main(int argc, char **argv) {
    m::ini(argc, argv);

    Particle *pp, *ppout;
    Particle *pp_hst;
    int n = 0, *starts, *counts;
    int3 dims = make_int3(4, 4, 2);
    clist::Clist clist;
    clist::Work work;

    ini(dims.x, dims.y, dims.z, /**/ &clist);
    ini_work(&clist, /**/ &work);

    pp_hst = (Particle*) malloc(MAXN * sizeof(Particle));
    CC(d::Malloc((void**) &pp, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &ppout, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &counts, clist.ncells * sizeof(Particle)));
    CC(d::Malloc((void**) &starts, clist.ncells * sizeof(Particle)));
       
    ini_1ppc(dims, /**/ &n, pp_hst);
    CC(d::Memcpy(pp, pp_hst, n * sizeof(Particle), H2D));

    build(n, n, pp, /**/ ppout, &clist, /*w*/ &work);

    CC(d::Memcpy(counts, clist.counts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(starts, clist.starts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(pp_hst, ppout, n * sizeof(Particle), D2H));

    verify(dims, starts, counts, ppout, n);
    

    CC(d::Free(pp));
    CC(d::Free(ppout));
    CC(d::Free(counts));
    CC(d::Free(starts));
    free(pp_hst);

    fin(/**/ &clist);
    fin_work(/**/ &work);

    
    m::fin();
}
