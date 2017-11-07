#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "bop_common.h"
#include "bop_reader.h"
extern "C" {
#include "bov.h"
}

struct Args {
    int lx, ly, lz;
    char *bop_s, *bop_c, *bov;
};

static void usg() {
    fprintf(stderr, "usg: u.color.density Lx Ly Lz <solvent.bop> <colors.bop> <out.bov>\n");
    exit(1);
}

static void parse(int argc, char **argv, /**/ Args *a) {
    if (argc != 7) usg();
    int iarg = 1;
    a->lx = atoi(argv[iarg++]);
    a->ly = atoi(argv[iarg++]);
    a->lz = atoi(argv[iarg++]);
    a->bop_s = argv[iarg++];
    a->bop_c = argv[iarg++];
    a->bov   = argv[iarg++];
}

static void p2m_1cid(float wx, float wy, float wz, int ix, int iy, int iz, int lx, int ly, int lz, /**/ float *grid) {
    int i;
    ix = (ix + lx) % lx;
    iy = (iy + ly) % ly;
    iz = (iz + lz) % lz;
    i = ix + lx * (iy + ly * iz);
    grid[i] += wx * wy * wz;
}

static void collect_p2m(long n, const float *pp, const int *cc, int lx, int ly, int lz, /**/ float *grid) {
    enum {X, Y, Z};
    long i, ix, iy, iz;
    float x, y, z;
    const float *r;

    memset(grid, 0, lx*ly*lz*sizeof(float));

    for (i = 0; i < n; ++i) {
        if (cc[i] == 0) continue;

        r = pp + 6 * i;

        ix = (int) r[X];
        iy = (int) r[Y];
        iz = (int) r[Z];

        x = r[X] - ix;
        y = r[Y] - iy;
        z = r[Z] - iz;

        p2m_1cid(1.f - x, 1.f - y, 1.f - z,     ix    , iy    , iz    ,    lx, ly, lz, /**/ grid);
        p2m_1cid(      x, 1.f - y, 1.f - z,     ix + 1, iy    , iz    ,    lx, ly, lz, /**/ grid);
        p2m_1cid(1.f - x,       y, 1.f - z,     ix    , iy + 1, iz    ,    lx, ly, lz, /**/ grid);
        p2m_1cid(      x,       y, 1.f - z,     ix + 1, iy + 1, iz    ,    lx, ly, lz, /**/ grid);

        p2m_1cid(1.f - x, 1.f - y,       z,     ix    , iy    , iz + 1,    lx, ly, lz, /**/ grid);
        p2m_1cid(      x, 1.f - y,       z,     ix + 1, iy    , iz + 1,    lx, ly, lz, /**/ grid);
        p2m_1cid(1.f - x,       y,       z,     ix    , iy + 1, iz + 1,    lx, ly, lz, /**/ grid);
        p2m_1cid(      x,       y,       z,     ix + 1, iy + 1, iz + 1,    lx, ly, lz, /**/ grid);
    }
}

int main(int argc, char **argv) {
    Args a;
    BopData bop_s, bop_c;
    BovDesc bov;
    float *grid;

    parse(argc, argv, /**/ &a);

    grid = (float*) malloc(a.lx * a.ly * a.lz * sizeof(float));
        
    init(&bop_s);
    init(&bop_c);

    read(a.bop_s, /**/ &bop_s);
    read(a.bop_c, /**/ &bop_c);

    collect_p2m(bop_s.n, bop_s.fdata, bop_c.idata, a.lx, a.ly, a.lz, /**/ grid);    
    
    bov.nx = a.lx; bov.ny = a.ly; bov.nz = a.lz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = 0;    bov.oy = 0;    bov.oz = 0;
    bov.data = grid;
    sprintf(bov.var, "color density");
    bov.ncmp = 1;    

    write_bov(a.bov, &bov);
    
    free(grid);
    
    finalize(&bop_s);
    finalize(&bop_c);

    return 0;
}
