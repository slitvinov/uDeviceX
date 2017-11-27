#include <mpi.h>

#include "conf.h"
#include "inc/conf.h"
#include "utils/mc.h"

#include "d/api.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "inc/type.h"
#include "inc/def.h"
#include "mpi/type.h"
#include "utils/os.h"
#include "utils/error.h"
#include "utils/efopen.h"

#include "imp.h"

namespace bop
{
void ini(Ticket *t) {
    if (m::rank == 0) os::mkdir(DUMP_BASE "/bop");
    const int L[3] = {XS, YS, ZS};        
    for (int c = 0; c < 3; ++c) t->mi[c] = (m::coords[c] + 0.5) * L[c];

    t->w_pp = new float[9*MAX_PART_NUM];
}

void fin(Ticket *t) {
    delete[] t->w_pp;
}

static void copy_shift(const Particle *pp, const long n, const int mi[3], /**/ float *w) {
    for (int j = 0; j < n; ++j)
    for (int d = 0; d < 3; ++d) {
        w[6 * j + d]     = pp[j].r[d] + mi[d];
        w[6 * j + 3 + d] = pp[j].v[d];
    }
}

static void copy_shift_with_forces(const Particle *pp, const Force *ff, const long n, const int mi[3], /**/ float *w) {
    for (int j = 0; j < n; ++j)
    for (int d = 0; d < 3; ++d) {
        w[9 * j + d]     = pp[j].r[d] + mi[d];
        w[9 * j + 3 + d] = pp[j].v[d];
        w[9 * j + 6 + d] = ff[j].f[d];
    }
}

#define PATTERN "%s-%05d"
    
static void header(const long n, const char *name, const int step, const char *type, const char *fields) {
    char fname[256] = {0};
    FILE *f;
    
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".bop", name, step / part_freq);
        
    UC(efopen(fname, "w", /**/ &f));

    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: " PATTERN ".values\n", name, step / part_freq);
    fprintf(f, "DATA_FORMAT: %s\n", type);
    fprintf(f, "VARIABLES: %s\n", fields);

    UC(efclose(f));
}

static void header_pp(const long n, const char *name, const int step) {
    header(n, name, step, "float", "x y z vx vy vz");
}

static void header_pp_ff(const long n, const char *name, const int step) {
    header(n, name, step, "float", "x y z vx vy vz fx fy fz");
}

static void header_ii(const long n, const char *name, const char *fields, const int step) {
    header(n, name, step, "int", fields);
}

static long write_data(const void *data, long n, size_t bytesperdata, MPI_Datatype datatype, const char *fname) {
    MPI_File f;
    MPI_Status status;
    MPI_Offset base, offset = 0;
    MPI_Offset len = n * bytesperdata;
    long ntot = 0;

    MC(m::Reduce(&n, &ntot, 1, MPI_LONG, MPI_SUM, 0, m::cart));
    MC(MPI_File_open(m::cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f));
    MC(MPI_File_set_size(f, 0));
    MC(MPI_File_get_position(f, &base)); 

    MC( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC( MPI_File_write_at_all(f, base + offset, data, n, datatype, &status) ); 
    MC( MPI_File_close(&f) );
    return ntot;
}
    
void parts(const Particle *pp, const long n, const char *name, const int step, Ticket *t) {
    copy_shift(pp, n, t->mi, /**/ t->w_pp);
        
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".values", name, step / part_freq);

    long ntot = write_data(t->w_pp, n, sizeof(Particle), datatype::particle, fname);
    if (m::rank == 0) header_pp(ntot, name, step);
}

void parts_forces(const Particle *pp, const Force *ff, const long n, const char *name, const int step, /*w*/ Ticket *t) {
    copy_shift_with_forces(pp, ff, n, t->mi, /**/ t->w_pp);
            
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".values", name, step / part_freq);

    long ntot = write_data(t->w_pp, n, sizeof(Particle) + sizeof(Force), datatype::partforce, fname);
    
    if (m::rank == 0) header_pp_ff(ntot, name, step);
}

static void intdata(const int *ii, const long n, const char *name, const char *fields, const int step) {
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".values", name, step / part_freq);

    long ntot = write_data(ii, n, sizeof(int), MPI_INT, fname);
    
    if (m::rank == 0) header_ii(ntot, name, fields, step);
}

void ids(const int *ii, const long n, const char *name, const int step) {
    intdata(ii, n, name, "id", step);
}

void colors(const int *ii, const long n, const char *name, const int step) {
    intdata(ii, n, name, "color", step);
}

#undef PATTERN
} // bop