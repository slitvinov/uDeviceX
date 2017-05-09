#include <mpi.h>
#include <sys/resource.h>
#include <utility>
#include <cell-lists.h>
#include <cuda-dpd.h>
#include <cstdio>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "m.h"     /* MPI */
#include "common.h"
#include "common.tmp.h"

bool Particle::initialized = false;
MPI_Datatype Particle::mytype;

void CellLists::build(Particle * const p, const int n, int * const order, const Particle * const src) {
    if (n > 0)
      build_clists_vanilla((float * )p, n, 1,
			   LX, LY, LZ,
			   -LX/2, -LY/2, -LZ/2,
			   order, start, count,  NULL, (float *)src);
    else {
        CC(cudaMemsetAsync(start, 0, sizeof(int) * ncells));
        CC(cudaMemsetAsync(count, 0, sizeof(int) * ncells));
    }
}

void diagnostics(Particle * particles, int n, int idstep) {
    enum {X, Y, Z};

    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 3; ++c)
            p[c] += particles[i].v[c];

    MC(MPI_Reduce(m::rank == 0 ? MPI_IN_PLACE : &p,
		  m::rank == 0 ? &p : NULL, 3,
		  MPI_DOUBLE, MPI_SUM, 0, m::cart) );
    double ke = 0;
    for(int i = 0; i < n; ++i)
        ke += pow(particles[i].v[0], 2) + pow(particles[i].v[1], 2) + pow(particles[i].v[2], 2);

    MC(MPI_Reduce(m::rank == 0 ? MPI_IN_PLACE : &ke,
		  &ke,
		  1, MPI_DOUBLE, MPI_SUM, 0, m::cart));
    MC(MPI_Reduce(m::rank == 0 ? MPI_IN_PLACE : &n,
		  &n, 1, MPI_INT, MPI_SUM, 0, m::cart));

    double M = 0, I = 0;
    for (int i = 0; i < n; ++i) {
        float *r = particles[i].r;
        float *v = particles[i].v;
        M += -r[X]*v[Z] + r[Z]*v[X];
        I += r[X]*r[X] + r[Z]*r[Z];
    }

    double kbt = 0.5 * ke / (n * 3. / 2);
    if (m::rank == 0) {
        static bool firsttime = true;
        FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
        firsttime = false;
        if (idstep == 0)
            fprintf(f, "# TSTEP\tKBT\tPX\tPY\tPZ\tM\tI\n");
#define FMT "%e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e"
        printf("\x1b[91m timestep: " FMT "\x1b[0m\n", idstep * dt, kbt, p[0], p[1], p[2], M, I);
        fprintf(f, FMT "\n", idstep * dt, kbt, p[0], p[1], p[2], M, I);
        fclose(f);
#undef FMT
    }
}
