namespace bop {
struct Ticket {
    float *w_pp;    // particle workspace
    int mi[3];      // global coordinates of my originx
};

void ini(Ticket *t);
void fin(Ticket *t);

void parts(MPI_Comm cart, const Particle *pp, long n, const char *name, int step, /*w*/ Ticket *t);
void parts_forces(MPI_Comm cart, const Particle *pp, const Force *ff, long n, const char *name, int step, /*w*/ Ticket *t);
void ids(MPI_Comm cart, const int *ii, long n, const char *name, int step);
void colors(MPI_Comm cart, const int *ii, long n, const char *name, int step);
}