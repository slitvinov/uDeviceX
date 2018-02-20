struct RbcParams;
struct RbcForce;
struct Force;

void rbc_force_ini(const MeshRead *cell, int seed, RbcForce**);
void rbc_force_fin(RbcForce*);

void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce*);
void rbc_force_set_stressfree(const char *fname, /**/ RbcForce*);

void rbc_force_set_rnd0(RbcForce *f);
void rbc_force_set_rnd1(RbcForce *f);

void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*);
void rbc_force_stat(/**/ float *pArea, float *pVolume);
