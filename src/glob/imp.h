void ini(MPI_Comm cart, Coords *c);
void fin(Coords *c);

/* domain sizes */

int xdomain(const Coords c);
int ydomain(const Coords c);
int zdomain(const Coords c);

/* [l]ocal to [c]enter */

float xl2xc(const Coords c, float xl);
float yl2yc(const Coords c, float yl);
float zl2zc(const Coords c, float zl);
void local2center(Coords c, float3 rl, /**/ float3 *rc);

/* [c]enter to [l]ocal  */

float xc2xl(const Coords c, float xc);
float yc2yl(const Coords c, float yc);
float zc2zl(const Coords c, float zc);
void center2local(Coords c, float3 rc, /**/ float3 *rl);

/* [l]ocal to [g]lobal */

float xl2xg(const Coords c, float xl);
float yl2yg(const Coords c, float yl);
float zl2zg(const Coords c, float zl);
void local2global(const Coords c, float3 rl, /**/ float3 *rg);

/* [g]lobal to [l]ocal */

float xg2xl(const Coords c, float xg);
float yg2yl(const Coords c, float yg);
float zg2zl(const Coords c, float zg);
void global2local(const Coords c, float3 rg, /**/ float3 *rl);