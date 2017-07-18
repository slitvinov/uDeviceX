namespace k_fsi {

struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
};

static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }
static __device__ void p2rv(const float2 *p, int i, /**/
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += 3*i;
    s0 = __ldg(p++); s1 = __ldg(p++); s2 = __ldg(p++);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa pp2p(float2 *pp, int i) {
    Pa p;
    p2rv(pp, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ void tex2rv(int i,
                              float  *x, float  *y, float  *z,
                              float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    i *= 3;
    s0 = tex1Dfetch(texSolventParticles, i++);
    s1 = tex1Dfetch(texSolventParticles, i++);
    s2 = tex1Dfetch(texSolventParticles, i++);

     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa tex2p(int i) {
    Pa p;
    tex2rv(i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ void pair(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    float3 r1 = make_float3( l.x,  l.y,  l.z), r2 = make_float3( r.x,  r.y,  r.z);
    float3 v1 = make_float3(l.vx, l.vy, l.vz), v2 = make_float3(r.vx, r.vy, r.vz);
    float3  f = force(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

static __device__ void bulk0(float2 *pp, int pid, int zplane, int n, float seed, float *ff0, float *ff1) {
    Map m;
    Pa l, r; /* "local" and "remote" particles */
    float x, y, z;
    l = pp2p(pp, pid);
    x = l.x; y = l.y; z = l.z;
    if (!p2map(zplane, n, x, y, z, /**/ &m)) return;
    float xforce = 0, yforce = 0, zforce = 0;
    for (int i = 0; !endp(m, i); ++i) {
        const int spid = m2id(m, i);
        r = tex2p(spid);
        const int sentry = 3 * spid;

        const float myrandnr = l::rnd::d::mean0var1ii(seed, pid, spid);
        float3 pos1 = make_float3(x, y, z), pos2 = make_float3(r.x, r.y, r.z);
        float3 vel1 = make_float3(l.vx, l.vy, l.vz), vel2 = make_float3(r.vx, r.vy, r.vz);
        const float3 strength = force(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2, vel1, vel2, myrandnr);
        const float xinteraction = strength.x;
        const float yinteraction = strength.y;
        const float zinteraction = strength.z;

        xforce += xinteraction;
        yforce += yinteraction;
        zforce += zinteraction;

        atomicAdd(ff1 + sentry, -xinteraction);
        atomicAdd(ff1 + sentry + 1, -yinteraction);
        atomicAdd(ff1 + sentry + 2, -zinteraction);
    }

    atomicAdd(ff0 + 3 * pid + 0, xforce);
    atomicAdd(ff0 + 3 * pid + 1, yforce);
    atomicAdd(ff0 + 3 * pid + 2, zforce);
}

__global__ void bulk(float2 *pp, int n0, int n1, float seed, float *ff0, float *ff1) {
    int gid, pid, zplane;
    gid    = threadIdx.x + blockDim.x * blockIdx.x;
    pid    = gid / 3;
    zplane = gid % 3;
    if (pid >= n0) return;
    bulk0(pp, pid, zplane, n1, seed, ff0, ff1);
}
}
