static __global__ void main(const tex3Dca texsdf, int n, const Particle *pp, /**/ int *labels) {
    enum {X, Y, Z};
    int pid;
    Particle p;
    float s;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    p = pp[pid];
    s = sdf(texsdf, p.r[X], p.r[Y], p.r[Z]);
    labels[pid] =
        s > 2 ? DEEP :
        s >=0 ? WALL :
                BULK;
}