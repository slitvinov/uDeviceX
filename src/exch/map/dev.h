namespace dev {

enum { MAX_DSTS = 7 };

__device__ int map_code(int3 L, const float r[3]) {
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_d2i(x, y, z);
}

__device__ int map_code_box(int3 L, float3 lo, float3 hi) {
    int x, y, z;
    x = -1 + (lo.x >= -L.x / 2) + (hi.x >= L.x / 2);
    y = -1 + (lo.y >= -L.y / 2) + (hi.y >= L.y / 2);
    z = -1 + (lo.z >= -L.z / 2) + (hi.z >= L.z / 2);

    return frag_d2i(x, y, z);
}

static __device__ void add_to_map(int soluteid, int pid, int fid, Map m) {
    int ientry, centry;
    centry = soluteid * NBAGS + fid;
    ientry = atomicAdd(m.counts + centry, 1);
    m.ids[fid][ientry] = pid;
}

static __device__ int add_faces(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            fids[j++] = frag_d32i(df);
        }
    }
    return j;
}

static __device__ int add_edges(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3])
            fids[j++] = frag_d32i(de);
    }
    return j;
}

static __device__ int add_cornr(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    if (d[X] && d[Y] && d[Z])
        fids[j++] = frag_d32i(d);
    return j;
}

__device__ int map_decode(int code, /**/ int fids[MAX_DSTS]) {
    int j = 0;
    const int d[3] = frag_i2d3(code);
    j = add_faces(j, d, /**/ fids);
    j = add_edges(j, d, /**/ fids);
    j = add_cornr(j, d, /**/ fids);
    return j;
}

static __device__ void warpexscan(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt;
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scan2d(const int *counts, const int *oldtcounts, /**/ int *tcounts, int *starts) {
    int t, cnt, newcnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        newcnt = cnt + oldtcounts[t];
        tcounts[t] = newcnt;
    }
    if (starts) warpexscan(cnt, t, /**/ starts);
}

__global__ void scan1d(const int *count, /**/ int *starts) {
    int t, cnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) cnt = count[t];
    if (starts) warpexscan(cnt, t, /**/ starts);
}

} // dev