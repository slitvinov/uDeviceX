namespace k_rex {
__device__ void scan_pad(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt = 32 * ((cnt + 31) / 32);
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scanA(const int *counts, const int *oldtcounts, /**/ int *tcounts, int *starts) {
    int t, cnt, newcount, scan, L;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        if (cnt > g::capacities[t]) g::failed = true;
        if (tcounts && oldtcounts) {
            newcount = cnt + oldtcounts[t];
            tcounts[t] = newcount;
            if (newcount > g::capacities[t]) g::failed = true;
        }
    }

    if (starts) {
        scan = cnt = 32 * ((cnt + 31) / 32);
        for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
        if (t < 27) starts[t] = scan - cnt;
    }
}

__global__ void scanB(const int *count, /**/ int *start) {
    int t, cnt, scan, L;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = count[t];
        if (cnt > g::capacities[t]) g::failed = true;
    }
    if (start) {
        scan = cnt = 32 * ((cnt + 31) / 32);
        for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
        if (t < 27) start[t] = scan - cnt;
    }
}

}
