static __device__ bool valid(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

__global__ void sample(int3 L, const int *__restrict__ cellsstart, const int *__restrict__ cellscount, __restrict__ const float2 *pp, /**/ float3 *gridv) {
    const int3 c = make_int3(threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z);

    if (valid(L, c)) {
        const int cid = get_cid(L, c);
        const float num = cellscount[cid];
        
        for (int pid = cellsstart[cid]; pid < cellsstart[cid] + num; pid++) {
            float2 tmp1 = pp[3*pid + 1];
            float2 tmp2 = pp[3*pid + 2];
            gridv[cid].x += tmp1.y / num;
            gridv[cid].y += tmp2.x / num;
            gridv[cid].z += tmp2.y / num;
        }
    }
}