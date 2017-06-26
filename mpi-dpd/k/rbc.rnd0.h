namespace k_rbc {
__device__ curandState rrnd[MAX_RND_NUM];

__global__ void rnd_ini() {
  return;
}

__dfi__ float3 frnd(float3 r1, float3 r2, int i1, int i2) {
  return make_float3(0, 0, 0);
}
} /* namespace k_rbc */
