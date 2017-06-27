namespace k_rbc {
__device__ float3 tri0(float3, float3, float3,
		       float, float, float, float);
__device__ float3 tri(float3 a, float3 b, float3 c,
		      int i1, int i2, int i3,
		      float area, float volume) {
  float l0, A0;
  A0 = RBCtotArea / (2.0 * RBCnv - 4.);
  l0 = sqrt(A0 * 4.0 / sqrt(3.0));
  return tri0(a, b, c,   l0, A0,   area, volume);
}
}
