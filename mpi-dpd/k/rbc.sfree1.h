namespace k_rbc {
  __device__   int  idx[MAX_VERT*RBCmd];
  __device__ float   ll[MAX_VERT*RBCmd];

__device__ float3 tri0(float3, float3, float3,
		       float, float, float, float);

__device__ float edge_len(int i, int j) {
  i *= RBCmd;
  while (idx[i] != j) i++;
  return ll[i];
}

__device__ float3 tri(float3 a, float3 b, float3 c,
		      int i1, int i2, int i3,
		      float area, float volume) {
  float l0, A0;
  l0 = edge_len(i1, i2);
  A0 = RBCtotArea / (2.0 * RBCnv - 4.);
  return tri0(a, b, c,   l0, A0,   area, volume);
}

}

