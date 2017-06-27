namespace k_rbc {
  __device__   int  idx[MAX_VERT*RBCmd];
  __device__ float   ll[MAX_VERT*RBCmd];

__device__ float3 tri0(float3, float3, float3,
		       float, float, float, float);

__device__ float edge_len(int i, int j) { /* length of an edge  */
  i *= RBCmd;
  while (idx[i] != j) i++;
  return ll[i];
}

__device__ float heron(float a, float b, float c) {
  float s;
  s = (a+b+c)/2;
  return sqrt(s*(s-a)*(s-b)*(s-c));
}

__device__ float tri_area(int i1, int i2, int i3) {
  float a, b, c;
  a = edge_len(i1, i2);
  b = edge_len(i1, i3);
  c = edge_len(i2, i3);
  return heron(a, b, c);
}

__device__ float3 tri(float3 a, float3 b, float3 c,
		      int i1, int i2, int i3,
		      float area, float volume) {
  float l0, A0;
  A0 = tri_area(i1, i2, i3);  float Aref = RBCtotArea / (2.0 * RBCnv - 4.);
  l0 = edge_len(i1, i2);      float lref = sqrt(Aref * 4.0 / sqrt(3.0));
  
  if (i1 == 42) {
    printf("A: %g %g\n", A0, Aref);
    printf("l: %g %g\n", l0, lref);
  }
  return tri0(a, b, c,   l0, A0,   area, volume);
}
} /* namespace k_rbc */
