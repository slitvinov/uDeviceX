/** Copyright 1993-2013 NVIDIA Corporation.  All rights reserved. **/

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(double b, float3 a)
{
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ double dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* yy := a*xx + yy */
inline __host__ __device__ void axpy(double a, float3 xx, float3 &yy) {
  yy.x += a * xx.x;
  yy.y += a * xx.y;
  yy.z += a * xx.z;
}
