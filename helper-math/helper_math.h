/** Copyright 1993-2013 NVIDIA Corporation.  All rights reserved. **/

#define IHD inline __host__ __device__
#define mf3(a, b, c) make_float3((a), (b), (c))

IHD float3 operator+(float3 a, float3 b) {
  double ax = a.x, ay = a.y, az = a.z;
  double bx = b.x, by = b.y, bz = b.z;
  return mf3(ax + bx, ay + by, az + bz);
}

IHD void operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

IHD float3 operator-(float3 a, float3 b) {
  double ax = a.x, ay = a.y, az = a.z;
  double bx = b.x, by = b.y, bz = b.z;
  return mf3(ax - bx, ay - by, az - bz);
}

IHD float3 operator*(double b, float3 a) {
  double ax = a.x, ay = a.y, az = a.z;
  return mf3(b * ax, b * ay, b * az);
}

IHD double dot(float3 a, float3 b) {
  double ax = a.x, ay = a.y, az = a.z;
  double bx = b.x, by = b.y, bz = b.z;
  return ax*bx + ay*by + az*bz;
}

IHD float3 cross(float3 a, float3 b) {
  double ax = a.x, ay = a.y, az = a.z;
  double bx = b.x, by = b.y, bz = b.z;
  return mf3(ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx);
}

#undef mf3
#undef IHD
