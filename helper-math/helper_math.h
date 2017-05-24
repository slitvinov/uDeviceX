/** Copyright 1993-2013 NVIDIA Corporation.  All rights reserved. **/

#define IHD inline __host__ __device__
#define mf3(a, b, c) make_float3((a), (b), (c))

IHD double dmult (double a, double b)           { return a * b; }
IHD double dplus (double a, double b)           { return a + b; }
IHD double dplus3(double a, double b, double c) { return a + b + c; }
IHD double dminus(double a, double b)           { return a - b; }

IHD float3 operator+(float3 a, float3 b)
{
  return mf3(a.x + b.x, a.y + b.y, a.z + b.z);
}

IHD void operator+=(float3 &a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

IHD float3 operator-(float3 a, float3 b)
{
  return mf3(a.x - b.x, a.y - b.y, a.z - b.z);
}

IHD float3 operator*(double b, float3 a)
{
  return mf3(b * a.x, b * a.y, b * a.z);
}

IHD double dot(float3 a, float3 b)
{
  return \
    a.x * b.x +
    a.y * b.y +
    a.z * b.z;
}

/* yy := a*xx + yy */
IHD void axpy(double a, float3 xx, float3 &yy) {
  yy.x += a * xx.x;
  yy.y += a * xx.y;
  yy.z += a * xx.z;
}

#undef mf3
#undef IHD
