#pragma once

#include <cuda_runtime.h>
#include "support.cuh"

__host__ __device__ inline
fptype norm(fptype3 in)
{
  return sqrt(in.x * in.x + in.y * in.y + in.z * in.z);
}

__host__ __device__ inline
fptype3 normalize(fptype3 in)
{
  auto normVal = norm(in);
  in.x /= normVal;
  in.y /= normVal;
  in.z /= normVal;
  return in;
}

__host__ __device__ inline
fptype dot(fptype3 a, fptype3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
fptype3 operator+(fptype3 a, fptype3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__host__ __device__ inline
fptype3& operator+=(fptype3& a, fptype3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__host__ __device__ inline
fptype3 operator-(fptype3 a, fptype3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__host__ __device__ inline
fptype3 operator*(fptype c, fptype3 v)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

__host__ __device__ inline
fptype3 operator*(fptype3 v, fptype c)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

__host__ __device__ inline
void linspace(fptype* out, fptype from, fptype to, std::size_t n)
{
  auto step = (to - from) / static_cast<fptype>(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
  {
    out[i] = from + static_cast<fptype>(i) * step;
  }
  out[n - 1] = to; // Don't rely on FP arithmetic to reach 'to'
}
