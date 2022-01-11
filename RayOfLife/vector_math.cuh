#pragma once

#include <cuda_runtime.h>

__host__ __device__ inline
float norm(float3 in)
{
  return sqrtf(in.x * in.x + in.y * in.y + in.z * in.z);
}

__host__ __device__ inline
float3 normalize(float3 in)
{
  auto normVal = norm(in);
  in.x /= normVal;
  in.y /= normVal;
  in.z /= normVal;
  return in;
}

__host__ __device__ inline
float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
float3 operator+(float3 a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__host__ __device__ inline
float3& operator+=(float3& a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__host__ __device__ inline
float3 operator-(float3 a, float3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__host__ __device__ inline
float3 operator*(float c, float3 v)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

__host__ __device__ inline
float3 operator*(float3 v, float c)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

__host__ __device__ inline
void linspace(float* out, float from, float to, std::size_t n)
{
  auto step = (to - from) / static_cast<float>(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
  {
    out[i] = from + static_cast<float>(i) * step;
  }
  out[n - 1] = to; // Don't rely on FP arithmetic to reach 'to'
}
