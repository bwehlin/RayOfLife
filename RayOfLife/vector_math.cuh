#pragma once

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
