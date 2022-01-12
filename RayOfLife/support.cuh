#pragma once

#include <cuda_runtime.h>

#define FLOAT_WIDTH 64

#if FLOAT_WIDTH == 32

using fptype = float;
using fptype2 = float2;
using fptype3 = float3;

__device__ __host__ __inline__ fptype2 makeFp2(fptype x, fptype y)
{
  return make_float2(x, y);
}

__device__ __host__ __inline__ fptype3 makeFp3(fptype x, fptype y, fptype z)
{
  return make_float3(x, y, z);
}

#else

using fptype = double;
using fptype2 = double2;
using fptype3 = double3;

__device__ __host__ __inline__ fptype2 makeFp2(fptype x, fptype y)
{
  return make_double2(x, y);
}

__device__ __host__ __inline__ fptype3 makeFp3(fptype x, fptype y, fptype z)
{
  return make_double3(x, y, z);
}

#endif
