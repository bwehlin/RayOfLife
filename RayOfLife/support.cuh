#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// FLOAT_WIDTH is defined in project build config

#ifndef FLOAT_WIDTH
#warning("FLOAT_WIDTH not set!")
#endif

#ifndef INT_WIDTH
#define INT_WIDTH 32
#endif

#if INT_WIDTH == 16
using itype = short;
using itype3 = short3;
#elif INT_WIDTH == 32
using itype = int;
using itype3 = int3;
#elif INT_WIDTH == 64
using itype = long int;
using itype3 = longlong3;
#endif

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

namespace rol
{
  // Apparently, there is no cuda::std::min/max as far as I can tell...

  __device__ __host__ __inline__ fptype min(fptype a, fptype b)
  {
    return a < b ? a : b;
  }

  __device__ __host__ __inline__ fptype max(fptype a, fptype b)
  {
    return a > b ? a : b;
  }

  class CudaError : public ::std::runtime_error
  {
  public:
    CudaError(cudaError_t err, const char* file, itype line)
      : ::std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + cudaGetErrorString(err))
    { }
  };
}

#define CHK_ERR(op) { cudaError_t err = op; if (err != cudaSuccess) { throw CudaError(err, __FILE__, __LINE__); } }
