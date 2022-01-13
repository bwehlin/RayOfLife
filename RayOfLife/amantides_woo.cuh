#pragma once

#include "support.cuh"

#include <cuda_runtime.h>
#include <cuda/std/limits>

namespace rol
{
  // We use the voxel traversal algorithm by Amantides and Woo.
  // Reference: Amanatides, John, and Andrew Woo. "A fast voxel traversal algorithm for ray tracing." Eurographics. Vol. 87. No. 3. 1987.

  struct AmantidesWooState
  {
    fptype3 tDelta;
    fptype3 tMax;
    int3 step;
    int3 pos;
  };

  __host__ __device__ __inline__ AmantidesWooState initAmantidesWoo(fptype3 origin, fptype3 direction, int nCellsPerDim)
  {
    AmantidesWooState state;

    fptype maxBoundaryCoord = static_cast<fptype>(nCellsPerDim) + 1.f;

    state.pos = make_int3(0, 0, 0);
    fptype tMin = 0;

    // Assume we are always viewing the cube from  (-inf, 0) x [0, nCellsPerDim] x [0, nCellsPerDim].
    // This way we can only hit the side of the cube that lies along the yz plane. If we don't hit
    // this plane with the ray, we don't hit the cube at all.

    if (origin.x >= 0.f && origin.x < maxBoundaryCoord)
    {
      // Ray starts inside cell cube. We don't bother checking for y and z because
      // if x is inside the cube, then we have entered the cube earlier and it means
      // we're bouncing around inside the cube already.

      state.pos.x = static_cast<int>(origin.x); // Assuming unit length for the cells
      state.pos.y = static_cast<int>(origin.y);
      state.pos.z = static_cast<int>(origin.z);
    }
    else
    {
      // x0 + tv = 0 => t = -x0/v

      tMin = -origin.x / direction.x;
      
      auto y = origin.y + tMin * direction.y;
      auto z = origin.z + tMin * direction.z;

      if (y >= 0.f && y < maxBoundaryCoord
        && z >= 0.f && z < maxBoundaryCoord)
      {
        state.pos.x = 0;
        state.pos.y = static_cast<int>(y);
        state.pos.z = static_cast<int>(z);
      }
      else
      {
        state.pos.x = cuda::std::numeric_limits<int>::max();
        return state;
      }
    }

    fptype targetX, targetY, targetZ;

    if (direction.x < 0)
    {
      state.step.x = -1;
      targetX = state.pos.x - 1;
    }
    else
    {
      state.step.x = 1;
      targetX = state.pos.x;
    }

    if (direction.y < 0)
    {
      state.step.y = -1;
      targetY = state.pos.y - 1;
    }
    else
    {
      state.step.y = 1;
      targetY = state.pos.y;
    }

    if (direction.z < 0)
    {
      state.step.z = -1;
      targetZ = state.pos.z - 1;
    }
    else
    {
      state.step.z = 1;
      targetZ = state.pos.z;
    }

    if (direction.x != 0.f)
    {
      state.tDelta.x = fptype{ 1.f } / direction.x;
      state.tMax.x = (fptype{ 1.f } + targetX - origin.x) / direction.x;
    }
    else
    {
      state.tDelta.x = cuda::std::numeric_limits<fptype>::infinity();
      state.tMax.x = cuda::std::numeric_limits<fptype>::infinity();
    }

    if (direction.y != 0.f)
    {
      state.tDelta.y = fptype{ 1.f } / direction.y;
      state.tMax.y = (fptype{ 1.f } + targetY - origin.y) / direction.y;
    }
    else
    {
      state.tDelta.y = cuda::std::numeric_limits<fptype>::infinity();
      state.tMax.y = cuda::std::numeric_limits<fptype>::infinity();
    }

    if (direction.z != 0.f)
    {
      state.tDelta.z = fptype{ 1.f } / direction.z;
      state.tMax.z = (fptype{ 1.f } + targetZ - origin.z) / direction.z;
    }
    else
    {
      state.tDelta.z = cuda::std::numeric_limits<fptype>::infinity();
      state.tMax.z = cuda::std::numeric_limits<fptype>::infinity();
    }

    if (state.tDelta.x < 0)
    {
      state.tDelta.x = -state.tDelta.x;
    }
    if (state.tDelta.y < 0)
    {
      state.tDelta.y = -state.tDelta.y;
    }
    if (state.tDelta.z < 0)
    {
      state.tDelta.z = -state.tDelta.z;
    }


    return state;
  }

  __host__ __device__ __inline__ void nextAwStep(AmantidesWooState& state)
  {
    if (state.tMax.x < state.tMax.y)
    {
      if (state.tMax.x < state.tMax.z)
      {
        state.pos.x += state.step.x;
        state.tMax.x += state.tDelta.x;
      }
      else
      {
        state.pos.z += state.step.z;
        state.tMax.z += state.tDelta.z;
      }
    }
    else
    {
      if (state.tMax.y < state.tMax.z)
      {
        state.pos.y += state.step.y;
        state.tMax.y += state.tDelta.y;
      }
      else
      {
        state.pos.z += state.step.z;
        state.tMax.z += state.tDelta.z;
      }
    }
  }
}
