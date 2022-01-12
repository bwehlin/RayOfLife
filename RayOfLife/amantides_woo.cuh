#pragma once

#include <cuda_runtime.h>
#include <cuda/std/limits>

namespace rol
{
  // We use the voxel traversal algorithm by Amantides and Woo.
  // Reference: Amanatides, John, and Andrew Woo. "A fast voxel traversal algorithm for ray tracing." Eurographics. Vol. 87. No. 3. 1987.

  struct AmantidesWooState
  {
    float3 tDelta;
    float3 tMax;
    int3 step;
    int3 pos;
  };

  __host__ __device__ AmantidesWooState initAmantidesWoo(float3 origin, float3 direction)
  {
    AmantidesWooState state;
    state.pos = make_int3(0, 0, 0); // TODO

    state.step.x = direction.x < 0 ? -1
      : direction.x > 0 ? 1
      : 0;

    state.step.y = direction.y < 0 ? -1
      : direction.y > 0 ? 1
      : 0;

    state.step.z = direction.z < 0 ? -1
      : direction.z > 0 ? 1
      : 0;

    state.tDelta.x = direction.x != 0.f ? 1.f / direction.x : cuda::std::numeric_limits<float>::infinity();
    state.tDelta.y = direction.y != 0.f ? 1.f / direction.y : cuda::std::numeric_limits<float>::infinity();
    state.tDelta.z = direction.z != 0.f ? 1.f / direction.z : cuda::std::numeric_limits<float>::infinity();

    state.tMax.x = 1.f / direction.x; // TODO
    state.tMax.y = 1.f / direction.y; // TODO
    state.tMax.z = 1.f / direction.z; // TODO

    return state;
  }

  __host__ __device__ void nextAwStep(AmantidesWooState& state)
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
