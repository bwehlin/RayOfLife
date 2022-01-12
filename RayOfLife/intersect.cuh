#pragma once

#include "support.cuh"
#include <cuda_runtime.h>
#include "scene_object.h"
#include "scene_data.cuh"
#include "camera.cuh"

namespace rol
{
  struct RayIntersection
  {
    fptype3 point;
    fptype3 normal;
    fptype3 color;
    bool hit;
  };

  __host__ __device__ RayIntersection traceRay(const fptype3& rayOrigin, const fptype3& rayDirection, const int3& gridPos, const SceneData& scene, const Camera& camera);
  __host__ __device__ fptype intersectSphere(const fptype3& rayOrigin, const fptype3& rayDirection, fptype3 sphereCenter, fptype sphereRadius);
}
