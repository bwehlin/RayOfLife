#pragma once

#include <cuda_runtime.h>

namespace rol
{
  struct Camera
  {
    float3 origin;
    float3 target;
  };

  inline Camera makeCamera(float3 origin, float3 target)
  {
    Camera camera;
    camera.origin = origin;
    camera.target = target;
    return camera;
  }
}
