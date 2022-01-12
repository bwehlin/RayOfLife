#pragma once

#include <cuda_runtime.h>

namespace rol
{
  struct Camera
  {
    float3 origin;
    float3 direction;
  };

  inline Camera makeCamera(float3 origin, float3 direction)
  {
    Camera camera;
    camera.origin = origin;
    camera.direction = direction;
    return camera;
  }
}
