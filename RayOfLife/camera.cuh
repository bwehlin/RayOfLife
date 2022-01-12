#pragma once

#include "support.cuh"
#include <cuda_runtime.h>

namespace rol
{
  struct Camera
  {
    fptype3 origin;
    fptype3 direction;
  };

  inline Camera makeCamera(fptype3 origin, fptype3 direction)
  {
    Camera camera;
    camera.origin = origin;
    camera.direction = direction;
    return camera;
  }
}
