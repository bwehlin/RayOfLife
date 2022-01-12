#pragma once

#include "support.cuh"
#include <cuda_runtime.h>
#include "scene_object.h"


__host__ __device__ fptype intersectSphere(fptype3 rayOrigin, fptype3 rayDirection, const rol::SphereData& sphere);
__host__ __device__ fptype intersectPlane(fptype3 rayOrigin, fptype3 rayDirection, const rol::PlaneData& plane);
