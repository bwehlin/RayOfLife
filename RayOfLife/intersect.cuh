#pragma once

#include <cuda_runtime.h>
#include "scene_object.h"


__host__ __device__ float intersectSphere(float3 rayOrigin, float3 rayDirection, const rol::SphereData& sphere);
__host__ __device__ float intersectPlane(float3 rayOrigin, float3 rayDirection, const rol::PlaneData& plane);
