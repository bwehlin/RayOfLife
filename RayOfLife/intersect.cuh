#pragma once

#include "support.cuh"
#include <cuda_runtime.h>
#include "scene_data.cuh"
#include "camera.cuh"
#include "vector_math.cuh"

#include <cuda/std/utility>
#include <cuda/std/limits>
#include <cuda/std/utility>

namespace rol
{
  struct RayIntersection
  {
    fptype3 point;
    fptype3 normal;
    fptype3 color;
    bool hit;
  };

  __host__ __device__ __inline__ fptype intersectSphere(const fptype3& rayOrigin, const fptype3& rayDirection, fptype3 sphereCenter, fptype sphereRadius)
  {
    auto a = dot(rayDirection, rayDirection);
    auto os = rayOrigin - sphereCenter;
    auto b = 2.f * dot(rayDirection, os);
    auto c = dot(os, os) - sphereRadius * sphereRadius;
    auto disc = b * b - 4 * a * c;
    if (disc > 0.f)
    {
      auto distSqrt = sqrt(disc);
      auto q = b < 0. ? (-b - distSqrt) / 2.f : (-b + distSqrt) / 2.f;
      auto t0 = q / a;
      auto t1 = c / q;
      if (t0 > t1)
      {
        cuda::std::swap(t0, t1);
      }
      if (t1 >= 0.)
      {
        return (t0 < 0.) ? t1 : t0;
      }
    }
    return cuda::std::numeric_limits<fptype>::infinity();
  }

  __host__ __device__ __inline__ RayIntersection traceRay(const fptype3& rayOrigin, const fptype3& rayDirection, const int3& gridPos, const SceneData& scene, const fptype3& cameraOrigin)
  {
    RayIntersection intersection;

    fptype3 sphereCenter;
    sphereCenter.x = static_cast<fptype>(gridPos.x) + .5f;
    sphereCenter.y = static_cast<fptype>(gridPos.y) + .5f;
    sphereCenter.z = static_cast<fptype>(gridPos.z) + .5f;

    auto sphereRadius = scene.sphereRaidus;

    auto t = intersectSphere(rayOrigin, rayDirection, sphereCenter, sphereRadius);
    if (t == cuda::std::numeric_limits<fptype>::infinity())
    {
      intersection.hit = false;
      return intersection;
    }

    intersection.point = rayOrigin + t * rayDirection;
    intersection.normal = normalize(intersection.point - sphereCenter);

    auto toLight = normalize(scene.lightPos - intersection.point);
    auto toCameraOrigin = normalize(cameraOrigin - intersection.point);

    auto objColor = scene.sphereColor;
    intersection.color = scene.ambient;

    auto diffuse = scene.sphereDiffuseC * rol::max(dot(intersection.normal, toLight), 0.f);
    auto diffuseColor = diffuse * objColor;
    intersection.color += diffuseColor;

    auto specular = scene.sphereSpecularC * rol::max(dot(intersection.normal, normalize(toLight + toCameraOrigin)), 0.f);

    // TODO: pow GPU
    specular = pow(specular, scene.specularK);
    intersection.color += specular * scene.lightColor;

    intersection.hit = true;

    return intersection;
  }
}
