#include "intersect.cuh"
#include "vector_math.cuh"

#include <cuda/std/utility>
#include <cuda/std/limits>

__host__ __device__
float intersectSphere(float3 rayOrigin, float3 rayDirection, const rol::SphereData& sphere)
{
  auto sphereCenter = sphere.position;
  auto sphereRadius = sphere.radius;

  auto a = dot(rayDirection, rayDirection);
  auto os = rayOrigin - sphereCenter;
  auto b = 2.f * dot(rayDirection, os);
  auto c = dot(os, os) - sphereRadius * sphereRadius;
  auto disc = b * b - 4 * a * c;
  if (disc > 0.f)
  {
    auto distSqrt = sqrtf(disc);
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
  return cuda::std::numeric_limits<float>::infinity();
}

__host__ __device__
float intersectPlane(float3 rayOrigin, float3 rayDirection, const rol::PlaneData& plane)
{
  auto denom = dot(rayDirection, plane.normal);
  if (fabsf(denom) < 1e-6f)
  {
    return cuda::std::numeric_limits<float>::infinity();
  }
  auto d = dot(plane.position - rayOrigin, plane.normal) / denom;
  if (d < 0.f)
  {
    return cuda::std::numeric_limits<float>::infinity();
  }
  return d;
}
