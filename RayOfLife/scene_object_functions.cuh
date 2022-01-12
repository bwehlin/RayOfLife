#pragma once

#include "scene_object.h"
#include "vector_math.cuh"
#include "support.cuh"

namespace rol
{

  __host__ __device__ inline
    fptype3 getNormal(const rol::SceneObject& obj, fptype3 pos)
  {
    switch (obj.type)
    {
    case rol::SceneObjectType::Sphere:
      return normalize(pos - obj.data.sphere.position);
    case rol::SceneObjectType::Plane:
      return normalize(obj.data.plane.normal);
    }
    return makeFp3(0.f, 0.f, 1.f);
  }

#if 0
  __host__ __device__ inline
    fptype3 cross(fptype3 a, fptype3 b)
  {
    return makeFp3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
  }
#endif

  __host__ __device__ inline
    fptype3 getColor(const rol::PlaneData& plane, fptype3 pos)
  {
    // if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),

    return (static_cast<int>(pos.x * 2.) % 2 == static_cast<int>(pos.z * 2.) % 2) ? makeFp3(1.f, 1.f, 1.f) : makeFp3(0.f, 0.f, 0.f);

    //return makeFp3(1.f, 1.f, 1.f);

    fptype3 referenceVec;
    referenceVec.y = 1.f;
    referenceVec.z = 1.f;
    referenceVec.x = -(plane.normal.y * referenceVec.y + plane.normal.z * referenceVec.z) / plane.normal.x;

    fptype3 posInPlane = pos - plane.position;

    auto x = dot(posInPlane, referenceVec);
    auto y = norm(posInPlane) / x;

    auto angle = cuda::std::acos(x / (norm(posInPlane) * norm(referenceVec)));
    if (angle < 0)
    {
      y = -y;
    }

    // w = 1.f
    auto w = plane.texture->w;
    auto h = plane.texture->pixels.size() / w;

    unsigned long ix = static_cast<int>(x * static_cast<fptype>(w) * 1000.f) % w;
    unsigned long iy = static_cast<int>(y * static_cast<fptype>(h) * 1000.f) % h;

    if (ix < 0)
    {
      ix = w + ix;
    }
    if (iy < 0)
    {
      iy = h + iy;
    }

    return plane.texture->pixels[iy * w + ix];
  }

  __host__ __device__ inline
    fptype3 getColor(const rol::SceneObject& obj, fptype3 pos)
  {
    switch (obj.type)
    {
    case rol::SceneObjectType::Sphere:
      return obj.data.sphere.colorRgb;
    case rol::SceneObjectType::Plane:
      return getColor(obj.data.plane, pos);
    }
    return makeFp3(1.f, 0.f, 0.f);
  }

  __host__ __device__ inline
    fptype getDiffuseC(const Scene& scene, const rol::SceneObject& obj)
  {
    return obj.diffuseC;
  }

  __host__ __device__ inline
    fptype getSpecularC(const Scene& scene, const rol::SceneObject& obj)
  {
    return obj.specularC;
  }

  __host__ __device__ inline
    fptype getReflection(const Scene& scene, const rol::SceneObject& obj)
  {
    return obj.reflection;
  }

}