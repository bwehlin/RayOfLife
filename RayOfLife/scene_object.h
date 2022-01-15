#pragma once

#include <cuda_runtime.h>
#include "support.cuh"
#include <vector>

namespace rol
{
  enum class SceneObjectType
  {
    Sphere = 1,
    Plane = 2
  };

  struct Texture
  {
    std::vector<fptype3> pixels;
    itype w;
  };

  struct PlaneData
  {
    fptype3 position;
    fptype3 normal;
    const Texture* texture;
  };

  struct SphereData
  {
    fptype3 position;
    fptype3 colorRgb;
    fptype radius;
  };

  struct SceneObject
  {
    SceneObjectType type;
    fptype reflection;
    fptype diffuseC;
    fptype specularC;
    union
    {
      PlaneData plane;
      SphereData sphere;
    } data;
  };

  inline SceneObject makeSphere(fptype3 pos, fptype radius, fptype3 color)
  {
    SceneObject ret;
    ret.type = SceneObjectType::Sphere;
    ret.reflection = 0.5f;
    ret.specularC = 1.f;
    ret.diffuseC = 1.f;
    ret.data.sphere.position = pos;
    ret.data.sphere.radius = radius;
    ret.data.sphere.colorRgb = color;
    return ret;
  }

  inline SceneObject makePlane(fptype3 pos, fptype3 normal)
  {
    SceneObject ret;
    ret.type = SceneObjectType::Plane;
    ret.reflection = .25f;
    ret.diffuseC = .75f;
    ret.specularC = .5f;
    ret.data.plane.position = pos;
    ret.data.plane.normal = normal;
    return ret;
  }
}
