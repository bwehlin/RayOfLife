#pragma once

#include <cuda_runtime.h>

namespace rol
{
  enum class SceneObjectType
  {
    Sphere = 1,
    Plane = 2
  };

  struct PlaneData
  {
    float3 position;
    float3 normal;
  };

  struct SphereData
  {
    float3 position;
    float3 colorRgb;
    float radius;
  };

  struct SceneObject
  {
    SceneObjectType type;
    float reflection;
    union
    {
      PlaneData plane;
      SphereData sphere;
    } data;
  };

  SceneObject makeSphere(float3 pos, float radius, float3 color)
  {
    SceneObject ret;
    ret.type = SceneObjectType::Sphere;
    ret.reflection = 0.5f;
    ret.data.sphere.position = pos;
    ret.data.sphere.radius = radius;
    ret.data.sphere.colorRgb = color;
    return ret;
  }
}
