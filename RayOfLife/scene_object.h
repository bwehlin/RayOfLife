#pragma once

#include <cuda_runtime.h>
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
    std::vector<float3> pixels;
    int w;
  };

  struct PlaneData
  {
    float3 position;
    float3 normal;
    const Texture* texture;
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
    float diffuseC;
    float specularC;
    union
    {
      PlaneData plane;
      SphereData sphere;
    } data;
  };

  inline SceneObject makeSphere(float3 pos, float radius, float3 color)
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

  inline SceneObject makePlane(float3 pos, float3 normal)
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
