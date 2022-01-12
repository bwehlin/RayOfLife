#pragma once

#include "support.cuh"

namespace rol
{
  struct SceneData
  {
    // Sphere material properties

    fptype3 sphereColor = { 1.f, .572f, .184f };

    fptype sphereDiffuseC = 1.f;
    fptype sphereSpecularC = 1.f;
    fptype sphereReflection = 0.5f;

    // Sphere geometry

    fptype sphereRaidus = .1f;

    // Lighting

    fptype3 ambient = makeFp3(0.05f, 0.05f, 0.05f);

    fptype3 lightPos = makeFp3(5.f, 5.f, -10.f);
    fptype3 lightColor = makeFp3(1.f, 1.f, 1.f);

    // Base specular

    fptype specularK = 50.f;
  };
}
