/*
This file contains a C++ port of the 'very simple ray tracing engine' at https://gist.github.com/rossant/6046463, which is licensed as follows:

MIT License
Copyright (c) 2017 Cyrille Rossant
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/


#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/bmp.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <boost/math/constants/constants.hpp>

#include "scene_object.h"
#include "intersect.cuh"
#include "vector_math.cuh"

#if 0
struct Scene
{
  std::vector<rol::SceneObject> objects;

  // Default light and material parameters.
  fptype3 ambient;
  fptype diffuseC = 1.f;
  fptype specularC = 1.f;
  fptype specularK = 50.f;

  fptype reflection = 1.f;

  fptype3 lightPos;
  fptype3 lightColor;

  fptype3 cameraOrigin;
};


Scene makeScene()
{
  Scene scene;

  scene.cameraOrigin = makeFp3(0.f, .35f, -1.f);
  scene.ambient = makeFp3(0.05f, 0.05f, 0.05f);

  scene.lightPos = makeFp3(5.f, 5.f, -10.f);
  scene.lightColor = makeFp3(1.f, 1.f, 1.f);

  scene.objects.emplace_back(rol::makeSphere(makeFp3(.75f, .1f, 1.f), .6f, makeFp3(0.f, 0.f, 1.f)));
  scene.objects.emplace_back(rol::makeSphere(makeFp3(-.75f, .1f, 2.25f), .6f, makeFp3(.5f, .223f, .5f)));
  scene.objects.emplace_back(rol::makeSphere(makeFp3(-2.75f, .1f, 3.5f), .6f, makeFp3(1.f, .572f, .184f)));

  scene.objects.emplace_back(rol::makePlane(makeFp3(0.f, -.5f, 0.f), makeFp3(0., 1., 0.)));

  return scene;
}

std::vector<fptype> linspace(fptype from, fptype to, std::size_t n)
{
  auto step = (to - from) / static_cast<fptype>(n);
  std::vector<fptype> ret;
  ret.resize(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
  {
    ret[i] = from + static_cast<fptype>(i) * step;
  }
  ret[n-1] = to; // Don't rely on FP arithmetic to reach 'to'
  return ret;
}

fptype intersect(fptype3 rayOrigin, fptype3 rayDirection, const rol::SceneObject& obj)
{
  switch (obj.type)
  {
  case rol::SceneObjectType::Sphere: return intersectSphere(rayOrigin, rayDirection, obj.data.sphere);
  case rol::SceneObjectType::Plane: return intersectPlane(rayOrigin, rayDirection, obj.data.plane);
  }

  return std::numeric_limits<fptype>::infinity();
}

struct RayIntersection
{
  std::vector<rol::SceneObject>::const_iterator obj;
  fptype3 point;
  fptype3 normal;
  fptype3 color;
};



RayIntersection traceRay(fptype3 rayOrigin, fptype3 rayDirection, const Scene& scene)
{
  RayIntersection intersection;

  // Find first point of intersection
  auto t = std::numeric_limits<fptype>::infinity();

  intersection.obj = scene.objects.end();
  for (auto it = scene.objects.begin(); it != scene.objects.end(); ++it)
  {
    auto tObj = intersect(rayOrigin, rayDirection, *it);
    if (tObj < t)
    {
      intersection.obj = it;
      t = tObj;
    }
  }

  if (intersection.obj == scene.objects.end())
  {
    return intersection;
  }

  intersection.point = rayOrigin + t * rayDirection;
  intersection.normal = getNormal(*intersection.obj, intersection.point);
  
  auto toLight = normalize(scene.lightPos - intersection.point);
  auto toCameraOrigin = normalize(scene.cameraOrigin - intersection.point);

  auto shadowTestVec = intersection.point + intersection.normal * 0.0001f;
  auto isShadowed = std::any_of(scene.objects.begin(), scene.objects.end(), [&shadowTestVec, &toLight](const rol::SceneObject& obj) {
    return intersect(shadowTestVec, toLight, obj) < std::numeric_limits<fptype>::infinity();
    });

  if (isShadowed)
  {
    // Shadowed
    intersection.obj = scene.objects.end();
    return intersection;
  }

  auto objColor = getColor(*intersection.obj, intersection.point);
  intersection.color = scene.ambient;
  
  auto diffuse = getDiffuseC(scene, *intersection.obj) * std::fmax(dot(intersection.normal, toLight), 0.f);
  auto diffuseColor = diffuse * objColor;
  intersection.color += diffuseColor;

  auto specular = getSpecularC(scene, *intersection.obj) * std::fmax(dot(intersection.normal, normalize(toLight + toCameraOrigin)), 0.f);
  specular = std::powf(specular, scene.specularK);
  intersection.color += specular * scene.lightColor;

  return intersection;
}

void render(boost::gil::rgb8_image_t& dstImage, const Scene& scene)
{
  auto aspectRatio = static_cast<fptype>(dstImage.width()) / static_cast<fptype>(dstImage.height());
  
  auto screenMin = makeFp2(-1.f, -1.f / aspectRatio + .25f);
  auto screenMax = makeFp2(1.f, 1.f / aspectRatio + .25f);

  auto wPixels = dstImage.width();
  auto hPixels = dstImage.height();

  auto xspace = linspace(screenMin.x, screenMax.x, wPixels);
  auto yspace = linspace(screenMin.y, screenMax.y, hPixels);

  auto maxDepth = 5; // Max reflections

  auto const& dstImageView = boost::gil::view(dstImage);

  for (auto iy = 0ul; iy < hPixels; ++iy)
  {
    auto y = yspace[iy];
    for (auto ix = 0ul; ix < wPixels; ++ix)
    {
      auto x = xspace[ix];
      auto cameraTarget = makeFp3(x, y, 0.f);

      auto rayOrigin = scene.cameraOrigin;
      auto rayDirection = normalize(cameraTarget - rayOrigin);

      auto color = makeFp3(0.f, 0.f, 0.f);
      auto reflection = 1.f;
      
      for (auto depth = 0; depth < maxDepth; ++depth)
      {
        auto intersection = traceRay(rayOrigin, rayDirection, scene);
        if (intersection.obj == scene.objects.end())
        {
          break;
        }

        rayOrigin = intersection.point + intersection.normal * 0.0001f;
        rayDirection = normalize(rayDirection - 2 * dot(rayDirection, intersection.normal) * intersection.normal);
        
        color += reflection * intersection.color;
        reflection *= getReflection(scene, *intersection.obj);
      }

      color.x = std::clamp(color.x, 0.f, 1.f) *255.f;
      color.y = std::clamp(color.y, 0.f, 1.f) *255.f;
      color.z = std::clamp(color.z, 0.f, 1.f) *255.f;
      
      dstImageView(ix, dstImage.height() - 1 - iy) = boost::gil::rgb8_pixel_t(
        static_cast<unsigned char>(color.x), 
        static_cast<unsigned char>(color.y),
        static_cast<unsigned char>(color.z));
    }
  }
}

#endif

#if 0
>> > def trace_ray(O, D) :
  # Find first point of intersection with the scene.
  t = intersect_sphere(O, D, position, radius)
  # No intersection ?
  if t == np.inf :
    return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient
    # Lambert shading(diffuse).
    col += diffuse * max(np.dot(N, toL), 0) * color
    # Blinn - Phong shading(specular).
    col += specular_c * color_light * \
    max(np.dot(N, normalize(toL + toO)), 0) \
    * *specular_k
    return col
#endif


#include "cpu_game.h"
#include "cpu_renderer.h"
#include "amantides_woo.cuh"

int main(int, char**)
{
  rol::CpuGame game(rol::make5766rule(), 16);

  auto camera = rol::makeCamera(makeFp3(-2.f, 0.f, 0.f), makeFp3(0.3f, 0.5f, 0.7f));

  rol::CpuRenderer renderer(640, 480);
  game.initRandomPrimordialSoup(2360);

  renderer.render(game, camera);

  renderer.saveFrameBmp("test.bmp");

  return EXIT_SUCCESS;
}
