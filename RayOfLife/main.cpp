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
#include <png.h>
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

#include "scene_object.cuh"

struct Scene
{
  std::vector<rol::SceneObject> objects;

  // Default light and material parameters.
  float3 ambient;
  float diffuseC = 1.f;
  float specularC = 1.f;
  float specularK = 50.f;

  float3 lightPos;
  float3 lightColor;

  float3 cameraOrigin;
};

Scene makeScene()
{
  Scene scene;

  scene.cameraOrigin = make_float3(0.f, .35f, -1.f);
  scene.ambient = make_float3(0.05f, 0.05f, 0.05f);

  scene.lightPos = make_float3(5.f, 5.f, -10.f);
  scene.lightPos = make_float3(1.f, 1.f, 1.f);

  scene.objects.emplace_back(rol::makeSphere(make_float3(.75f, .1f, 1.f), .6f, make_float3(0.f, 0.f, 1.f)));
  scene.objects.emplace_back(rol::makeSphere(make_float3(-.75f, .1f, 2.25f), .6f, make_float3(.5f, .223f, .5f)));
  scene.objects.emplace_back(rol::makeSphere(make_float3(-2.75f, .1f, 3.5f), .6f, make_float3(1.f, .572f, .184f)));
  
  return scene;
}

float3 normalize(float3 in)
{
  auto norm = sqrtf(in.x * in.x + in.y * in.y + in.z * in.z);
  in.x /= norm;
  in.y /= norm;
  in.z /= norm;
  return in;
}

float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 operator+(float3 a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

float3& operator+=(float3& a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

float3 operator-(float3 a, float3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

float3 operator*(float c, float3 v)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

float3 operator*(float3 v, float c)
{
  v.x *= c;
  v.y *= c;
  v.z *= c;
  return v;
}

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
      std::swap(t0, t1);
    }
    if (t1 >= 0.)
    {
      return (t0 < 0.) ? t1 : t0;
    }
  }
  return std::numeric_limits<float>::infinity();
}

std::vector<float> linspace(float from, float to, std::size_t n)
{
  auto step = (to - from) / static_cast<float>(n);
  std::vector<float> ret;
  ret.resize(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
  {
    ret[i] = from + static_cast<float>(i) * step;
  }
  ret[n-1] = to; // Don't rely on FP arithmetic to reach 'to'
  return ret;
}

float intersect(float3 rayOrigin, float3 rayDirection, const rol::SceneObject& obj)
{
  switch (obj.type)
  {
  case rol::SceneObjectType::Sphere: return intersectSphere(rayOrigin, rayDirection, obj.data.sphere);
  }

  return std::numeric_limits<float>::infinity();
}

struct RayIntersection
{
  std::vector<rol::SceneObject>::const_iterator obj;
  float3 point;
  float3 normal;
  float3 color;
};

float3 getNormal(const rol::SceneObject& obj, float3 pos)
{
  switch (obj.type)
  {
  case rol::SceneObjectType::Sphere:
    return normalize(pos - obj.data.sphere.position);
  }
  return make_float3(0.f, 0.f, 1.f);
}

float3 getColor(const rol::SceneObject& obj, float3 /*pos*/)
{
  switch (obj.type)
  {
  case rol::SceneObjectType::Sphere:
    return obj.data.sphere.colorRgb;
  }
  return make_float3(1.f, 0.f, 0.f);
}

float getDiffuseC(const Scene& scene, const rol::SceneObject& obj)
{
  return scene.diffuseC;
}

float getSpecularC(const Scene& scene, const rol::SceneObject& obj)
{
  return scene.specularC;
}

RayIntersection traceRay(float3 rayOrigin, float3 rayDirection, const Scene& scene)
{
  RayIntersection intersection;

  // Find first point of intersection
  auto t = std::numeric_limits<float>::infinity();

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
    return intersect(shadowTestVec, toLight, obj) < std::numeric_limits<float>::infinity();
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
  intersection.color += make_float3(diffuse, diffuse, diffuse);

  auto specular = getSpecularC(scene, *intersection.obj) * std::fmax(dot(intersection.normal, normalize(toLight + toCameraOrigin)), 0.f);
  specular = std::powf(specular, scene.specularK);
  intersection.color += specular * scene.lightColor;

  return intersection;
}

void render(boost::gil::rgb8_image_t dstImage, const Scene& scene)
{
  auto aspectRatio = static_cast<float>(dstImage.width()) / static_cast<float>(dstImage.height());
  
  auto screenMin = make_float2(-1.f, -1.f / aspectRatio + .25f);
  auto screenMax = make_float2(1.f, 1.f / aspectRatio + .25f);

  auto wPixels = dstImage.width();
  auto hPixels = dstImage.height();

  auto xspace = linspace(screenMin.x, screenMax.x, wPixels);
  auto yspace = linspace(screenMin.y, screenMax.y, hPixels);

  auto maxDepth = 5; // Max reflections

  for (auto iy = 0ul; iy < hPixels; ++iy)
  {
    auto y = yspace[iy];
    for (auto ix = 0ul; ix < wPixels; ++ix)
    {
      auto x = xspace[ix];
      auto cameraTarget = make_float3(x, y, 0.f);

      auto rayOrigin = scene.cameraOrigin;
      auto rayDirection = normalize(cameraTarget - rayOrigin);

      auto color = make_float3(0.f, 0.f, 0.f);
      auto reflection = 1.f;
      
      for (auto depth = 0; depth < maxDepth; ++depth)
      {
        traceRay(rayOrigin, rayDirection, scene);
      }
    }
  }
}

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

int main(int, char**)
{
  boost::gil::rgb8_image_t image(1024, 768);
  auto const & view = boost::gil::view(image);

  for (int row = 0; row < 240; ++row)
  {
    for (int col = 0; col < 240; ++col)
    {
      view(row, col) = boost::gil::rgb8_pixel_t(0, 0, 0);
    }
  }

  for (int row = 0; row < 240; ++row)
  {
    for (int col = 0; col < 320 / 3; ++col)
    {
      view(col, row) = boost::gil::rgb8_pixel_t(row, 0, 0);
      view(col+320/3, row) = boost::gil::rgb8_pixel_t(0, row, 0);
      view(col+320/3*2, row) = boost::gil::rgb8_pixel_t(0, 0, row);
    }
  }
  
  boost::gil::write_view("test.bmp", view, boost::gil::bmp_tag());

  return EXIT_SUCCESS;
}
