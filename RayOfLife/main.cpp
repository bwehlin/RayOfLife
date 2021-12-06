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

#include <boost/math/constants/constants.hpp>

#include "scene_object.h"



struct Scene
{
  std::vector<rol::SceneObject> objects;

  // Default light and material parameters.
  float3 ambient;
  float diffuseC = 1.f;
  float specularC = 1.f;
  float specularK = 50.f;

  float reflection = 1.f;

  float3 lightPos;
  float3 lightColor;

  float3 cameraOrigin;

  // std::list gives iterator stability on emplace_back.
  // Data locality is within a single texture anyway.
  std::list<rol::Texture> textures;
};

rol::Texture* readTexture(Scene& scene, const char* path)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);

  boost::gil::image_read_settings<boost::gil::bmp_tag> settings;
  boost::gil::rgb8_image_t img;
  boost::gil::read_image(in, img, settings);

  auto const& view = boost::gil::view(img);

  auto & texture = scene.textures.emplace_back();

  texture.w = img.width();
  texture.pixels.resize(img.width() * img.height());

  for (auto iy = img.height() - 1; iy != -1; --iy)
  {
    for (auto ix = 0; ix < img.width(); ++ix)
    {
      auto& pixel = texture.pixels[iy * texture.w + ix];
      
      auto pixel8 = view(ix, iy);
      pixel.x = pixel8[0];
      pixel.y = pixel8[1];
      pixel.z = pixel8[2];
    }
  }

  return &texture;
}

Scene makeScene()
{
  Scene scene;

  scene.cameraOrigin = make_float3(0.f, .35f, -1.f);
  scene.ambient = make_float3(0.05f, 0.05f, 0.05f);

  scene.lightPos = make_float3(5.f, 5.f, -10.f);
  scene.lightColor = make_float3(1.f, 1.f, 1.f);

  scene.objects.emplace_back(rol::makeSphere(make_float3(.75f, .1f, 1.f), .6f, make_float3(0.f, 0.f, 1.f)));
  scene.objects.emplace_back(rol::makeSphere(make_float3(-.75f, .1f, 2.25f), .6f, make_float3(.5f, .223f, .5f)));
  scene.objects.emplace_back(rol::makeSphere(make_float3(-2.75f, .1f, 3.5f), .6f, make_float3(1.f, .572f, .184f)));
  
  auto* groundTexture = readTexture(scene, "groundtexture.bmp");
  scene.objects.emplace_back(rol::makePlane(make_float3(0.f, -.5f, 0.f), make_float3(0., 1., 0.), groundTexture));

  return scene;
}

float norm(float3 in)
{
  return sqrtf(in.x * in.x + in.y * in.y + in.z * in.z);
}

float3 normalize(float3 in)
{
  auto normVal = norm(in);
  in.x /= normVal;
  in.y /= normVal;
  in.z /= normVal;
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

float intersectPlane(float3 rayOrigin, float3 rayDirection, const rol::PlaneData& plane)
{
  auto denom = dot(rayDirection, plane.normal);
  if (fabsf(denom) < 1e-6f)
  {
    return std::numeric_limits<float>::infinity();
  }
  auto d = dot(plane.position - rayOrigin, plane.normal) / denom;
  if (d < 0.f)
  {
    return std::numeric_limits<float>::infinity();
  }
  return d;
#if 0
  def intersect_plane(O, D, P, N) :
  # Return the distance from O to the intersection of the ray(O, D) with the
    # plane(P, N), or +inf if there is no intersection.
    # Oand P are 3D points, Dand N(normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
  return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0 :
      return np.inf
      return d
#endif
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
  case rol::SceneObjectType::Plane: return intersectPlane(rayOrigin, rayDirection, obj.data.plane);
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
  case rol::SceneObjectType::Plane:
    return normalize(obj.data.plane.normal);
  }
  return make_float3(0.f, 0.f, 1.f);
}

#if 0
float3 cross(float3 a, float3 b)
{
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x);
}
#endif

float3 getColor(const rol::PlaneData& plane, float3 pos)
{
  // if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),

  return (static_cast<int>(pos.x * 2.) % 2 == static_cast<int>(pos.z * 2.) % 2) ? make_float3(1.f, 1.f, 1.f) : make_float3(0.f, 0.f, 0.f);

  //return make_float3(1.f, 1.f, 1.f);

  float3 referenceVec;
  referenceVec.y = 1.f;
  referenceVec.z = 1.f;
  referenceVec.x = -(plane.normal.y * referenceVec.y + plane.normal.z * referenceVec.z) / plane.normal.x;

  float3 posInPlane = pos - plane.position;

  auto x = dot(posInPlane, referenceVec);
  auto y = norm(posInPlane) / x;
  
  auto angle = std::acosf(x / (norm(posInPlane) * norm(referenceVec)));
  if (angle < 0)
  {
    y = -y;
  }

  // w = 1.f
  auto w = plane.texture->w;
  auto h = plane.texture->pixels.size() / w;

  int ix = static_cast<int>(x * static_cast<float>(w) * 1000.f) % w;
  int iy = static_cast<int>(y * static_cast<float>(h) * 1000.f) % h;
  
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

float3 getColor(const rol::SceneObject& obj, float3 pos)
{
  switch (obj.type)
  {
  case rol::SceneObjectType::Sphere:
    return obj.data.sphere.colorRgb;
  case rol::SceneObjectType::Plane:
    return getColor(obj.data.plane, pos);
  }
  return make_float3(1.f, 0.f, 0.f);
}

float getDiffuseC(const Scene& scene, const rol::SceneObject& obj)
{
  return obj.diffuseC;
}

float getSpecularC(const Scene& scene, const rol::SceneObject& obj)
{
  return obj.specularC;
}

float getReflection(const Scene& scene, const rol::SceneObject& obj)
{
  return obj.reflection;
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
  auto diffuseColor = diffuse * objColor;
  intersection.color += diffuseColor;

  auto specular = getSpecularC(scene, *intersection.obj) * std::fmax(dot(intersection.normal, normalize(toLight + toCameraOrigin)), 0.f);
  specular = std::powf(specular, scene.specularK);
  intersection.color += specular * scene.lightColor;

  return intersection;
}

void render(boost::gil::rgb8_image_t& dstImage, const Scene& scene)
{
  auto aspectRatio = static_cast<float>(dstImage.width()) / static_cast<float>(dstImage.height());
  
  auto screenMin = make_float2(-1.f, -1.f / aspectRatio + .25f);
  auto screenMax = make_float2(1.f, 1.f / aspectRatio + .25f);

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
      auto cameraTarget = make_float3(x, y, 0.f);

      auto rayOrigin = scene.cameraOrigin;
      auto rayDirection = normalize(cameraTarget - rayOrigin);

      auto color = make_float3(0.f, 0.f, 0.f);
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

      dstImageView(ix, dstImage.height() - 1 - iy) = boost::gil::rgb8_pixel_t(color.x, color.y, color.z);
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
  
  render(image, makeScene());

  boost::gil::write_view("test.bmp", view, boost::gil::bmp_tag());

  return EXIT_SUCCESS;
}
