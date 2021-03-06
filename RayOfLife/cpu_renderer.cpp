#include "cpu_renderer.h"
#include "support.cuh"

#include <stdexcept>

#include "vector_math.cuh"
#include "amantides_woo.cuh"
#include "intersect.cuh"
#include "game.cuh"

rol::CpuRenderer::CpuRenderer(size_t w, size_t h)
  : Renderer(w, h)
{
  m_imageData = std::make_unique<fptype3[]>(w * h);
}

const fptype3*
rol::CpuRenderer::imageData() const
{
  return m_imageData.get();
}

void
rol::CpuRenderer::produceFrame(const Game& game, const Camera& camera,
  const fptype2& screenMin, const fptype2& screenMax)
{
  auto wPixels = width();
  auto hPixels = height();
  auto subpixels = subpixelCount();

  std::vector<fptype> xspace(wPixels * subpixels);
  std::vector<fptype> yspace(hPixels * subpixels);

  linspace(xspace.data(), screenMin.x, screenMax.x, wPixels * subpixels);
  linspace(yspace.data(), screenMin.y, screenMax.y, hPixels * subpixels);


#pragma omp parallel for
  for (auto iy = 0; iy < hPixels; ++iy)
  {
    for (auto ix = 0; ix < wPixels; ++ix)
    {
      fptype3 pixelval = makeFp3(0.f, 0.f, 0.f);

      for (auto subpixel = 0; subpixel < subpixels; ++subpixel)
      {
        for (auto subpixel2 = 0; subpixel2 < subpixels; ++subpixel2)
        {
          pixelval += renderPixel(ix, iy, xspace[ix * subpixels + subpixel], yspace[iy * subpixels + subpixel2], game, camera);
        }
      }
      pixelval.x /= static_cast<fptype>(subpixels * subpixels);
      pixelval.y /= static_cast<fptype>(subpixels * subpixels);
      pixelval.z /= static_cast<fptype>(subpixels * subpixels);

      m_imageData[iy * wPixels + ix] = pixelval;
    }
  }
}

/*
This file started its life as a C++ port of the 'very simple ray tracing engine'
available at https://gist.github.com/rossant/6046463, which is licensed as follows:

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

// TODO: Move to use equivalent __host__ __device__ functions.

fptype3
rol::CpuRenderer::renderPixel(itype ix, itype iy, 
  fptype x, fptype y, 
  const Game& game, const Camera& camera)
{
  auto rayOrigin = camera.origin;
  auto cameraTarget = makeFp3(camera.origin.x + 1.f, x, y); // TODO: Viewing planes...

  auto rayDirection = normalize(cameraTarget - rayOrigin);

  auto color = makeFp3(0.f, 0.f, 0.f);
  fptype reflection = 1.f;

  auto cellsPerDim = game.cellsPerDim();
  AmantidesWooState awstate;
  rol::initAmantidesWoo(awstate, rayOrigin, rayDirection, cellsPerDim);
  if (awstate.pos.x != 0)
  {
    // Ray from origin does not hit cell grid
    return color;
  }

  auto depth = maxDepth();
  while (depth--)
  {
    auto intersection = castRay(awstate, rayOrigin, rayDirection, game, camera);
    if (!intersection.hit)
    {
      break;
    }

    rayOrigin = intersection.point + intersection.normal * static_cast<fptype>(0.0001f);
    rayDirection = normalize(rayDirection - 2 * dot(rayDirection, intersection.normal) * intersection.normal);

    rol::initAmantidesWoo(awstate, rayOrigin, rayDirection, cellsPerDim);
    rol::nextAwStep(awstate);

    color += reflection * intersection.color;
    reflection *= m_scene.sphereReflection;
  }

  return color;
}

rol::RayIntersection
rol::CpuRenderer::castRay(AmantidesWooState& awstate, fptype3 rayOrigin, fptype3 rayDirection, const Game& game, const Camera& camera)
{
  auto cellsPerDim = game.cellsPerDim();
  while (true)
  {
    if (awstate.pos.x < 0 || awstate.pos.x >= cellsPerDim
      || awstate.pos.y < 0 || awstate.pos.y >= cellsPerDim
      || awstate.pos.z < 0 || awstate.pos.z >= cellsPerDim)
    {
      // We have fallen out of the cell grid
      RayIntersection intersection;
      intersection.hit = false;
      return intersection;
    }

    if (game.isAlive(awstate.pos.x, awstate.pos.y, awstate.pos.z))
    {
      auto intersection = traceRay(rayOrigin, rayDirection, awstate.pos, m_scene, camera.origin);
      if (intersection.hit)
      {
        return intersection;
      }
    }

    nextAwStep(awstate);
  }

  RayIntersection intersection;
  intersection.hit = false;
  return intersection;
}
