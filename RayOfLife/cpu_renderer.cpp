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
rol::CpuRenderer::render(const Game& game, const Camera& camera)
{
  auto aspect = aspectRatio();

  auto screenMin = makeFp2(-1.f, -1.f / aspect + .25f);
  auto screenMax = makeFp2(1.f, 1.f / aspect + .25f);

  auto wPixels = width();
  auto hPixels = height();

  std::vector<fptype> xspace(wPixels);
  std::vector<fptype> yspace(hPixels);

  linspace(xspace.data(), screenMin.x, screenMax.x, wPixels);
  linspace(yspace.data(), screenMin.y, screenMax.y, hPixels);

  for (auto iy = 0ul; iy < hPixels; ++iy)
  {
    for (auto ix = 0ul; ix < wPixels; ++ix)
    {
      renderPixel(ix, iy, xspace, yspace, game, camera);
    }
  }
}

void
rol::CpuRenderer::renderPixel(int ix, int iy, 
  const std::vector<fptype>& xspace, const std::vector<fptype>& yspace, 
  const Game& game, const Camera& camera)
{
  auto y = yspace[iy];
  auto x = xspace[ix];

  auto rayOrigin = camera.origin;
  auto cameraTarget = makeFp3(0.f, x, y); // TODO: Viewing planes...

  auto rayDirection = normalize(cameraTarget - rayOrigin);

  auto color = makeFp3(0.f, 0.f, 0.f);
  auto reflection = 1.f;

  auto& pixel = m_imageData[iy * width() + ix];
  pixel.x = 0;
  pixel.y = 0;
  pixel.z = 0;

  auto cellsPerDim = game.cellsPerDim();
  auto awstate = rol::initAmantidesWoo(camera.origin, rayDirection, cellsPerDim);
  if (awstate.pos.x != 0)
  {
    // Ray from origin does not hit cell grid
    return;
  }

  auto reflection = 1.f;

  auto depth = maxDepth();
  while (depth--)
  {
    auto intersection = castRay(awstate, rayOrigin, rayDirection, game, camera);
    if (!intersection.hit)
    {

    }
  }

#if 0
  for (auto depth = 0; depth < maxDepth(); ++depth)
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

  color.x = std::clamp(color.x, 0.f, 1.f) * 255.f;
  color.y = std::clamp(color.y, 0.f, 1.f) * 255.f;
  color.z = std::clamp(color.z, 0.f, 1.f) * 255.f;

  dstImageView(ix, dstImage.height() - 1 - iy) = boost::gil::rgb8_pixel_t(
    static_cast<unsigned char>(color.x),
    static_cast<unsigned char>(color.y),
    static_cast<unsigned char>(color.z));
#endif
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
      return;
    }

    if (game.isAlive(awstate.pos.x, awstate.pos.y, awstate.pos.z))
    {
      auto intersection = traceRay(rayOrigin, rayDirection, awstate.pos, m_scene, camera);
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
