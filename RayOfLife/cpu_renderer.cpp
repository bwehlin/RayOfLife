#include "cpu_renderer.h"
#include "support.cuh"
#include <stdexcept>

#include "vector_math.cuh"

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
      renderPixel(ix, iy, xspace, yspace, camera);
    }
  }
}

void
rol::CpuRenderer::renderPixel(int ix, int iy, 
  const std::vector<fptype>& xspace, const std::vector<fptype>& yspace, 
  const Camera& camera)
{
  auto y = yspace[iy];
  auto x = xspace[ix];

  auto rayOrigin = camera.origin;
  auto cameraTarget = makeFp3(x, y, 0.f); // TODO: Viewing planes...

  auto rayDirection = normalize(cameraTarget - rayOrigin);

  auto color = makeFp3(0.f, 0.f, 0.f);
  auto reflection = 1.f;

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
