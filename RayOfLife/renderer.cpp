#include "renderer.h"

#include "vector_math.cuh"

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

#include <chrono>
#include <iostream>

rol::Renderer::Renderer(size_t w, size_t h)
  : m_width(w), m_height(h)
{

}

void
rol::Renderer::saveFrameBmp(const char* filename)
{
  auto w = width();
  auto h = height();

  boost::gil::rgb8_image_t image(w, h);
  auto const& dstImageView = boost::gil::view(image);

  const fptype3* data = imageData();

  // Sure, we could parallelize this, but in the grand scheme of things, how much
  // time will an in-memory copy+scaling of w*h pixels (parallelizable) take
  // compared to a memory->disk transfer (not parallelizable)?
  for (size_t y = 0ul; y < h; ++y)
  {
    auto baseIdx = y * w;
    auto yinv = h - 1 - y; // GIL images are 'upside down' compared to our view.
    for (size_t x = 0ul; x < w; ++x)
    {
      auto idx = baseIdx + x;
      
      auto fr = data[idx].x * 255.;
      auto fg = data[idx].y * 255.;
      auto fb = data[idx].z * 255.;

      auto r = std::clamp(static_cast<int>(fr), 0, 255);
      auto g = std::clamp(static_cast<int>(fg), 0, 255);
      auto b = std::clamp(static_cast<int>(fb), 0, 255);

      dstImageView(x, yinv) = boost::gil::rgb8_pixel_t(
        static_cast<unsigned char>(r),
        static_cast<unsigned char>(g),
        static_cast<unsigned char>(b));
    }
  }

  auto const& view = boost::gil::view(image);
  boost::gil::write_view(filename, view, boost::gil::bmp_tag());
}

void
rol::Renderer::render(const Game& game, const Camera& camera)
{
  auto aspect = aspectRatio();

  auto screenMin = makeFp2(camera.origin.y - 0.5f, camera.origin.z - 0.5f / aspect);
  auto screenMax = makeFp2(camera.origin.y + 0.5f, camera.origin.z + 0.5f / aspect);

  using clock = std::chrono::steady_clock;
  auto start = clock::now();
  produceFrame(game, camera, screenMin, screenMax);
  auto end = clock::now();

  using ftime = std::chrono::duration<double>;
  ftime frameTime = end - start;
  auto frameSeconds = frameTime.count();
  m_lastFrameSecs = frameSeconds;

  if (frameSeconds > 1.)
  {
    std::cout << "Frame time: " << frameSeconds << "s" << std::endl;
  }
  else if (frameSeconds > 1e-3)
  {
    std::cout << "Frame time: " << frameSeconds * 1e3 << "ms" << std::endl;
  }
  else
  {
    std::cout << "Frame time: " << frameSeconds * 1e6 << "us" << std::endl;
  }
}
